"""Autoregressive selection of three distinct residual query vertices."""
from math import log1p

import torch
from torch import nn
from torch.distributions import Categorical


class TriplePointer(nn.Module):
    def __init__(self, device, anchor_dim=0, width=32, architecture="attention"):
        super().__init__()
        self.token = nn.Linear(5, width)
        self.architecture = architecture
        if architecture == "attention":
            self.relations = nn.Linear(2, 1)
            self.encoder = nn.TransformerEncoderLayer(width, 4, width*2, dropout=0., batch_first=True)
        else:
            self.relations = nn.Sequential(nn.Linear(2, width), nn.ReLU())
            self.encoder = nn.Sequential(nn.Linear(width*2, width*2), nn.ReLU(),
                                         nn.Linear(width*2, width))
            if architecture == "message":
                self.message = nn.Linear(width, width)
        self.anchor = nn.Linear(anchor_dim, width, bias=False) if anchor_dim else None
        self.pointer = nn.Linear(width, width, bias=False)
        self.update = nn.GRUCell(width, width)
        self.to(device)

    def forward(self, space, domains, anchor, *, sample=False, witness=None, forced=None):
        vertices = sorted(domains)
        sizes = [len(domains[u]) for u in vertices]
        device = self.token.weight.device
        pairs = [[(float(v in space.q.adj_set[u]),
                   len(domains[u] & domains[v]) / min(len(domains[u]), len(domains[v])))
                  for v in vertices] for u in vertices]
        tokens = [[log1p(w), log1p(space.q.degree[u]), log1p(len(space.q.adj_set[u] & domains.keys())),
                   log1p(len(space.cand_set[u])), len(vertices)/space.q.n] for u,w in zip(vertices,sizes)]
        h = self.token(torch.tensor(tokens, device=device, dtype=torch.float32))
        relations = self.relations(torch.tensor(pairs, device=device))
        if self.architecture == "attention":
            h = self.encoder(h.unsqueeze(0), src_mask=relations.squeeze(-1))[0]
        else:
            if self.architecture == "message":
                relations = relations * self.message(h).unsqueeze(0)
            h = self.encoder(torch.cat((h, relations.mean(dim=1)), dim=-1))
        state = h.mean(0)
        if self.anchor is not None and anchor is not None:
            state = state + self.anchor(torch.as_tensor(anchor, device=device))
        selected = []
        logprob = h.new_zeros(())
        mask = torch.zeros(len(vertices), device=device, dtype=torch.bool)
        for step in range(3):
            logits = (h @ self.pointer(state)).masked_fill(mask, float('-inf'))
            dist = Categorical(logits=logits)
            if witness is not None:
                valid = torch.tensor([u in witness for u in vertices], device=device) & ~mask
                logprob = logprob + torch.logsumexp(dist.logits.masked_fill(~valid, float('-inf')), dim=0)
                choice = Categorical(logits=logits.detach().masked_fill(~valid, float('-inf'))).sample()
            else:
                if forced is not None:
                    choice = logits.new_tensor(vertices.index(forced[step]), dtype=torch.long)
                else:
                    choice = dist.sample() if sample else logits.argmax()
                logprob = logprob + dist.log_prob(choice)
            selected.append(choice)
            mask = mask.scatter(0, choice.reshape(1), True)
            state = self.update(h.index_select(0, choice.reshape(1)), state.unsqueeze(0))[0]
        return tuple(vertices[i] for i in torch.stack(selected).tolist()), logprob
