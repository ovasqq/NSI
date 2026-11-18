"""Score AC-3 from the current assignment and residual domain tokens."""

from __future__ import annotations

from heapq import nsmallest

import torch
from torch import nn

from models.encoding import NUM_QVERT, collate, encode_state
from models.neugn import NeuGN
from search.pruning import Domains


def encode_residual(qenc, mapping: list[int], domains: Domains, assigned: int,
                    num_g: int, candidates_per_domain: int):
    vertices = list(domains)
    return dict(
        state=encode_state(qenc, mapping, None, 0, num_g),
        vertices=vertices,
        widths=[len(domains[u]) for u in vertices],
        candidates=[nsmallest(candidates_per_domain, domains[u]) for u in vertices],
        assigned=assigned,
        image=mapping[assigned],
    )


def collate_residual(rows, device):
    batch = collate([row["state"] for row in rows])
    shape = (len(rows), max(len(row["vertices"]) for row in rows))
    width = max(len(candidates) for row in rows for candidates in row["candidates"])
    vertices = torch.zeros(shape, dtype=torch.long)
    widths = torch.zeros(shape)
    candidates = torch.zeros((*shape, width), dtype=torch.long)
    candidate_mask = torch.zeros_like(candidates, dtype=torch.bool)
    for i, row in enumerate(rows):
        n = len(row["vertices"])
        vertices[i, :n] = torch.tensor(row["vertices"])
        widths[i, :n] = torch.tensor(row["widths"])
        for j, ids in enumerate(row["candidates"]):
            candidates[i, j, :len(ids)] = torch.tensor(ids)
            candidate_mask[i, j, :len(ids)] = True
    batch.update(res_vertices=vertices, res_widths=widths, res_candidates=candidates,
                 candidate_mask=candidate_mask, assigned=torch.tensor([r["assigned"] for r in rows]),
                 image=torch.tensor([r["image"] for r in rows]))
    return {key: value.to(device) for key, value in batch.items()}


class ResidualArcHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.check = nn.Parameter(torch.zeros(d))
        self.residual = nn.Parameter(torch.zeros(d))
        self.edge_bias = nn.Parameter(torch.zeros(()))
        self.projection = nn.Linear(2 * d + 1, d)
        self.score = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 1))

    @staticmethod
    def query_context(model: NeuGN, batch):
        if model.qs is None:
            h_q = model.query_vector(batch)
            vertices = h_q[:, None].expand(-1, batch["q_labels"].shape[1], -1)
            return h_q, vertices
        vertices = model.qs.vertices(batch["q_labels"], batch["q_adj"])
        h_q = vertices.masked_fill(~batch["q_mask"][..., None], float("-inf")).amax(dim=1)
        return h_q, vertices

    def forward(self, model: NeuGN, batch, query_context=None) -> torch.Tensor:
        h_q, query_vertices = self.query_context(model, batch) if query_context is None else query_context
        vertices, widths = batch["res_vertices"], batch["res_widths"]
        owners = query_vertices.gather(1, vertices[..., None].expand(-1, -1, model.cfg.d))
        candidates = model.content_emb(batch["res_candidates"])
        mask = batch["candidate_mask"][..., None]
        pooled = (candidates * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1)
        residual = self.projection(torch.cat((owners, pooled, widths.log1p()[..., None]), dim=-1))
        residual = residual + self.residual + model.qvert_emb(vertices % NUM_QVERT)

        check = (model.anchor + self.check + model.content_emb(batch["image"])
                 + model.qvert_emb(batch["assigned"] % NUM_QVERT))
        prefix = torch.cat((h_q[:, None], check[:, None], model.mapping_tokens(batch)), dim=1)
        p = prefix.shape[1]
        if p > model.pos_emb.num_embeddings:
            raise ValueError("Euler sequence exceeds the NeuGN checkpoint's positional embeddings")
        prefix = prefix + model.pos_emb(torch.arange(p, device=prefix.device))
        tokens = model.norm_in(torch.cat((prefix, residual), dim=1))

        n = vertices.shape[1]
        rows = batch["q_adj"].gather(1, vertices[..., None].expand(-1, -1, batch["q_adj"].shape[-1]))
        edges = (rows.gather(2, vertices[:, None].expand(-1, n, -1)) != 0)
        edges = edges & (vertices[:, :, None] != vertices[:, None, :])
        keep = torch.cat((torch.ones_like(batch["tok_mask"][:, :1]).expand(-1, 2),
                          batch["tok_mask"], widths > 0), dim=1)
        bias = tokens.new_zeros((tokens.shape[0], tokens.shape[1], tokens.shape[1]))
        bias[:, p:, p:] = edges * self.edge_bias
        bias = bias.masked_fill(~keep[:, None, :], float("-inf"))
        bias = bias.repeat_interleave(model.cfg.heads, dim=0)
        return self.score(model.encoder(tokens, mask=bias, is_causal=False)[:, 1]).squeeze(-1)
