"""NeuGN with explicit domain-pair tokens and relation-biased attention."""

from copy import deepcopy

import torch
from torch import nn

from models.encoding import collate, encode_state
from models.fragment import encode as sample_domains
from models.residual_arc import ResidualArcHead


def encode_pairs(space, qenc, mapping, domains, options, candidates, max_pairs):
    depth = sum(v >= 0 for v in mapping)
    target = space.order[depth]
    row = sample_domains(
        space, domains, depth, mapping, candidates, options, torch.zeros(1)
    )
    # Bound tokens globally, taking one candidate per domain before a second.
    ownership = row["ownership"]
    pools = [ownership[i].nonzero().flatten().tolist() for i in range(len(domains))]
    pairs = []
    for offset in range(max(map(len, pools), default=0)):
        for i, pool in enumerate(pools):
            if offset < len(pool) and len(pairs) < max_pairs:
                pairs.append((i, pool[offset]))
        if len(pairs) == max_pairs:
            break
    owners = torch.tensor([i for i, _ in pairs], dtype=torch.long)
    ids = torch.tensor([int(row["ids"][j]) for _, j in pairs], dtype=torch.long)
    query_edges = row["query_edges"]
    same_owner = owners[:, None] == owners[None, :]
    same_image = ids[:, None] == ids[None, :]
    required = query_edges[owners[:, None], owners[None, :]]
    actual = torch.tensor(
        [[int(w) in space.g.adj_set[int(v)] for w in ids] for v in ids],
        dtype=torch.bool,
    )
    # A learned multi-hot relation bias per attention head.
    relations = torch.stack(
        (
            same_owner,
            same_image & ~same_owner,
            required & actual & ~same_image,
            required & (~actual | same_image),
        ),
        dim=-1,
    )
    row.update(
        state=encode_state(qenc, mapping, target, 0, space.g.n),
        owners=owners,
        pair_ids=ids,
        relations=relations,
        target=target,
        local=space.local_candidates(depth, mapping, set(v for v in mapping if v >= 0)),
    )
    return row


class PairTransformer(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.neugn = deepcopy(backbone)
        d = backbone.cfg.d
        self.check = nn.Parameter(torch.zeros(d))
        self.types = nn.Embedding(2, d)
        self.stats = nn.Linear(4, d)
        self.relations = nn.Parameter(torch.zeros(4, backbone.cfg.heads))
        self.belongs = nn.Parameter(torch.zeros(backbone.cfg.heads))
        self.select = nn.Sequential(nn.Linear(3 * d + 1, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, row):
        model = self.neugn
        device = model.anchor.device
        batch = {k: v.to(device) for k, v in collate([row["state"]]).items()}
        hq, qvectors = ResidualArcHead.query_context(model, batch)
        vertices = row["vertices"].to(device)
        owners = row["owners"].to(device)
        ids = row["pair_ids"].to(device)
        qtokens = (
            qvectors[0, vertices]
            + model.qvert_emb(vertices % model.qvert_emb.num_embeddings)
            + self.stats(row["stats"].to(device))
            + self.types.weight[0]
        )
        ptokens = qtokens[owners] + model.content_emb(ids) + self.types.weight[1]
        prefix = torch.cat(
            (
                hq[:, None],
                model.anchor[None, None],
                (model.anchor + self.check)[None, None],
                model.mapping_tokens(batch),
            ),
            dim=1,
        )
        p = prefix.shape[1]
        if p > model.pos_emb.num_embeddings:
            raise ValueError(
                "mapping sequence exceeds the checkpoint positional vocabulary"
            )
        prefix = prefix + model.pos_emb(torch.arange(p, device=device))[None]
        tokens = model.norm_in(torch.cat((prefix, qtokens[None], ptokens[None]), dim=1))
        n, k = len(vertices), len(ids)
        heads = model.cfg.heads
        bias = tokens.new_zeros((heads, tokens.shape[1], tokens.shape[1]))
        if k:
            relation_bias = (
                row["relations"].to(device=device, dtype=tokens.dtype) @ self.relations
            )
            bias[:, p + n :, p + n :] = relation_bias.permute(2, 0, 1)
            membership = torch.arange(n, device=device)[:, None] == owners[None]
            bias[:, p : p + n, p + n :] = membership[None] * self.belongs[:, None, None]
            bias[:, p + n :, p : p + n] = (
                membership.T[None] * self.belongs[:, None, None]
            )
        # The same query-edge relation is also visible between residual Q tokens.
        bias[:, p : p + n, p : p + n] = (
            row["query_edges"].to(device)[None] * self.belongs[:, None, None]
        )
        hidden = model.encoder(tokens, mask=bias, is_causal=False)
        ranks = model.out(hidden[:, 1])[0]
        actions = row["actions"].to(device=device, dtype=tokens.dtype)
        size = actions.sum(-1, keepdim=True)
        qpool = actions @ hidden[0, p : p + n] / size.clamp_min(1)
        pair_membership = actions[:, owners]
        ppool = (
            pair_membership
            @ hidden[0, p + n :]
            / pair_membership.sum(-1, keepdim=True).clamp_min(1)
        )
        check = hidden[0, 2].expand(len(actions), -1)
        scores = self.select(
            torch.cat((qpool, ppool, check, size / 4), dim=-1)
        ).flatten()
        return ranks, scores
