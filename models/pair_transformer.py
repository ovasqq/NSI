"""NeuGN with explicit domain-pair tokens and relation-biased attention."""

from copy import deepcopy

import torch
from torch import nn

from models.encoding import collate, encode_state
from models.fragment import encode as sample_domains
from models.neugn import NeuGN
from models.residual_arc import ResidualArcHead


def encode_pairs(
    space,
    qenc,
    mapping,
    domains,
    options,
    candidates,
    max_pairs,
    *,
    relations_on_device=False,
):
    depth = sum(v >= 0 for v in mapping)
    target = space.order[depth]
    row = sample_domains(
        space,
        domains,
        depth,
        mapping,
        candidates,
        options,
        torch.zeros(1),
        include_data_edges=False,
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
    relations = None
    if not relations_on_device:
        same_owner = owners[:, None] == owners[None, :]
        same_image = ids[:, None] == ids[None, :]
        required = row["query_edges"][owners[:, None], owners[None, :]]
        actual = torch.tensor(
            [[int(w) in space.g.adj_set[int(v)] for w in ids] for v in ids],
            dtype=torch.bool,
        )
        relations = torch.stack(
            (
                same_owner,
                same_image & ~same_owner,
                required & actual & ~same_image,
                required & (~actual | same_image),
            ),
            dim=-1,
        )
    # These dense sample tensors are unnecessary once pair owners are known.
    for name in ("ownership", "data_edges", "ids", "anchor", "image"):
        row.pop(name, None)
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
    def __init__(self, backbone, *, from_scratch=False):
        super().__init__()
        self.neugn = NeuGN(backbone.cfg) if from_scratch else deepcopy(backbone)
        d = backbone.cfg.d
        self.register_buffer("graph_edges", None, persistent=False)
        self.check = nn.Parameter(torch.zeros(d))
        self.types = nn.Embedding(2, d)
        self.stats = nn.Linear(4, d)
        self.relations = nn.Parameter(torch.zeros(4, backbone.cfg.heads))
        self.belongs = nn.Parameter(torch.zeros(backbone.cfg.heads))
        self.select = nn.Sequential(nn.Linear(3 * d + 1, d), nn.GELU(), nn.Linear(d, 1))

    def cache_graph(self, graph):
        if graph.n * graph.n > 512 * 1024 * 1024:
            return False
        adjacency = torch.zeros((graph.n, graph.n), dtype=torch.bool)
        for vertex, neighbors in enumerate(graph.adj):
            adjacency[vertex, torch.from_numpy(neighbors)] = True
        self.graph_edges = adjacency.to(self.neugn.anchor.device)
        return True

    def prepare(self, row):
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
            if row["relations"] is None:
                if self.graph_edges is None:
                    raise ValueError(
                        "device relation construction requires a cached graph"
                    )
                same_owner = owners[:, None] == owners[None, :]
                same_image = ids[:, None] == ids[None, :]
                edges = row["query_edges"].to(device)
                required = edges[owners[:, None], owners[None, :]]
                actual = self.graph_edges[ids[:, None], ids[None, :]]
                relations = torch.stack(
                    (
                        same_owner,
                        same_image & ~same_owner,
                        required & actual & ~same_image,
                        required & (~actual | same_image),
                    ),
                    dim=-1,
                )
            else:
                relations = row["relations"].to(device)
            relation_bias = relations.to(tokens.dtype) @ self.relations
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
        return tokens[0], bias, p, n, owners

    def forward(self, row):
        ranks, scores = self.forward_batch([row])
        return ranks[0], scores[0]

    def forward_batch(self, rows):
        model = self.neugn
        device = model.anchor.device
        prepared = [self.prepare(row) for row in rows]
        maximum = max(tokens.shape[0] for tokens, *_ in prepared)
        tokens = torch.stack(
            [
                torch.nn.functional.pad(item[0], (0, 0, 0, maximum - item[0].shape[0]))
                for item in prepared
            ]
        )
        masks = []
        for item in prepared:
            length = item[0].shape[0]
            mask = torch.nn.functional.pad(
                item[1], (0, maximum - length, 0, maximum - length)
            )
            mask[:, :, length:] = float("-inf")
            masks.append(mask)
        hidden = model.encoder(tokens, mask=torch.cat(masks), is_causal=False)
        ranks = model.out(hidden[:, 1])
        scores = []
        for index, (row, item) in enumerate(zip(rows, prepared)):
            _, _, p, n, owners = item
            output = hidden[index, : item[0].shape[0]]
            scores.append(self.fragment_scores(row, output, p, n, owners))
        return ranks, scores

    def fragment_scores(self, row, output, p, n, owners):
        device = output.device
        actions = row["actions"].to(device=device, dtype=output.dtype)
        size = actions.sum(-1, keepdim=True)
        qpool = actions @ output[p : p + n] / size.clamp_min(1)
        pair_membership = actions[:, owners]
        ppool = (
            pair_membership
            @ output[p + n :]
            / pair_membership.sum(-1, keepdim=True).clamp_min(1)
        )
        check = output[2].expand(len(actions), -1)
        scores = self.select(
            torch.cat((qpool, ppool, check, size / 4), dim=-1)
        ).flatten()
        return scores
