"""Candidate-conditioned fragment scoring on a bounded sample of the residual graph."""

from __future__ import annotations

from itertools import islice
from math import log1p, prod
from random import Random

import torch
from torch import nn

from search.enumerate import SearchSpace
from search.pruning import Domains


def proposals(space: SearchSpace, domains: Domains, limit: int) -> list[tuple[int, ...]]:
    """Grow connected fragments with two cheap orderings and deduplicate them."""
    adj = space.q.adj_set
    widths = {u: len(values) for u, values in domains.items()}
    small_first = sorted(domains, key=lambda u: (widths[u], u))
    dense_first = sorted(domains, key=lambda u: (-sum(v in domains for v in adj[u]), widths[u], u))
    groups = set()
    result = []
    for small, dense in zip(small_first, dense_first):
        for root, prefer_dense in ((small, False), (dense, True)):
            members = {root}
            boundary = set(adj[root].intersection(domains))
            while boundary and len(members) < 4:
                next_vertex = min(boundary, key=lambda u: (
                    -len(adj[u] & members) if prefer_dense else widths[u], widths[u], u))
                members.add(next_vertex)
                boundary.update(adj[next_vertex].intersection(domains))
                boundary.difference_update(members)
                fragment = tuple(sorted(members))
                if fragment not in groups:
                    groups.add(fragment)
                    result.append(fragment)
                    if len(result) == limit:
                        return result
    return result


def manual_fragment(space: SearchSpace, domains: Domains, options: list[tuple[int, ...]]) -> tuple[int, ...]:
    def priority(fragment):
        edges = sum(v in space.q.adj_set[u] for i, u in enumerate(fragment) for v in fragment[:i])
        return (-edges + len(fragment) - 1, prod(len(domains[u]) for u in fragment), fragment)

    return min(options, key=priority)


def encode(space: SearchSpace, domains: Domains, depth: int, mapping: list[int],
           limit: int, options: list[tuple[int, ...]], anchor: torch.Tensor,
           *, include_data_edges: bool = True):
    """Sample bounded domain neighborhoods. Shared candidate IDs are shared nodes."""
    q, g = space.q, space.g
    vertices = list(domains)
    u = space.order[depth - 1]
    image = mapping[u]
    rng = Random((depth << 32) + image)
    samples = {}
    for x, domain in domains.items():
        if len(domain) <= limit:
            samples[x] = set(domain)
            continue
        pool = space.cand[x]
        selected = set()
        for i in rng.sample(range(len(pool)), min(len(pool), 4 * limit)):
            v = int(pool[i])
            if v in domain:
                selected.add(v)
                if len(selected) >= (limit + 1) // 2:
                    break
        samples[x] = selected

    seeds = {x: sorted(s) for x, s in samples.items()}
    for x, domain in domains.items():
        selected = samples[x]
        if len(selected) == limit or len(domain) <= limit:
            continue
        sources = []
        for y in sorted(q.adj_set[x].intersection(domains)):
            if seeds[y]:
                sources.append(rng.choice(seeds[y]))
        rng.shuffle(sources)
        for v in sources[:limit]:
            neighbors = g.adj[v]
            for i in rng.sample(range(len(neighbors)), min(len(neighbors), 4)):
                w = int(neighbors[i])
                if w in domain:
                    selected.add(w)
                if len(selected) == limit:
                    break
            if len(selected) == limit:
                break
        for v in islice(domain, limit):
            if len(selected) == limit:
                break
            selected.add(v)

    ids = sorted(set().union(*samples.values()))
    ids_set = set(ids)
    index = {v: i for i, v in enumerate(ids)}
    n, m = len(vertices), len(ids)
    ownership = torch.zeros((n, m), dtype=torch.bool)
    data_edges = torch.zeros((m, m) if include_data_edges else (0, 0), dtype=torch.bool)
    query_edges = torch.zeros((n, n), dtype=torch.bool)
    query_index = {x: i for i, x in enumerate(vertices)}
    for i, x in enumerate(vertices):
        ownership[i, [index[v] for v in domains[x] & ids_set]] = True
        query_edges[i, [query_index[y] for y in q.adj_set[x] if y in query_index]] = True
    if include_data_edges:
        for i, v in enumerate(ids):
            data_edges[i, [index[w] for w in g.adj_set[v] & ids_set]] = True
    actions = torch.zeros((len(options), n), dtype=torch.bool)
    for i, fragment in enumerate(options):
        actions[i, [query_index[x] for x in fragment]] = True
    coverage = ownership.sum(dim=1).tolist()
    stats = torch.tensor([[log1p(len(domains[x])), count / len(domains[x]),
                           sum(mapping[y] >= 0 for y in q.adj_set[x]), int(x in q.adj_set[u])]
                          for x, count in zip(vertices, coverage)], dtype=torch.float32)
    return dict(anchor=anchor, image=image, vertices=torch.tensor(vertices), ids=torch.tensor(ids),
                ownership=ownership, data_edges=data_edges, query_edges=query_edges, actions=actions,
                stats=stats)


def batch_rows(rows, device):
    n = max(len(row["vertices"]) for row in rows)
    m = max(len(row["ids"]) for row in rows)
    a = max(len(row["actions"]) for row in rows)
    shapes = dict(vertices=(n,), ids=(m,), ownership=(n, m), data_edges=(m, m),
                  query_edges=(n, n), actions=(a, n), stats=(n, 4))
    batch = {key: rows[0][key].new_zeros((len(rows), *shape)) for key, shape in shapes.items()}
    batch["anchor"] = torch.stack([row["anchor"] for row in rows])
    batch["image"] = torch.tensor([row["image"] for row in rows])
    for i, row in enumerate(rows):
        for key in shapes:
            value = row[key]
            batch[key][(i, *(slice(d) for d in value.shape))] = value
    return {key: value.to(device) for key, value in batch.items()}


def mean_neighbors(edges: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
    weights = edges.to(nodes.dtype)
    return (weights @ nodes) / weights.sum(dim=-1, keepdim=True).clamp_min(1)


class FragmentHead(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        hidden = 32
        self.data = nn.Linear(width, hidden)
        self.candidate = nn.Linear(2 * hidden, hidden)
        self.query = nn.Linear(width + hidden + 4, hidden)
        self.context = nn.Linear(2 * width, hidden)
        self.score = nn.Sequential(nn.Linear(4 * hidden + 3, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, neugn, batch):
        candidates = self.data(neugn.content_emb(batch["ids"]))
        neighborhood = mean_neighbors(batch["data_edges"], candidates)
        candidates = self.candidate(torch.cat((candidates, neighborhood), dim=-1)).relu()
        owners = mean_neighbors(batch["ownership"], candidates)
        vertices = neugn.qvert_emb(batch["vertices"] % neugn.qvert_emb.num_embeddings)
        query = self.query(torch.cat((vertices, owners, batch["stats"]), dim=-1)).relu()
        neighbors = mean_neighbors(batch["query_edges"], query)
        context = self.context(torch.cat((batch["anchor"], neugn.content_emb(batch["image"])), dim=-1))
        actions = batch["actions"]
        pooled = mean_neighbors(actions, torch.cat((query, neighbors), dim=-1))
        # Pool only edges internal to each proposed action, so the score can depend
        # on the combination, not just on the individual query vertices.
        links = actions[:, :, :, None] & actions[:, :, None, :] & batch["query_edges"][:, None]
        pair_features = query[:, :, None, :] * query[:, None, :, :]
        pair = mean_neighbors(links.flatten(2), pair_features.flatten(1, 2))
        membership = batch["ownership"].to(query.dtype)
        counts = membership.sum(-1)
        pairs = (counts[:, :, None] * counts[:, None, :]).clamp_min(1)
        support = (membership @ batch["data_edges"].to(query.dtype) @ membership.transpose(1, 2)) / pairs
        overlap = (membership @ membership.transpose(1, 2)) / pairs
        relations = torch.stack((support, overlap), dim=-1)
        relations = mean_neighbors(links.flatten(2), relations.flatten(1, 2))
        size = actions.sum(dim=-1, keepdim=True)
        return self.score(torch.cat((pooled, pair, context[:, None].expand_as(pair), relations, size / 4),
                                    dim=-1)).squeeze(-1)
