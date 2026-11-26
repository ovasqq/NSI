"""Shared incremental features and lightweight candidate/check heads for DFS."""

from dataclasses import dataclass, field
from math import log1p

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.residual_policy import residual_features


def vertex_features(graph):
    degree = np.log1p(graph.degree).astype(np.float32)
    neighbors = np.asarray([degree[adj].mean() if len(adj) else 0. for adj in graph.adj],
                           dtype=np.float32)
    label = graph.labels.astype(np.float32) / max(1, int(graph.labels.max()))
    return np.stack((degree, neighbors, label, degree*degree), axis=1)


@dataclass
class Frame:
    domains: dict
    mass: np.ndarray
    mapped: np.ndarray
    arc_consistent: bool = False
    matching: dict | None = None
    scheduled_action: str = "skip"
    delay: int | None = None
    hidden: np.ndarray | None = None
    support_sizes: dict = field(default_factory=dict)
    ranking_context: tuple | None = None
    action_input: tuple | None = None


class StateFeatures:
    def __init__(self, space, data_features):
        self.space = space
        self.data = data_features
        self.query = vertex_features(space.q)
        self.residual = residual_features(space)
        self.degree = space.q.degree.tolist()

    def _contribution(self, u, width):
        return np.asarray((log1p(width), width == 1, log1p(width)*self.query[u, 0],
                           self.degree[u]), dtype=np.float32)

    def frame(self, domains, parent=None, assigned=None):
        if parent is None:
            mass = np.zeros(4, dtype=np.float32)
            for u, domain in domains.items():
                mass += self._contribution(u, len(domain))
            mapped = np.zeros(self.data.shape[1], dtype=np.float32)
        else:
            u, image = assigned
            mass = parent.mass - self._contribution(u, len(parent.domains[u]))
            for v, domain in domains.items():
                previous = parent.domains[v]
                if len(previous) != len(domain):
                    mass += self._contribution(v, len(domain))-self._contribution(v, len(previous))
            mapped = parent.mapped + self.data[image]
        matching = parent.matching if parent is not None else None
        return Frame(domains, mass, mapped, matching=matching)

    def support_size(self, frame, image, u):
        key = (image, u)
        if key not in frame.support_sizes:
            frame.support_sizes[key] = len(frame.domains[u] & self.space.g.adj_set[image])
        return frame.support_sizes[key]

    def encode(self, depth, frame, *, graph=False):
        next_u = self.space.order[depth]
        domains = frame.domains
        sizes = [len(domain) for domain in domains.values()]
        base = np.concatenate((
            np.asarray((depth / self.space.q.n, len(domains) / self.space.q.n,
                        log1p(min(sizes)), log1p(max(sizes)), log1p(len(domains[next_u]))),
                       dtype=np.float32),
            self.query[next_u], frame.mass / len(domains), frame.mapped / max(depth, 1),
        ))
        if not graph:
            return base
        adjacent = self.space.q.adj_set[next_u]
        live_neighbors = [v for v in adjacent if v in domains]
        # One neighborhood aggregation in the residual query; local domain
        # supports in G are kept separate in candidate_features.
        neighbor_width = sum(log1p(len(domains[v])) for v in live_neighbors)
        structure = self.residual(depth, domains)
        supports = [self.support_size(frame, image, v)
                    for image in sorted(domains[next_u])[:3] for v in sorted(live_neighbors)[:2]]
        return np.concatenate((base, np.log1p(np.asarray(structure, dtype=np.float32)),
                               np.asarray((len(live_neighbors), neighbor_width,
                                           log1p(min(supports, default=0)),
                                           log1p(sum(supports))), dtype=np.float32)))

    def candidate_features(self, depth, frame, ids, *, graph=False):
        base = self.data[list(ids)]
        if not graph:
            return base
        u = self.space.order[depth]
        neighbors = [v for v in self.space.q.adj_set[u] if v in frame.domains]
        supports = []
        for image in ids:
            sizes = [self.support_size(frame, image, v) for v in neighbors]
            supports.append((log1p(min(sizes, default=0)), log1p(sum(sizes))))
        return np.column_stack((base, np.asarray(supports, dtype=np.float32)))


class JointModel(nn.Module):
    def __init__(self, state_dim, candidate_dim, actions, width=24):
        super().__init__()
        self.state = nn.Linear(state_dim, width)
        self.rank = nn.Linear(width, candidate_dim)
        self.check = nn.Linear(width, actions)

    def forward(self, states):
        hidden = self.state(states).relu()
        return self.rank(hidden), self.check(hidden)


@dataclass
class Inference:
    state_weight: np.ndarray
    state_bias: np.ndarray
    rank_weight: np.ndarray
    rank_bias: np.ndarray
    check_weight: np.ndarray
    check_bias: np.ndarray

    @classmethod
    def from_model(cls, model):
        return cls(*(p.detach().cpu().numpy().copy() for p in model.parameters()))

    def encode(self, features):
        return np.maximum(self.state_weight @ features + self.state_bias, 0.)

    def scores(self, hidden, candidates):
        return candidates @ (self.rank_weight @ hidden + self.rank_bias)

    def costs(self, hidden):
        return self.check_weight @ hidden + self.check_bias


def train_joint(rank_rows, check_rows, *, graph=False, epochs=6, batch_size=128,
                seed=19, device="cpu"):
    index = int(graph)
    rstate, differences = zip(*[(r[index], r[2+index]) for r in rank_rows])
    cstate, costs = zip(*[(r[index], r[2]) for r in check_rows])

    def tensor(values):
        return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device)
    rstate, differences, cstate, costs = map(tensor, (rstate, differences, cstate, costs))
    torch.manual_seed(seed)
    model = JointModel(rstate.shape[1], differences.shape[1], costs.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    # Rank losses compare a known feasible child against sampled alternatives.
    # Signed logarithms keep long/censored branches from dominating the loss.
    costs = costs.sign() * (costs.abs() / 10).log1p()
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        rank_batches = rng.permutation(len(rstate))
        check_batches = rng.permutation(len(cstate))
        for offset in range(0, max(len(rank_batches), len(check_batches)), batch_size):
            rids = rank_batches[np.arange(offset, offset+batch_size) % len(rank_batches)]
            cids = check_batches[np.arange(offset, offset+batch_size) % len(check_batches)]
            rank, _ = model(rstate[rids])
            _, check = model(cstate[cids])
            loss = F.softplus(-(rank*differences[rids]).sum(dim=1)).mean()
            loss = loss + F.smooth_l1_loss(check, costs[cids])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.cpu().eval()
