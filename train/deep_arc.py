"""Train a CPU MLP to score AC-3 after forward checking and matching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from search.enumerate import SearchSpace
from search.pruning import Domains


def feature_factory(space: SearchSpace):
    q = space.q
    residual_edges = [0] * (q.n + 1)
    for u, v in q.edges():
        residual_edges[min(space.pos[u], space.pos[v])] += 1
    for i in range(q.n - 1, -1, -1):
        residual_edges[i] += residual_edges[i + 1]
    density = 2 * q.m / max(1, q.n * (q.n - 1))

    def features(depth: int, domains: Domains) -> tuple[float, ...]:
        widths = [len(d) for d in domains.values()]
        n = len(widths)
        total = sum(widths)
        union = len(set().union(*domains.values()))
        return (q.n, density, depth, n, residual_edges[depth], min(widths), max(widths),
                total / n, union / n, total / union, widths.count(1) / n)

    return features


@dataclass(frozen=True)
class ArcReadout:
    center: np.ndarray
    scale: np.ndarray
    first: np.ndarray
    bias: np.ndarray
    last: np.ndarray
    intercept: float

    def score(self, x: np.ndarray):
        h = np.maximum(((x - self.center) / self.scale) @ self.first + self.bias, 0)
        return h @ self.last + self.intercept

    def policy(self, threshold: float):
        def factory(space: SearchSpace):
            features = feature_factory(space)

            def should_check(depth: int, domains: Domains) -> bool:
                return bool(self.score(np.asarray(features(depth, domains), dtype=np.float32)) > threshold)

            return should_check

        return factory


def fit(features: np.ndarray, gains: np.ndarray, cohorts: np.ndarray, epochs: int, seed: int) -> ArcReadout:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    center = features.mean(0)
    scale = features.std(0).clip(1e-5)
    x = torch.from_numpy((features - center) / scale)
    y = torch.from_numpy(np.sign(gains) * np.log1p(np.abs(gains)))
    # Equal total loss weight for each size/density cohort that contributed examples.
    _, group, counts = np.unique(cohorts, return_inverse=True, return_counts=True)
    weights = torch.from_numpy((len(x) / (len(counts) * counts[group])).astype(np.float32))
    model = nn.Sequential(nn.Linear(x.shape[1], 32), nn.ReLU(), nn.Linear(32, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(1, epochs + 1):
        total = 0.0
        for ids in np.array_split(rng.permutation(len(x)), (len(x) + 255) // 256):
            optimizer.zero_grad()
            loss = (weights[ids] * (model(x[ids])[:, 0] - y[ids]).square()).mean()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ids)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs}: weighted loss {total / len(x):.6f}", flush=True)
    with torch.no_grad():
        return ArcReadout(center, scale, model[0].weight.numpy().T.copy(), model[0].bias.numpy().copy(),
                          model[2].weight.numpy()[0].copy(), model[2].bias.item())


def save(path: Path, readout: ArcReadout, threshold: float) -> None:
    np.savez(path, **vars(readout), threshold=threshold)


def load(path: Path) -> tuple[ArcReadout, float]:
    with np.load(path, allow_pickle=False) as data:
        values = {k: data[k] for k in ("center", "scale", "first", "bias", "last")}
        return ArcReadout(**values, intercept=data["intercept"].item()), data["threshold"].item()
