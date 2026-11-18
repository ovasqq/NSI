"""Train and serve the residual AC-3 head on the frozen NeuGN encoder."""

from __future__ import annotations

import numpy as np
import torch

from models.encoding import encode_query
from models.residual_arc import ResidualArcHead, collate_residual, encode_residual
from queries.euler import euler_sequence
from search.pruning import DeepPruning


class ResidualArcReadout:
    def __init__(self, backbone, candidates_per_domain: int):
        self.backbone = backbone.eval().requires_grad_(False)
        self.device = backbone.anchor.device
        self.head = ResidualArcHead(backbone.cfg.d).to(self.device)
        self.candidates_per_domain = candidates_per_domain

    def score(self, rows):
        self.head.eval()
        with torch.inference_mode():
            return np.concatenate([
                self.head(self.backbone, collate_residual(rows[i:i + 32], self.device)).cpu().numpy()
                for i in range(0, len(rows), 32)
            ])

    def deep_policy(self, depth: int, threshold: float) -> DeepPruning:
        def factory(space):
            qenc = encode_query(space.q, euler_sequence(space.q))
            context = None

            def should_check(depth, domains, mapping):
                nonlocal context
                with torch.inference_mode():
                    row = encode_residual(qenc, mapping, domains, space.order[depth - 1],
                                          self.backbone.cfg.num_g, self.candidates_per_domain)
                    batch = collate_residual([row], self.device)
                    if context is None:
                        context = self.head.query_context(self.backbone, batch)
                    return bool(self.head(self.backbone, batch, context)[0].item() > threshold)

            return should_check

        return DeepPruning(depth, "adaptive", state_arc_policy_factory=factory)


def fit(backbone, rows, gains, cohorts, epochs, seed, candidates_per_domain):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    readout = ResidualArcReadout(backbone, candidates_per_domain)
    y = np.sign(gains) * np.log1p(np.abs(gains))
    _, groups, counts = np.unique(cohorts, return_inverse=True, return_counts=True)
    weights = (len(rows) / (len(counts) * counts[groups])).astype(np.float32)
    optimizer = torch.optim.AdamW(readout.head.parameters(), lr=1e-3)
    for epoch in range(1, epochs + 1):
        total = 0.0
        for ids in np.array_split(rng.permutation(len(rows)), (len(rows) + 31) // 32):
            batch = collate_residual([rows[i] for i in ids], readout.device)
            pred = readout.head(backbone, batch)
            target = torch.as_tensor(y[ids], device=readout.device)
            weight = torch.as_tensor(weights[ids], device=readout.device)
            loss = (weight * (pred - target).square()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.detach().item() * len(ids)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs}: weighted loss {total / len(rows):.6f}", flush=True)
    readout.head.eval()
    return readout


def save(path, readout, threshold):
    torch.save(dict(head=readout.head.state_dict(), threshold=threshold,
                    candidates_per_domain=readout.candidates_per_domain), path)


def load(path, backbone):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    readout = ResidualArcReadout(backbone, saved["candidates_per_domain"])
    readout.head.load_state_dict(saved["head"])
    readout.head.eval()
    return readout, saved["threshold"]
