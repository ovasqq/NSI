"""NeuGN as a reorder hook for search.enumerate.enumerate_matches.

At search depth < max_depth the local candidates C_M(u) are sorted by the
model's logit for "u ↦ v", descending (ties: smaller id first, via a stable
sort of the id-ordered list). Deeper levels keep the baseline id order.
The hook only permutes, so completeness is untouched.
"""

from __future__ import annotations

import time

import torch

from models.encoding import collate, encode_query, encode_state
from models.neugn import NeuGN
from queries.euler import euler_sequence
from search.graph import Graph


class NeuGNReorder:
    def __init__(self, model: NeuGN, q: Graph, max_depth: int = 10,
                 device: str | torch.device = "cpu", shift: int = 0):
        self.model, self.max_depth, self.device, self.shift = model, max_depth, device, shift
        self.num_g = model.cfg.num_g
        # Fixed per query: Euler sequence from vertex 0, shift 0.
        self.qenc = encode_query(q, euler_sequence(q))
        self.calls = 0
        self.seconds = 0.0
        with torch.inference_mode():
            probe = self._batch([-1] * q.n, 0)
            self.h_q = model.query_vector(probe)

    def _batch(self, mapping: list[int], target: int) -> dict[str, torch.Tensor]:
        state = encode_state(self.qenc, mapping, target, self.shift, self.num_g)
        return {k: v.to(self.device) for k, v in collate([state]).items()}

    def scores(self, mapping: list[int], u: int, cands: list[int]) -> torch.Tensor:
        with torch.inference_mode():
            logits = self.model(self._batch(mapping, u), self.h_q)[0]
            return logits[torch.tensor(cands, device=logits.device)].cpu()

    def __call__(self, depth: int, u: int, mapping: list[int], cands: list[int]) -> list[int]:
        if depth >= self.max_depth:
            return cands
        t0 = time.perf_counter()
        s = self.scores(mapping, u, cands)
        order = torch.argsort(-s, stable=True).tolist()
        self.calls += 1
        self.seconds += time.perf_counter() - t0
        return [cands[i] for i in order]
