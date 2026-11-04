"""Ordering by p/c (Simon & Kadane 1975): expected time to the first solution
is minimised by trying candidates in decreasing p_i / c_i, where p_i is the
probability that candidate i's subtree contains a solution and c_i the cost
of searching it.

Here, without any new training:
- p̂_i — NeuGN's probability renormalised over the local candidates
  (softmax over their logits); a proxy for "the subtree has a solution";
- ĉ_i — Knuth's estimate of the size of the subtree under u ↦ v_i.

score_i = log p̂_i − α · log ĉ_i, sorted descending (ties: smaller id).
The p/c score is used at depths < pc_depth (probes are expensive); at
depths pc_depth … max_depth−1 plain NeuGN ordering; deeper — baseline order.
Without a model (`neugn=None`) this is the "1/c only" ordering: smallest
estimated subtree first, baseline order below pc_depth.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from models.hook import NeuGNReorder
from search.enumerate import SearchSpace
from search.knuth import knuth_estimate


class PCReorder:
    def __init__(self, space: SearchSpace, neugn: NeuGNReorder | None = None,
                 pc_depth: int = 3, probes: int = 4, alpha: float = 1.0, seed: int = 0):
        self.space, self.neugn = space, neugn
        self.pc_depth, self.probes, self.alpha = pc_depth, probes, alpha
        self.rng = np.random.default_rng(seed)
        self.probe_steps = 0  # charged as extra search work, see search/knuth.py
        self.calls = 0

    @property
    def seconds(self) -> float:
        return self.neugn.seconds if self.neugn else 0.0

    def log_costs(self, depth: int, u: int, mapping: list[int], cands: list[int]) -> list[float]:
        used = {v for v in mapping if v >= 0}
        out = []
        for v in cands:
            mapping[u] = v
            used.add(v)
            est, steps = knuth_estimate(self.space, depth + 1, mapping, used, self.rng, self.probes)
            used.discard(v)
            mapping[u] = -1
            self.probe_steps += steps
            out.append(math.log(est))
        return out

    def __call__(self, depth: int, u: int, mapping: list[int], cands: list[int]) -> list[int]:
        if depth >= self.pc_depth:
            return self.neugn(depth, u, mapping, cands) if self.neugn else cands
        self.calls += 1
        score = -self.alpha * torch.tensor(self.log_costs(depth, u, mapping, cands))
        if self.neugn is not None:
            score = score + torch.log_softmax(self.neugn.scores(mapping, u, cands), 0)
        order = torch.argsort(-score, stable=True).tolist()
        return [cands[i] for i in order]
