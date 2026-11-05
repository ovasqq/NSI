"""Oracle ordering hooks: rank candidates by the *true* p/c values computed by
`search.oracle.label_subtrees`.

  * 'p'      — solvable candidates first (has_solution = 1), then dead ends;
  * 'inv_c'  — smallest subtree first (1/c, ignores p entirely);
  * 'pc'     — the Simon–Kadane rule: solvable first (sorted by subtree size),
               then dead ends (sorted by subtree size).

Every mode only permutes C_M(u), so completeness is untouched (same argument
as NeuGN). At depth ≥ `max_depth` the baseline id order is kept.
"""

from __future__ import annotations

from search.enumerate import SearchSpace
from search.oracle import OracleLabels


class OracleReorder:
    def __init__(self, space: SearchSpace, labels: OracleLabels,
                 mode: str = "pc", max_depth: int = 3):
        assert mode in ("p", "inv_c", "pc")
        self.space, self.labels, self.mode, self.max_depth = space, labels, mode, max_depth
        self.calls = 0

    def __call__(self, depth: int, u: int, mapping: list[int], cands: list[int]) -> list[int]:
        if depth >= self.max_depth:
            return cands
        key = tuple(mapping[w] for w in self.space.order[:depth])
        info = self.labels.get(key)
        if info is None:  # unlabeled (only possible if labels were pruned)
            return cands
        lookup = {v: (sol, size) for v, sol, size, _ in info}
        self.calls += 1

        def keyfn(v: int) -> tuple:
            sol, size = lookup[v]
            if self.mode == "p":
                return (-sol, v)
            if self.mode == "inv_c":
                return (size, v)
            return (-sol, size, v)  # 'pc'

        return sorted(cands, key=keyfn)
