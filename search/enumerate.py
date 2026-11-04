"""Backtracking enumeration with a candidate-reordering hook.

Definition of a *step* (used for FMS and every budget in this project):
one attempted extension (u, v) of the partial mapping, i.e. one iteration of
the loop over local candidates C_M(u) — equivalently, one non-root node of the
search tree. The step that completes an embedding is counted. Vertices of G
already used by the mapping are excluded from C_M(u) beforehand and are not
counted.

C_M(u) for u = φ(i), i > 0: every v ∈ C(u), not yet used, adjacent in G to
M(u') for every backward neighbour u' of u (neighbours of u earlier in φ).
For i = 0, C_M(u) = C(u). Without a hook, C_M(u) is visited in increasing id.

The hook may only permute C_M(u); with `validate=True` this is checked on
every call, so a hook can never prune the search (completeness).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from search.graph import Graph

# hook(depth, u, mapping, candidates) -> permutation of candidates.
# `mapping[w]` is the image of query vertex w, or -1 if w is not mapped yet.
ReorderHook = Callable[[int, int, list[int], list[int]], Sequence[int]]


@dataclass
class SearchResult:
    num_matches: int = 0
    steps: int = 0
    # match_steps[k] = step counter when the (k+1)-th embedding was found.
    match_steps: list[int] = field(default_factory=list)
    embeddings: list[tuple[int, ...]] = field(default_factory=list)
    # True iff the whole search tree was traversed (no limit was hit).
    complete: bool = False

    @property
    def first_match_step(self) -> int | None:
        return self.match_steps[0] if self.match_steps else None


class _Stop(Exception):
    pass


class SearchSpace:
    """Static part of a search: (Q, G, C, φ) plus the backward neighbours of
    every position. Shared by the enumerator and by hooks that look ahead
    (e.g. Knuth probes), so both compute C_M(u) identically."""

    def __init__(self, q: Graph, g: Graph, cand: list[np.ndarray], order: list[int]):
        assert sorted(order) == list(range(q.n))
        self.q, self.g, self.cand, self.order = q, g, cand, order
        self.pos = {u: i for i, u in enumerate(order)}
        self.backward = [[w for w in q.adj[u].tolist() if self.pos[w] < self.pos[u]] for u in order]
        self.cand_set = [frozenset(c.tolist()) for c in cand]

    def local_candidates(self, i: int, mapping: list[int], used: set[int]) -> list[int]:
        """C_M(φ(i)) in increasing id order."""
        u = self.order[i]
        if not self.backward[i]:
            return [v for v in self.cand[u].tolist() if v not in used]
        g = self.g
        images = [mapping[w] for w in self.backward[i]]
        pivot = min(images, key=lambda v: g.degree[v])
        others = [g.adj_set[v] for v in images if v != pivot]
        cu = self.cand_set[u]
        return [
            v
            for v in g.adj[pivot].tolist()
            if v in cu and v not in used and all(v in s for s in others)
        ]


def enumerate_matches(
    q: Graph,
    g: Graph,
    cand: list[np.ndarray],
    order: list[int],
    reorder: ReorderHook | None = None,
    *,
    max_matches: int | None = None,
    max_steps: int | None = None,
    store_embeddings: bool = False,
    validate: bool = True,
) -> SearchResult:
    n = q.n
    space = SearchSpace(q, g, cand, order)
    mapping = [-1] * n
    used: set[int] = set()
    res = SearchResult()

    def dfs(i: int) -> None:
        u = order[i]
        local = space.local_candidates(i, mapping, used)
        if reorder is not None and len(local) > 1:
            new = list(reorder(i, u, mapping, local))
            if validate and sorted(new) != local:
                raise ValueError("reorder hook must return a permutation of its candidates")
            local = new
        for v in local:
            if max_steps is not None and res.steps >= max_steps:
                raise _Stop
            res.steps += 1
            mapping[u] = v
            used.add(v)
            if i + 1 == n:
                res.num_matches += 1
                res.match_steps.append(res.steps)
                if store_embeddings:
                    res.embeddings.append(tuple(mapping))
                if max_matches is not None and res.num_matches >= max_matches:
                    raise _Stop
            else:
                dfs(i + 1)
            used.discard(v)
            mapping[u] = -1

    try:
        if all(len(c) for c in cand):
            dfs(0)
        res.complete = True
    except _Stop:
        # Stopping exactly on the last leaf still leaves the tree unfinished
        # from the enumerator's point of view; that is fine for all our uses.
        pass
    return res
