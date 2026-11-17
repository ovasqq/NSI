"""Backtracking enumeration with a candidate-reordering hook.

Definition of a *step* (used for FMS and the step budget):
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
from time import perf_counter
from typing import Callable, Sequence

import numpy as np

from search.graph import Graph
from search.pruning import PRUNING_MODES, PruningStats, ResidualPruner

# hook(depth, u, mapping, candidates) -> permutation of candidates.
# `mapping[w]` is the image of query vertex w, or -1 if w is not mapped yet.
ReorderHook = Callable[[int, int, list[int], list[int]], Sequence[int]]


@dataclass(frozen=True, slots=True)
class BranchState:
    depth: int  # depth of the vertex just assigned, before descending
    vertex: int
    candidate: int
    rank: int
    width: int
    step: int
    steps_remaining: int | None
    backtracks: int
    last_deadend_steps: int


PruningPolicy = Callable[[BranchState], bool]
# Observers may copy the live mapping/used. The finish flag means the DFS
# returned from this branch or found a match before being interrupted.
BranchObserver = Callable[[BranchState, list[int], set[int]], Callable[[bool], None] | None]


@dataclass
class SearchResult:
    num_matches: int = 0
    steps: int = 0
    # match_steps[k] = step counter when the (k+1)-th embedding was found.
    match_steps: list[int] = field(default_factory=list)
    embeddings: list[tuple[int, ...]] = field(default_factory=list)
    # True iff all branches were explored or proved infeasible (no limit was hit).
    complete: bool = False
    pruning: PruningStats | None = None
    timed_out: bool = False
    pruning_policy_seconds: float = 0.0

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
    pruning: str = "none",
    prune_after_steps: int = 0,
    deadline: float | None = None,
    pruning_policy: PruningPolicy | None = None,
    branch_observer: BranchObserver | None = None,
) -> SearchResult:
    """Skip pruning for the first prune_after_steps attempts. The optional
    deadline uses perf_counter() and is checked between search operations."""
    if pruning not in PRUNING_MODES:
        raise ValueError(f"unknown pruning mode: {pruning!r}")
    if prune_after_steps < 0:
        raise ValueError("prune_after_steps must be nonnegative")
    if pruning_policy is not None and pruning == "none":
        raise ValueError("a pruning policy requires an exact pruning mode")
    n = q.n
    space = SearchSpace(q, g, cand, order)
    pruner = ResidualPruner(space, pruning) if pruning != "none" else None
    mapping = [-1] * n
    used: set[int] = set()
    res = SearchResult(pruning=pruner.stats if pruner is not None else None)
    track_branches = pruning_policy is not None or branch_observer is not None
    backtracks = last_deadend_steps = 0

    def check_deadline(limit: float) -> None:
        if perf_counter() >= limit:
            res.timed_out = True
            raise _Stop

    def dfs(i: int) -> None:
        nonlocal backtracks, last_deadend_steps
        u = order[i]
        local = space.local_candidates(i, mapping, used)
        if reorder is not None and len(local) > 1:
            new = list(reorder(i, u, mapping, local))
            if validate and sorted(new) != local:
                raise ValueError("reorder hook must return a permutation of its candidates")
            local = new
        for rank, v in enumerate(local):
            if deadline is not None:
                check_deadline(deadline)
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
                state = None
                if track_branches:
                    state = BranchState(i, u, v, rank, len(local), res.steps,
                                        None if max_steps is None else max_steps - res.steps,
                                        backtracks, last_deadend_steps)
                rejected = False
                should_check = pruner is not None and res.steps > prune_after_steps
                if should_check and pruning_policy is not None:
                    started = perf_counter()
                    should_check = pruning_policy(state)
                    res.pruning_policy_seconds += perf_counter() - started
                    if deadline is not None:
                        check_deadline(deadline)
                if should_check:
                    rejected = pruner.rejects(i + 1, mapping, used)
                    if deadline is not None:
                        check_deadline(deadline)
                if not rejected:
                    if track_branches:
                        first_match_count = res.num_matches
                        finish = branch_observer(state, mapping, used) if branch_observer else None
                        completed = False
                        try:
                            dfs(i + 1)
                            completed = True
                        finally:
                            found = res.num_matches > first_match_count
                            if finish is not None:
                                finish(completed or found)
                        if not found:
                            backtracks += 1
                            last_deadend_steps = res.steps - state.step
                    else:
                        dfs(i + 1)
                elif track_branches:
                    backtracks += 1
                    last_deadend_steps = 0
            used.discard(v)
            mapping[u] = -1

    try:
        if deadline is not None:
            check_deadline(deadline)
        if all(len(c) for c in cand):
            dfs(0)
        res.complete = True
    except _Stop:
        # Stopping exactly on the last leaf still leaves the tree unfinished
        # from the enumerator's point of view; that is fine for all our uses.
        pass
    return res
