"""Sound feasibility checks for a partial subgraph embedding."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import prod
from time import perf_counter
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from search.enumerate import SearchSpace


PRUNING_MODES = ("none", "forward", "matching", "arc", "domains", "motifs")
DEEP_MODES = ("none", "forward", "matching", "matching_arc", "adaptive")
Domains = dict[int, frozenset[int]]
ArcPolicy = Callable[[int, Domains], bool]
StateArcPolicy = Callable[[int, Domains, list[int]], bool]
FragmentSelector = Callable[[int, Domains, list[int]], tuple[int, ...]]
# Limit the Cartesian product of a motif's domains before attempting its search.
MAX_MOTIF_PRODUCT = 4096


@dataclass(frozen=True)
class DeepPruning:
    """Override domains after the first full_depth assignments."""

    full_depth: int
    mode: str
    arc_policy_factory: Callable[[SearchSpace], ArcPolicy] | None = None
    state_arc_policy_factory: Callable[[SearchSpace], StateArcPolicy] | None = None

    def __post_init__(self):
        if self.full_depth < 0 or self.mode not in DEEP_MODES:
            raise ValueError("invalid deep pruning configuration")
        policies = int(self.arc_policy_factory is not None) + int(self.state_arc_policy_factory is not None)
        if policies != int(self.mode == "adaptive"):
            raise ValueError("adaptive deep pruning requires exactly one AC-3 policy")


@dataclass
class PruningStageStats:
    calls: int = 0
    seconds: float = 0.0


@dataclass
class PruningStats:
    calls: int = 0
    seconds: float = 0.0
    motif_checks: int = 0
    arc_policy_checks: int = 0
    cuts: dict[str, int] = field(default_factory=dict)
    stages: dict[str, PruningStageStats] = field(default_factory=lambda: {
        name: PruningStageStats()
        for name in ("build_domains", "arc_consistency", "matching", "motifs")
    })

    def record_stage(self, name: str, started: float) -> None:
        elapsed = perf_counter() - started
        stage = self.stages.get(name)
        if stage is None:
            stage = self.stages[name] = PruningStageStats()
        stage.calls += 1
        stage.seconds += elapsed


class ResidualPruner:
    def __init__(self, space: SearchSpace, mode: str, deep: DeepPruning | None = None):
        started = perf_counter()
        if deep is not None and mode != "domains":
            raise ValueError("deep pruning overrides require domains as the base mode")
        self.space = space
        self.deep = deep
        self.stats = PruningStats()
        self.arc_policy = None
        self.state_arc_policy = None
        if deep is not None and deep.arc_policy_factory is not None:
            self.arc_policy = deep.arc_policy_factory(space)
        if deep is not None and deep.state_arc_policy_factory is not None:
            self.state_arc_policy = deep.state_arc_policy_factory(space)
        if self.arc_policy is not None or self.state_arc_policy is not None:
            self.stats.stages["arc_policy"] = PruningStageStats()
        self.use_arc = mode in ("arc", "domains", "motifs")
        self.use_matching = mode in ("matching", "domains", "motifs")
        self.motifs = self._find_motifs() if mode == "motifs" else ()
        self.stats.seconds = perf_counter() - started

    def rejects(self, depth: int, mapping: list[int], used: set[int],
                fragment: FragmentSelector | None = None, fragment_budget: int = 0) -> bool:
        mode = self.deep.mode if self.deep is not None and depth > self.deep.full_depth else None
        if mode == "none":
            return False
        started = perf_counter()
        reason = self._conflict(depth, mapping, used, mode, fragment, fragment_budget)
        self.stats.calls += 1
        self.stats.seconds += perf_counter() - started
        if reason is None:
            return False
        self.stats.cuts[reason] = self.stats.cuts.get(reason, 0) + 1
        return True

    def _conflict(self, depth: int, mapping: list[int], used: set[int], mode: str | None,
                  fragment: FragmentSelector | None, fragment_budget: int) -> str | None:
        started = perf_counter()
        domains = self.build_domains(depth, mapping, used)
        self.stats.record_stage("build_domains", started)
        if domains is None:
            return "empty_domain"

        if fragment is not None:
            started = perf_counter()
            vertices = fragment(depth, domains, mapping)
            self.stats.record_stage("fragment_select", started)
            if vertices:
                started = perf_counter()
                outcome = check_fragment(self.space, domains, vertices, fragment_budget)
                self.stats.record_stage("fragment_check", started)
                if outcome is False:
                    return "fragment"
                if outcome is None:
                    stage = self.stats.stages.setdefault("fragment_exhausted", PruningStageStats())
                    stage.calls += 1

        if mode == "forward":
            return None
        if mode in ("matching", "matching_arc", "adaptive"):
            if not self.covers_matching(domains):
                return "all_different"
            if mode == "matching_arc":
                return self.extra_arc_conflict(domains)
            if mode == "adaptive":
                started = perf_counter()
                check = (self.state_arc_policy(depth, domains, mapping) if self.state_arc_policy is not None
                         else self.arc_policy(depth, domains))
                self.stats.record_stage("arc_policy", started)
                if check:
                    self.stats.arc_policy_checks += 1
                    return self.extra_arc_conflict(domains)
            return None

        if self.use_arc and not self.check_arcs(domains):
            return "arc_consistency"

        if self.use_matching and not self.covers_matching(domains):
            return "all_different"

        if self.motifs:
            started = perf_counter()
            reason = self._check_motifs(domains)
            self.stats.record_stage("motifs", started)
            return reason
        return None

    def build_domains(
        self, depth: int, mapping: list[int], used: set[int]
    ) -> dict[int, frozenset[int]] | None:
        space = self.space
        g_adj = space.g.adj_set
        q_adj = space.q.adj_set
        domains = {}
        for u in space.order[depth:]:
            mapped_neighbours = [g_adj[mapping[w]] for w in q_adj[u] if mapping[w] != -1]
            candidates = space.cand_set[u].intersection(*mapped_neighbours).difference(used)
            if not candidates:
                return None
            domains[u] = candidates
        return domains

    def covers_matching(self, domains: Domains) -> bool:
        started = perf_counter()
        covered = _covers_all(domains)
        self.stats.record_stage("matching", started)
        return covered

    def check_arcs(self, domains: Domains) -> bool:
        started = perf_counter()
        consistent = self._propagate_arcs(domains)
        self.stats.record_stage("arc_consistency", started)
        return consistent

    def extra_arc_conflict(self, domains: Domains) -> str | None:
        """Run AC-3 after a successful matching and recheck if domains shrink."""
        before = domains.copy()
        if not self.check_arcs(domains):
            return "arc_consistency"
        if any(domains[u] is not old for u, old in before.items()) and not self.covers_matching(domains):
            return "all_different"
        return None

    def _propagate_arcs(self, domains: dict[int, frozenset[int]]) -> bool:
        # AC-3: a candidate must have at least one support across every query edge.
        g_adj = self.space.g.adj_set
        q_adj = self.space.q.adj_set
        arcs = deque((u, w) for u in domains for w in q_adj[u] if w in domains)
        while arcs:
            u, w = arcs.popleft()
            old = domains[u]
            support = domains[w]
            revised = frozenset(v for v in old if not g_adj[v].isdisjoint(support))
            if not revised:
                return False
            if len(revised) < len(old):
                domains[u] = revised
                arcs.extend((other, u) for other in q_adj[u] if other != w and other in domains)
        return True

    def _check_motifs(self, domains: dict[int, frozenset[int]]) -> str | None:
        # Try the smallest domain product for each motif size. The domains used
        # here are the reduced domains of this branch only.
        for group in self.motifs:
            choices = (
                (prod(len(domains[u]) for u in motif), motif)
                for motif in group if all(u in domains for u in motif)
            )
            cost, motif = min(choices, default=(MAX_MOTIF_PRODUCT + 1, ()))
            if cost <= MAX_MOTIF_PRODUCT:
                self.stats.motif_checks += 1
                if not self._motif_has_embedding(motif, domains):
                    return "triangle" if len(motif) == 3 else "four_cycle"
        return None

    def _find_motifs(self) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
        q = self.space.q
        triangles = [
            (a, b, c) for a, b in q.edges() for c in q.adj_set[a] & q.adj_set[b] if c > b
        ]
        four_cycles = []
        for a in range(q.n):
            # a is the smallest vertex, b its opposite. The two other
            # vertices are both neighbours of a and b and have no chord.
            for b in range(a + 1, q.n):
                if b in q.adj_set[a]:
                    continue
                sides = sorted(c for c in q.adj_set[a] & q.adj_set[b] if c > a)
                four_cycles.extend(
                    (a, c, b, d) for i, c in enumerate(sides) for d in sides[i + 1:]
                    if d not in q.adj_set[c]
                )
        return triangles, four_cycles

    def _motif_has_embedding(self, motif: tuple[int, ...], domains: dict[int, frozenset[int]]) -> bool:
        q_adj = self.space.q.adj_set
        g_adj = self.space.g.adj_set
        order = sorted(motif, key=lambda u: (len(domains[u]), u))
        embedding: dict[int, int] = {}

        def visit(i: int) -> bool:
            if i == len(order):
                return True
            u = order[i]
            candidates = domains[u].intersection(*(g_adj[v] for w, v in embedding.items() if w in q_adj[u]))
            for v in candidates.difference(embedding.values()):
                embedding[u] = v
                if visit(i + 1):
                    return True
                del embedding[u]
            return False

        return visit(0)


def check_fragment(space: SearchSpace, domains: Domains, vertices: tuple[int, ...],
                   budget: int) -> bool | None:
    """Find a non-induced embedding, exhaust all choices, or return unknown.

    Each candidate inspection costs one unit, including rejected candidates.
    Iterating the original domains keeps even unsuccessful work bounded.
    """
    if not 2 <= len(vertices) <= 4 or len(set(vertices)) != len(vertices):
        raise ValueError("expected 2–4 distinct residual vertices")
    order = sorted(vertices, key=lambda u: (len(domains[u]), u))
    earlier = [[j for j, w in enumerate(order[:i]) if w in space.q.adj_set[u]]
               for i, u in enumerate(order)]
    images = []

    def visit(i: int) -> bool | None:
        nonlocal budget
        if i == len(order):
            return True
        for v in domains[order[i]]:
            if budget <= 0:
                return None
            budget -= 1
            if v in images or any(images[j] not in space.g.adj_set[v] for j in earlier[i]):
                continue
            images.append(v)
            outcome = visit(i + 1)
            images.pop()
            if outcome is not False:
                return outcome
        return False

    return visit(0)


def _covers_all(domains: dict[int, frozenset[int]]) -> bool:
    """Is there a bipartite matching that covers every remaining query vertex?"""
    owner: dict[int, int] = {}

    def augment(u: int, seen: set[int]) -> bool:
        for v in domains[u]:
            if v in seen:
                continue
            seen.add(v)
            if v not in owner or augment(owner[v], seen):
                owner[v] = u
                return True
        return False

    return all(augment(u, set()) for u in sorted(domains, key=lambda u: (len(domains[u]), u)))
