"""Query extraction by random walk, as in NeuGN and the Sun & Luo benchmark.

A query is the subgraph of G *induced* by the distinct vertices a random walk
visits. Because it is cut out of G, the walk itself is a known embedding
(`Query.origin`), so every such query is feasible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from search.graph import Graph

DENSE_MIN_AVG_DEGREE = 3.0  # NeuGN: dense iff average degree >= 3


@dataclass
class Query:
    graph: Graph
    # origin[i] = vertex of G that query vertex i was cut from.
    origin: list[int]

    @property
    def avg_degree(self) -> float:
        return 2 * self.graph.m / self.graph.n

    @property
    def density(self) -> str:
        return "dense" if self.avg_degree >= DENSE_MIN_AVG_DEGREE else "sparse"


def random_walk_vertices(
    g: Graph, start: int, size: int, rng: np.random.Generator, max_moves: int | None = None
) -> list[int] | None:
    """Distinct vertices in visiting order, or None if the walk did not reach
    `size` distinct vertices within `max_moves` moves (default 100·size)."""
    walk, seen, v = [start], {start}, start
    for _ in range(max_moves if max_moves is not None else 100 * size):
        if len(walk) == size:
            break
        nb = g.adj[v]
        if len(nb) == 0:
            return None
        v = int(nb[rng.integers(len(nb))])
        if v not in seen:
            seen.add(v)
            walk.append(v)
    return walk if len(walk) == size else None


def sample_query(
    g: Graph, size: int, rng: np.random.Generator, start: int | None = None
) -> Query | None:
    if start is None:
        start = int(rng.integers(g.n))
    walk = random_walk_vertices(g, start, size, rng)
    if walk is None:
        return None
    return Query(g.induced_subgraph(walk), walk)


def generate_query_set(
    g: Graph,
    size: int,
    density: str | None,
    count: int,
    seed: int,
    max_attempts: int = 200_000,
) -> list[Query]:
    """`count` queries of `size` vertices; `density` in {"dense", "sparse", None}
    is enforced by rejection sampling. May return fewer than `count` if the
    attempt limit is hit (e.g. sparse induced queries in a very dense graph);
    callers must check the length."""
    rng = np.random.default_rng(seed)
    out: list[Query] = []
    for _ in range(max_attempts):
        if len(out) == count:
            break
        q = sample_query(g, size, rng)
        if q is not None and (density is None or q.density == density):
            out.append(q)
    return out
