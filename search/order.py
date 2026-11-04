"""Matching orders φ for connected queries. Every vertex after the first has at
least one earlier neighbour (a backward neighbour), so local candidates are
always computed by intersection, never from the full C(u)."""

from __future__ import annotations

import numpy as np

from search.graph import Graph


def gql_order(q: Graph, cand: list[np.ndarray]) -> list[int]:
    """GQL-style (Sun & Luo 2020): start from argmin |C(u)|, then repeatedly
    take the unvisited neighbour of the visited set with the fewest candidates.
    Ties: larger query degree, then smaller id."""
    key = lambda u: (len(cand[u]), -int(q.degree[u]), u)
    order = [min(range(q.n), key=key)]
    seen = {order[0]}
    frontier = set(q.adj[order[0]].tolist())
    while len(order) < q.n:
        if not frontier:
            raise ValueError("query graph is not connected")
        u = min(frontier, key=key)
        order.append(u)
        seen.add(u)
        frontier.discard(u)
        frontier.update(w for w in q.adj[u].tolist() if w not in seen)
    return order


def ri_order(q: Graph, cand: list[np.ndarray] | None = None) -> list[int]:
    """RI-style (Bonnici et al. 2013), structure only: start from the max-degree
    vertex, then take the vertex with most backward neighbours. Ties: more
    neighbours adjacent to a backward neighbour, then larger degree, then id."""
    order = [max(range(q.n), key=lambda u: (int(q.degree[u]), -u))]
    seen = {order[0]}
    while len(order) < q.n:
        best, best_key = None, None
        for u in range(q.n):
            if u in seen:
                continue
            nb = q.adj[u].tolist()
            back = sum(w in seen for w in nb)
            if back == 0:
                continue
            near = sum(
                w not in seen and any(x in seen for x in q.adj[w].tolist()) for w in nb
            )
            k = (back, near, int(q.degree[u]), -u)
            if best_key is None or k > best_key:
                best, best_key = u, k
        if best is None:
            raise ValueError("query graph is not connected")
        order.append(best)
        seen.add(best)
    return order


ORDERS = {"gql": gql_order, "ri": ri_order}
