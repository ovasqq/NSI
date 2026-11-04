"""Eulerian serialisation of a query (NeuGN's input to the Transformer).

`networkx.eulerize` duplicates a minimum-weight set of existing edges so that
every degree becomes even; an Eulerian circuit of the result visits every
edge of Q at least once. Consecutive tokens are therefore always adjacent in
Q, and the set of consecutive pairs is exactly E(Q), so Q is recoverable from
the sequence (together with the vertex count).
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from search.graph import Graph


def euler_sequence(q: Graph, rng: np.random.Generator | None = None) -> list[int]:
    """Query-vertex sequence along an Eulerian circuit of eulerize(Q).

    The start vertex is uniform at random if `rng` is given, else vertex 0.
    The circuit is closed: the last token equals the first one.
    """
    if q.n == 1:
        return [0]
    g = q.to_networkx()
    multi = nx.MultiGraph(g) if nx.is_eulerian(g) else nx.eulerize(g)
    start = int(rng.integers(q.n)) if rng is not None else 0
    circuit = nx.eulerian_circuit(multi, source=start)
    return [start] + [b for _, b in circuit]


def edges_of_sequence(seq: list[int]) -> set[frozenset[int]]:
    return {frozenset((a, b)) for a, b in zip(seq, seq[1:])}
