"""Candidate filtering: C(u) ⊆ V(G) for every query vertex u.

Both filters are sound: if f is an embedding then f(u) ∈ C(u) for every u.
"""

from __future__ import annotations

import numpy as np

from search.graph import Graph


def ldf_filter(q: Graph, g: Graph) -> list[np.ndarray]:
    """Label and Degree Filter: label(v) = label(u) and deg(v) ≥ deg(u)."""
    empty = np.empty(0, dtype=np.int64)
    out = []
    for u in range(q.n):
        same = g.by_label.get(int(q.labels[u]), empty)
        out.append(same[g.degree[same] >= q.degree[u]])
    return out


def nlf_filter(q: Graph, g: Graph) -> list[np.ndarray]:
    """LDF + Neighbour Label Frequency: for every label l, v has at least as
    many l-labelled neighbours as u does."""
    out = []
    for u, cand in enumerate(ldf_filter(q, g)):
        need = q.neighbor_label_counts(u)
        keep = [
            v
            for v in cand.tolist()
            if all(g.neighbor_label_counts(v)[l] >= k for l, k in need.items())
        ]
        out.append(np.array(keep, dtype=np.int64))
    return out


FILTERS = {"ldf": ldf_filter, "nlf": nlf_filter}
