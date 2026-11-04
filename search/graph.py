"""Undirected vertex-labelled graph used both for data graphs G and queries Q."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import networkx as nx
import numpy as np


@dataclass
class Graph:
    """Vertices are 0..n-1. `adj[v]` is a sorted int64 array of neighbours."""

    labels: np.ndarray
    adj: list[np.ndarray]
    adj_set: list[frozenset[int]] = field(init=False, repr=False)
    degree: np.ndarray = field(init=False, repr=False)
    by_label: dict[int, np.ndarray] = field(init=False, repr=False)
    _nlf: list[Counter] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.labels = np.asarray(self.labels, dtype=np.int64)
        assert len(self.adj) == len(self.labels)
        self.adj_set = [frozenset(a.tolist()) for a in self.adj]
        self.degree = np.array([len(a) for a in self.adj], dtype=np.int64)
        self.by_label = {
            int(l): np.flatnonzero(self.labels == l) for l in np.unique(self.labels)
        }

    @property
    def n(self) -> int:
        return len(self.labels)

    @property
    def m(self) -> int:
        return int(self.degree.sum()) // 2

    @property
    def num_labels(self) -> int:
        return len(self.by_label)

    def neighbor_label_counts(self, v: int) -> Counter:
        """NLF profile of v: label -> number of neighbours with that label (cached)."""
        if self._nlf is None:
            self._nlf = [Counter(self.labels[a].tolist()) for a in self.adj]
        return self._nlf[v]

    def has_edge(self, a: int, b: int) -> bool:
        return b in self.adj_set[a]

    def edges(self) -> Iterable[tuple[int, int]]:
        for a, nb in enumerate(self.adj):
            for b in nb[nb > a].tolist():
                yield a, b

    @classmethod
    def from_edges(cls, n: int, edges: Iterable[tuple[int, int]], labels) -> "Graph":
        """Builds a simple graph: self-loops dropped, duplicate/reversed edges merged."""
        nbrs: list[set[int]] = [set() for _ in range(n)]
        for a, b in edges:
            a, b = int(a), int(b)
            if a == b:
                continue
            nbrs[a].add(b)
            nbrs[b].add(a)
        adj = [np.array(sorted(s), dtype=np.int64) for s in nbrs]
        return cls(np.asarray(labels), adj)

    @classmethod
    def from_networkx(cls, g: nx.Graph, label_attr: str = "label") -> "Graph":
        """Relabels nodes to 0..n-1 in sorted order of the original node keys."""
        nodes = sorted(g.nodes)
        idx = {x: i for i, x in enumerate(nodes)}
        labels = [g.nodes[x][label_attr] for x in nodes]
        return cls.from_edges(len(nodes), ((idx[a], idx[b]) for a, b in g.edges), labels)

    def to_networkx(self, label_attr: str = "label") -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from((v, {label_attr: int(l)}) for v, l in enumerate(self.labels))
        g.add_edges_from(self.edges())
        return g

    def induced_subgraph(self, vertices: list[int]) -> "Graph":
        """Query induced by `vertices`; query vertex i corresponds to vertices[i]."""
        idx = {v: i for i, v in enumerate(vertices)}
        edges = [
            (idx[a], idx[b])
            for a in vertices
            for b in self.adj[a].tolist()
            if b in idx and a < b
        ]
        return Graph.from_edges(len(vertices), edges, self.labels[vertices])
