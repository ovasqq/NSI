"""Indexed equivalent of NLF for repeated queries on one fixed graph."""

from collections import OrderedDict

import numpy as np

from search.graph import Graph


class NLFIndex:
    def __init__(self, graph: Graph, cache_size=4096):
        self.graph = graph
        self.cache_size = cache_size
        labels = sorted(graph.by_label)
        self.columns = {label: index for index, label in enumerate(labels)}
        self.counts = np.zeros((graph.n, len(labels)), dtype=np.int64)
        for vertex in range(graph.n):
            for label, count in graph.neighbor_label_counts(vertex).items():
                self.counts[vertex, self.columns[int(label)]] = count
        self.cache = OrderedDict()

    def filter(self, query: Graph):
        result = []
        for u in range(query.n):
            needs = tuple(sorted(query.neighbor_label_counts(u).items()))
            key = (int(query.labels[u]), int(query.degree[u]), needs)
            candidates = self.cache.get(key)
            if candidates is None:
                candidates = self.graph.by_label.get(
                    key[0], np.empty(0, dtype=np.int64)
                )
                candidates = candidates[self.graph.degree[candidates] >= key[1]]
                for label, required in needs:
                    if label not in self.columns:
                        candidates = candidates[:0]
                        break
                    candidates = candidates[
                        self.counts[candidates, self.columns[label]] >= required
                    ]
                candidates.flags.writeable = False
                self.cache[key] = candidates
                if len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
            else:
                self.cache.move_to_end(key)
            result.append(candidates)
        return result
