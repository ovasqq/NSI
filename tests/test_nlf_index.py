import numpy as np

from search.filter import nlf_filter
from search.graph import Graph
from search.nlf_index import NLFIndex


def test_indexed_nlf_matches_original_on_varied_labels_and_graphs():
    rng = np.random.default_rng(24)
    for _ in range(50):
        n = 18
        edges = [(u, v) for u in range(n) for v in range(u) if rng.random() < 0.25]
        graph = Graph.from_edges(n, edges, rng.choice([-7, 2, 100], n))
        index = NLFIndex(graph, cache_size=5)
        for _ in range(6):
            size = int(rng.integers(2, 9))
            query = graph.induced_subgraph(rng.choice(n, size, replace=False).tolist())
            expected = nlf_filter(query, graph)
            for _ in range(2):
                actual = index.filter(query)
                for old, new in zip(expected, actual):
                    np.testing.assert_array_equal(old, new)
            assert len(index.cache) <= 5
        absent = Graph.from_edges(2, [(0, 1)], [-7, 999])
        for old, new in zip(nlf_filter(absent, graph), index.filter(absent)):
            np.testing.assert_array_equal(old, new)
