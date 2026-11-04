"""Stage-2 checks: query generation, storage, Eulerian serialisation."""

import random

import networkx as nx
import numpy as np
import pytest

from queries.euler import edges_of_sequence, euler_sequence
from queries.random_walk import generate_query_set, sample_query
from queries.storage import load_query_set, save_query_set
from search.enumerate import enumerate_matches
from search.filter import nlf_filter
from search.graph import Graph
from search.order import gql_order


@pytest.fixture(scope="module")
def data_graph() -> Graph:
    g = nx.connected_watts_strogatz_graph(300, 6, 0.3, seed=0)
    rng = random.Random(0)
    for v in g.nodes:
        g.nodes[v]["label"] = rng.randrange(4)
    return Graph.from_networkx(g)


def test_sampled_query_is_induced_and_feasible(data_graph):
    g = data_graph
    rng = np.random.default_rng(0)
    for _ in range(100):
        size = int(rng.integers(3, 15))
        q = sample_query(g, size, rng)
        assert q is not None and q.graph.n == size == len(set(q.origin))
        assert nx.is_connected(q.graph.to_networkx())
        o = q.origin
        for a in range(size):
            assert q.graph.labels[a] == g.labels[o[a]]
            for b in range(a + 1, size):
                assert q.graph.has_edge(a, b) == g.has_edge(o[a], o[b])
        # The known embedding is among those the enumerator finds.
        cand = nlf_filter(q.graph, g)
        res = enumerate_matches(q.graph, g, cand, gql_order(q.graph, cand), store_embeddings=True)
        assert tuple(o) in set(res.embeddings)


def test_density_rejection_and_reproducibility(data_graph, tmp_path):
    for density in ("dense", "sparse"):
        qs = generate_query_set(data_graph, 8, density, 20, seed=1)
        assert len(qs) == 20 and all(q.density == density for q in qs)
    a = generate_query_set(data_graph, 8, None, 10, seed=5)
    b = generate_query_set(data_graph, 8, None, 10, seed=5)
    assert [q.origin for q in a] == [q.origin for q in b]

    path = tmp_path / "set.json"
    save_query_set(path, a, {"x": 1})
    loaded, meta = load_query_set(path)
    assert meta == {"x": 1}
    for q, r in zip(a, loaded):
        assert q.origin == r.origin
        assert q.graph.labels.tolist() == r.graph.labels.tolist()
        assert sorted(q.graph.edges()) == sorted(r.graph.edges())


def test_euler_sequence_encodes_query(data_graph):
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = sample_query(data_graph, int(rng.integers(2, 20)), rng).graph
        seq = euler_sequence(q, rng)
        assert seq[0] == seq[-1]
        assert set(seq) == set(range(q.n))
        # Consecutive tokens are edges of Q, and every edge of Q appears.
        assert edges_of_sequence(seq) == {frozenset(e) for e in q.edges()}
        # Eulerization only duplicates edges, so length >= m + 1.
        assert len(seq) >= q.m + 1
