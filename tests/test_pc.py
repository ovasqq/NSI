"""p/c ordering: Knuth estimate is unbiased; the p/c hook keeps completeness."""

import random

import networkx as nx
import numpy as np
import pytest
import torch

from models.hook import NeuGNReorder
from models.neugn import NeuGN, NeuGNConfig
from models.pc_hook import PCReorder
from queries.random_walk import sample_query
from search.enumerate import SearchSpace, enumerate_matches
from search.filter import nlf_filter
from search.graph import Graph
from search.knuth import knuth_estimate
from search.order import gql_order


@pytest.fixture(scope="module")
def g() -> Graph:
    nxg = nx.connected_watts_strogatz_graph(120, 6, 0.3, seed=2)
    rng = random.Random(2)
    for v in nxg.nodes:
        nxg.nodes[v]["label"] = rng.randrange(2)
    return Graph.from_networkx(nxg)


def instances(g, count, sizes=(3, 6), seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        q = sample_query(g, int(rng.integers(*sizes)), rng).graph
        cand = nlf_filter(q, g)
        yield q, cand, gql_order(q, cand)


def test_knuth_estimate_is_unbiased_and_restores_state(g):
    rng = np.random.default_rng(0)
    for q, cand, order in instances(g, 10):
        space = SearchSpace(q, g, cand, order)
        true_nodes = enumerate_matches(q, g, cand, order).steps + 1  # + root
        mapping, used = [-1] * q.n, set()
        est, steps = knuth_estimate(space, 0, mapping, used, rng, probes=4000)
        assert mapping == [-1] * q.n and used == set()
        assert steps >= 4000
        assert abs(est - true_nodes) / true_nodes < 0.15


@pytest.mark.parametrize("with_model", [False, True])
def test_pc_hook_preserves_completeness(g, with_model):
    torch.manual_seed(0)
    model = NeuGN(NeuGNConfig(num_g=g.n, num_labels=2, d=32, layers=1, heads=4, ffn=64)).eval()
    for q, cand, order in instances(g, 15, seed=1):
        base = enumerate_matches(q, g, cand, order, store_embeddings=True)
        nav = NeuGNReorder(model, q) if with_model else None
        hook = PCReorder(SearchSpace(q, g, cand, order), nav, pc_depth=3, probes=2)
        res = enumerate_matches(q, g, cand, order, hook, store_embeddings=True)
        assert set(res.embeddings) == set(base.embeddings)
        assert res.steps == base.steps
        assert hook.probe_steps > 0 or base.steps <= q.n


def test_knuth_only_prefers_smaller_subtrees(g):
    """No model, many probes: the root candidate ranked first has (nearly)
    the smallest subtree, measured exactly by forcing it first."""
    for q, cand, order in instances(g, 5, sizes=(4, 6), seed=3):
        space = SearchSpace(q, g, cand, order)
        root = space.local_candidates(0, [-1] * q.n, set())
        if len(root) < 2:
            continue
        ranked = PCReorder(space, None, pc_depth=1, probes=3000)(0, order[0], [-1] * q.n, root)
        exact = {}
        for v in root:
            only_v = SearchSpace(q, g, [np.array([v]) if u == order[0] else c
                                       for u, c in enumerate(cand)], order)
            exact[v] = enumerate_matches(q, g, only_v.cand, order).steps
        assert exact[ranked[0]] <= min(exact.values()) * 1.3 + 2
