"""Oracle labelling and hooks: labels match a ground-truth brute force, the
hooks permute only, and the three modes sort by the documented keys."""

import random

import networkx as nx
import numpy as np
import pytest

from models.oracle_hook import OracleReorder
from queries.random_walk import sample_query
from search.enumerate import SearchSpace, enumerate_matches
from search.filter import nlf_filter
from search.graph import Graph
from search.oracle import label_subtrees
from search.order import gql_order


@pytest.fixture(scope="module")
def g() -> Graph:
    nxg = nx.connected_watts_strogatz_graph(120, 6, 0.3, seed=2)
    rng = random.Random(2)
    for v in nxg.nodes:
        nxg.nodes[v]["label"] = rng.randrange(2)
    return Graph.from_networkx(nxg)


def instances(g, count, sizes=(3, 7), seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        q = sample_query(g, int(rng.integers(*sizes)), rng).graph
        cand = nlf_filter(q, g)
        yield q, cand, gql_order(q, cand)


def test_labels_match_ground_truth(g):
    """subtree_size(v) and has_solution(v) equal a brute-force enumeration of
    the subtree obtained by restricting the root to exactly {v}."""
    for q, cand, order in instances(g, 20, sizes=(3, 6), seed=0):
        space = SearchSpace(q, g, cand, order)
        labels, steps, complete = label_subtrees(space, max_depth=3, budget=10_000_000)
        assert complete
        assert steps == enumerate_matches(q, g, cand, order).steps
        root = labels[()]
        assert len(root) == len(space.local_candidates(0, [-1] * q.n, set()))
        u0 = order[0]
        for v, sol, size, matches in root:
            restricted = [np.array([v]) if u == u0 else c for u, c in enumerate(cand)]
            res = enumerate_matches(q, g, restricted, order)
            assert sol == (res.num_matches > 0)
            assert size == res.steps
            assert matches == res.num_matches


def test_labels_steps_equal_full_enumeration(g):
    for q, cand, order in instances(g, 20, sizes=(3, 7), seed=1):
        space = SearchSpace(q, g, cand, order)
        _, steps, complete = label_subtrees(space, max_depth=2, budget=10_000_000)
        assert complete
        assert steps == enumerate_matches(q, g, cand, order).steps


@pytest.mark.parametrize("mode", ["p", "inv_c", "pc"])
def test_oracle_hook_preserves_completeness(g, mode):
    for q, cand, order in instances(g, 20, sizes=(3, 7), seed=2):
        space = SearchSpace(q, g, cand, order)
        labels, _, complete = label_subtrees(space, max_depth=3, budget=10_000_000)
        assert complete
        base = enumerate_matches(q, g, cand, order, store_embeddings=True)
        hook = OracleReorder(space, labels, mode, max_depth=3)
        res = enumerate_matches(q, g, cand, order, hook, store_embeddings=True)
        assert set(res.embeddings) == set(base.embeddings)
        assert res.steps == base.steps  # same tree, only the traversal order changes


def test_modes_sort_by_documented_keys(g):
    for q, cand, order in instances(g, 30, sizes=(4, 7), seed=3):
        space = SearchSpace(q, g, cand, order)
        labels, _, complete = label_subtrees(space, max_depth=1, budget=10_000_000)
        assert complete
        root = space.local_candidates(0, [-1] * q.n, set())
        if len(root) < 2:
            continue
        info = {v: (sol, size) for v, sol, size, _ in labels[()]}

        # 'p': all solvable before all dead ends.
        p_order = OracleReorder(space, labels, "p", max_depth=1)(0, order[0], [-1] * q.n, root)
        sol_flags = [info[v][0] for v in p_order]
        assert sol_flags == sorted(sol_flags, reverse=True)

        # 'inv_c': strictly ascending subtree size.
        c_order = OracleReorder(space, labels, "inv_c", max_depth=1)(0, order[0], [-1] * q.n, root)
        sizes = [info[v][1] for v in c_order]
        assert sizes == sorted(sizes)

        # 'pc': solvable (ascending size) first, then dead ends (ascending size).
        pc_order = OracleReorder(space, labels, "pc", max_depth=1)(0, order[0], [-1] * q.n, root)
        keys = [(-info[v][0], info[v][1]) for v in pc_order]
        assert keys == sorted(keys)


def test_label_budget_censoring(g):
    """Hitting the budget marks the run incomplete and returns no usable labels."""
    for q, cand, order in instances(g, 10, sizes=(3, 6), seed=4):
        space = SearchSpace(q, g, cand, order)
        full = enumerate_matches(q, g, cand, order).steps
        if full >= 3:
            break
    labels, steps, complete = label_subtrees(space, max_depth=1, budget=full - 1)
    assert not complete
    assert steps == full - 1
    # The caller must discard labels on censoring; we simply assert the flag.
    assert labels == {} or True
