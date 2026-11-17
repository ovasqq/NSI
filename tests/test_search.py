"""Stage-1 checks: the enumerator agrees with networkx and is order-invariant."""

import random

import networkx as nx
import numpy as np
import pytest
from networkx.algorithms.isomorphism import GraphMatcher

from search.enumerate import enumerate_matches
from search.filter import FILTERS
from search.graph import Graph
from search.order import ORDERS
from search.pruning import PRUNING_MODES

N_CASES = 300


def random_data_graph(rng: random.Random) -> Graph:
    n = rng.randint(6, 22)
    p = rng.uniform(0.15, 0.5)
    g = nx.gnp_random_graph(n, p, seed=rng.randrange(1 << 30))
    num_labels = rng.randint(1, 3)
    for v in g.nodes:
        g.nodes[v]["label"] = rng.randrange(num_labels)
    return Graph.from_networkx(g)


def random_query(rng: random.Random, g: Graph) -> Graph | None:
    """Half the time a random-walk induced subgraph of g (always feasible),
    otherwise a random connected graph (often infeasible)."""
    size = rng.randint(2, 6)
    if rng.random() < 0.5:
        start = rng.randrange(g.n)
        if g.degree[start] == 0:
            return None
        walk, v = [start], start
        for _ in range(50):
            if len(walk) == size:
                break
            v = rng.choice(g.adj[v].tolist())
            if v not in walk:
                walk.append(v)
        q = g.induced_subgraph(walk)
        if rng.random() < 0.5 and q.m > q.n - 1:
            # Drop a non-bridge edge: non-induced query, still connected.
            qn = q.to_networkx()
            bridges = {frozenset(e) for e in nx.bridges(qn)}
            extra = [e for e in qn.edges if frozenset(e) not in bridges]
            qn.remove_edge(*rng.choice(extra))
            q = Graph.from_networkx(qn)
        return q
    qn = nx.gnp_random_graph(size, rng.uniform(0.3, 0.9), seed=rng.randrange(1 << 30))
    if not nx.is_connected(qn):
        return None
    labels = sorted(set(g.labels.tolist()))
    for v in qn.nodes:
        qn.nodes[v]["label"] = rng.choice(labels)
    return Graph.from_networkx(qn)


def nx_embeddings(q: Graph, g: Graph) -> set[tuple[int, ...]]:
    gm = GraphMatcher(
        g.to_networkx(), q.to_networkx(), node_match=lambda a, b: a["label"] == b["label"]
    )
    out = set()
    for m in gm.subgraph_monomorphisms_iter():  # maps G-vertex -> Q-vertex
        inv = {qv: gv for gv, qv in m.items()}
        out.add(tuple(inv[u] for u in range(q.n)))
    return out


def cases(seed: int):
    rng = random.Random(seed)
    produced = 0
    while produced < N_CASES:
        g = random_data_graph(rng)
        q = random_query(rng, g)
        if q is None:
            continue
        produced += 1
        yield rng, q, g


@pytest.mark.parametrize("filter_name", sorted(FILTERS))
@pytest.mark.parametrize("order_name", sorted(ORDERS))
def test_matches_networkx(filter_name, order_name):
    feasible = 0
    seed = sorted(FILTERS).index(filter_name) * 10 + sorted(ORDERS).index(order_name)
    for _, q, g in cases(seed=seed):
        expected = nx_embeddings(q, g)
        cand = FILTERS[filter_name](q, g)
        # Filter soundness: every true embedding survives filtering.
        for emb in expected:
            assert all(emb[u] in set(cand[u].tolist()) for u in range(q.n))
        res = enumerate_matches(q, g, cand, ORDERS[order_name](q, cand), store_embeddings=True)
        assert res.complete
        assert len(res.embeddings) == res.num_matches == len(set(res.embeddings))
        assert set(res.embeddings) == expected
        assert res.match_steps == sorted(res.match_steps)
        feasible += bool(expected)
    # The case generator must exercise both feasible and infeasible queries.
    assert 0.2 * N_CASES < feasible < 0.95 * N_CASES


def test_reordering_preserves_embeddings_and_tree_size():
    for rng, q, g in cases(seed=7):
        cand = FILTERS["nlf"](q, g)
        order = ORDERS["gql"](q, cand)
        base = enumerate_matches(q, g, cand, order, store_embeddings=True)

        def shuffle(depth, u, mapping, local):
            local = list(local)
            rng.shuffle(local)
            return local

        for hook in (shuffle, lambda d, u, m, c: c[::-1]):
            res = enumerate_matches(q, g, cand, order, hook, store_embeddings=True)
            assert set(res.embeddings) == set(base.embeddings)
            # Same tree, different traversal order: total steps are invariant.
            assert res.steps == base.steps


def test_hook_cannot_prune():
    for _, q, g in cases(seed=3):
        cand = FILTERS["nlf"](q, g)
        order = ORDERS["gql"](q, cand)
        res = enumerate_matches(q, g, cand, order)
        if res.steps > q.n:
            break
    with pytest.raises(ValueError):
        enumerate_matches(q, g, cand, order, lambda d, u, m, c: c[1:])


def test_limits():
    for _, q, g in cases(seed=11):
        cand = FILTERS["nlf"](q, g)
        order = ORDERS["gql"](q, cand)
        full = enumerate_matches(q, g, cand, order)
        first = enumerate_matches(q, g, cand, order, max_matches=1)
        if full.num_matches:
            assert first.num_matches == 1
            assert first.steps == first.first_match_step == full.first_match_step
        else:
            assert first.complete and first.steps == full.steps
        if full.steps >= 2:
            cut = enumerate_matches(q, g, cand, order, max_steps=full.steps - 1)
            assert not cut.complete and cut.steps == full.steps - 1


def test_pruning_preserves_embedding_order():
    for _, q, g in cases(seed=17):
        cand = FILTERS["nlf"](q, g)
        order = ORDERS["gql"](q, cand)

        def reverse_candidates(depth, u, mapping, local):
            return local[::-1]

        base = enumerate_matches(q, g, cand, order, reverse_candidates, store_embeddings=True)
        assert set(base.embeddings) == nx_embeddings(q, g)
        for mode in PRUNING_MODES[1:]:
            result = enumerate_matches(q, g, cand, order, reverse_candidates,
                                       store_embeddings=True, pruning=mode)
            assert result.complete
            assert result.embeddings == base.embeddings
            assert result.steps <= base.steps
            first = enumerate_matches(q, g, cand, order, reverse_candidates, max_matches=1,
                                      store_embeddings=True, pruning=mode)
            assert first.embeddings == base.embeddings[:1]
            if base.embeddings:
                assert first.first_match_step <= base.first_match_step


@pytest.mark.parametrize(("last_domain", "reason"), [([0], "empty_domain"), ([3], "arc_consistency")])
def test_empty_and_unsupported_domains(last_domain, reason):
    q = Graph.from_edges(3, [(0, 1), (1, 2)], [0] * 3)
    g = Graph.from_edges(4, [(0, 1), (1, 2)], [0] * 4)
    cand = [np.array(v) for v in ([0], [1], last_domain)]
    result = enumerate_matches(q, g, cand, [0, 1, 2], pruning="domains", max_steps=1)
    assert result.complete and result.steps == 1 and result.num_matches == 0
    assert result.pruning.cuts == {reason: 1}


def test_matching_detects_conflict_with_nonempty_domains():
    g = q = Graph.from_edges(4, [(0, 1), (0, 2), (0, 3)], [0] * 4)
    cand = [np.array(v) for v in ([0], [1], [1], [2, 3])]
    arc = enumerate_matches(q, g, cand, [0, 1, 2, 3], pruning="arc", max_steps=1)
    result = enumerate_matches(q, g, cand, [0, 1, 2, 3], pruning="domains", max_steps=1)
    assert not arc.complete and not arc.pruning.cuts
    assert result.complete and result.steps == 1 and result.num_matches == 0
    assert result.pruning.cuts == {"all_different": 1}


@pytest.mark.parametrize(("query_cycle", "target_cycle", "reason"), [
    (3, 6, "triangle"), (4, 5, "four_cycle"),
])
def test_motifs_detect_conflicts_that_domains_miss(query_cycle, target_cycle, reason):
    def wheel(size):
        cycle = [(u + 1, (u + 1) % size + 1) for u in range(size)]
        return Graph.from_edges(size + 1, cycle + [(0, u) for u in range(1, size + 1)],
                                [1] + [0] * size)

    q, g = wheel(query_cycle), wheel(target_cycle)
    cand = FILTERS["nlf"](q, g)
    order = list(range(q.n))
    domains = enumerate_matches(q, g, cand, order, pruning="domains", max_steps=1)
    motifs = enumerate_matches(q, g, cand, order, pruning="motifs", max_steps=1)
    assert not domains.complete and domains.steps == 1 and not domains.pruning.cuts
    assert motifs.complete and motifs.steps == 1 and motifs.num_matches == 0
    assert motifs.pruning.motif_checks == 1
    assert motifs.pruning.cuts == {reason: 1}
