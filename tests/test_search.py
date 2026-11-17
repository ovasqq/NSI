"""Stage-1 checks: the enumerator agrees with networkx and is order-invariant."""

import random
from time import perf_counter

import networkx as nx
import numpy as np
import pytest
from networkx.algorithms.isomorphism import GraphMatcher

from search.enumerate import SearchSpace, enumerate_matches, enumerate_space
from search.filter import FILTERS
from search.graph import Graph
from search.order import ORDERS
from search.pruning import PRUNING_MODES, DeepPruning, ResidualPruner

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
    stages = result.pruning.stages
    assert stages["build_domains"].calls == 1
    assert stages["arc_consistency"].calls == (reason == "arc_consistency")
    assert stages["matching"].calls == 0


def test_matching_detects_conflict_with_nonempty_domains():
    g = q = Graph.from_edges(4, [(0, 1), (0, 2), (0, 3)], [0] * 4)
    cand = [np.array(v) for v in ([0], [1], [1], [2, 3])]
    arc = enumerate_matches(q, g, cand, [0, 1, 2, 3], pruning="arc", max_steps=1)
    assert not arc.complete and not arc.pruning.cuts
    assert arc.pruning.stages["matching"].calls == 0
    for mode in ("matching", "domains"):
        result = enumerate_matches(q, g, cand, [0, 1, 2, 3], pruning=mode, max_steps=1)
        assert result.complete and result.steps == 1 and result.num_matches == 0
        assert result.pruning.cuts == {"all_different": 1}
        stages = result.pruning.stages
        assert stages["matching"].calls == 1
        assert stages["arc_consistency"].calls == (mode == "domains")
        assert all(stage.seconds >= 0 for stage in stages.values())
        assert sum(stage.seconds for stage in stages.values()) <= result.pruning.seconds


def test_deep_arc_gate_uses_matched_domains_and_rechecks_after_shrinkage():
    q = Graph.from_edges(4, [(0, 1), (0, 2), (0, 3), (1, 2)], [0] * 4)
    g = Graph.from_edges(5, [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2)], [0] * 5)
    candidates = [np.array(row) for row in ([0], [1, 2, 3], [1, 2, 4], [1, 2])]
    space = SearchSpace(q, g, candidates, list(range(q.n)))
    gate_inputs = []

    def gate(depth, domains):
        gate_inputs.append((depth, domains.copy()))
        return True

    pruner = ResidualPruner(space, "domains", DeepPruning(0, "adaptive", lambda _: gate))
    assert pruner.rejects(1, [0, -1, -1, -1], {0})
    assert gate_inputs == [(1, {1: frozenset({1, 2, 3}), 2: frozenset({1, 2, 4}), 3: frozenset({1, 2})})]
    assert pruner.stats.cuts == {"all_different": 1}
    assert pruner.stats.stages["matching"].calls == 2
    assert pruner.stats.stages["build_domains"].calls == 1

    gate_inputs.clear()
    candidates[1:3] = [np.array([1]), np.array([1])]
    space = SearchSpace(q, g, candidates, list(range(q.n)))
    pruner = ResidualPruner(space, "domains", DeepPruning(0, "adaptive", lambda _: gate))
    assert pruner.rejects(1, [0, -1, -1, -1], {0})
    assert pruner.stats.cuts == {"all_different": 1}
    assert pruner.stats.stages["arc_consistency"].calls == 0
    assert not gate_inputs


def test_deep_pruning_boundary_counts_assignments():
    q = g = Graph.from_edges(4, [(0, 1), (1, 2), (2, 3)], [0] * 4)
    cand = [np.array([i]) for i in range(q.n)]
    options = {"domains": None, "none": DeepPruning(2, "none"),
               "forward": DeepPruning(2, "forward"), "matching": DeepPruning(2, "matching"),
               "adaptive": DeepPruning(2, "adaptive", lambda _: lambda depth, domains: False)}
    for mode, deep in options.items():
        result = enumerate_matches(q, g, cand, list(range(q.n)), pruning="domains",
                                   deep_pruning=deep, store_embeddings=True)
        assert result.complete and result.embeddings == [(0, 1, 2, 3)]
        assert result.pruning.calls == (2 if mode == "none" else 3)
        assert result.pruning.stages["arc_consistency"].calls == (3 if mode == "domains" else 2)
        assert result.pruning.stages["matching"].calls == (2 if mode in ("none", "forward") else 3)


def test_delayed_pruning_continues_from_current_state():
    g = q = Graph.from_edges(3, [(0, 1), (1, 2)], [0] * 3)
    cand = [np.array([u]) for u in range(q.n)]
    for delay, expected_calls in ((0, 2), (1, 1), (3, 0)):
        result = enumerate_matches(q, g, cand, [0, 1, 2], pruning="matching",
                                   prune_after_steps=delay, store_embeddings=True)
        assert result.complete and result.steps == 3
        assert result.embeddings == [(0, 1, 2)]
        assert result.pruning.calls == expected_calls


def test_deadline_does_not_report_an_exhausted_tree():
    g = q = Graph.from_edges(2, [(0, 1)], [0] * 2)
    cand = [np.array([0]), np.array([1])]
    result = enumerate_matches(q, g, cand, [0, 1], deadline=perf_counter() - 1)
    assert result.timed_out and not result.complete and result.steps == 0


def test_branch_observation_distinguishes_first_match_from_step_limit():
    g = q = Graph.from_edges(3, [(0, 1), (1, 2)], [0] * 3)
    cand = [np.array([0]), np.array([1]), np.array([2])]
    observed = []

    def observe(branch, mapping, used):
        assert mapping[branch.vertex] == branch.candidate and branch.candidate in used
        return lambda resolved: observed.append((branch.depth, resolved))

    enumerate_matches(q, g, cand, [0, 1, 2], max_matches=1, max_steps=2, branch_observer=observe)
    assert observed == [(1, False), (0, False)]
    observed.clear()
    enumerate_matches(q, g, cand, [0, 1, 2], max_matches=1, branch_observer=observe)
    assert observed == [(1, True), (0, True)]


def test_policy_requests_a_check_without_deciding_feasibility():
    q = Graph.from_edges(3, [(0, 1), (1, 2)], [0] * 3)
    g = Graph.from_edges(4, [(0, 1), (1, 2)], [0] * 4)
    cand = [np.array([0]), np.array([1]), np.array([2, 3])]
    for accept in (False, True):
        result = enumerate_matches(q, g, cand, [0, 1, 2], pruning="domains", store_embeddings=True,
                                   pruning_policy=lambda branch: accept)
        assert result.complete and result.embeddings == [(0, 1, 2)]
        assert result.pruning.calls == (2 if accept else 0)


def test_branch_replay_preserves_prefix_and_checks_descendants():
    edges = [(0, 1), (1, 2), (2, 3)]
    q = Graph.from_edges(4, edges, [0] * 4)
    g = Graph.from_edges(5, edges, [0] * 5)
    space = SearchSpace(q, g, [np.arange(5) for _ in range(4)], [3, 2, 1, 0])
    visited = []

    def visit(branch, mapping):
        if branch.depth == 0:
            visited.append(mapping[3])

    original = enumerate_space(space, pruning="domains", before_check=visit)
    assert original.pruning.cuts["empty_domain"] >= 1
    assert 4 in visited
    replay = enumerate_space(space, pruning="domains", prefix=(3, 2), store_embeddings=True)
    assert replay.complete and replay.embeddings == [(0, 1, 2, 3)]
    assert replay.steps == 2 and replay.pruning.calls == 1


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
    assert motifs.pruning.stages["motifs"].calls == 1
    assert motifs.pruning.cuts == {reason: 1}
