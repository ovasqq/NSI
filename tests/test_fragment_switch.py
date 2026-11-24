from types import SimpleNamespace

import pytest

from experiments import fragment_switch as switch
from search.graph import Graph


@pytest.fixture
def space():
    q = Graph.from_edges(5, [(0, 1), (1, 2), (0, 2), (3, 4)], [0]*5)
    return SimpleNamespace(q=q)


def test_threshold_and_baselines(space):
    domains = {u: set(range(w)) for u, w in enumerate([4, 4, 4, 2, 2])}
    assert switch.select(space, domains, "switch", 64, 12) == ("manual", (0, 1, 2))
    assert switch.select(space, domains, "switch", 4, 12) == ("edge", (3, 4))
    assert switch.select(space, domains, "manual", None, 12) == ("manual", (0, 1, 2))
    assert switch.select(space, domains, "edge", None, 12) == ("edge", (3, 4))


def test_skip_before_proposals(space, monkeypatch):
    def unexpected(*args):
        pytest.fail("selector built unused proposals")
    monkeypatch.setattr(switch, "proposals", unexpected)
    assert switch.select(space, {0: {1, 2}, 1: {3, 4}}, "switch", 3, 12) == ("skip", ())
    assert switch.select(space, {0: {1}, 3: {2}}, "switch", 100, 12) == ("skip", ())


def test_cheap_edge_outside_proposals(space, monkeypatch):
    monkeypatch.setattr(switch, "proposals", lambda *a: [(0, 1), (0, 1, 2)])
    domains = {u: set(range(w)) for u, w in enumerate([4, 4, 4, 2, 2])}
    assert switch.select(space, domains, "switch", 4, 12) == ("skip", ())


def test_residual_features_include_topology_and_follow_depth(space):
    from experiments.residual_policy import residual_features

    space.order = (4, 0, 1, 2, 3)
    extract = residual_features(space)
    domains = {u: range(w) for u, w in enumerate([2, 3, 4, 5, 1])}
    assert extract(0, domains) == (5, 4, 1, 5, 3, 1)
    del domains[4]
    assert extract(1, domains) == (4, 3, 1, 6, 3, 1)
    domains[0] = range(5)
    assert extract(1, domains) == (4, 3, 1, 12, 1, 0)


def test_residual_ensemble_learns_check_or_skip_from_costs():
    import numpy as np
    from experiments.residual_policy import cost_action, fit_cost_ensemble

    features = np.repeat([[0], [1]], 32, axis=0)
    costs = np.repeat([[100, 10, 120], [10, 100, 120]], 32, axis=0)
    ensemble = fit_cost_ensemble(features, costs)
    assert cost_action(ensemble, (0,)) == "edge"
    assert cost_action(ensemble, (1,)) == "skip"


def test_incremental_domains_match_rebuild_across_siblings_and_prefixes():
    import numpy as np
    from search.enumerate import SearchSpace, enumerate_space
    from search.pruning import ResidualPruner

    q = Graph.from_edges(4, [(0, 1), (1, 2)], [0]*4)
    g = Graph.from_edges(6, [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)], [0]*6)
    space = SearchSpace(q, g, [np.arange(6)]*4, [0, 1, 3, 2])
    pruner = ResidualPruner(space, "forward")
    def advance(depth, mapping, used, parent):
        expected = pruner.build_domains(depth, mapping, used)
        if parent is None:
            return expected
        u = space.order[depth-1]
        child = pruner.extend_domains(parent, u, mapping[u])
        assert child == expected
        return child
    for prefix in ((), (0,)):
        for limit in (None, 14):
            args = dict(prefix=prefix, max_steps=limit, store_embeddings=True,
                        reorder=lambda d, u, m, c: reversed(c))
            base = enumerate_space(space, pruning="forward", **args)
            result = enumerate_space(space, branch_domains=advance, **args)
            assert (result.embeddings, result.match_steps, result.steps, result.complete) == (
                base.embeddings, base.match_steps, base.steps, base.complete)


def test_budgeted_fragment_reuse_matches_original(monkeypatch):
    graph = Graph.from_edges(6, [(u, v) for u in range(6) for v in range(u)], [0]*6)
    query = SimpleNamespace(graph=Graph.from_edges(4, [(0, 1), (1, 2), (2, 3)], [0]*4))
    class FixedOrder:
        seconds = 0
        def __call__(self, depth, vertex, mapping, local):
            return local
    monkeypatch.setattr(switch, "NeuGNReorder", lambda *args: FixedOrder())
    args = SimpleNamespace(depth=1, device="cpu", budget=100, fragment_budget=1,
                           proposals=12, time_limit=10)
    records = []
    for strategy in ("rebuild", "reuse", "incremental"):
        args.domain_strategy = strategy
        r = switch.run(query, graph, None, args, "switch", 256)
        assert r["counts"]["unknown"] > 0
        records.append((r["steps"], r["fms"], r["counts"]))
    assert records[0] == records[1] == records[2]


def test_delayed_proof_cuts_active_ancestor_and_restores_siblings():
    import numpy as np
    from search.enumerate import SearchSpace, enumerate_space
    from search.pruning import ResidualPruner, check_fragment

    query = Graph.from_edges(4, [(0, 2), (0, 3), (2, 3)], [0]*4)
    graph = Graph.from_edges(6, [(0, 1), (0, 2), (3, 4), (4, 5), (3, 5)], [0]*6)
    space = SearchSpace(query, graph, [np.arange(6)]*4, [0, 1, 2, 3])
    pruner = ResidualPruner(space, "forward")
    proofs = []

    def advance(depth, mapping, used, parent):
        if parent is None:
            return pruner.build_domains(depth, mapping, used)
        u = space.order[depth-1]
        return pruner.extend_domains(parent, u, mapping[u])

    def late(depth, domains):
        if depth != 1 or check_fragment(space, domains, (2, 3), 100) is not False:
            return False
        assert 1 in domains  # The isolated vertex has not been assigned in the saved ancestor.
        proofs.append(depth)
        return True

    baseline = enumerate_space(space, branch_domains=advance, store_embeddings=True)
    adaptive = enumerate_space(space, branch_domains=advance, store_embeddings=True,
                               late_check=late, late_depth=1, late_steps=1)
    assert proofs
    assert adaptive.complete and adaptive.embeddings == baseline.embeddings
    assert adaptive.steps < baseline.steps
