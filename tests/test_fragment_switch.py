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
