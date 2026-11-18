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
