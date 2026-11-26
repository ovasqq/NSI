from itertools import combinations, product
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.triple_pointer import TriplePointer
from search.graph import Graph
from search.triple_check import TripleCheck


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_triples_against_exhaustive_reference(device):
    graph = Graph.from_edges(5, [(0,1),(1,2),(2,0),(3,4)], [0]*5)
    q = Graph.from_edges(5, [(0,1),(1,2),(0,2),(2,3)], [0]*5)
    space = SimpleNamespace(q=q)
    triples = list(combinations(range(5), 3))
    checker = TripleCheck(graph, device, 125)
    rng = np.random.default_rng(27)
    for _ in range(15):
        domains = {u:frozenset(np.where(rng.random(5)<.5)[0].tolist()) for u in range(5)}
        expected = [any(len(set(images)) == 3 and all(
            triple[j] not in q.adj_set[triple[i]] or images[j] in graph.adj_set[images[i]]
            for i,j in combinations(range(3),2)) for images in product(*(domains[u] for u in triple)))
                    for triple in triples]
        assert checker.many(space, domains, triples) == expected
    checker.budget = 1
    assert checker.many(space, {u:frozenset(range(5)) for u in range(5)}, [(0,1,2)]) == [None]


def test_pointer_selects_distinct_vertices_and_replays_log_probability():
    graph = Graph.from_edges(4, [(0,1),(1,2),(2,3)], [0]*4)
    domains = {u:frozenset([0,u+1]) for u in range(4)}
    space = SimpleNamespace(q=graph, cand_set=domains)
    model = TriplePointer("cpu", 4)
    anchor = np.ones(4, np.float32)
    triple, lp = model(space, domains, anchor, sample=True)
    assert len(set(triple)) == 3
    _, actual = model(space, domains, anchor, forced=triple)
    torch.testing.assert_close(lp, actual)
    selected, imitation = model(space, domains, anchor, witness=(0,2,3))
    assert set(selected) == {0,2,3}
    (-imitation-actual).backward()
    assert model.relations.weight.grad is not None


def test_variable_pointer_replays_stop_and_can_select_the_whole_residual():
    graph = Graph.from_edges(5, [(0, 1), (1, 2), (2, 3)], [0]*5)
    domains = {u: frozenset([0, u+1]) for u in range(5)}
    space = SimpleNamespace(q=graph, cand_set=domains)
    model = TriplePointer("cpu", variable_size=True)
    with torch.no_grad():
        model.stop.weight.zero_()
        model.stop.bias.fill_(100.)
    subset, logprob = model(space, domains, None)
    assert len(set(subset)) == 2
    _, replayed = model(space, domains, None, forced=subset)
    torch.testing.assert_close(logprob, replayed)
    whole, imitation = model(space, domains, None, witness=tuple(domains))
    assert set(whole) == set(domains)
    assert imitation.isfinite()


def test_fragment_larger_than_four_handles_conflict_and_budget():
    from search.pruning import check_fragment

    graph = Graph.from_edges(5, [], [0]*5)
    space = SimpleNamespace(q=graph, g=graph)
    vertices = tuple(range(5))
    domains = {u: frozenset(range(4)) for u in vertices}
    assert check_fragment(space, domains, vertices, 1) is None
    assert check_fragment(space, domains, vertices, 4096) is False
    domains[4] = frozenset({4})
    assert check_fragment(space, domains, vertices, 4096) is True
