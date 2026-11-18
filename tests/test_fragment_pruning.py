from itertools import permutations

import numpy as np
import torch

from models.fragment import FragmentHead, batch_rows, encode, proposals
from models.neugn import NeuGN, NeuGNConfig
from search.enumerate import SearchSpace, enumerate_space
from search.graph import Graph
from search.pruning import ResidualPruner, check_fragment


def test_fragment_proves_a_triangle_impossible_after_arc_consistency():
    q = Graph.from_edges(3, [(0, 1), (1, 2), (2, 0)], [0] * 3)
    g = Graph.from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)], [0] * 4)
    space = SearchSpace(q, g, [np.arange(4)] * 3, [0, 1, 2])
    domains = {u: frozenset(range(4)) for u in range(3)}
    pruner = ResidualPruner(space, "domains")
    assert pruner.check_arcs(domains)
    assert pruner.covers_matching(domains)
    assert check_fragment(space, domains, (0, 1, 2), 1) is None
    assert check_fragment(space, domains, (0, 1, 2), 200) is False
    assert check_fragment(space, domains, (0, 1), 200) is True


def test_fragment_search_preserves_all_noninduced_embeddings():
    q = Graph.from_edges(4, [(1, 2), (2, 3), (3, 1)], [0] * 4)
    g = Graph.from_edges(7, [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 4)], [0] * 7)
    space = SearchSpace(q, g, [np.arange(7)] * 4, list(range(4)))
    expected = {p for p in permutations(range(7), 4) if all(g.has_edge(p[u], p[v]) for u, v in q.edges())}

    def select(depth, domains, mapping):
        return (1, 2, 3) if depth == 1 else ()

    for budget in (1, 1000):
        result = enumerate_space(space, pruning="forward", fragment=select, fragment_budget=budget,
                                 store_embeddings=True)
        assert result.complete
        assert set(result.embeddings) == expected
        if budget == 1:
            assert "fragment" not in result.pruning.cuts
            assert result.pruning.stages["fragment_exhausted"].calls > 0
        else:
            assert result.pruning.cuts["fragment"] == 3


def test_head_sees_shared_candidate_ids_and_updates_without_touching_neugn():
    q = Graph.from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 1)], [0] * 4)
    g = Graph.from_edges(5, [(0, 1), (0, 2), (1, 2), (2, 3)], [0] * 5)
    space = SearchSpace(q, g, [np.arange(5)] * 4, [0, 3, 1, 2])
    domains = {1: frozenset({1, 2}), 2: frozenset({2, 3}), 3: frozenset({1, 2, 3, 4})}
    options = proposals(space, domains, 12)
    assert (1, 2, 3) in options
    anchor = torch.zeros(8)
    row = encode(space, domains, 1, [0, -1, -1, -1], 2, options, anchor)
    shared = row["ids"].tolist().index(2)
    assert row["ownership"][:, shared].tolist() == [True, True, True]
    data_vertex = row["ids"].tolist().index(1)
    assert row["data_edges"][shared, data_vertex]
    shorter = encode(space, {1: frozenset({1}), 2: frozenset({2})}, 2, [0, -1, -1, 4],
                     2, [(1, 2)], anchor)
    model = NeuGN(NeuGNConfig(5, 1, d=8, heads=2, ffn=16)).eval().requires_grad_(False)
    head = FragmentHead(8)
    scores = head(model, batch_rows([row, shorter], "cpu"))
    assert scores.shape == (2, len(options))
    alone = head(model, batch_rows([shorter], "cpu"))
    torch.testing.assert_close(scores[1, 0], alone[0, 0])
    scores.sum().backward()
    assert head.score[-1].weight.grad is not None
    assert all(parameter.grad is None for parameter in model.parameters())
