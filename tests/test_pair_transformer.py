import numpy as np
import torch
from types import SimpleNamespace

from experiments.pair_search import Search
from models.neugn import NeuGN, NeuGNConfig
from models.pair_transformer import PairTransformer, encode_pairs
from models.encoding import encode_query
from queries.euler import euler_sequence
from search.enumerate import SearchSpace, enumerate_space
from search.graph import Graph


def example():
    q = Graph.from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 1)], [0] * 4)
    g = Graph.from_edges(5, [(0, 1), (0, 2), (1, 2), (2, 3)], [0] * 5)
    space = SearchSpace(q, g, [np.arange(5)] * 4, list(range(4)))
    domains = {1: frozenset({1, 2}), 2: frozenset({2, 3}), 3: frozenset({1, 2, 3})}
    row = encode_pairs(
        space,
        encode_query(q, euler_sequence(q)),
        [0, -1, -1, -1],
        domains,
        [(1, 2), (1, 2, 3)],
        2,
        6,
    )
    return (space, row)


def test_pair_relations_and_joint_training():
    space, row = example()
    ids = row["pair_ids"].tolist()
    owners = row["owners"].tolist()
    assert len(ids) <= 6
    for i in range(len(ids)):
        for j in range(len(ids)):
            rel = row["relations"][i, j]
            assert bool(rel[1]) == (ids[i] == ids[j] and owners[i] != owners[j])
    backbone = (
        NeuGN(NeuGNConfig(5, 1, d=8, heads=2, ffn=16, layers=1))
        .eval()
        .requires_grad_(False)
    )
    model = PairTransformer(backbone).requires_grad_(True).eval()
    ranks, scores = model(row)
    assert ranks.shape == (5,)
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()
    (ranks.square().mean() + scores.square().mean()).backward()
    assert model.relations.grad is not None
    assert model.neugn.encoder.layers[0].linear1.weight.grad is not None
    assert all((p.grad is None for p in backbone.parameters()))


def test_branch_check_preserves_exhaustive_search_and_handles_deadline():
    space, _ = example()
    baseline = enumerate_space(space, store_embeddings=True)
    calls = []

    def check(depth, mapping, used):
        calls.append((depth, tuple(mapping), set(used)))
        assert len(used) == depth
        return False

    result = enumerate_space(space, branch_check=check, store_embeddings=True)
    assert result.complete
    assert result.embeddings == baseline.embeddings
    assert calls
    stopped = enumerate_space(space, branch_check=check, deadline=0.0)
    assert stopped.timed_out and (not stopped.complete)


def test_empty_fragment_list_still_ranks():
    _, row = example()
    row["actions"] = torch.zeros((0, len(row["vertices"])), dtype=torch.bool)
    backbone = NeuGN(NeuGNConfig(5, 1, d=8, heads=2, ffn=16, layers=1))
    model = PairTransformer(backbone).eval()
    ranks, scores = model(row)
    assert ranks.shape == (5,) and scores.shape == (0,)


def test_joint_search_preserves_embeddings_with_active_checks():
    torch.set_num_threads(1)
    space, _ = example()
    backbone = NeuGN(NeuGNConfig(5, 1, d=8, heads=2, ffn=16, layers=1)).eval()
    model = PairTransformer(backbone).eval()
    args = SimpleNamespace(
        depth=1,
        device="cpu",
        proposals=12,
        candidates=2,
        max_pairs=6,
        fragment_budget=1000,
    )
    expected = enumerate_space(space, store_embeddings=True)
    session = Search(space, backbone, model, args, "pair_joint", "always")
    result = enumerate_space(
        space, session.reorder, branch_check=session.check, store_embeddings=True
    )
    assert result.complete
    assert set(result.embeddings) == set(expected.embeddings)
    assert session.metrics["checks"] > 0
    assert session.metrics["transformer_calls"] > 0
