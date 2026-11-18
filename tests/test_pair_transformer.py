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


def test_from_scratch_does_not_copy_teacher_parameters():
    teacher = NeuGN(NeuGNConfig(5, 1, d=8, heads=2, ffn=16, layers=1))
    with torch.no_grad():
        for parameter in teacher.parameters():
            parameter.fill_(17)
    student = PairTransformer(teacher, from_scratch=True)
    assert not torch.equal(student.neugn.content_emb.weight, teacher.content_emb.weight)
    assert not torch.equal(
        student.neugn.encoder.layers[0].linear1.weight,
        teacher.encoder.layers[0].linear1.weight,
    )
    _, row = example()
    ranks, scores = student(row)
    (ranks.square().mean() + scores.square().mean()).backward()
    assert student.neugn.content_emb.weight.grad is not None
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_online_stream_generates_new_states_each_epoch():
    from train.pair_stream import PairStateStream

    graph = Graph.from_edges(8, [(i, (i + 1) % 8) for i in range(8)], [0] * 8)
    teacher = NeuGN(NeuGNConfig(8, 1, d=8, heads=2, ffn=16, layers=1)).eval()
    args = SimpleNamespace(
        depth=2,
        sizes=[6],
        states_per_epoch=3,
        reference_batch=4,
        epochs=2,
        seed=0,
        device="cpu",
        candidates=2,
        max_pairs=8,
        proposals=3,
        fragment_budget=32,
        budget=1000,
        horizon_ms=1.0,
    )
    stream = PairStateStream(graph, teacher, {}, args)
    first, second = list(stream.epoch(0)), list(stream.epoch(1))
    assert len(first) == len(second) == 3
    assert all(row["local"] for row in first + second)
    assert all(len(row["owners"]) <= args.max_pairs for row in first + second)
    assert any(
        not torch.equal(a["pair_ids"], b["pair_ids"])
        or not torch.equal(a["state"]["content"], b["state"]["content"])
        for a, b in zip(first, second)
    )
    args.states_per_epoch = 0
    assert PairStateStream.epoch_size(graph, args) == 8


def test_batched_padding_and_cached_relations_match_single_state():
    space, row = example()
    teacher = NeuGN(NeuGNConfig(5, 1, d=8, heads=2, ffn=16, layers=1)).eval()
    model = PairTransformer(teacher).eval()
    shorter = encode_pairs(
        space,
        encode_query(space.q, euler_sequence(space.q)),
        [0, 1, -1, -1],
        {2: frozenset({2, 3}), 3: frozenset({2})},
        [(2, 3)],
        2,
        4,
    )
    expected_rank, expected_scores = model(shorter)
    ranks, scores = model.forward_batch([row, shorter])
    torch.testing.assert_close(ranks[1], expected_rank)
    torch.testing.assert_close(scores[1], expected_scores)
    original_rank, original_scores = model(row)
    assert model.cache_graph(space.g)
    row["relations"] = None
    cached_rank, cached_scores = model(row)
    torch.testing.assert_close(cached_rank, original_rank)
    torch.testing.assert_close(cached_scores, original_scores)
    model.train()
    rank, scores = model.forward_batch([row, shorter])
    (rank.square().mean() + sum(score.square().mean() for score in scores)).backward()
    assert model.relations.grad is not None


def test_vectorized_batch_matches_original_per_state_computation():
    _, row = example()
    teacher = NeuGN(NeuGNConfig(5, 1, d=8, heads=2, ffn=16, layers=1)).eval()
    model = PairTransformer(teacher).eval()
    tokens, bias, p, n, owners = model.prepare(row)
    output = model.neugn.encoder(tokens[None], mask=bias, is_causal=False)[0]
    reference_rank = model.neugn.out(output[1])
    reference_score = model.fragment_scores(row, output, p, n, owners)
    rank, score = model(row)
    torch.testing.assert_close(rank, reference_rank)
    torch.testing.assert_close(score, reference_score)


def test_vectorized_loss_matches_mean_of_individual_losses():
    from models.pair_batch import collate_pairs
    from train.pair_loss import pair_loss
    from torch.nn import functional as F

    _, row = example()
    row["gains"] = torch.linspace(-0.1, 0.2, len(row["actions"]))
    row["teacher"] = torch.randn(len(row["local"]))
    shorter = dict(
        row,
        actions=row["actions"][:0],
        gains=torch.empty(0),
        local=row["local"][:1],
        teacher=row["teacher"][:1],
        target_image=1,
    )
    rows = [row, shorter]
    batch = collate_pairs(rows)
    ranks = torch.randn(2, 5, requires_grad=True)
    scores = torch.randn(2, len(row["actions"]), requires_grad=True)
    actual = pair_loss(ranks, scores, batch, torch.zeros_like(ranks))
    references = []
    for i, r in enumerate(rows):
        ids = torch.tensor(r["local"])
        loss = F.kl_div(
            F.log_softmax(ranks[i, ids], -1),
            F.softmax(r["teacher"], -1),
            reduction="sum",
        )
        if "target_image" in r:
            loss = loss + F.cross_entropy(
                ranks[i : i + 1], torch.tensor([r["target_image"]])
            )
        if len(r["actions"]):
            gain = r["gains"]
            values = scores[i, : len(gain)]
            target = gain.sign() * gain.abs().log1p()
            loss = (
                loss
                + (
                    F.smooth_l1_loss(values, target, reduction="none")
                    * (1 + 2 * (gain > 0))
                ).mean()
            )
            best = int(gain.argmax()) if gain.max() > 0 else len(gain)
            loss = loss + 0.1 * F.cross_entropy(
                torch.cat((values, values.new_zeros(1)))[None], torch.tensor([best])
            )
        references.append(loss)
    expected = torch.stack(references).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(ranks.grad).all()
    assert torch.isfinite(scores.grad).all()
