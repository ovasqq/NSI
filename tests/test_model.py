"""Stage-3 checks: token format, training examples, model, search hook."""

import random

import networkx as nx
import numpy as np
import pytest
import torch

from models.encoding import cls_id, collate, encode_query, encode_state, pad_id
from models.hook import NeuGNReorder
from models.neugn import NeuGN, NeuGNConfig
from models.neugn_gate import CandidateGate
from queries.euler import euler_sequence
from queries.random_walk import sample_query
from search.enumerate import BranchState, enumerate_matches
from search.filter import nlf_filter
from search.graph import Graph
from search.order import gql_order
from train.dataset import TrainingStream, collate_examples, make_example


@pytest.fixture(scope="module")
def g() -> Graph:
    nxg = nx.connected_watts_strogatz_graph(200, 6, 0.3, seed=1)
    rng = random.Random(1)
    for v in nxg.nodes:
        nxg.nodes[v]["label"] = rng.randrange(3)
    return Graph.from_networkx(nxg)


def small_model(g: Graph, **kw) -> NeuGN:
    torch.manual_seed(0)
    cfg = NeuGNConfig(num_g=g.n, num_labels=3, d=32, layers=1, heads=4, ffn=64, **kw)
    return NeuGN(cfg).eval()


def test_encode_state_tokens(g):
    rng = np.random.default_rng(0)
    q = sample_query(g, 8, rng)
    seq = euler_sequence(q.graph, rng)
    mapping = list(q.origin)
    mapping[3] = mapping[5] = -1
    s = encode_state(encode_query(q.graph, seq), mapping, target=3, shift=60, num_g=g.n)
    for j, w in enumerate(seq):
        if w == 3:
            assert s["content"][j] == cls_id(g.n)
        elif w == 5:
            assert s["content"][j] == pad_id(g.n)
        else:
            assert s["content"][j] == q.origin[w]
        assert s["qvert"][j] == (w + 60) % 64
        assert s["tok_label"][j] == q.graph.labels[w]
    # Normalised GCN adjacency is symmetric with self-loops.
    a = s["q_adj"]
    assert np.allclose(a, a.T) and (np.diag(a) > 0).all()


def test_training_examples_are_consistent(g):
    rng = np.random.default_rng(0)
    for _ in range(200):
        start = int(rng.integers(g.n))
        state, target_id = make_example(g, start, g.n, rng)
        content = state["content"]
        assert (content == cls_id(g.n)).any()
        # The target is never revealed, and mapped tokens are real G ids
        # with the right label.
        assert target_id not in content.tolist()
        real = content < g.n
        assert (g.labels[content[real]] == state["tok_label"][real]).all()
        # CLS positions carry the target's label.
        assert (state["tok_label"][content == cls_id(g.n)] == g.labels[target_id]).all()


def test_stream_skips_small_components():
    nxg = nx.disjoint_union(nx.path_graph(10), nx.path_graph(3))
    for v in nxg.nodes:
        nxg.nodes[v]["label"] = 0
    small = Graph.from_networkx(nxg)
    stream = TrainingStream(small, seed=0)
    assert set(stream.starts.tolist()) == set(range(10))
    batch = collate_examples([next(iter(stream)) for _ in range(4)])
    assert batch["content"].shape[0] == 4


@pytest.mark.parametrize("kw", [{}, {"use_gcn": False}, {"token_labels": False}])
def test_model_forward_and_learning(g, kw):
    model = small_model(g, **kw).train()
    rng = np.random.default_rng(0)
    items = [make_example(g, int(rng.integers(g.n)), g.n, rng) for _ in range(16)]
    batch = collate_examples(items)
    logits = model(batch)
    assert logits.shape == (16, g.n)
    # Padding must not change the result: a sample alone vs inside a batch.
    model.eval()
    alone = model(collate_examples(items[:1]))
    together = model(batch)[:1]
    assert torch.allclose(alone, together, atol=1e-5)
    # A few steps on one batch reduce the loss (gradients flow everywhere).
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(30):
        loss = torch.nn.functional.cross_entropy(model(batch), batch["target"])
        first = loss.item() if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first


def test_hook_preserves_completeness(g):
    model = small_model(g)
    rng = np.random.default_rng(3)
    for _ in range(20):
        q = sample_query(g, int(rng.integers(4, 9)), rng).graph
        cand = nlf_filter(q, g)
        order = gql_order(q, cand)
        base = enumerate_matches(q, g, cand, order, store_embeddings=True)
        hook = NeuGNReorder(model, q, max_depth=10)
        res = enumerate_matches(q, g, cand, order, hook, store_embeddings=True)
        assert set(res.embeddings) == set(base.embeddings)
        assert res.steps == base.steps


def test_candidate_gate_follows_ranked_candidates(g):
    class CheckByEmbedding(torch.nn.Module):
        def forward(self, state, candidates):
            return candidates[..., 0]

    model = small_model(g)
    q = sample_query(g, 6, np.random.default_rng(1)).graph
    candidates = [0, 1, 2, 3]
    with torch.no_grad():
        model.content_emb.weight[candidates, 0] = torch.tensor([-1., 1., -1., 1.])
        model.out.weight.zero_()
        model.out.bias[candidates] = torch.tensor([2., 4., 3., 1.])
    hook = NeuGNReorder(model, q, max_depth=2, check_head=CheckByEmbedding())
    assert hook(0, 0, [-1] * q.n, candidates) == [1, 2, 0, 3]
    policy = CandidateGate.for_query(hook, g, q)
    decisions = [policy(BranchState(0, 0, v, i, 4, i + 1, None, 0, 0))
                 for i, v in enumerate([1, 2, 0, 3])]
    assert decisions == [True, False, False, True]
    assert policy(BranchState(2, 0, 0, 0, 4, 5, None, 0, 0))
    assert policy(BranchState(0, 0, 0, 0, 1, 6, None, 0, 0))
    assert (hook.gate_decisions, hook.gate_skips) == (4, 2)


def test_hook_ranks_by_model_scores(g):
    model = small_model(g)
    rng = np.random.default_rng(4)
    q = sample_query(g, 6, rng).graph
    hook = NeuGNReorder(model, q)
    cands = list(range(0, 40, 3))
    out = hook(0, 0, [-1] * q.n, cands)
    s = hook.scores([-1] * q.n, 0, cands)
    assert sorted(out) == cands
    assert [s[cands.index(v)] for v in out] == sorted(s.tolist(), reverse=True)
    traced = NeuGNReorder(model, q, capture_state=True)
    assert traced(0, 0, [-1] * q.n, cands) == out
    anchor, ranked_scores = traced.ranking_context[0]
    assert anchor.shape == (model.cfg.d,)
    torch.testing.assert_close(ranked_scores, s.sort(descending=True).values)
    # Beyond max_depth the baseline order is kept.
    assert hook(10, 0, [-1] * q.n, cands) == cands
