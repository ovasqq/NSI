"""Inductive model checks: features, forward, inductivity, learning, hook."""

import random

import networkx as nx
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from models.inductive import InductiveConfig, InductiveNeuGN, g_features
from models.inductive_hook import InductiveReorder
from queries.random_walk import sample_query
from search.enumerate import enumerate_matches
from search.filter import nlf_filter
from search.graph import Graph
from search.order import gql_order
from train.train_inductive import collate_inductive, make_examples


@pytest.fixture(scope="module")
def g() -> Graph:
    nxg = nx.connected_watts_strogatz_graph(120, 6, 0.3, seed=2)
    rng = random.Random(2)
    for v in nxg.nodes:
        nxg.nodes[v]["label"] = rng.randrange(3)
    return Graph.from_networkx(nxg)


def small_model(g: Graph, **kw) -> InductiveNeuGN:
    torch.manual_seed(0)
    cfg = InductiveConfig(num_labels=3, d=32, layers=1, heads=4, ffn=64, **kw)
    return InductiveNeuGN(cfg, *g_features(g, 3)).eval()


def test_g_features(g):
    labels, logdeg, nlf = g_features(g, 3)
    assert (labels == g.labels).all()
    assert np.allclose(logdeg, np.log1p(g.degree))
    assert nlf.shape == (g.n, 3)
    assert np.allclose(nlf.sum(1), 1.0)  # connected graph, normalized by degree


def test_model_has_no_vertex_id_parameters(g):
    model = small_model(g)
    for p in model.parameters():
        for dim in p.shape:
            assert dim != g.n, f"trainable parameter {tuple(p.shape)} depends on |V(G)|"


def test_forward_shapes_and_padding_invariance(g):
    model = small_model(g)
    rng = np.random.default_rng(0)
    items = []
    while len(items) < 4:
        items.extend(make_examples(g, rng, 4, 6, label_depth=2, budget=10_000_000))
    batch = collate_inductive(items)
    s = model(batch)
    assert s.shape == (len(items), 32)
    scores = model.score(s, batch["cand"])
    assert scores.shape == batch["cand"].shape
    alone = collate_inductive(items[:1])
    assert torch.allclose(model(alone), s[:1], atol=1e-5)


def test_learning(g):
    model = small_model(g).train()
    rng = np.random.default_rng(1)
    items = []
    while len(items) < 8:
        items.extend(make_examples(g, rng, 4, 6, label_depth=2, budget=10_000_000))
    batch = collate_inductive(items)

    def loss_fn():
        s = model(batch)
        scores = model.score(s, batch["cand"])
        m = batch["cand_mask"]
        return F.mse_loss(scores[m], batch["target"][m])

    first = loss_fn().item()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(30):
        loss = loss_fn()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss_fn().item() < 0.9 * first


def test_hook_preserves_completeness(g):
    model = small_model(g)
    rng = np.random.default_rng(3)
    for _ in range(20):
        q = sample_query(g, int(rng.integers(4, 8)), rng).graph
        cand = nlf_filter(q, g)
        order = gql_order(q, cand)
        base = enumerate_matches(q, g, cand, order, store_embeddings=True)
        hook = InductiveReorder(model, q, max_depth=3)
        res = enumerate_matches(q, g, cand, order, hook, store_embeddings=True)
        assert set(res.embeddings) == set(base.embeddings)
        assert res.steps == base.steps


def test_hook_ranks_by_scores(g):
    model = small_model(g)
    rng = np.random.default_rng(4)
    q = sample_query(g, 6, rng).graph
    hook = InductiveReorder(model, q, max_depth=3)
    cands = list(range(0, 40, 3))
    out = hook(0, 0, [-1] * q.n, cands)
    s = hook.scores([-1] * q.n, 0, cands)
    assert sorted(out) == cands
    assert [s[cands.index(v)].item() for v in out] == sorted(s.tolist(), reverse=True)
    assert hook(3, 0, [-1] * q.n, cands) == cands  # beyond max_depth
