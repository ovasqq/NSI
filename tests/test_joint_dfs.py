from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.joint_dfs import StateFeatures, vertex_features
from models.neugn import NeuGN, NeuGNConfig
from search.enumerate import SearchSpace, enumerate_space
from search.graph import Graph
from search.joint_checks import CHECK_ACTIONS, run_check
from search.pruning import ResidualPruner, check_fragment, matching_with_reuse
from search.residual_proofs import ProofCache


def test_domain_projection_uses_same_checkpoint_weights():
    model = NeuGN(NeuGNConfig(12, 2, d=8, heads=2))
    state = torch.randn(2, 8)
    indices = torch.tensor([[4, 1, 9], [2, 8, 0]])
    full = model.out(state)
    torch.testing.assert_close(model.candidate_logits(state, indices), full.gather(1, indices))
    torch.testing.assert_close(model.candidate_logits(state, indices[0]), full[:, indices[0]])


def space_of(query, graph):
    return SearchSpace(query, graph, [np.arange(graph.n)]*query.n, list(range(query.n)))


def test_incremental_features_and_checks_match_on_sibling_frames():
    query = Graph.from_edges(4, [(0, 1), (1, 2), (2, 3)], [0]*4)
    graph = Graph.from_edges(6, [(u, v) for u, v in combinations(range(6), 2)
                                if (u+v) % 3], [0]*6)
    space = space_of(query, graph)
    pruner = ResidualPruner(space, "forward")
    features = StateFeatures(space, vertex_features(graph))
    root = features.frame(dict(enumerate(space.cand_set)))
    assert not run_check(pruner, root, "both")
    for image in (1, 2, 3):
        domains = pruner.extend_domains(root.domains, 0, image)
        child = features.frame(domains, root, (0, image))
        np.testing.assert_allclose(child.mass, features.frame(domains).mass, atol=1e-6)
        rebuilt = domains.copy()
        assert pruner.check_arcs(rebuilt)
        assert pruner.check_arcs(domains, parent=root.domains, residues={})
        assert rebuilt == domains
        matching, proof = matching_with_reuse(domains, root.matching)
        assert proof is None and len(set(matching.values())) == len(domains)
        assert all(v in domains[u] for u, v in matching.items())


def test_all_actions_preserve_known_embedding():
    query = Graph.from_edges(4, [(0, 1), (1, 2), (0, 2), (2, 3)], [0]*4)
    graph = Graph.from_edges(5, [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4)], [0]*5)
    pruner = ResidualPruner(space_of(query, graph), "forward")
    domains = {u: frozenset({u, (u+1) % 5}) for u in range(4)}
    for action in CHECK_ACTIONS:
        frame = SimpleNamespace(domains=domains, arc_consistent=False, matching=None)
        assert run_check(pruner, frame, action, cache=ProofCache(), budget=128) is None, action
        assert all(u in frame.domains[u] for u in domains)


@pytest.mark.parametrize("adaptive", (False, True))
def test_scheduler_preserves_embeddings(adaptive):
    graph = Graph.from_edges(5, [(1, 2), (1, 3), (2, 3)], [0]*5)
    query = Graph.from_edges(3, [(0, 1), (0, 2), (1, 2)], [0]*3)
    # Allow 0 for the first vertex even though it has no feasible continuation.
    space = space_of(query, graph)
    snapshots = {}
    proofs = ProofCache()
    pruner = ResidualPruner(space, "forward")

    def advance(depth, mapping, used, parent):
        new = (pruner.build_domains(depth, mapping, used) if parent is None
               else pruner.extend_domains(parent, space.order[depth-1], mapping[space.order[depth-1]]))
        if new is not None:
            snapshots[depth] = new
        return new

    baseline = enumerate_space(space, branch_domains=advance, store_embeddings=True)
    calls = []

    def check(depth, domains):
        calls.append(depth)
        frame = SimpleNamespace(domains=domains, arc_consistent=False, matching=None)
        return run_check(pruner, frame, "cascade", cache=proofs, propagate=False) is not None

    def jump(depth):
        return next((i for i in range(1, depth+1) if proofs.conflict(snapshots[i])), None)

    options = {"late_schedule": lambda depth, domains: 0 if min(domains[1]) == 0 else 1} if adaptive else {}
    result = enumerate_space(space, branch_domains=advance, late_check=check, late_steps=0,
                             late_depth=1, backjump_depth=jump, store_embeddings=True, **options)
    assert calls and result.complete
    assert result.embeddings == baseline.embeddings


def test_fragment_proof_can_jump_over_an_irrelevant_assignment():
    graph = Graph.from_edges(7, [(0, 4), (1, 5), (5, 6)], [0]*7)
    query = Graph.from_edges(4, [(0, 2), (2, 3)], [0]*4)
    candidates = [np.asarray(v) for v in ([0, 1], [2, 3], [4, 5], [6])]
    space = SearchSpace(query, graph, candidates, [0, 1, 2, 3])
    pruner = ResidualPruner(space, "forward")
    cache = ProofCache()
    snapshots, targets = {}, []

    def advance(depth, mapping, used, parent):
        new = (pruner.build_domains(depth, mapping, used) if parent is None
               else pruner.extend_domains(parent, depth-1, mapping[depth-1]))
        if new is not None:
            snapshots[depth] = new
            if depth == 2 and check_fragment(space, new, (2, 3), 32) is False:
                cache.add_fragment((2, 3), new)
                return None
        return new

    def jump(depth):
        if depth and cache.conflict(snapshots[1]) is not None:
            targets.append(1)
            return 1
        return None

    baseline = enumerate_space(space, branch_domains=advance, store_embeddings=True)
    closures = []
    result = enumerate_space(space, branch_domains=advance, backjump_depth=jump,
                             ancestor_closed=lambda *event: closures.append(event), store_embeddings=True)
    assert closures == [("backjump", 1, 1)]
    assert targets == [1]
    assert result.complete and result.embeddings == baseline.embeddings
    assert result.steps < baseline.steps


def test_joint_adaptive_runs_the_requested_deferred_cascade():
    from experiments.joint_search import run
    from search.joint_checks import MODEL_ACTIONS

    query = Graph.from_edges(7, [(0, 5), (0, 6), (5, 6)], [0]*7)
    graph = Graph.from_edges(10, [(u, v) for u in range(5) for v in range(5, 10)], [0]*10)
    space = space_of(query, graph)

    class Defer:
        @staticmethod
        def encode(features):
            return features

        @staticmethod
        def scores(hidden, candidates):
            return np.zeros(len(candidates))

        @staticmethod
        def costs(hidden):
            costs = np.zeros(len(MODEL_ACTIONS)-1)
            costs[MODEL_ACTIONS.index("defer_64")-1] = -1
            return costs

    args = SimpleNamespace(seconds=None, budget=5000, fragment_budget=256)
    result = run(space, None, vertex_features(graph), args, "joint_adaptive", light=Defer())
    assert not result["solved"]
    assert result["defer_64"] >= result["check_cascade"] > 0
    assert result["closure_steps_deferred"] == 64 * result["closed_deferred"]
    assert result["proved_cascade"] == result["closed_deferred"]


def test_replay_same_unsat_prefix_and_censoring():
    from experiments.joint_search import run

    query = Graph.from_edges(4, [(1, 2), (2, 3), (1, 3)], [0]*4)
    graph = Graph.from_edges(6, [(u, v) for u in range(3) for v in range(3, 6)], [0]*6)
    space = space_of(query, graph)
    args = SimpleNamespace(seconds=None, budget=1000, fragment_budget=256)
    data = vertex_features(graph)
    base = run(space, None, data, args, "joint_rank", prefix=(0,))
    checked = run(space, None, data, args, "joint_fixed", prefix=(0,))
    assert base["outcome"] == checked["outcome"] == "unsat"
    assert checked["steps"] == 0 < base["steps"]
    assert checked["direct_cuts"] == checked["closed_immediate"] == 1
    args.budget = 1
    assert run(space, None, data, args, "joint_rank", prefix=(0,))["outcome"] == "step_limit"


def test_shared_arc_labels_match_cpu_and_batched_continuations():
    from random import Random

    from experiments.joint_search import arc_labels
    from search.batched_arcs import sparse_adjacency

    query = Graph.from_edges(3, [(0, 1), (1, 2)], [0]*3)
    graph = Graph.from_edges(5, [(0, 1), (1, 2), (2, 3)], [0]*5)
    pruner = ResidualPruner(space_of(query, graph), "forward")
    frames = [
        SimpleNamespace(domains={0: frozenset({0, 4}), 1: frozenset({1}), 2: frozenset({2})},
                        matching=None),
        SimpleNamespace(domains={0: frozenset({4}), 1: frozenset({1}), 2: frozenset({2})},
                        matching=None),
    ]
    cpu = arc_labels(pruner, frames, None, 2, 1, Random(19))
    batched = arc_labels(pruner, frames, sparse_adjacency(graph), 2, 2, Random(19))
    assert [r[0] for r in batched] == [r[0] for r in cpu] == [
        {0: frozenset({0}), 1: frozenset({1}), 2: frozenset({2})}, None,
    ]
    assert sum(r[2] for r in batched) == 1
    for frame, (reduced, _, _) in zip(frames, batched, strict=True):
        for action in ("arc", "both", "cascade"):
            actual = "arc" if reduced is None else run_check(
                pruner, frame, action, propagate=False, arc_domains=reduced,
            )
            assert actual == run_check(pruner, frame, action, propagate=False)


def test_neugn_action_head_reuses_only_current_parent_context():
    from experiments.joint_search import run

    query = Graph.from_edges(5, list(combinations(range(5), 2)), [0]*5)
    graph = Graph.from_edges(7, list(combinations(range(7), 2)), [0]*7)
    space = space_of(query, graph)
    teacher = NeuGN(NeuGNConfig(7, 1, d=8, heads=2, layers=1, ffn=16)).eval()
    data = vertex_features(graph)
    args = SimpleNamespace(seconds=None, budget=100, fragment_budget=256, depth=1, device="cpu")
    seen = []

    class Skip:
        def costs(self, features, anchor):
            seen.append((bool(features[-1]), None if anchor is None else len(anchor)))
            return np.zeros(3)

    forward = run(space, teacher, data, args, "neugn_forward", store_embedding=True)
    checked = run(space, teacher, data, args, "neugn_head_anchor", store_embedding=True,
                  neugn_heads={"neugn_head_anchor": Skip()})
    assert forward["embedding"] == checked["embedding"]
    assert forward["steps"] == checked["steps"]
    assert checked["rank_calls"] == checked["anchor_decisions"] == 1
    assert seen == [(True, 8), (False, None), (False, None)]


def test_full_depth_neugn_reuses_check_results_on_equal_sibling_domains():
    from experiments.joint_search import run

    query = Graph.from_edges(4, list(combinations(range(4), 2)), [0]*4)
    graph = Graph.from_edges(6, [(a, b) for a, b in combinations(range(6), 2)
                                 if a//2 != b//2], [0]*6)
    space = space_of(query, graph)
    teacher = NeuGN(NeuGNConfig(6, 1, d=8, heads=2, layers=1, ffn=16)).eval()
    args = SimpleNamespace(seconds=None, budget=1000, fragment_budget=256, depth=1,
                           device="cpu", neugn_full_depth_pilot=True)

    class Forest:
        def costs(self, features, anchor):
            return np.array([0., 0., -1.])

    def trial(**options):
        return run(space, teacher, vertex_features(graph), args, "neugn_head_anchor",
                   neugn_heads={"neugn_head_anchor": Forest()}, **options)

    cached, uncached = trial(), trial(cache_proofs=False)
    limited = trial(rank_all_depths=False)
    assert cached["outcome"] == uncached["outcome"] == "unsat"
    assert cached["steps"] == uncached["steps"]
    assert cached["cache_result_hits"] > 0
    assert cached["check_forest"] + cached["cache_result_hits"] == uncached["check_forest"]
    assert cached["rank_calls"] > limited["rank_calls"] == 1
    assert cached["anchor_decisions"] > limited["anchor_decisions"]
