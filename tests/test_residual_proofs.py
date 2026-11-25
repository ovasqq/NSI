from itertools import product
from types import SimpleNamespace

from search.graph import Graph
from search.pruning import ResidualPruner, hall_conflict
from search.residual_proofs import ProofCache, local_lad_conflict


def test_hall_witness_matches_brute_force_on_small_domains():
    values = (frozenset({0}), frozenset({1}), frozenset({0, 1}), frozenset({2}))
    for choice in product(values, repeat=3):
        domains = dict(enumerate(choice))
        witness = hall_conflict(domains)
        can_match = any(len(set(images)) == 3 for images in product(*choice))
        assert (witness is None) == can_match
        if witness is not None:
            variables, images = witness
            assert len(set(variables)) == len(variables) > len(images)
            assert images == frozenset().union(*(domains[u] for u in variables))


def test_proof_cache_requires_scope_and_subset():
    cache = ProofCache(limit=1)
    cache.add_hall((0, 1), frozenset({2}))
    assert cache.conflict({0: frozenset({2}), 1: frozenset({2})}) == "hall"
    assert cache.conflict({0: frozenset({2})}) is None
    assert cache.conflict({0: frozenset({2}), 1: frozenset({2, 3})}) is None
    cache.add_fragment((0, 1), {0: frozenset({0, 1}), 1: frozenset({2})})
    assert cache.conflict({0: frozenset({1}), 1: frozenset({2})}) == "fragment"
    assert cache.conflict({0: frozenset({2}), 1: frozenset({2})}) is None


def test_local_lad_proves_conflict_after_arc_and_matching_pass():
    query = Graph.from_edges(3, [(0, 1), (0, 2)], [0]*3)
    graph = Graph.from_edges(4, [(0, 2), (1, 3)], [0]*4)
    pruner = ResidualPruner(SimpleNamespace(q=query, g=graph), "forward")
    domains = {0: frozenset({0, 1}), 1: frozenset({2, 3}), 2: frozenset({2, 3})}
    assert pruner.check_arcs(domains.copy())
    assert pruner.covers_matching(domains)
    assert local_lad_conflict(pruner, domains)
    assert not local_lad_conflict(pruner, domains, max_width=1)
