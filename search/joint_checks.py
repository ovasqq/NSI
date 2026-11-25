"""Exact actions shared by fixed, learned and cost-labeling search policies."""

from itertools import islice
from math import prod

from models.fragment import manual_fragment, proposals
from search.pruning import check_fragment, matching_with_reuse
from search.residual_proofs import forest_conflict, local_lad_conflict, triangle_edge_conflict


CHECK_ACTIONS = (
    "skip", "arc", "matching", "both", "lad", "lad_alt", "edge",
    "fragment", "fragment_alt", "forest", "triangle_join", "cascade",
)
DEFER_DELAYS = (64, 128, 256)
MODEL_ACTIONS = (*CHECK_ACTIONS, *(f"defer_{d}" for d in DEFER_DELAYS))


def run_check(pruner, frame, action, *, parent=None, cache=None, residues=None, budget=256,
              propagate=True, reuse_matching=True):
    """Return a conflict reason, keeping successful propagation on this frame."""
    space = pruner.space
    domains = frame.domains
    reduced = domains
    if action in ("arc", "both", "cascade"):
        reduced = domains.copy()
        consistent_parent = parent.domains if parent is not None and parent.arc_consistent else None
        if not pruner.check_arcs(reduced, parent=consistent_parent, residues=residues):
            return "arc"
        if propagate:
            frame.domains = reduced
            frame.arc_consistent = True
    if action in ("matching", "both", "cascade"):
        matching, witness = matching_with_reuse(reduced, frame.matching if reuse_matching else None)
        if witness is not None:
            if cache is not None:
                cache.add_hall(*witness)
            return "matching"
        if propagate:
            frame.matching = matching
    if action in ("lad", "lad_alt", "cascade"):
        if local_lad_conflict(pruner, reduced, offset=2 if action == "lad_alt" else 0):
            return "lad"
    # A successful full AC-3 already guarantees feasibility on every forest
    # relaxation (which drops injectivity). Try the forest independently.
    if action == "forest" and forest_conflict(space, reduced):
        return "forest"
    if action in ("edge", "fragment", "fragment_alt", "triangle_join", "cascade"):
        options = proposals(space, reduced, 12)
        if action == "edge":
            options = sorted((f for f in options if len(f) == 2),
                             key=lambda f: (prod(len(reduced[u]) for u in f), f))
            selected = options[:1]
        elif action == "triangle_join":
            adjacency = space.q.adj_set
            selected = [f for f in options if len(f) == 3
                        and f[1] in adjacency[f[0]] and f[2] in adjacency[f[0]] & adjacency[f[1]]][:1]
        else:
            selected = []
            for _ in range(2 if action in ("fragment_alt", "cascade") else 1):
                if options:
                    choice = manual_fragment(space, reduced, options)
                    selected.append(choice)
                    options.remove(choice)
            if action == "fragment_alt":
                selected = selected[1:]
        for fragment in islice(selected, 2):
            if action == "triangle_join":
                failed = triangle_edge_conflict(space, reduced, fragment, budget) is True
            else:
                failed = check_fragment(space, reduced, fragment, budget) is False
            if failed:
                if cache is not None:
                    cache.add_fragment(fragment, reduced)
                return action if action == "triangle_join" else "fragment"
    return None
