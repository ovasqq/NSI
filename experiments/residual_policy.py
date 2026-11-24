"""Early check policies that use the topology of the remaining query."""

from functools import cache
from itertools import combinations

import numpy as np


RESIDUAL_FEATURES = (
    "remaining_vertices", "remaining_edges", "remaining_triangles",
    "min_edge_product", "small_domain_edges", "small_domain_triangles",
)


def residual_features(space, small_domain=4):
    @cache
    def topology(depth):
        vertices = sorted(space.order[depth:])
        adjacency = space.q.adj_set
        edges = tuple((u, v) for u, v in combinations(vertices, 2) if v in adjacency[u])
        later = dict.fromkeys(vertices, 0)
        for u, v in edges:
            later[u] |= 1 << v
        # Each triangle is counted once, at the edge between its two lowest IDs.
        edges = tuple((u, v, later[u] & later[v]) for u, v in edges)
        return edges, sum(shared.bit_count() for _, _, shared in edges)

    def extract(depth, domains):
        edges, triangles = topology(depth)
        widths = {u: len(values) for u, values in domains.items()}
        small = sum(1 << u for u, width in widths.items() if width <= small_domain)
        min_product = None
        small_edges = small_triangles = 0
        for u, v, shared in edges:
            left, right = widths[u], widths[v]
            product = left * right
            if min_product is None or product < min_product:
                min_product = product
            if left <= small_domain and right <= small_domain:
                small_edges += 1
                small_triangles += (shared & small).bit_count()
        return len(widths), len(edges), triangles, min_product or 0, small_edges, small_triangles

    return extract


def fit_cost_ensemble(features, costs, rounds=8, min_leaf=16, learning_rate=.25):
    """Boost shared decision stumps on edge and fragment costs relative to skip."""
    targets = costs[:, 1:] - costs[:, :1]
    base = targets.mean(axis=0)
    residual = targets - base
    splits = []
    for column, values in enumerate(features.T):
        pivots = np.unique(np.quantile(values, np.linspace(.1, .9, 9), method="nearest"))
        for pivot in pivots:
            left = values <= pivot
            if min(left.sum(), (~left).sum()) >= min_leaf:
                splits.append((column, int(pivot), left))

    stumps = []
    for _ in range(rounds):
        best_gain = 0.
        best = None
        for column, pivot, left in splits:
            left_mean = residual[left].mean(axis=0)
            right_mean = residual[~left].mean(axis=0)
            gain = left.sum() * (left_mean @ left_mean) + (~left).sum() * (right_mean @ right_mean)
            if gain > best_gain:
                best_gain = gain
                best = column, pivot, left, left_mean, right_mean
        if best is None:
            break
        column, pivot, left, left_step, right_step = best
        left_step *= learning_rate
        right_step *= learning_rate
        residual[left] -= left_step
        residual[~left] -= right_step
        stumps.append((column, pivot, left_step.tolist(), right_step.tolist()))
    return base.tolist(), stumps


def cost_action(ensemble, features):
    (edge, fragment), stumps = ensemble
    for column, pivot, left, right in stumps:
        edge_step, fragment_step = left if features[column] <= pivot else right
        edge += edge_step
        fragment += fragment_step
    if edge >= 0 and fragment >= 0:
        return "skip"
    return "edge" if edge <= fragment else "manual"
