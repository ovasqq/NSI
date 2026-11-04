"""Knuth's (1975) estimate of search-tree size by random probes.

From a node, walk down picking a uniformly random child at every level until
a leaf or a dead end. With branching factors d_1, d_2, … along the walk,
1 + d_1 + d_1·d_2 + … is an unbiased estimate of the number of nodes in the
subtree (Knuth, "Estimating the efficiency of backtrack programs"). Averaging
over several probes reduces the variance, which is still huge for skewed
trees, so callers use it on a log scale.

Each level of a probe computes one C_M(u) — about the work of one search
step — so a probe of L levels is charged L steps (`probe_steps`).
"""

from __future__ import annotations

import numpy as np

from search.enumerate import SearchSpace


def knuth_estimate(
    space: SearchSpace, i: int, mapping: list[int], used: set[int],
    rng: np.random.Generator, probes: int,
) -> tuple[float, int]:
    """Estimated size of the subtree rooted at the current node, whose next
    position to fill is i (the node itself counts as 1). Returns
    (estimate, probe steps spent). `mapping`/`used` are restored on return."""
    n = len(space.order)
    total, steps = 0.0, 0
    for _ in range(probes):
        est, prod, assigned = 1.0, 1.0, []
        for j in range(i, n):
            lc = space.local_candidates(j, mapping, used)
            steps += 1
            if not lc:
                break
            prod *= len(lc)
            est += prod
            v = lc[int(rng.integers(len(lc)))]
            u = space.order[j]
            mapping[u] = v
            used.add(v)
            assigned.append((u, v))
        for u, v in assigned:
            mapping[u] = -1
            used.discard(v)
        total += est
    return total / probes, steps
