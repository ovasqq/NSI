"""Oracle labels for the p/c rule (stage 4) and the cardinality regression
target (idea 2).

A full enumeration of the search tree records, for every node at depth
< `max_depth`, the *true* values of each local candidate:

  * has_solution(v)  ∈ {0, 1} — committing u ↦ v leads to ≥1 embedding;
  * subtree_size(v)  ∈ ℕ      — number of search nodes ("steps") in the
                                subtree rooted at the node where u ↦ v is
                                committed (the node itself counts as 1);
  * num_matches(v)   ∈ ℕ      — number of complete embeddings in that subtree
                                (cardinality). 0 ⟺ has_solution = 0.

`has_solution` and `subtree_size` are the "true p" and "true c" of the
Simon–Kadane rule; `num_matches` is the richer, order-invariant regression
target of the inductive candidate scorer (NeuSO-style cardinality).

All three are intrinsic to a node: the set of reachable nodes depends only on
the fixed filter C(u), the order φ and the pruning inside `local_candidates`,
and NOT on the order in which candidates are tried. Hence a single id-order
pass labels every node for every ordering (the same argument that makes
reordering completeness-preserving).

Returns (labels, steps, complete):

  * labels   — {state → [(candidate, has_solution, subtree_size, num_matches)]}
               in id order; `state` = tuple of images of φ[:i] (i = depth);
  * steps    — total nodes enumerated, equal to `enumerate_matches(...).steps`;
  * complete — False iff the budget was hit. In that case some subtrees were
               not enumerated and the labels are discarded: the caller must
               treat the query as censored (skip it).
"""

from __future__ import annotations

from search.enumerate import SearchSpace

# state -> list of (candidate, has_solution, subtree_size, num_matches), in id order.
OracleLabels = dict[tuple[int, ...], list[tuple[int, bool, int, int]]]


def label_subtrees(
    space: SearchSpace,
    max_depth: int,
    budget: int,
) -> tuple[OracleLabels, int, bool]:
    q = space.q
    order = space.order
    n = q.n
    mapping = [-1] * n
    used: set[int] = set()
    labels: OracleLabels = {}
    steps = 0
    stopped = False

    def dfs(i: int) -> tuple[int, bool, int]:
        """Returns (subtree_size, has_solution, num_matches) of state i."""
        nonlocal steps, stopped
        if stopped:
            return 0, False, 0
        u = order[i]
        local = space.local_candidates(i, mapping, used)
        node_labels: list[tuple[int, bool, int, int]] = []
        total = 0
        has_sol = False
        n_matches = 0
        for v in local:
            if steps >= budget:
                stopped = True
                break
            steps += 1
            mapping[u] = v
            used.add(v)
            if i + 1 == n:
                child_size, child_sol, child_matches = 1, True, 1
            else:
                sub_size, sub_sol, sub_matches = dfs(i + 1)
                child_size, child_sol, child_matches = 1 + sub_size, sub_sol, sub_matches
            total += child_size
            has_sol = has_sol or child_sol
            n_matches += child_matches
            node_labels.append((v, child_sol, child_size, child_matches))
            used.discard(v)
            mapping[u] = -1
        if not stopped and i < max_depth:
            key = tuple(mapping[w] for w in order[:i])
            labels[key] = node_labels
        return total, has_sol, n_matches

    if all(len(c) for c in space.cand):
        dfs(0)
    return labels, steps, not stopped
