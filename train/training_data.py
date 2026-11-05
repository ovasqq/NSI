"""Generate, save and load the labeled training set (idea 2).

A labeled set is a fixed collection of random-walk queries with their
per-candidate `num_matches` (cardinality) labels, computed once by full
enumeration and saved to disk. This makes training reproducible and reusable
(no re-enumeration), and records detailed per-depth statistics for the
presentation.

Saved JSON layout:

    {
      "meta":       {graph, seed, n_queries, min_size, max_size, label_depth, budget},
      "queries":    [{labels, edges, origin, order, cand_count, states}],
      "statistics": {pos_rate_by_depth, mean_cardinality_by_depth, cardinality_histogram, ...}
    }

Each state in `states` is {key, target, candidates: [[candidate, num_matches]]};
`key` is the tuple of images of order[:depth], so the full partial mapping and
the model input are reconstructable deterministically.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from queries.random_walk import random_walk_vertices
from search.enumerate import SearchSpace
from search.filter import nlf_filter
from search.oracle import label_subtrees
from search.order import gql_order


def _card_bucket(matches: int) -> str:
    if matches == 0:
        return "0"
    if matches == 1:
        return "1"
    if matches <= 5:
        return "2-5"
    if matches <= 10:
        return "6-10"
    if matches <= 100:
        return "11-100"
    return ">100"


def generate_labeled_set(g, seed: int, n_queries: int, min_size: int = 5,
                         max_size: int = 15, label_depth: int = 3,
                         budget: int = 5_000_000, max_attempts: int | None = None):
    """Generate `n_queries` feasible queries, label them, return (queries, statistics)."""
    rng = np.random.default_rng(seed)
    max_attempts = max_attempts or n_queries * 100
    queries = []
    per_depth = {d: {"n_sol": 0, "n_tot": 0, "sum_card": 0} for d in range(label_depth)}
    card_hist: Counter = Counter()
    n_states = n_examples = attempts = 0
    while len(queries) < n_queries and attempts < max_attempts:
        attempts += 1
        size = int(rng.integers(min_size, max_size + 1))
        start = int(rng.integers(g.n))
        walk = random_walk_vertices(g, start, size, rng)
        if walk is None:
            continue
        q = g.induced_subgraph(walk)
        cand = nlf_filter(q, g)
        if any(len(c) == 0 for c in cand):
            continue
        order = gql_order(q, cand)
        labels, _, complete = label_subtrees(SearchSpace(q, g, cand, order), label_depth, budget)
        if not complete:
            continue
        states = []
        for key, cand_list in labels.items():
            depth = len(key)
            states.append({
                "key": list(key),
                "target": int(order[depth]),
                "candidates": [[int(v), int(m)] for v, _, _, m in cand_list],
            })
            for _, _, _, m in cand_list:
                per_depth[depth]["n_tot"] += 1
                per_depth[depth]["n_sol"] += 1 if m > 0 else 0
                per_depth[depth]["sum_card"] += m
                card_hist[_card_bucket(m)] += 1
                n_examples += 1
            n_states += 1
        queries.append({
            "labels": q.labels.tolist(),
            "edges": [list(e) for e in q.edges()],
            "origin": list(walk),
            "order": list(order),
            "cand_count": [len(c) for c in cand],
            "states": states,
        })
    statistics = {
        "n_queries": len(queries),
        "n_attempts": attempts,
        "n_states": n_states,
        "n_examples": n_examples,
        "pos_rate_by_depth": {str(d): per_depth[d]["n_sol"] / max(per_depth[d]["n_tot"], 1)
                              for d in per_depth},
        "mean_cardinality_by_depth": {str(d): per_depth[d]["sum_card"] / max(per_depth[d]["n_tot"], 1)
                                      for d in per_depth},
        "cardinality_histogram": dict(card_hist),
    }
    return queries, statistics


def save_labeled_set(path, g_name: str, seed: int, queries, statistics, **meta) -> Path:
    payload = {"meta": {"graph": g_name, "seed": seed, **meta},
               "queries": queries, "statistics": statistics}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def load_labeled_set(path) -> dict:
    return json.loads(Path(path).read_text())
