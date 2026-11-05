"""Measure how much the matching order φ matters (stage 4b).

Unlike candidate reordering, φ changes the search tree itself, so the total
number of enumeration steps is order-dependent. This script runs the same
small queries with different φ and compares:

  * total steps  — the tree size under full enumeration (the clean φ signal);
  * FMS          — steps to the first embedding (baseline id candidate order).

Orders: GQL (data-aware greedy, current default), RI (structure-only), and
k random permutations (the "bad" end of the spectrum, to show the spread).

    python -m experiments.phi --graph hamster --sizes 8 10 --densities dense sparse

Reuses the oracle query sets data/queries/<graph>/oracle_size<N>_<density>.json
so the numbers are directly comparable with the stage-4 oracle experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from data.loaders import load
from queries.storage import load_query_set
from search.enumerate import enumerate_matches
from search.filter import nlf_filter
from search.graph import Graph
from search.order import gql_order, ri_order

RESULTS_DIR = Path(__file__).resolve().parent / "results"
QUERY_DIR = Path(__file__).resolve().parent.parent / "data" / "queries"


def oracle_query_path(graph: str, size: int, density: str) -> Path:
    return QUERY_DIR / graph / f"oracle_size{size}_{density}.json"


def censored_median(values: list[int | None]) -> float:
    if not values:
        return math.nan
    return statistics.median(math.inf if v is None else v for v in values)


def run_query(q: Graph, g: Graph, args, rng: np.random.Generator) -> dict:
    cand = nlf_filter(q, g)
    gql = gql_order(q, cand)
    ri = ri_order(q, cand)
    rands = [rng.permutation(q.n).tolist() for _ in range(args.random_orders)]

    # Tree size under GQL first: if it exceeds the budget, φ cannot help
    # (GQL is near-best), so we skip the more expensive alternatives.
    gql_full = enumerate_matches(q, g, cand, gql, max_matches=None, max_steps=args.enum_budget)
    row: dict = dict(n=q.n, m=q.m,
                     gql_total=gql_full.steps if gql_full.complete else None)

    def totals(order: list[int]) -> int | None:
        res = enumerate_matches(q, g, cand, order, max_matches=None, max_steps=args.enum_budget)
        return res.steps if res.complete else None

    def fms(order: list[int]) -> int | None:
        res = enumerate_matches(q, g, cand, order, max_matches=1, max_steps=args.search_budget)
        return res.first_match_step

    row["gql_fms"] = fms(gql)
    row["ri_fms"] = fms(ri)
    row["rand_fms"] = [fms(o) for o in rands]

    if gql_full.complete:
        row["ri_total"] = totals(ri)
        row["rand_total"] = [totals(o) for o in rands]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="hamster")
    ap.add_argument("--sizes", nargs="+", type=int, default=[8, 10])
    ap.add_argument("--densities", nargs="+", default=["dense", "sparse"])
    ap.add_argument("--random-orders", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enum-budget", type=int, default=3_000_000)
    ap.add_argument("--search-budget", type=int, default=1_000_000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    g = load(args.graph)
    for size in args.sizes:
        for density in args.densities:
            queries, _meta = load_query_set(oracle_query_path(args.graph, size, density))
            queries = queries[: args.limit]
            records = []
            for i, q in enumerate(queries):
                rng = np.random.default_rng(args.seed * 1_000_003 + i)
                records.append(dict(query=i, **run_query(q.graph, g, args, rng)))

            measured = [r for r in records if r["gql_total"] is not None]

            def geomean_ratio(a, b):
                ratios = [x / y for x, y in zip(a, b)
                          if x is not None and y is not None and y > 0]
                return math.exp(sum(map(math.log, ratios)) / len(ratios)) if ratios else math.nan

            gql_total = [r["gql_total"] for r in measured]
            ri_total = [r.get("ri_total") for r in measured]
            rand_best_total = [
                min((t for t in r["rand_total"] if t is not None), default=None)
                for r in measured
            ]

            def median_fms(key):
                return censored_median([r[key] for r in records])

            ri_better = ri_worse = ri_n = 0
            for x, y in zip(gql_total, ri_total):
                if x is not None and y is not None:
                    ri_n += 1
                    ri_better += y < x
                    ri_worse += y > x
            rb_better = sum(1 for x, y in zip(gql_total, rand_best_total)
                            if x is not None and y is not None and y < x)

            summary = dict(
                queries=len(records),
                gql_complete=len(measured),
                median_total_gql=censored_median(gql_total),
                median_total_ri=censored_median(ri_total),
                median_total_rand_best=censored_median(rand_best_total),
                ri_vs_gql=dict(better=ri_better, worse=ri_worse, tied=ri_n - ri_better - ri_worse,
                               n=ri_n, geomean=geomean_ratio(ri_total, gql_total)),
                rand_best_vs_gql=dict(better=rb_better, of=len(measured),
                                      geomean=geomean_ratio(rand_best_total, gql_total)),
                median_fms_gql=median_fms("gql_fms"),
                median_fms_ri=median_fms("ri_fms"),
                median_fms_rand=censored_median([censored_median(r["rand_fms"]) for r in records]),
            )

            tag = f"{args.graph}_phi_size{size}_{density}"
            if args.tag:
                tag += f"_{args.tag}"
            out = RESULTS_DIR / f"{tag}.json"
            out.write_text(json.dumps(dict(config=vars(args), summary=summary, records=records), indent=1))
            print(f"== {args.graph} size={size} {density}: GQL complete {summary['gql_complete']}/{summary['queries']} "
                  f"-> {out.name}")
            print(f"  total steps:  GQL {summary['median_total_gql']:>10,.0f}   "
                  f"RI {summary['median_total_ri']:>10,.0f}   rand(best) {summary['median_total_rand_best']:>10,.0f}")
            print(f"  FMS:          GQL {summary['median_fms_gql']:>10,.0f}   "
                  f"RI {summary['median_fms_ri']:>10,.0f}   rand(med) {summary['median_fms_rand']:>10,.0f}")
            rv = summary["ri_vs_gql"]
            print(f"  RI vs GQL (total): better {rv['better']} worse {rv['worse']} tied {rv['tied']} "
                  f"(n={rv['n']}), geomean RI/GQL = {rv['geomean']:.3f}")
            rb = summary["rand_best_vs_gql"]
            print(f"  rand_best vs GQL: better {rb['better']}/{rb['of']}, geomean = {rb['geomean']:.3f}")


if __name__ == "__main__":
    main()
