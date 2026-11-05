"""Stage-4 oracle experiment: the ceiling of the p/c rule (go/no-go).

For every *small* query that can be fully enumerated within a budget, compute
the true p (has_solution) and c (subtree size) of every candidate at the top
levels (`search.oracle.label_subtrees`), then order candidates by:

  * baseline  — id order;
  * neugn     — NeuGN logits (learned; in-distribution on small queries);
  * oracle_p  — solvable candidates first;
  * oracle_1c — smallest subtree first (ignores p);
  * oracle_pc — solvable first, then smallest subtree (the Simon–Kadane rule).

FMS (steps to the first embedding) is compared across methods on exactly the
same queries. If oracle_pc does not clearly beat NeuGN, idea 1 must be
revisited before training any model.

    python -m experiments.oracle --graph hamster --sizes 8 10 12 --density dense \
        --count 100 --checkpoint checkpoints/hamster/paper/model.pt

Writes one JSON per (size, density) under experiments/results/.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from data.loaders import load
from models.hook import NeuGNReorder
from models.oracle_hook import OracleReorder
from queries.random_walk import generate_query_set
from queries.storage import QUERY_DIR, load_query_set, save_query_set
from search.enumerate import SearchSpace, enumerate_matches
from search.filter import FILTERS
from search.graph import Graph
from search.oracle import label_subtrees
from search.order import ORDERS

RESULTS_DIR = Path(__file__).resolve().parent / "results"

ALL_METHODS = ["baseline", "neugn", "oracle_p", "oracle_1c", "oracle_pc"]


def oracle_query_path(graph: str, size: int, density: str) -> Path:
    return QUERY_DIR / graph / f"oracle_size{size}_{density}.json"


def censored_median(values: list[int | None]) -> float:
    if not values:
        return math.nan
    return statistics.median(math.inf if v is None else v for v in values)


def pairwise(a: list[int | None], b: list[int | None]) -> dict:
    """Per-query comparison of method A vs B on queries solved by both.
    geomean < 1 means A is faster."""
    ratios, better, worse = [], 0, 0
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        ratios.append(x / y)
        better += x < y
        worse += x > y
    geomean = math.exp(sum(map(math.log, ratios)) / len(ratios)) if ratios else math.nan
    return dict(better=better, worse=worse, tied=len(ratios) - better - worse,
                n=len(ratios), geomean=geomean)


def run_one_query(q: Graph, g: Graph, model, args) -> dict:
    cand = FILTERS[args.filter](q, g)
    order = ORDERS[args.order](q, cand)
    space = SearchSpace(q, g, cand, order)
    labels, label_steps, complete = label_subtrees(space, args.oracle_depth, args.enum_budget)
    if not complete:
        return dict(n=q.n, m=q.m, skipped="enumeration_over_budget", label_steps=label_steps)

    root = labels[()]
    n_solvable_root = sum(1 for _, sol, _, _ in root)
    row: dict = dict(
        n=q.n, m=q.m, label_steps=label_steps,
        root_candidates=len(root), root_solvable=n_solvable_root,
    )
    methods = ALL_METHODS if model is not None else [m for m in ALL_METHODS if m != "neugn"]
    hooks = {
        "baseline": None,
        "neugn": NeuGNReorder(model, q, args.max_depth, args.device) if model is not None else None,
        "oracle_p": OracleReorder(space, labels, "p", args.oracle_depth),
        "oracle_1c": OracleReorder(space, labels, "inv_c", args.oracle_depth),
        "oracle_pc": OracleReorder(space, labels, "pc", args.oracle_depth),
    }
    for method in methods:
        hook = hooks[method]
        res = enumerate_matches(q, g, cand, order, hook, max_matches=1, max_steps=args.search_budget)
        row[method] = dict(fms=res.first_match_step, steps=res.steps, complete=res.complete)
    return row


def summarize(records: list[dict], methods: list[str], args) -> dict:
    solved = [r for r in records if "baseline" in r]
    skipped = len(records) - len(solved)
    out = dict(queries=len(records), labeled=len(solved), skipped=skipped)
    fms = {m: [r[m]["fms"] for r in solved] for m in methods}
    for m in methods:
        out[m] = dict(
            solved=sum(v is not None for v in fms[m]),
            median_fms=censored_median(fms[m]),
        )
    for ref in ("baseline", "neugn"):
        if ref not in methods:
            continue
        out[f"vs_{ref}"] = {
            m: pairwise(fms[m], fms[ref]) for m in methods if m != ref
        }
    # how much headroom there is at all: baseline spread
    out["baseline"]["median_steps"] = censored_median([r["baseline"]["steps"] for r in solved])
    out["median_label_steps"] = censored_median([r["label_steps"] for r in solved])
    out["median_root_candidates"] = censored_median([r["root_candidates"] for r in solved])
    out["median_root_solvable"] = censored_median([r["root_solvable"] for r in solved])
    return out


def fmt(x: float) -> str:
    return "∞" if math.isinf(x) else f"{x:,.0f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="hamster")
    ap.add_argument("--sizes", nargs="+", type=int, default=[8, 10, 12])
    ap.add_argument("--densities", nargs="+", default=["dense", "sparse"])
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=12345, help="base seed for query generation")
    ap.add_argument("--filter", default="nlf", choices=sorted(FILTERS))
    ap.add_argument("--order", default="gql", choices=sorted(ORDERS))
    ap.add_argument("--oracle-depth", type=int, default=3, help="oracle reorders depths < this")
    ap.add_argument("--max-depth", type=int, default=10, help="NeuGN reorders depths < this")
    ap.add_argument("--checkpoint", default=None, help="NeuGN model.pt (skip neugn if omitted)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--enum-budget", type=int, default=5_000_000, help="budget for full enumeration")
    ap.add_argument("--search-budget", type=int, default=1_000_000, help="FMS budget per query")
    ap.add_argument("--limit", type=int, default=None, help="use only first N queries")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    g = load(args.graph)
    model = None
    if args.checkpoint:
        from train.train_neugn import load_model
        model = load_model(Path(args.checkpoint), args.device)
    methods = ALL_METHODS if model is not None else [m for m in ALL_METHODS if m != "neugn"]

    for size in args.sizes:
        for density in args.densities:
            seed = args.seed * 1_000_003 + size * 10 + args.densities.index(density)
            qpath = oracle_query_path(args.graph, size, density)
            queries = generate_query_set(g, size, density, args.count, seed)
            save_query_set(qpath, queries, dict(graph=args.graph, size=size, density=density,
                                                seed=seed, requested=args.count, purpose="oracle"))
            queries = queries[: args.limit]
            records = [run_one_query(q.graph, g, model, args) for q in queries]
            summary = summarize(records, methods, args)

            tag = f"{args.graph}_oracle_size{size}_{density}_{args.filter}_{args.order}"
            if args.tag:
                tag += f"_{args.tag}"
            out = RESULTS_DIR / f"{tag}.json"
            out.write_text(json.dumps(dict(config=vars(args), query_set=dict(seed=seed),
                                           summary=summary, records=records), indent=1))
            print(f"== {args.graph} size={size} {density}: {summary['labeled']}/{summary['queries']} "
                  f"labeled (skipped {summary['skipped']}) -> {out.name}")
            if summary["labeled"] == 0:
                print("  (no query fully enumerated within budget; skipping)")
                continue
            header = "  " + "".join(f"{m:>12}" for m in methods)
            print(header)
            print("  " + "".join(f"{fmt(summary[m]['median_fms']):>12}" for m in methods))
            print(f"  median label_steps={fmt(summary['median_label_steps'])}  "
                  f"median root_cands={fmt(summary['median_root_candidates'])}  "
                  f"median root_solvable={fmt(summary['median_root_solvable'])}")
            for ref in ("baseline", "neugn"):
                if ref not in methods:
                    continue
                for m in methods:
                    if m == ref:
                        continue
                    pw = summary[f"vs_{ref}"][m]
                    print(f"  {m:>9} vs {ref:<8} better={pw['better']:>3} worse={pw['worse']:>3} "
                          f"geomean={pw['geomean']:.2f}")


if __name__ == "__main__":
    main()
