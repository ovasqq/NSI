"""FMS protocol: steps until the first (and the first k) embeddings.

    python -m experiments.fms --graph hamster --size 20 --density dense

For every query of a stored query set, the enumerator runs until k embeddings
are found or the step budget is exhausted. A query not solved within the
budget is *censored*: its FMS is only known to exceed the budget. The median
is computed with censored values set to +inf, so it is exact as long as fewer
than half of the queries are censored, and reported as "> budget" otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

from data.loaders import load
from queries.storage import load_query_set, query_set_path
from search.enumerate import ReorderHook, enumerate_matches
from search.filter import FILTERS
from search.graph import Graph
from search.order import ORDERS
from search.pruning import PRUNING_MODES

RESULTS_DIR = Path(__file__).resolve().parent / "results"


HookFactory = Callable[[Graph, Graph, list, list[int]], ReorderHook | None]


def run_query(
    q: Graph,
    g: Graph,
    factory: HookFactory,
    *,
    filter_name: str = "nlf",
    order_name: str = "gql",
    k: int = 1,
    budget: int = 1_000_000,
    pruning: str = "none",
) -> dict:
    t0 = time.perf_counter()
    cand = FILTERS[filter_name](q, g)
    order = ORDERS[order_name](q, cand)
    hook = factory(q, g, cand, order)
    t1 = time.perf_counter()
    res = enumerate_matches(q, g, cand, order, hook, max_matches=k, max_steps=budget,
                            pruning=pruning)
    t2 = time.perf_counter()
    extra = {}
    if hasattr(hook, "calls"):
        extra.update(model_calls=hook.calls, model_seconds=hook.seconds)
    if hasattr(hook, "probe_steps"):
        # Look-ahead work is charged as search steps; with k = 1 all probes
        # happen before the first match. Over budget in total = censored.
        total = None if res.first_match_step is None else res.first_match_step + hook.probe_steps
        extra.update(probe_steps=hook.probe_steps,
                     fms_with_probes=total if total is not None and total <= budget else None)
    if res.pruning is not None:
        extra.update(pruning_calls=res.pruning.calls, pruning_seconds=res.pruning.seconds,
                     motif_checks=res.pruning.motif_checks, pruning_cuts=res.pruning.cuts)
    return dict(
        n=q.n,
        m=q.m,
        fms=res.first_match_step,  # None = censored (> budget)
        match_steps=res.match_steps,  # steps at the 1st..k-th embedding
        steps=res.steps,
        complete=res.complete,
        filter_seconds=t1 - t0,
        search_seconds=t2 - t1,
        total_seconds=t2 - t0,
        **extra,
    )


def censored_median(values: list[int | None]) -> float:
    """Median with None treated as +inf (see module docstring)."""
    return statistics.median(math.inf if v is None else v for v in values)


def summarize(records: list[dict], k: int, budget: int) -> dict:
    fms = [r["fms"] for r in records]
    kth = [r["match_steps"][k - 1] if len(r["match_steps"]) >= k else None for r in records]
    cuts: Counter[str] = Counter()
    for r in records:
        cuts.update(r.get("pruning_cuts", {}))
    return dict(
        queries=len(records),
        solved=sum(v is not None for v in fms),
        median_fms=censored_median(fms),
        median_steps_to_k=censored_median(kth),
        # A tree exhausted without k matches is not a budget failure.
        exhausted_without_k=sum(r["complete"] and len(r["match_steps"]) < k for r in records),
        steps_per_second=sum(r["steps"] for r in records)
        / max(sum(r["search_seconds"] for r in records), 1e-9),
        budget=budget,
        total_seconds=sum(r.get("total_seconds", r.get("filter_seconds", 0.0) + r["search_seconds"])
                          for r in records),
        **({"pruning_calls": sum(r["pruning_calls"] for r in records),
            "pruning_seconds": sum(r["pruning_seconds"] for r in records),
            "motif_checks": sum(r["motif_checks"] for r in records),
            "pruning_cuts": dict(cuts)} if records and "pruning_calls" in records[0] else {}),
        **({"median_fms_with_probes": censored_median([r["fms_with_probes"] for r in records]),
            "solved_with_probes": sum(r["fms_with_probes"] is not None for r in records)}
           if records and "fms_with_probes" in records[0] else {}),
    )


def random_hook(seed: int) -> ReorderHook:
    """Sanity baseline: uniformly random order at every depth."""
    rng = np.random.default_rng(seed)

    def hook(depth, u, mapping, local):
        return [local[i] for i in rng.permutation(len(local))]

    return hook


def make_hook_factory(args, g) -> HookFactory:
    """Returns (q, g, cand, order) -> hook. Hooks are per query: NeuGN
    precomputes h_Q and the Euler sequence, Knuth probes need (C, φ)."""
    if args.method == "baseline":
        return lambda q, g, cand, order: None
    if args.method == "random":
        hook = random_hook(args.seed)
        return lambda q, g, cand, order: hook
    if args.method == "inductive":
        from models.inductive_hook import InductiveReorder
        from train.train_inductive import load_model as load_inductive

        model = load_inductive(Path(args.checkpoint), g, args.device)
        return lambda q, g, cand, order: InductiveReorder(model, q, args.max_depth,
                                                          args.device, cand=cand)

    from models.hook import NeuGNReorder
    from models.pc_hook import PCReorder
    from search.enumerate import SearchSpace

    model = None
    if args.method in ("neugn", "neugn_pc"):
        from train.train_neugn import load_model

        model = load_model(Path(args.checkpoint), args.device)
    if args.method == "neugn":
        return lambda q, g, cand, order: NeuGNReorder(model, q, args.max_depth, args.device)
    if args.method in ("knuth", "neugn_pc"):
        def factory(q, g, cand, order):
            nav = NeuGNReorder(model, q, args.max_depth, args.device) if model else None
            return PCReorder(SearchSpace(q, g, cand, order), nav, args.pc_depth,
                             args.probes, args.alpha, args.seed)
        return factory
    raise ValueError(f"unknown method {args.method!r}")


def fmt(x: float) -> str:
    return "> budget" if math.isinf(x) else f"{x:,.0f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="hamster")
    ap.add_argument("--size", type=int, default=20)
    ap.add_argument("--density", default="dense")
    ap.add_argument("--method", default="baseline", choices=["baseline", "random", "neugn", "knuth", "neugn_pc", "inductive"])
    ap.add_argument("--checkpoint", default=None, help="model.pt for --method neugn/inductive")
    ap.add_argument("--max-depth", type=int, default=10, help="NeuGN reorders depths < this")
    ap.add_argument("--device", default="cpu", help="inference device for NeuGN")
    ap.add_argument("--pc-depth", type=int, default=3, help="p/c scoring at depths < this")
    ap.add_argument("--probes", type=int, default=4, help="Knuth probes per candidate")
    ap.add_argument("--alpha", type=float, default=1.0, help="score = log p - alpha log c")
    ap.add_argument("--query-set", default=None,
                    help="path to a query-set JSON instead of the standard one (e.g. validation)")
    ap.add_argument("--tag", default=None, help="extra suffix for the result file name")
    ap.add_argument("--filter", default="nlf", choices=sorted(FILTERS))
    ap.add_argument("--order", default="gql", choices=sorted(ORDERS))
    ap.add_argument("--pruning", default="none", choices=PRUNING_MODES,
                    help="exact residual checks: arc consistency, plus matching (domains), or plus motifs")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--budget", type=int, default=1_000_000)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N queries")
    ap.add_argument("--progress", action="store_true", help="show per-query progress (requires tqdm)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    g = load(args.graph)
    qpath = Path(args.query_set) if args.query_set else query_set_path(args.graph, args.size, args.density)
    queries, qmeta = load_query_set(qpath)
    queries = queries[: args.limit]
    factory = make_hook_factory(args, g)

    query_iter = enumerate(queries)
    if args.progress:
        try:
            from tqdm import tqdm
        except ImportError:
            ap.error("--progress requires tqdm; install it with python -m pip install tqdm")
        description = f"{args.graph} {args.size} {args.density} / {args.method} / {args.pruning}"
        query_iter = tqdm(query_iter, total=len(queries), desc=description, unit="query",
                          mininterval=2, dynamic_ncols=True)

    records = []
    for i, q in query_iter:
        r = run_query(q.graph, g, factory, filter_name=args.filter, order_name=args.order,
                      k=args.k, budget=args.budget, pruning=args.pruning)
        records.append(dict(query=i, **r))

    summary = summarize(records, args.k, args.budget)
    tag = f"{args.graph}_size{args.size}_{args.density}_{args.method}_{args.filter}_{args.order}_k{args.k}"
    if args.pruning != "none":
        tag += f"_pruning_{args.pruning}"
    if args.tag:
        tag += f"_{args.tag}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{tag}.json"
    out.write_text(json.dumps(dict(config=vars(args), query_set=qmeta,
                                   summary=summary, records=records), indent=1))
    print(f"{tag}: solved {summary['solved']}/{summary['queries']}, "
          f"median FMS {fmt(summary['median_fms'])}, "
          f"median steps to {args.k} {fmt(summary['median_steps_to_k'])}, "
          + (f"with probes: solved {summary['solved_with_probes']}, median "
             f"{fmt(summary['median_fms_with_probes'])}, " if "solved_with_probes" in summary else "")
          + f"{summary['total_seconds']:.3f}s total, "
          + f"{summary['steps_per_second']:,.0f} steps/s -> {out.name}")


if __name__ == "__main__":
    main()
