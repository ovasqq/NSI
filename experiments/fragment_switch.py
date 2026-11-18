"""Calibrate a deep fragment selector and benchmark four policies on held-out queries."""

import argparse
import csv
import json
from collections import Counter
from math import prod, isfinite
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
from tqdm import tqdm

from data.loaders import load
from experiments.deep_pruning import load_splits, write_json
from experiments.fragment_pruning import search_space
from models.fragment import proposals, manual_fragment
from models.hook import NeuGNReorder
from models.pruning_controller import checkpoint_digest
from search.enumerate import enumerate_space
from search.pruning import ResidualPruner, check_fragment
from train.train_neugn import load_model


def select(space, domains, mode, threshold, limit):
    if mode == "switch" and not any(
        len(domain) * len(domains[v]) <= threshold
        for u, domain in domains.items() for v in space.q.adj_set[u] if v > u and v in domains
    ):
        return "skip", ()

    options = proposals(space, domains, limit)
    if not options:
        return "skip", ()
    if mode != "edge":
        hand = manual_fragment(space, domains, options)
        if mode == "manual" or prod(len(domains[u]) for u in hand) <= threshold:
            return "manual", hand
    edges = (f for f in options if len(f) == 2)
    edge = min(edges, key=lambda f: (len(domains[f[0]]) * len(domains[f[1]]), f), default=())
    if edge and (mode == "edge" or len(domains[edge[0]]) * len(domains[edge[1]]) <= threshold):
        return "edge", edge
    return "skip", ()


def run(query, graph, model, args, mode, threshold):
    start = perf_counter()
    space = search_space(query.graph, graph)
    hook = NeuGNReorder(model, space.q, args.depth, args.device)
    pruner = ResidualPruner(space, "forward")
    counts = Counter()
    domain_seconds = select_seconds = check_seconds = 0.0

    def check(depth, mapping, used):
        nonlocal domain_seconds, select_seconds, check_seconds
        before = perf_counter()
        domains = pruner.build_domains(depth, mapping, used)
        domain_seconds += perf_counter() - before
        counts["domain_calls"] += 1
        if domains is None:
            counts["domain_cuts"] += 1
            return True
        if mode == "forward" or depth < args.depth or len(domains) < 2:
            return False
        before = perf_counter()
        action, fragment = select(space, domains, mode, threshold, args.proposals)
        select_seconds += perf_counter() - before
        counts[action] += 1
        if not fragment:
            return False
        before = perf_counter()
        outcome = check_fragment(space, domains, fragment, args.fragment_budget)
        check_seconds += perf_counter() - before
        if outcome is False:
            counts[action + "_cuts"] += 1
            return True
        if outcome is None:
            counts["unknown"] += 1
        return False

    search_start = perf_counter()
    result = enumerate_space(space, hook, branch_check=check, max_matches=1,
                             max_steps=args.budget, deadline=start + args.time_limit)
    end = perf_counter()
    return dict(mode=mode, threshold=threshold, seconds=end-start, search_seconds=end-search_start,
                solved=bool(result.num_matches), timed_out=result.timed_out, steps=result.steps,
                fms=result.first_match_step, complete=result.complete, counts=dict(counts),
                domain_seconds=domain_seconds, select_seconds=select_seconds,
                check_seconds=check_seconds, rank_seconds=hook.seconds)


def measure(sets, split_name, policies, repeats, graph, model, args):
    for rep in range(1, repeats + 1):
        rng = np.random.default_rng([args.seed, rep, int(split_name == "test")])
        records = []
        for (size, density), split in sets.items():
            for index, q in tqdm(split[split_name], desc=f"{split_name} {rep} {size}/{density}"):
                for i in rng.permutation(len(policies)):
                    mode, threshold = policies[i]
                    records.append(dict(size=size, density=density, query=index,
                                        **run(q, graph, model, args, mode, threshold)))
        write_json(args.output / f"{split_name}_{rep}.json", records)
        yield records


def summarize(records):
    table = []
    for size, density in [("all", "all"), *sorted({(r["size"], r["density"]) for r in records})]:
        group = records if size == "all" else [
            r for r in records if (r["size"], r["density"]) == (size, density)]
        key = lambda r: (r["size"], r["density"], r["query"])
        manual = {key(r): r for r in group if r["mode"] == "manual" and r["solved"]}
        for mode in ("forward", "manual", "edge", "switch"):
            rows = [r for r in group if r["mode"] == mode]
            common = [r for r in rows if r["solved"] and key(r) in manual]
            common_ms = [r["seconds"] * 1000 for r in common]
            delta = [(r["seconds"] - manual[key(r)]["seconds"]) * 1000 for r in common]
            steps = [r["steps"] for r in rows if r["solved"]]
            counts = Counter()
            for r in rows:
                counts.update(r["counts"])
            table.append(dict(size=size, density=density, mode=mode, queries=len(rows),
                              solved=sum(r["solved"] for r in rows), seconds=sum(r["seconds"] for r in rows),
                              median_attempt_ms=1000*median(r["seconds"] for r in rows),
                              common_with_manual=len(common), median_common_ms=median(common_ms) if common else None,
                              p95_common_ms=float(np.percentile(common_ms, 95)) if common else None,
                              median_paired_delta_ms=median(delta) if common else None,
                              steps=sum(r["steps"] for r in rows), median_fms_solved=median(steps) if steps else None,
                              timeouts=sum(r["timed_out"] for r in rows),
                              **{k: sum(r[k] for r in rows) for k in (
                                  "domain_seconds", "select_seconds", "check_seconds", "rank_seconds")},
                              **{k: counts[k] for k in ("domain_calls", "domain_cuts", "manual", "edge",
                                                        "skip", "manual_cuts", "edge_cuts", "unknown")}))
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "query-dir", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--graph", default="wikics")
    p.add_argument("--device", default="cpu")
    p.add_argument("--sizes", type=int, nargs="+", default=[16, 24, 32])
    p.add_argument("--thresholds", type=int, nargs="+", default=[64, 256, 1024])
    p.add_argument("--time-limit", type=float, default=.5)
    for name, default in dict(depth=10, budget=100000, proposals=12, fragment_budget=256,
                              seed=0, train_queries=100, validation_queries=20, test_queries=80,
                              validation_repeats=2, repeats=3).items():
        p.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    args = p.parse_args()
    if min(*args.thresholds, *args.sizes, args.budget, args.depth, args.proposals, args.fragment_budget,
           args.validation_queries, args.test_queries, args.validation_repeats, args.repeats, args.time_limit) <= 0:
        p.error("positive sizes, budgets, thresholds and repeat counts required")
    if not isfinite(args.time_limit) or min(args.seed, args.train_queries) < 0:
        p.error("invalid time limit, seed or split offset")
    if len(set(args.sizes)) != len(args.sizes) or len(set(args.thresholds)) != len(args.thresholds):
        p.error("duplicate size or threshold")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    graph = load(args.graph)
    model = load_model(args.checkpoint, args.device)
    if model.cfg.num_g != graph.n:
        p.error("graph size does not match checkpoint")
    sets = load_splits(graph, args, query_dir=args.query_dir)
    if any(not s["validation"] for s in sets.values()):
        p.error("empty validation group after duplicate removal")
    query_hashes = {f.name: checkpoint_digest(f) for size, density in sets
                    for f in [args.query_dir / f"size{size}_{density}.json"]}
    protocol = dict(vars(args), checkpoint_digest=checkpoint_digest(args.checkpoint),
                    query_digests=query_hashes, torch_version=str(torch.__version__))
    write_json(args.output / "protocol.json", json.loads(json.dumps(protocol, default=str)))
    choices = [("switch", t) for t in args.thresholds]
    # Only validation queries are used for warmup and threshold selection.
    for split in sets.values():
        q = split["validation"][0][1]
        for mode, threshold in [("forward", None), ("manual", None), ("edge", None), choices[0]]:
            run(q, graph, model, args, mode, threshold)
    trials = []
    for rep, rows in enumerate(measure(sets, "validation", choices, args.validation_repeats, graph, model, args), 1):
        for t in args.thresholds:
            group = [r for r in rows if r["threshold"] == t]
            trials.append(dict(rep=rep, threshold=t, solved=sum(r["solved"] for r in group),
                               seconds=sum(r["seconds"] for r in group)))
    def priority(t):
        rows = [r for r in trials if r["threshold"] == t]
        return -sum(r["solved"] for r in rows), sum(r["seconds"] for r in rows), t
    threshold = min(args.thresholds, key=priority)
    write_json(args.output / "calibration.json", dict(selected=threshold, trials=trials))
    print(f"Selected T={threshold}", flush=True)
    modes = [("forward", None), ("manual", None), ("edge", None), ("switch", threshold)]
    aggregate = []
    for rep, rows in enumerate(measure(sets, "test", modes, args.repeats, graph, model, args), 1):
        for entry in summarize(rows):
            aggregate.append(dict(rep=rep, **entry))
            if entry["size"] == "all":
                print(f"rep {rep} {entry['mode']}: {entry['solved']}/{entry['queries']}, {entry['seconds']:.3f}s")
    with (args.output / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=aggregate[0])
        writer.writeheader()
        writer.writerows(aggregate)
    print(f"Finished: {args.output / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
