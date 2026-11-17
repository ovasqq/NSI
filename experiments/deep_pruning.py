"""Train, calibrate and benchmark selective deep AC-3 on Hamster."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from data.loaders import load
from experiments.fms import run_query, summarize
from models.hook import NeuGNReorder
from models.pruning_controller import checkpoint_digest
from queries.random_walk import generate_query_set
from queries.storage import load_query_set, query_set_path, save_query_set
from search.enumerate import SearchSpace, enumerate_space
from search.filter import nlf_filter
from search.order import gql_order
from search.pruning import DeepPruning, ResidualPruner
from train import deep_arc
from train.train_neugn import load_model

MODES = ("domains", "none", "forward", "matching", "adaptive")
DENSITIES = ("dense", "sparse")


def fingerprint(query):
    return tuple(query.graph.labels.tolist()), tuple(query.graph.edges())


def load_splits(g, args):
    sets = {}
    train_end = args.train_queries
    val_end = train_end + args.validation_queries
    test_end = val_end + args.test_queries
    for size in args.sizes:
        for i, density in enumerate(DENSITIES):
            path = query_set_path("hamster", size, density)
            if not path.exists():
                path = args.output / "queries" / path.name
                if not path.exists():
                    seed = size * 10 + i
                    queries = generate_query_set(g, size, density, 200, seed)
                    save_query_set(path, queries, dict(graph="hamster", size=size, density=density, seed=seed))
                    print(f"Generated {len(queries)} queries in {path}", flush=True)
            queries, meta = load_query_set(path)
            if len(queries) < test_end:
                raise ValueError(f"{path} contains {len(queries)} queries; need {test_end}")
            indexed = list(enumerate(queries[:test_end]))
            sets[size, density] = dict(train=indexed[:train_end], validation=indexed[train_end:val_end],
                                      test=indexed[val_end:], metadata=meta)
    # Preserve the test IDs and prevent exact duplicates from crossing splits.
    excluded = {fingerprint(q) for split in sets.values() for _, q in split["test"]}
    for split in sets.values():
        split["validation"] = [(i, q) for i, q in split["validation"] if fingerprint(q) not in excluded]
    excluded.update(fingerprint(q) for split in sets.values() for _, q in split["validation"])
    for split in sets.values():
        split["train"] = [(i, q) for i, q in split["train"] if fingerprint(q) not in excluded]
    return sets


def collect(g, model, sets, args):
    features, gains, cohorts = [], [], []
    rng = np.random.default_rng(args.seed)
    horizon = args.horizon_ms / 1000
    matching = DeepPruning(args.depth, "matching")
    full = DeepPruning(args.depth, "adaptive", lambda _: lambda depth, domains: True)
    for (size, density), split in sets.items():
        count_before = len(gains)
        positives_before = sum(gain > 0 for gain in gains)
        censored = 0
        if size > args.depth + 1:
            for _, query in split["train"]:
                q = query.graph
                cand = nlf_filter(q, g)
                if not all(map(len, cand)):
                    continue
                order = gql_order(q, cand)
                space = SearchSpace(q, g, cand, order)
                hook = NeuGNReorder(model, q, device=args.device, max_depth=args.depth)
                pruner = ResidualPruner(space, "domains")
                feature_fn = deep_arc.feature_factory(space)
                states = {}
                for policy in (matching, full):
                    sampled = []
                    visits = eligible_visits = 0

                    def sample(branch, mapping):
                        nonlocal visits, eligible_visits
                        if branch.depth < args.depth:
                            return
                        visits += 1
                        if visits > 8 and visits % args.stride:
                            return
                        # Give late visits the same chance as earlier sampled visits.
                        eligible_visits += 1
                        slot = (eligible_visits - 1 if eligible_visits <= args.samples
                                else rng.integers(eligible_visits))
                        if slot < args.samples:
                            state = (tuple(mapping[u] for u in order[:branch.depth + 1]), branch.steps_remaining)
                            if slot == len(sampled):
                                sampled.append(state)
                            else:
                                sampled[slot] = state

                    enumerate_space(space, hook, max_matches=1, max_steps=args.budget, pruning="domains",
                                    deep_pruning=policy, before_check=sample,
                                    deadline=perf_counter() + args.trace_seconds)
                    for prefix, remaining in sampled:
                        states[prefix] = max(states.get(prefix, 0), remaining)
                for prefix, remaining in states.items():
                    depth = len(prefix)
                    mapping = [-1] * q.n
                    for u, v in zip(order, prefix):
                        mapping[u] = v
                    domains = pruner.build_domains(depth, mapping, set(prefix))
                    if domains is None or not pruner.covers_matching(domains):
                        continue
                    row = feature_fn(depth, domains)
                    start = perf_counter()
                    conflict = pruner.extra_arc_conflict(domains)
                    check_ms = (perf_counter() - start) * 1000
                    continuation_ms = 0.0
                    if conflict:
                        start = perf_counter()
                        replay = enumerate_space(space, max_matches=1, max_steps=remaining, prefix=prefix,
                                                 pruning="domains", deep_pruning=matching, deadline=start + horizon)
                        elapsed = (perf_counter() - start) * 1000
                        if not replay.complete and not replay.timed_out:
                            censored += 1
                            continue
                        continuation_ms = min(elapsed, args.horizon_ms)
                    features.append(row)
                    gains.append(continuation_ms - check_ms)
                    cohorts.append(f"{size}_{density}")
        print(f"Collect {size}/{density}: {len(gains) - count_before} states, "
              f"{sum(gain > 0 for gain in gains) - positives_before} profitable checks, {censored} censored",
              flush=True)
    if not gains:
        raise ValueError("no branches reached the adaptive stage; increase training queries or use larger sizes")
    arrays = np.asarray(features, dtype=np.float32), np.asarray(gains, dtype=np.float32), np.asarray(cohorts)
    np.savez_compressed(args.output / "traces.npz", features=arrays[0], gains_ms=arrays[1], cohorts=arrays[2])
    return arrays


def results(g, model, queries, args, deep):
    def factory(q, _g, _cand, _order):
        return NeuGNReorder(model, q, device=args.device, max_depth=args.depth)

    return [dict(query=i, **run_query(query.graph, g, factory, budget=args.budget, pruning="domains",
                                     deep_pruning=deep, time_limit=args.time_limit)) for i, query in queries]


def calibrate(g, model, sets, args, readout, features):
    scores = readout.score(features)
    thresholds = np.unique(np.r_[-np.inf, np.quantile(scores, [.05, .20, .40, .60, .80, .95]), 0, np.inf])
    validation = [row for (size, _), split in sets.items() if size > args.depth + 1
                  for row in split["validation"]]
    if not validation:
        raise ValueError("no held-out validation queries can reach the adaptive stage")
    trials = []
    for threshold in thresholds:
        deep = DeepPruning(args.depth, "adaptive", readout.policy(float(threshold)))
        summary = summarize(results(g, model, validation, args, deep), 1, args.budget)
        name = "always" if np.isneginf(threshold) else "never" if np.isposinf(threshold) else float(threshold)
        trials.append(dict(threshold=name, solved=summary["solved"], seconds=summary["total_seconds"],
                           decisions=summary["pruning_stages"]["arc_policy"]["calls"],
                           checks=summary["arc_policy_checks"]))
        print(f"Validation {name}: {summary['solved']}/{len(validation)} solved, "
              f"{summary['total_seconds']:.3f}s, {summary['arc_policy_checks']} deep AC-3 calls", flush=True)
    best = min(range(len(trials)), key=lambda i: (-trials[i]["solved"], trials[i]["seconds"]))
    write_json(args.output / "calibration.json", dict(queries=len(validation), selected=trials[best], trials=trials))
    return float(thresholds[best])


def write_json(path, data):
    path.write_text(json.dumps(data, indent=1, allow_nan=False) + "\n")


def test(g, model, sets, args, readout, threshold):
    configs = dict(domains=None, none=DeepPruning(args.depth, "none"),
                   forward=DeepPruning(args.depth, "forward"), matching=DeepPruning(args.depth, "matching"),
                   adaptive=DeepPruning(args.depth, "adaptive", readout.policy(threshold)))
    table = []
    for (size, density), split in sets.items():
        for mode in MODES:
            records = results(g, model, split["test"], args, configs[mode])
            summary = summarize(records, 1, args.budget)
            for key in ("median_fms", "median_steps_to_k"):
                if not math.isfinite(summary[key]):
                    summary[key] = None
            stages = summary["pruning_stages"]
            solved_ms = [r["total_seconds"] * 1000 for r in records if r["fms"] is not None]
            row = dict(size=size, density=density, mode=mode, queries=len(records), solved=summary["solved"],
                       timed_out=summary["timed_out"], total_seconds=summary["total_seconds"],
                       median_solved_ms=float(np.median(solved_ms)) if solved_ms else None,
                       median_fms=summary["median_fms"], dfs_steps=sum(r["steps"] for r in records),
                       pruning_calls=summary["pruning_calls"], arc_calls=stages["arc_consistency"]["calls"],
                       arc_seconds=stages["arc_consistency"]["seconds"], matching_calls=stages["matching"]["calls"],
                       matching_seconds=stages["matching"]["seconds"],
                       gate_decisions=stages.get("arc_policy", {}).get("calls"),
                       gate_checks=summary.get("arc_policy_checks"),
                       gate_seconds=stages.get("arc_policy", {}).get("seconds"))
            table.append(row)
            write_json(args.output / f"size{size}_{density}_{mode}.json",
                       dict(config=dict(size=size, density=density, mode=mode, depth=args.depth,
                                        budget=args.budget, time_limit=args.time_limit, device=args.device),
                            query_set=split["metadata"], summary=summary, records=records))
            print(f"Test {size}/{density}/{mode}: {row['solved']}/{row['queries']} solved, "
                  f"{row['total_seconds']:.3f}s", flush=True)
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/hamster/paper/model.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--phase", choices=("all", "train", "test"), default="all")
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 10, 12, 16, 20, 24, 32])
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--train-queries", type=int, default=100)
    parser.add_argument("--validation-queries", type=int, default=20)
    parser.add_argument("--test-queries", type=int, default=80)
    parser.add_argument("--samples", type=int, default=160, help="training states per query and collection policy")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--trace-seconds", type=float, default=0.65)
    parser.add_argument("--horizon-ms", type=float, default=50.0)
    parser.add_argument("--time-limit", type=float, default=0.5)
    parser.add_argument("--budget", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if (args.depth < 0 or args.seed < 0 or min(args.sizes) < 2
            or len(set(args.sizes)) != len(args.sizes)
            or min(args.train_queries, args.validation_queries, args.test_queries,
                   args.samples, args.stride, args.budget, args.epochs) < 1
            or any(not math.isfinite(v) or v <= 0 for v in (args.horizon_ms, args.trace_seconds, args.time_limit))):
        parser.error("invalid sizes, split, training or search limits")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    model_file = args.output / "arc_mlp.npz"
    protocol_file = args.output / "protocol.json"
    protocol = dict(checkpoint_digest=checkpoint_digest(args.checkpoint), sizes=args.sizes,
                    train=args.train_queries, validation=args.validation_queries, test=args.test_queries,
                    depth=args.depth, time_limit=args.time_limit, budget=args.budget, device=args.device)
    if args.phase == "test" and json.loads(protocol_file.read_text()) != protocol:
        raise ValueError("test configuration differs from the calibrated run")
    g = load("hamster")
    model = load_model(args.checkpoint, args.device)
    sets = load_splits(g, args)
    if args.phase == "test":
        readout, threshold = deep_arc.load(model_file)
    else:
        features, gains, cohorts = collect(g, model, sets, args)
        print(f"Training on {len(gains)} labeled states", flush=True)
        readout = deep_arc.fit(features, gains, cohorts, args.epochs, args.seed)
        threshold = calibrate(g, model, sets, args, readout, features)
        deep_arc.save(model_file, readout, threshold)
        write_json(protocol_file, protocol)
    if args.phase != "train":
        test(g, model, sets, args, readout, threshold)
        print(f"Saved {len(sets) * len(MODES)} runs to {args.output / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
