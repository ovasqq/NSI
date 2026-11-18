"""Train, calibrate and benchmark selective deep AC-3."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from data.loaders import LOADERS, load
from experiments.fms import run_query, summarize
from models.hook import NeuGNReorder
from models.pruning_controller import checkpoint_digest
from models.residual_arc import encode_residual
from queries.random_walk import generate_query_set
from queries.storage import load_query_set, query_set_path, save_query_set
from search.enumerate import SearchSpace, enumerate_space
from search.filter import nlf_filter
from search.order import gql_order
from search.pruning import DeepPruning, ResidualPruner
from train import deep_arc, residual_arc
from train.train_neugn import load_model

MODES = ("domains", "none", "forward", "matching", "matching_arc", "adaptive")
COMPARISON_MODES = ("neugn", "matching_arc", "adaptive", "transformer")
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
            path = query_set_path(args.graph, size, density)
            if not path.exists():
                path = args.output / "queries" / path.name
                if not path.exists():
                    seed = size * 10 + i
                    queries = generate_query_set(g, size, density, 200, seed)
                    save_query_set(path, queries, dict(graph=args.graph, size=size, density=density, seed=seed))
                    print(f"Generated {len(queries)} queries in {path}", flush=True)
            queries, meta = load_query_set(path)
            if meta.get("graph", args.graph) != args.graph:
                raise ValueError(f"{path} is for graph {meta['graph']}, expected {args.graph}")
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
    residual_rows = []
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
                    if args.gate in ("transformer", "both"):
                        residual = encode_residual(hook.qenc, mapping, domains, order[depth - 1],
                                                   model.cfg.num_g, args.candidates_per_domain)
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
                    if args.gate in ("transformer", "both"):
                        residual_rows.append(residual)
                    gains.append(continuation_ms - check_ms)
                    cohorts.append(f"{size}_{density}")
        print(f"Collect {size}/{density}: {len(gains) - count_before} states, "
              f"{sum(gain > 0 for gain in gains) - positives_before} profitable checks, {censored} censored",
              flush=True)
    if not gains:
        raise ValueError("no branches reached the adaptive stage; increase training queries or use larger sizes")
    arrays = np.asarray(features, dtype=np.float32), np.asarray(gains, dtype=np.float32), np.asarray(cohorts)
    np.savez_compressed(args.output / "traces.npz", features=arrays[0], gains_ms=arrays[1], cohorts=arrays[2])
    return arrays[0], residual_rows, arrays[1], arrays[2]


def results(g, model, queries, args, deep, pruning="domains"):
    def factory(q, _g, _cand, _order):
        return NeuGNReorder(model, q, device=args.device, max_depth=args.depth)

    return [dict(query=i, **run_query(query.graph, g, factory, budget=args.budget, pruning=pruning,
                                     deep_pruning=deep, time_limit=args.time_limit)) for i, query in queries]


def gate_policy(depth, gate, readout, threshold):
    if gate == "transformer":
        return readout.deep_policy(depth, threshold)
    return DeepPruning(depth, "adaptive", readout.policy(threshold))


def calibrate(g, model, sets, args, gate, readout, features, output):
    scores = readout.score(features)
    thresholds = np.unique(np.r_[-np.inf, np.quantile(scores, [.05, .20, .40, .60, .80, .95]), 0, np.inf])
    validation = [row for (size, _), split in sets.items() if size > args.depth + 1
                  for row in split["validation"]]
    if not validation:
        raise ValueError("no held-out validation queries can reach the adaptive stage")
    trials = []
    for threshold in thresholds:
        deep = gate_policy(args.depth, gate, readout, float(threshold))
        summary = summarize(results(g, model, validation, args, deep), 1, args.budget)
        name = "always" if np.isneginf(threshold) else "never" if np.isposinf(threshold) else float(threshold)
        trials.append(dict(threshold=name, solved=summary["solved"], seconds=summary["total_seconds"],
                           decisions=summary["pruning_stages"]["arc_policy"]["calls"],
                           checks=summary["arc_policy_checks"]))
        print(f"Validation {name}: {summary['solved']}/{len(validation)} solved, "
              f"{summary['total_seconds']:.3f}s, {summary['arc_policy_checks']} deep AC-3 calls", flush=True)
    best = min(range(len(trials)), key=lambda i: (-trials[i]["solved"], trials[i]["seconds"]))
    write_json(output, dict(queries=len(validation), selected=trials[best], trials=trials))
    return float(thresholds[best])


def write_json(path, data):
    path.write_text(json.dumps(data, indent=1, allow_nan=False) + "\n")


def test(g, model, sets, args, policies):
    configs = dict(neugn=None, domains=None, none=DeepPruning(args.depth, "none"),
                   forward=DeepPruning(args.depth, "forward"), matching=DeepPruning(args.depth, "matching"),
                   matching_arc=DeepPruning(args.depth, "matching_arc"),
                   adaptive=policies.get("mlp", policies.get("transformer")),
                   transformer=policies.get("transformer"))
    output = args.results_output or args.output
    output.mkdir(parents=True, exist_ok=True)
    table = []
    for (size, density), split in sets.items():
        if args.warmup:
            for mode in args.test_modes:
                pruning = "none" if mode == "neugn" else "domains"
                results(g, model, split["validation"][:1], args, configs[mode], pruning)
        for mode in args.test_modes:
            pruning = "none" if mode == "neugn" else "domains"
            gate = "transformer" if mode == "transformer" else None
            if mode == "adaptive":
                gate = "mlp" if "mlp" in policies else "transformer"
            records = results(g, model, split["test"], args, configs[mode], pruning)
            summary = summarize(records, 1, args.budget)
            for key in ("median_fms", "median_steps_to_k"):
                if not math.isfinite(summary[key]):
                    summary[key] = None
            stages = summary.get("pruning_stages", {})
            arcs = stages.get("arc_consistency", dict(calls=0, seconds=0.0))
            matching = stages.get("matching", dict(calls=0, seconds=0.0))
            solved_ms = [r["total_seconds"] * 1000 for r in records if r["fms"] is not None]
            row = dict(size=size, density=density, mode=mode, queries=len(records), solved=summary["solved"],
                       timed_out=summary["timed_out"], total_seconds=summary["total_seconds"],
                       median_solved_ms=float(np.median(solved_ms)) if solved_ms else None,
                       median_fms=summary["median_fms"], dfs_steps=sum(r["steps"] for r in records),
                       pruning_calls=summary.get("pruning_calls", 0), arc_calls=arcs["calls"],
                       arc_seconds=arcs["seconds"], matching_calls=matching["calls"],
                       matching_seconds=matching["seconds"],
                       gate_decisions=stages.get("arc_policy", {}).get("calls"),
                       gate_checks=summary.get("arc_policy_checks"),
                       gate_seconds=stages.get("arc_policy", {}).get("seconds"))
            table.append(row)
            write_json(output / f"size{size}_{density}_{mode}.json",
                       dict(config=dict(graph=args.graph, size=size, density=density, mode=mode, depth=args.depth,
                                        gate=gate,
                                        pruning=pruning, budget=args.budget, time_limit=args.time_limit,
                                        device=args.device, warmup=args.warmup),
                            query_set=split["metadata"], summary=summary, records=records))
            print(f"Test {size}/{density}/{mode}: {row['solved']}/{row['queries']} solved, "
                  f"{row['total_seconds']:.3f}s", flush=True)
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"Saved {len(table)} runs to {output / 'summary.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results-output", type=Path, help="write test results separately from training files")
    parser.add_argument("--graph", choices=sorted(LOADERS), default="hamster")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--phase", choices=("all", "train", "test"), default="all")
    parser.add_argument("--gate", choices=("mlp", "transformer", "both"), default="mlp")
    parser.add_argument("--candidates-per-domain", type=int, default=16,
                        help="maximum candidate embeddings per residual token in the Transformer gate")
    parser.add_argument("--test-modes", choices=(*MODES, "neugn", "transformer"), nargs="+")
    parser.add_argument("--warmup", action="store_true", help="run one untimed validation query per test mode and group")
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
    parser.add_argument("--transformer-epochs", type=int, default=30, help="Transformer head epochs with --gate both")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = Path("checkpoints") / args.graph / "paper" / "model.pt"
    if args.test_modes is None:
        args.test_modes = COMPARISON_MODES if args.gate == "both" else MODES
    if (args.depth < 0 or args.seed < 0 or min(args.sizes) < 2
            or len(set(args.sizes)) != len(args.sizes)
            or len(set(args.test_modes)) != len(args.test_modes)
            or min(args.train_queries, args.validation_queries, args.test_queries,
                   args.samples, args.stride, args.budget, args.epochs, args.transformer_epochs,
                   args.candidates_per_domain) < 1
            or any(not math.isfinite(v) or v <= 0 for v in (args.horizon_ms, args.trace_seconds, args.time_limit))):
        parser.error("invalid sizes, split, training or search limits")
    if "transformer" in args.test_modes and args.gate == "mlp":
        parser.error("the transformer test mode requires --gate transformer or --gate both")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    gates = ("mlp", "transformer") if args.gate == "both" else (args.gate,)
    protocol_file = args.output / "protocol.json"
    protocol = dict(checkpoint_digest=checkpoint_digest(args.checkpoint), sizes=args.sizes,
                    train=args.train_queries, validation=args.validation_queries, test=args.test_queries,
                    depth=args.depth, time_limit=args.time_limit, budget=args.budget, device=args.device)
    if args.gate != "mlp":
        protocol["gate"] = args.gate
    if args.graph != "hamster":
        protocol["graph"] = args.graph
    if args.phase == "test" and json.loads(protocol_file.read_text()) != protocol:
        raise ValueError("test configuration differs from the calibrated run")
    g = load(args.graph)
    model = load_model(args.checkpoint, args.device)
    if model.cfg.num_g != g.n or model.cfg.num_labels <= int(g.labels.max()):
        raise ValueError(f"{args.checkpoint} has an incompatible vertex or label vocabulary for {args.graph}")
    sets = load_splits(g, args)
    if args.phase != "test":
        aggregates, residual_rows, gains, cohorts = collect(g, model, sets, args)
        print(f"Training on {len(gains)} labeled states", flush=True)

    policies = {}
    for gate in gates:
        transformer = gate == "transformer"
        model_file = args.output / ("arc_transformer.pt" if transformer else "arc_mlp.npz")
        if args.phase == "test":
            readout, threshold = (residual_arc.load(model_file, model) if transformer
                                  else deep_arc.load(model_file))
        else:
            print(f"Training {gate} gate", flush=True)
            features = residual_rows if transformer else aggregates
            if transformer:
                epochs = args.transformer_epochs if args.gate == "both" else args.epochs
                readout = residual_arc.fit(model, features, gains, cohorts, epochs, args.seed,
                                           args.candidates_per_domain)
            else:
                readout = deep_arc.fit(features, gains, cohorts, args.epochs, args.seed)
            calibration = args.output / (f"calibration_{gate}.json" if len(gates) > 1 else "calibration.json")
            threshold = calibrate(g, model, sets, args, gate, readout, features, calibration)
            (residual_arc if transformer else deep_arc).save(model_file, readout, threshold)
        policies[gate] = gate_policy(args.depth, gate, readout, threshold)
    if args.phase != "test":
        write_json(protocol_file, protocol)
    if args.phase != "train":
        test(g, model, sets, args, policies)


if __name__ == "__main__":
    main()
