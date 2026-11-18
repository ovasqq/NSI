"""Collect, train, calibrate and benchmark candidate-conditioned fragment pruning."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from data.loaders import LOADERS, load
from experiments.deep_pruning import load_splits, write_json
from experiments.fms import summarize
from models.fragment import FragmentHead, batch_rows, encode, manual_fragment, proposals
from models.hook import NeuGNReorder
from models.pruning_controller import checkpoint_digest
from search.enumerate import SearchSpace, enumerate_space
from search.filter import nlf_filter
from search.order import gql_order
from search.pruning import ResidualPruner, check_fragment
from train.train_neugn import load_model

MODES = ("neugn", "domains", "forward", "manual", "learned")


def search_space(q, g):
    candidates = nlf_filter(q, g)
    return SearchSpace(q, g, candidates, gql_order(q, candidates))


def ranker(model, q, args, capture=False):
    return NeuGNReorder(model, q, args.depth, args.device, capture_state=capture)


def collect(sets, g, model, args):
    rng = np.random.default_rng(args.seed)
    rows, gains, stats = [], [], []
    for (size, density), split in sets.items():
        group = dict(size=size, density=density, states=0, fragments=0, refutations=0,
                     exhausted=0, profitable=0, profitable_manual=0, censored=0)
        for _, query in split["train"]:
            space = search_space(query.graph, g)
            hook = ranker(model, space.q, args, capture=True)
            continuator = ranker(model, space.q, args)
            pruner = ResidualPruner(space, "forward")
            sampled = []
            eligible = 0

            def sample(branch, mapping):
                nonlocal eligible
                d = branch.depth + 1
                if d > args.depth or branch.width < 2 or d + 1 >= space.q.n:
                    return
                eligible += 1
                index = eligible - 1 if eligible <= args.samples else rng.integers(eligible)
                if index >= args.samples:
                    return
                prefix = tuple(mapping[u] for u in space.order[:d])
                anchor, _ = hook.ranking_context[d - 1]
                item = prefix, branch.steps_remaining, anchor
                if index == len(sampled):
                    sampled.append(item)
                else:
                    sampled[index] = item

            enumerate_space(space, hook, pruning="forward", max_matches=1, max_steps=args.budget,
                            deadline=perf_counter() + args.trace_seconds, before_check=sample)
            for prefix, remaining, anchor in sampled:
                depth = len(prefix)
                mapping = [-1] * space.q.n
                for x, image in zip(space.order, prefix):
                    mapping[x] = image
                domains = pruner.build_domains(depth, mapping, set(prefix))
                if domains is None:
                    continue
                options = proposals(space, domains, args.proposals)
                if not options:
                    continue
                check_times, proofs = [], []
                for vertices in options:
                    start = perf_counter()
                    outcome = check_fragment(space, domains, vertices, args.fragment_budget)
                    check_times.append((perf_counter() - start) * 1000)
                    proofs.append(outcome is False)
                    group["exhausted"] += int(outcome is None)
                continuation = 0.0
                if any(proofs):
                    start = perf_counter()
                    replay = enumerate_space(space, continuator, pruning="forward", prefix=prefix,
                                             max_matches=1, max_steps=remaining,
                                             deadline=start + args.horizon_ms / 1000)
                    continuation = min((perf_counter() - start) * 1000, args.horizon_ms)
                    group["censored"] += int(not replay.complete)
                    if replay.num_matches:
                        raise RuntimeError("a fragment refuted a branch with an embedding")
                rows.append(encode(space, domains, depth, mapping, args.candidates, options, anchor))
                values = (np.asarray(proofs) * continuation - check_times).astype(np.float32)
                gains.append(torch.from_numpy(values))
                group["states"] += 1
                group["fragments"] += len(options)
                group["refutations"] += sum(proofs)
                group["profitable"] += int(values.max() > 0)
                hand = manual_fragment(space, domains, options)
                group["profitable_manual"] += int(values[options.index(hand)] > 0)
        stats.append(group)
        print(f"Collect {size}/{density}: {group['states']} states, {group['refutations']} refutations, "
              f"{group['profitable']} states with positive gain", flush=True)
    write_json(args.output / "collection.json", stats)
    if not any(group["profitable"] for group in stats):
        raise ValueError("no profitable fragments found; see collection.json")
    return rows, gains


def fit(model, rows, gains, args):
    torch.manual_seed(args.seed)
    head = FragmentHead(model.cfg.d).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.002)
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        indices = rng.permutation(len(rows))
        loss_sum = 0.0
        for start in range(0, len(rows), args.batch_size):
            ids = indices[start:start + args.batch_size]
            batch = batch_rows([rows[i] for i in ids], args.device)
            target = torch.nn.utils.rnn.pad_sequence([gains[i] for i in ids], batch_first=True).to(args.device)
            keep = batch["actions"].any(dim=-1)
            target = target[keep]
            target = target.sign() * target.abs().log1p()
            score = head(model, batch)[keep]
            # Most actions never refute a state. Keep those negatives while
            # increasing the weight of observed savings.
            weights = 1 + 2 * (target > 0)
            loss = (F.smooth_l1_loss(score, target, reduction="none") * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(ids)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            print(f"Epoch {epoch + 1}/{args.epochs}: loss {loss_sum / len(rows):.5f}", flush=True)
    return head.eval()


def run(query, g, model, args, mode, head=None, threshold=None):
    start = perf_counter()
    space = search_space(query.graph, g)
    learned = mode == "learned" and threshold is not None
    hook = ranker(model, space.q, args, capture=learned)
    select = None
    preprocessing = model_time = 0.0
    if mode == "manual" or learned:
        def select(depth, domains, mapping):
            nonlocal preprocessing, model_time
            t0 = perf_counter()
            options = proposals(space, domains, args.proposals)
            if not options or not learned:
                choice = manual_fragment(space, domains, options) if options else ()
                preprocessing += perf_counter() - t0
                return choice
            anchor, _ = hook.ranking_context[depth - 1]
            row = encode(space, domains, depth, mapping, args.candidates, options, anchor)
            batch = batch_rows([row], args.device)
            t1 = perf_counter()
            preprocessing += t1 - t0
            with torch.inference_mode():
                scores = head(model, batch)[0].tolist()
            model_time += perf_counter() - t1
            best = max(range(len(options)), key=scores.__getitem__)
            return options[best] if threshold == "always" or scores[best] > threshold else ()

    begin_search = perf_counter()
    pruning = {"neugn": "none", "domains": "domains"}.get(mode, "forward")
    result = enumerate_space(space, hook, pruning=pruning, max_matches=1, max_steps=args.budget,
                             deadline=start + args.time_limit, fragment=select,
                             fragment_budget=args.fragment_budget, fragment_depth=args.depth)
    end = perf_counter()
    record = dict(fms=result.first_match_step, match_steps=result.match_steps, steps=result.steps,
                  complete=result.complete, timed_out=result.timed_out, total_seconds=end - start,
                  search_seconds=end - begin_search, model_calls=hook.calls, model_seconds=hook.seconds,
                  fragment_preprocessing_seconds=preprocessing, fragment_model_seconds=model_time)
    if result.pruning is not None:
        pruning = result.pruning
        record.update(pruning_calls=pruning.calls, pruning_seconds=pruning.seconds, motif_checks=0,
                      pruning_cuts=pruning.cuts,
                      pruning_stages={name: asdict(stage) for name, stage in pruning.stages.items()})
    return record


def calibrate(sets, g, model, head, args):
    queries = [q for split in sets.values() for _, q in split["validation"]]
    if not queries:
        raise ValueError("no distinct validation queries remain")
    trials = []
    for threshold in (None, "always", -0.025, 0.0, 0.025, 0.10, 0.25, 0.50):
        records = [run(q, g, model, args, "learned", head, threshold) for q in queries]
        summary = summarize(records, 1, args.budget)
        stage = summary.get("pruning_stages", {}).get("fragment_check", {})
        trial = dict(threshold=threshold, solved=summary["solved"], seconds=summary["total_seconds"],
                     checks=stage.get("calls", 0))
        trials.append(trial)
        print(f"Validation {threshold}: {trial['solved']}/{len(queries)} solved, "
              f"{trial['seconds']:.3f}s, {trial['checks']} checks", flush=True)
    def priority(trial):
        return -trial["solved"], trial["seconds"]

    active = [trial for trial in trials if trial["checks"]]
    best = min(trials, key=priority)
    # Keep the learned selector active for the test comparison even if forward
    # alone wins on validation. Benchmark measures both as independent modes.
    selected = min(active, key=priority) if active else trials[1]
    write_json(args.output / "calibration.json",
               dict(queries=len(queries), selected=selected, best_overall=best, trials=trials))
    return selected["threshold"]


def benchmark(sets, g, model, head, threshold, protocol, args):
    rng = np.random.default_rng(args.seed)
    for repeat in range(1, args.repeats + 1):
        output = args.output / f"test_rep{repeat}"
        output.mkdir(exist_ok=True)
        table = []
        for (size, density), split in sets.items():
            cohort = {}
            for mode in rng.permutation(MODES).tolist():
                if split["validation"]:
                    run(split["validation"][0][1], g, model, args, mode, head, threshold)
                records = [dict(query=i, **run(q, g, model, args, mode, head, threshold))
                           for i, q in split["test"]]
                cohort[mode] = records
                summary = summarize(records, 1, args.budget)
                for key in ("median_fms", "median_steps_to_k"):
                    if not math.isfinite(summary[key]):
                        summary[key] = None
                stages = summary.get("pruning_stages", {})
                table.append(dict(size=size, density=density, mode=mode, queries=len(records),
                                  solved=summary["solved"], timed_out=summary["timed_out"],
                                  seconds=summary["total_seconds"], median_fms=summary["median_fms"],
                                  dfs_steps=sum(r["steps"] for r in records),
                                  domain_seconds=stages.get("build_domains", {}).get("seconds", 0),
                                  ac3_seconds=stages.get("arc_consistency", {}).get("seconds", 0),
                                  fragment_checks=stages.get("fragment_check", {}).get("calls", 0),
                                  fragment_seconds=stages.get("fragment_check", {}).get("seconds", 0),
                                  fragment_exhausted=stages.get("fragment_exhausted", {}).get("calls", 0),
                                  fragment_cuts=summary.get("pruning_cuts", {}).get("fragment", 0),
                                  selector_seconds=stages.get("fragment_select", {}).get("seconds", 0),
                                  preprocessing_seconds=sum(r["fragment_preprocessing_seconds"] for r in records),
                                  selector_model_seconds=sum(r["fragment_model_seconds"] for r in records)))
                write_json(output / f"size{size}_{density}_{mode}.json",
                           dict(config=dict(protocol, mode=mode, threshold=threshold),
                                query_set=split["metadata"], summary=summary, records=records))
                print(f"Test {repeat}/{size}/{density}/{mode}: {summary['solved']}/{len(records)} solved, "
                      f"{summary['total_seconds']:.3f}s", flush=True)
            solved_by_neugn = {r["query"] for r in cohort["neugn"] if r["fms"] is not None}
            for row in table[-len(MODES):]:
                paired = [r for r in cohort[row["mode"]] if r["query"] in solved_by_neugn]
                solved_times = [r["total_seconds"] * 1000 for r in paired if r["fms"] is not None]
                row.update(neugn_solved_queries=len(paired), neugn_subset_solved=len(solved_times),
                           neugn_subset_seconds=sum(r["total_seconds"] for r in paired),
                           neugn_subset_median_solved_ms=median(solved_times) if solved_times else None)
        with (output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", choices=sorted(LOADERS), default="wikics")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 16, 24, 32])
    parser.add_argument("--phase", choices=("all", "train", "test"), default="all")
    for name, default in (("depth", 10), ("train-queries", 100), ("validation-queries", 20),
                          ("test-queries", 80), ("samples", 32), ("candidates", 8), ("proposals", 12),
                          ("fragment-budget", 256), ("budget", 100_000), ("epochs", 20),
                          ("batch-size", 32), ("repeats", 3), ("seed", 0)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--time-limit", type=float, default=0.5)
    parser.add_argument("--trace-seconds", type=float, default=1)
    parser.add_argument("--horizon-ms", type=float, default=30)
    args = parser.parse_args()
    if (min(args.sizes) < 3 or len(set(args.sizes)) != len(args.sizes) or args.seed < 0
            or min(args.depth, args.train_queries, args.validation_queries, args.test_queries, args.samples,
                   args.candidates, args.proposals, args.fragment_budget, args.budget, args.epochs,
                   args.batch_size, args.repeats) < 1
            or any(not math.isfinite(v) or v <= 0 for v in (args.time_limit, args.trace_seconds, args.horizon_ms))):
        parser.error("invalid sizes, split, training or search limits")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {key: value for key, value in vars(args).items()
                if key not in ("phase", "checkpoint", "output", "repeats")}
    protocol["checkpoint_digest"] = checkpoint_digest(args.checkpoint)
    g = load(args.graph)
    model = load_model(args.checkpoint, args.device).eval().requires_grad_(False)
    if model.cfg.num_g != g.n or model.cfg.num_labels <= int(g.labels.max()):
        parser.error("checkpoint vertex or label vocabulary is incompatible with the data graph")
    sets = load_splits(g, args)
    path = args.output / "fragment_head.pt"
    if args.phase == "test":
        checkpoint = torch.load(path, map_location=args.device, weights_only=True)
        if checkpoint["protocol"] != protocol:
            parser.error("test configuration differs from the calibrated run")
        head = FragmentHead(model.cfg.d).to(args.device)
        head.load_state_dict(checkpoint["head"])
        threshold = checkpoint["threshold"]
    else:
        rows, gains = collect(sets, g, model, args)
        head = fit(model, rows, gains, args)
        threshold = calibrate(sets, g, model, head, args)
        torch.save(dict(head=head.state_dict(), threshold=threshold, protocol=protocol), path)
        write_json(args.output / "protocol.json", protocol)
    if args.phase != "train":
        benchmark(sets, g, model, head.eval(), threshold, protocol, args)


if __name__ == "__main__":
    main()
