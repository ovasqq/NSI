"""First-embedding benchmark for full-depth ranking and proof selection."""

import argparse
import json
import shutil
import subprocess
from collections import Counter
from copy import copy
from math import isfinite
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter

import numpy as np
import torch
from tqdm import tqdm

from data.loaders import load
from experiments.deep_pruning import fingerprint
from experiments.fragment_pruning import search_space
from experiments.fragment_switch import check_selected_fragment, select
from experiments.late_check_pilot import choose, compare_paired, fit_tree
from experiments.residual_policy import fit_cost_ensemble, predict_cost_deltas
from models.hook import NeuGNReorder
from models.joint_dfs import Inference, StateFeatures, train_joint, vertex_features
from models.pruning_controller import checkpoint_digest
from queries.storage import load_query_set
from search.enumerate import SearchSpace, enumerate_space
from search.joint_checks import CHECK_ACTIONS, DEFER_DELAYS, MODEL_ACTIONS, run_check
from search.nlf_index import NLFIndex
from search.order import gql_order
from search.pruning import ResidualPruner
from search.residual_proofs import ProofCache
from train.train_neugn import load_model


BASELINES = ("neugn_original", "neugn", "neugn_domain", "neugn_full_domain", "switch")
JOINT_MODES = ("joint_rank", "joint_graph_rank", "joint_fixed", "joint_tree", "joint_ensemble",
               "joint_mlp", "joint_graph", "joint_adaptive")


def indexed_space(query, index):
    candidates = index.filter(query)
    return SearchSpace(query, index.graph, candidates, gql_order(query, candidates))


def run(space, teacher, data_features, args, mode, *, light=None, graph_model=None,
        costs=None, observe=None, store_embedding=False, margin=0., delay=None,
        reuse=True, cache_proofs=True):
    start = perf_counter()
    deadline = start + args.seconds if args.seconds is not None else None
    pruner = ResidualPruner(space, "forward")
    stats = Counter()
    representation = None
    frames = {}
    cache = None
    residues = {} if reuse else None
    use_graph = mode in ("joint_graph_rank", "joint_graph")
    adaptive = mode == "joint_adaptive" or delay is not None
    shared = graph_model if use_graph else light
    joint = mode not in BASELINES
    checking = mode not in ("joint_rank", "joint_graph_rank") or delay is not None
    if joint:
        representation = StateFeatures(space, data_features)
        cache = ProofCache() if checking and cache_proofs else None
        frames[0] = representation.frame(dict(enumerate(space.cand_set)))
    else:
        hook = NeuGNReorder(teacher, space.q, None if mode == "neugn_full_domain" else args.depth,
                            args.device, domain_only=mode in ("neugn_domain", "neugn_full_domain"))

    def hidden(depth, frame):
        if frame.hidden is None:
            frame.hidden = shared.encode(representation.encode(depth, frame, graph=use_graph))
        return frame.hidden

    def rank(depth, u, mapping, local):
        began = perf_counter()
        if shared is None:
            ordered = local
        else:
            frame = frames[depth]
            scores = shared.scores(hidden(depth, frame),
                                  representation.candidate_features(depth, frame, local, graph=use_graph))
            ordered = [local[i] for i in np.argsort(-scores, kind="stable")]
            stats["rank_calls"] += 1
        stats["rank_seconds"] += perf_counter()-began
        return ordered

    def proof(depth, frame, action, propagate):
        began = perf_counter()
        stats["check_"+action] += 1
        parent = frames.get(depth-1) if reuse else None
        result = run_check(pruner, frame, action, parent=parent, cache=cache,
                           residues=residues, budget=args.fragment_budget, propagate=propagate,
                           reuse_matching=reuse)
        elapsed = perf_counter()-began
        stats["check_seconds"] += elapsed
        stats["seconds_"+action] += elapsed
        if result is not None:
            stats["cut_"+result] += 1
        return result

    def decision(depth, frame):
        if mode == "joint_fixed" or delay is not None:
            return "cascade"
        if mode in ("joint_tree", "joint_ensemble"):
            features = representation.encode(depth, frame)
            if mode == "joint_tree":
                return choose(costs[mode], features)
            predictions = predict_cost_deltas(costs[mode], features)
            index = int(np.argmin(predictions))
            return CHECK_ACTIONS[index+1] if predictions[index] < 0 else "skip"
        predictions = shared.costs(hidden(depth, frame))
        usable = predictions if adaptive else predictions[:len(CHECK_ACTIONS)-1]
        index = int(np.argmin(usable))
        return MODEL_ACTIONS[index+1] if usable[index] < -margin else "skip"

    def advance(depth, mapping, used, domains):
        started = perf_counter()
        u = space.order[depth-1]
        parent = frames[depth-1]
        new = pruner.extend_domains(parent.domains, u, mapping[u])
        if new is None:
            stats["cut_empty"] += 1
            stats["state_seconds"] += perf_counter()-started
            return None
        frame = representation.frame(new, parent, (u, mapping[u]))
        frames[depth] = frame
        stats["state_seconds"] += perf_counter()-started
        if len(new) < 2 or not checking or observe is not None:
            return new
        if cache is not None and cache.entries:
            started = perf_counter()
            conflict = cache.conflict(new)
            stats["cache_seconds"] += perf_counter()-started
            if conflict is not None:
                stats["cache_"+conflict+"_cuts"] += 1
                return None
        started = perf_counter()
        action = decision(depth, frame)
        stats["decision_seconds"] += perf_counter()-started
        stats["decisions"] += 1
        if delay is not None or action.startswith("defer_"):
            frame.delay = delay if delay is not None else int(action.split("_")[1])
            frame.scheduled_action = "cascade"
            stats["defer_"+str(frame.delay)] += 1
            return new
        if action == "skip":
            stats["skips"] += 1
            return new
        if proof(depth, frame, action, propagate=True) is not None:
            return None
        if frame.domains is not new and frame.domains != new:
            started = perf_counter()
            updated = representation.frame(frame.domains, parent, (u, mapping[u]))
            updated.arc_consistent, updated.matching = frame.arc_consistent, frame.matching
            frames[depth] = updated
            stats["state_seconds"] += perf_counter()-started
        return frame.domains

    def backjump(depth):
        # A recorded proof can jump only when it applies to an earlier
        # frame's entire domain snapshot.
        if not cache.entries or depth < 1:
            return None
        started = perf_counter()
        target = next((i for i in range(depth+1) if cache.conflict(frames[i].domains)), None)
        stats["cache_seconds"] += perf_counter()-started
        if target is not None:
            stats["backjumps"] += 1
            stats["levels_jumped"] += depth-target
        return target

    def switch_advance(depth, mapping, used, domains):
        u = space.order[depth-1]
        new = (pruner.build_domains(depth, mapping, used) if domains is None
               else pruner.extend_domains(domains, u, mapping[u]))
        if new is None or depth < args.depth or len(new) < 2:
            return new
        began = perf_counter()
        _, fragment = select(space, new, "switch", args.threshold, 12)
        stats["decision_seconds"] += perf_counter()-began
        if fragment:
            began = perf_counter()
            outcome = check_selected_fragment(pruner, depth, mapping, used, new, fragment,
                                              args.fragment_budget, incremental=True)
            stats["check_seconds"] += perf_counter()-began
            if outcome is False:
                return None
        return new

    callbacks = {}
    if joint:
        callbacks["branch_domains"] = advance
        if cache is not None and observe is None:
            callbacks["backjump_depth"] = backjump
        if adaptive:
            callbacks.update(
                late_check=lambda depth, domains: proof(
                    depth, frames[depth], frames[depth].scheduled_action, False,
                ) is not None,
                late_schedule=lambda depth, domains: frames[depth].delay,
                late_depth=1,
            )
        if observe is not None:
            callbacks.update(
                late_observer=lambda depth, domains, step: observe(representation, frames[depth], depth, step),
                late_depth=1, late_steps=0,
            )
    elif mode == "switch":
        callbacks["branch_domains"] = switch_advance
    result = enumerate_space(space, rank if joint else hook, max_matches=1,
                             max_steps=args.budget, deadline=deadline,
                             store_embeddings=store_embedding, **callbacks)
    if not joint:
        stats.update(rank_seconds=hook.seconds, rank_calls=hook.calls)
    return dict(seconds=perf_counter()-start, solved=result.num_matches > 0,
                fms=result.first_match_step, steps=result.steps, timed_out=result.timed_out,
                embedding=result.embeddings[0] if result.embeddings else None, **stats)


def rank_examples(space, data, embedding, rng, negatives):
    if embedding is None or len(embedding) != space.q.n:
        return
    features = StateFeatures(space, data)
    pruner = ResidualPruner(space, "forward")
    frame = features.frame(dict(enumerate(space.cand_set)))
    for depth, u in enumerate(space.order):
        image = int(embedding[u])
        if image not in frame.domains[u]:
            return
        others = sorted(frame.domains[u] - {image})
        if others:
            ids = [image, *rng.sample(others, min(negatives, len(others)))]
            light, graph = (features.candidate_features(depth, frame, ids, graph=g)
                            for g in (False, True))
            simple, structural = (features.encode(depth, frame, graph=g) for g in (False, True))
            for a, b in zip(light[0]-light[1:], graph[0]-graph[1:], strict=True):
                yield simple, structural, a, b
        if depth + 1 < space.q.n:
            child = pruner.extend_domains(frame.domains, u, image)
            if child is None:
                return
            frame = features.frame(child, frame, (u, image))


def collect(train, index, data, teacher, args):
    rank_rows, check_rows, diagnostics = [], [], []
    rng = Random(args.seed)
    training_args = copy(args)
    training_args.seconds = None
    for query_id, query in enumerate(tqdm(train, desc="Label all DFS depths", disable=not args.progress)):
        space = indexed_space(query.graph, index)
        samples, seen = {}, Counter()

        def observe(representation, frame, depth, step):
            seen[depth] += 1
            sample = samples.setdefault(depth, [])
            slot = seen[depth]-1 if seen[depth] <= args.samples else rng.randrange(seen[depth])
            if slot >= args.samples:
                return None
            row = [representation, frame, step, None, None, None]
            if slot == len(sample):
                sample.append(row)
            else:
                sample[slot] = row
            start = perf_counter()

            def finish(end, complete):
                row[3:] = [end-step, (perf_counter()-start)*1e6, not complete]

            return finish

        observed = run(space, teacher, data, training_args, "joint_rank", observe=observe,
                       store_embedding=True)
        embeddings = {tuple(query.origin)}
        if observed["embedding"] is not None:
            embeddings.add(observed["embedding"])
        for embedding in sorted(embeddings):
            rank_rows.extend(rank_examples(space, data, embedding, rng, args.negatives))

        pruner = ResidualPruner(space, "forward")
        for depth, sample in samples.items():
            for features, frame, start, remaining, micros, censored in sample:
                outcomes = {}
                actions = list(CHECK_ACTIONS[1:])
                rng.shuffle(actions)
                for action in actions:
                    tick = perf_counter()
                    reason = run_check(pruner, frame, action, propagate=False,
                                       budget=args.fragment_budget)
                    outcomes[action] = ((perf_counter()-tick)*1e6, reason)
                value = micros + args.step_value_us*remaining
                immediate = [outcomes[action][0] - value*(outcomes[action][1] is not None)
                             for action in CHECK_ACTIONS[1:]]
                cascade_us, cascade_reason = outcomes["cascade"]
                # Postponed checks may never be reached. For reached checks,
                # approximate the remaining time from the observed step fraction.
                delayed = [
                    0. if remaining <= d else
                    cascade_us - (cascade_reason is not None)*value*(1-d/remaining)
                    for d in DEFER_DELAYS
                ]
                check_rows.append((features.encode(depth, frame),
                                   features.encode(depth, frame, graph=True), immediate + delayed))
                diagnostics.append(dict(query=query_id, depth=depth, step=start, steps=remaining,
                                        remaining_us=micros, censored=censored, checks=outcomes))
    if not rank_rows or not check_rows:
        raise ValueError("Training split contains no rank pairs or eligible proof states")
    return rank_rows, check_rows, diagnostics


def paired_summary(records, modes):
    result = {}
    for mode in modes:
        trials = [r[mode] for r in records]
        solved = [ts[0]["fms"] for ts in trials if ts[0]["fms"] is not None]
        counters = {key for ts in trials for key in ts[0] if
                    key.startswith(("check_", "cut_", "defer_", "cache_", "seconds_"))
                    or key in ("rank_calls", "decisions", "skips", "backjumps", "levels_jumped")}
        phase_times = ("prepare_seconds", "state_seconds", "rank_seconds", "decision_seconds")
        result[mode] = dict(
            queries=len(trials), solved=sum(ts[0]["solved"] for ts in trials),
            solved_by_repeat=[sum(ts[i]["solved"] for ts in trials) for i in range(len(trials[0]))],
            median_fms_solved=median(solved) if solved else None,
            median_ms=1000*median(median(t["seconds"] for t in ts) for ts in trials),
            seconds=sum(median(t["seconds"] for t in ts) for ts in trials),
            steps=sum(ts[0].get("steps") or 0 for ts in trials) if mode != "glasgow" else None,
            **{k: sum(median(t.get(k, 0.) for t in ts) for ts in trials)
               for k in (*phase_times, *sorted(counters))},
        )
    return result


def write_glasgow(graph, path):
    with path.open("w") as stream:
        stream.writelines(f"{i},,{int(label)}\n" for i, label in enumerate(graph.labels))
        stream.writelines(f"{u},{v}\n" for u, v in graph.edges())


def glasgow(binary, query_path, graph_path, seconds):
    start = perf_counter()
    try:
        proc = subprocess.run([binary, "--format", "csv", str(query_path), str(graph_path)],
                              capture_output=True, text=True, timeout=seconds, check=True)
    except subprocess.TimeoutExpired:
        return dict(seconds=perf_counter()-start, solved=False, timed_out=True, fms=None, steps=None)
    fields = {key.strip(): value.strip() for line in proc.stdout.splitlines() if "=" in line
              for key, value in (line.split("=", 1),)}
    status = fields.get("status", "").lower()
    if status not in ("true", "false", "aborted"):
        raise ValueError(f"Unrecognized Glasgow status: {proc.stdout[:500]!r}")
    return dict(seconds=perf_counter()-start, solved=status == "true",
                timed_out=status == "aborted", fms=None, steps=None)


def read_queries(args):
    sources = {}
    for size in args.sizes:
        for density in ("dense", "sparse"):
            name = f"size{size}_{density}.json"
            source, meta = load_query_set(args.query_dir / name)
            if meta.get("graph") != args.graph:
                raise ValueError(f"{name} belongs to a different graph")
            sources[size, density] = source
    test = []
    for (size, density), source in sources.items():
        offset = args.test_offset
        if args.test_query_dir is not None:
            source, meta = load_query_set(args.test_query_dir / f"size{size}_{density}.json")
            if meta.get("graph") != args.graph:
                raise ValueError("External test queries belong to a different graph")
            offset = 0
        if len(source) < offset+args.queries:
            raise ValueError(f"Not enough test queries for {size} {density}")
        test.extend((size, density, i, source[i]) for i in range(offset, offset+args.queries))
    seen = {fingerprint(q) for _, _, _, q in test}
    validation, train = [], []
    for target, start, end in ((validation, args.train_queries, args.train_queries+args.validation_queries),
                               (train, 0, args.train_queries)):
        for (size, density), source in sources.items():
            if len(source) < end:
                raise ValueError(f"Not enough training/validation queries for {size} {density}")
            for i in range(start, end):
                query = source[i]
                key = fingerprint(query)
                if key in seen:
                    continue
                seen.add(key)
                target.append((size, density, i, query) if target is validation else query)
    if not train or not validation:
        raise ValueError("No disjoint training or validation queries")
    return train, validation, test


def benchmark(items, modes, index_filter, teacher, data, args, runtime, path, repetitions,
              external=None):
    rng = Random(args.seed)
    records = []
    with path.open("w", buffering=1) as stream:
        for size, density, index, query in tqdm(items, desc=path.stem, disable=not args.progress):
            start = perf_counter()
            space = indexed_space(query.graph, index_filter)
            prepare_seconds = perf_counter()-start
            prepared = {name: (space, prepare_seconds) for name in modes}
            if "neugn_original" in modes:
                start = perf_counter()
                original = search_space(query.graph, index_filter.graph)
                prepared["neugn_original"] = (original, perf_counter()-start)
            common_deadline = args.seconds
            trial_args = copy(args)
            if external is not None:
                trial_args.budget = None
            trials = {name: [] for name in modes}
            if external is not None:
                binary, target_path = external
                query_path = path.parent / "glasgow" / f"{size}_{density}_{index}.csv"
                write_glasgow(query.graph, query_path)
            for _ in range(repetitions):
                shuffled = list(modes)
                rng.shuffle(shuffled)
                for name in shuffled:
                    if name == "glasgow":
                        trial = glasgow(binary, query_path, target_path, common_deadline)
                    else:
                        mode, options = modes[name]
                        trial_space, setup = prepared[name]
                        trial_args.seconds = (max(0., common_deadline-setup)
                                              if common_deadline is not None else None)
                        trial = run(trial_space, teacher, data, trial_args, mode, **runtime, **options)
                        trial["seconds"] += setup
                        trial["prepare_seconds"] = setup
                    trials[name].append(trial)
            record = dict(size=size, density=density, index=index, **trials)
            records.append(record)
            stream.write(json.dumps(record)+"\n")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "query-dir", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--test-query-dir", type=Path)
    parser.add_argument("--graph", default="wikics")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 16, 24, 32])
    for key, value in dict(depth=10, train_queries=60, validation_queries=30, queries=80,
                           test_offset=120, budget=30000, repeats=3, samples=3, negatives=4,
                           epochs=6, seed=19, threshold=256, fragment_budget=256).items():
        parser.add_argument("--"+key.replace("_", "-"), type=int, default=value)
    parser.add_argument("--seconds", type=float, help="Common end-to-end wall limit per test query.")
    parser.add_argument("--step-value-us", type=float, default=10.)
    parser.add_argument("--glasgow-bin", help="Path to a Glasgow Subgraph Solver executable.")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if min(args.depth, args.train_queries, args.validation_queries, args.queries,
           args.budget, args.repeats, args.samples, args.negatives, args.epochs,
           args.threshold, args.fragment_budget) <= 0 or args.seed < 0 or min(args.sizes) < 3:
        parser.error("use positive counts, sizes ≥3 and nonnegative seed")
    if args.test_offset < args.train_queries+args.validation_queries and not args.test_query_dir:
        parser.error("source splits overlap")
    if len(set(args.sizes)) != len(args.sizes):
        parser.error("query sizes must be distinct")
    if args.seconds is not None and (not isfinite(args.seconds) or args.seconds <= 0):
        parser.error("wall limit must be positive and finite")
    if not isfinite(args.step_value_us) or args.step_value_us < 0:
        parser.error("step value must be nonnegative and finite")
    binary = shutil.which(args.glasgow_bin) if args.glasgow_bin else None
    if args.glasgow_bin and (binary is None or args.seconds is None):
        parser.error("Glasgow comparison needs an executable and --seconds")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("use an empty output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    graph = load(args.graph)
    started = perf_counter()
    index = NLFIndex(graph)
    data = vertex_features(graph)
    graph_index_seconds = perf_counter()-started
    teacher = load_model(args.checkpoint, args.device)
    if teacher.cfg.num_g != graph.n:
        parser.error("checkpoint graph size mismatch")
    train, validation, test = read_queries(args)
    with torch.no_grad():
        rank_rows, check_rows, labels = collect(train, index, data, teacher, args)
    with (args.out/"training_labels.jsonl").open("w") as stream:
        stream.writelines(json.dumps(row)+"\n" for row in labels)
    print(f"Rank pairs {len(rank_rows)}; proof states {len(check_rows)}", flush=True)
    light = train_joint(rank_rows, check_rows, epochs=args.epochs, seed=args.seed, device=args.device)
    graph_model = train_joint(rank_rows, check_rows, graph=True, epochs=args.epochs,
                              seed=args.seed, device=args.device)
    x = np.asarray([row[0] for row in check_rows])
    y = np.asarray([[0., *row[2][:len(CHECK_ACTIONS)-1]] for row in check_rows])
    tree = fit_tree(x, y, actions=CHECK_ACTIONS)
    ensemble = fit_cost_ensemble(x, y, rounds=16)
    torch.save(dict(light=light.state_dict(), graph=graph_model.state_dict(),
                    proof_actions=MODEL_ACTIONS, tree=tree, ensemble=ensemble),
               args.out/"joint_models.pt")
    runtime = dict(light=Inference.from_model(light), graph_model=Inference.from_model(graph_model),
                   costs={"joint_tree": tree, "joint_ensemble": ensemble})
    candidate_modes = {"joint_fixed": ("joint_fixed", {})}
    for mode in ("joint_mlp", "joint_graph", "joint_adaptive"):
        for margin in (0., .5, 1.):
            candidate_modes[f"{mode}_{margin:g}"] = mode, {"margin": margin}
    with torch.no_grad():
        validation_records = benchmark(
            validation, candidate_modes, index, teacher, data, args, runtime,
            args.out/"validation.jsonl", 1,
        )
    valve = paired_summary(validation_records, candidate_modes)
    pairs = compare_paired(validation_records, candidate_modes, "joint_fixed")
    selected = {}
    modes = {name: (name, {}) for name in (*BASELINES, *JOINT_MODES)}
    for mode in ("joint_mlp", "joint_graph", "joint_adaptive"):
        best = min((name for name in pairs if candidate_modes[name][0] == mode),
                   key=lambda name: (len(pairs[name]["only_reference"]), valve[name]["seconds"]))
        selected[mode] = candidate_modes[best][1]["margin"]
        modes[mode] = candidate_modes[best]
    (args.out/"calibration.json").write_text(json.dumps(dict(margins=selected, summary=valve,
                                                             paired=pairs), indent=2)+"\n")
    for delay in DEFER_DELAYS:
        modes[f"joint_fixed_{delay}"] = "joint_rank", {"delay": delay}
    modes["joint_fixed_rebuild"] = "joint_fixed", {"reuse": False}
    modes["joint_fixed_no_cache"] = "joint_fixed", {"cache_proofs": False}
    external = None
    if binary is not None:
        (args.out/"glasgow").mkdir()
        target = args.out/"glasgow"/"target.csv"
        write_glasgow(graph, target)
        external = binary, target
        modes["glasgow"] = None
    with torch.no_grad():
        records = benchmark(test, modes, index, teacher, data, args, runtime,
                            args.out/"test_records.jsonl", args.repeats, external)
    summary = paired_summary(records, modes)
    by_group = {f"{size}_{density}": paired_summary(
        [r for r in records if r["size"] == size and r["density"] == density], modes)
        for size, density in sorted({(size, density) for size, density, _, _ in test})}
    result = dict(params={k: str(v) for k, v in vars(args).items()}, summary=summary,
                  by_group=by_group, selected_margins=selected,
                  graph_index_seconds=graph_index_seconds,
                  checkpoint_sha256=checkpoint_digest(args.checkpoint),
                  source_sha256={path.name: checkpoint_digest(path)
                                 for path in sorted(args.query_dir.glob("size*.json"))},
                  test_sha256={path.name: checkpoint_digest(path)
                               for path in sorted(args.test_query_dir.glob("size*.json"))}
                  if args.test_query_dir else None,
                  test_step_budget=None if external else args.budget,
                  paired_vs_neugn_original=compare_paired(records, modes, "neugn_original"),
                  paired_vs_switch=compare_paired(records, modes, "switch"),
                  paired_vs_fixed=compare_paired(records, modes, "joint_fixed"),
                  paired_graph_rank=compare_paired(records, ("joint_graph_rank",), "joint_rank"),
                  paired_graph_checks=compare_paired(records, ("joint_graph",), "joint_graph_rank"))
    (args.out/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
