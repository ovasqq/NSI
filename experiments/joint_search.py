"""First-embedding benchmark for full-depth ranking and proof selection."""

import argparse
import json
import shutil
import subprocess
from collections import Counter, OrderedDict
from copy import copy
from math import isfinite
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter

import numpy as np
import torch
from tqdm import tqdm

from data.dblp import load_dblp_proxy15
from data.loaders import load
from experiments.deep_pruning import fingerprint
from experiments.fragment_pruning import search_space
from experiments.fragment_switch import check_selected_fragment, select
from experiments.late_check_pilot import choose, compare_paired, fit_tree
from experiments.residual_policy import fit_cost_ensemble, predict_cost_deltas
from models.hook import NeuGNReorder
from models.neugn_action import ACTIONS as NEUGN_ACTIONS, fit_head, inputs as neugn_action_inputs
from models.joint_dfs import Inference, StateFeatures, train_joint, vertex_features
from models.pruning_controller import checkpoint_digest
from queries.storage import load_query_set
from search.batched_arcs import pack_states, propagate, sparse_adjacency, unpack_domains
from search.enumerate import SearchResult, SearchSpace, enumerate_space
from search.joint_checks import CHECK_ACTIONS, DEFER_DELAYS, MODEL_ACTIONS, run_check
from search.nlf_index import NLFIndex
from search.order import gql_order
from search.pruning import ResidualPruner
from search.residual_proofs import ProofCache
from train.train_neugn import load_model


BASELINES = ("neugn_original", "neugn", "neugn_domain", "neugn_full_domain", "switch")
JOINT_MODES = ("joint_rank", "joint_graph_rank", "joint_fixed", "joint_tree", "joint_ensemble",
               "joint_mlp", "joint_graph", "joint_adaptive")


NEUGN_HEAD_MODES = ("neugn_head_domains", "neugn_head_anchor")
NEUGN_FORWARD_MODES = ("neugn_forward", *NEUGN_HEAD_MODES)
PILOT_MODES = ("neugn_domain", "neugn_forward", "switch", *NEUGN_HEAD_MODES,
               "joint_graph_rank", "joint_graph")


def indexed_space(query, index):
    candidates = index.filter(query)
    return SearchSpace(query, index.graph, candidates, gql_order(query, candidates))


def run(space, teacher, data_features, args, mode, *, light=None, graph_model=None,
        costs=None, observe=None, store_embedding=False, margin=0., delay=None,
        reuse=True, cache_proofs=True, prefix=(), sample_prefix=None, neugn_heads=None,
        rank_all_depths=None, triple_policy=None, count_matches=False):
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
    neugn_forward = mode in NEUGN_FORWARD_MODES
    pilot_switch = mode == "switch" and getattr(args, "seven_mode_pilot", False)
    capture_anchor = mode == "neugn_head_anchor" or (neugn_forward and (observe is not None or
                      getattr(triple_policy, "uses_anchor", False)))
    capture_scores = capture_anchor or mode == "neugn_head_domains"
    checking = mode not in ("joint_rank", "joint_graph_rank", "neugn_forward", "id_forward") or delay is not None
    full_depth_pilot = getattr(args, "neugn_full_depth_pilot", False)
    all_depths = full_depth_pilot if rank_all_depths is None else rank_all_depths
    memo = OrderedDict() if full_depth_pilot and mode in NEUGN_HEAD_MODES and cache_proofs else None
    if joint:
        representation = StateFeatures(space, data_features)
        cache = ProofCache() if checking and cache_proofs and (not neugn_forward or memo is not None) else None
        frames[0] = representation.frame(dict(enumerate(space.cand_set)))
    if not joint or neugn_forward:
        rank_limit = None if mode == "neugn_full_domain" or (neugn_forward and all_depths) else args.depth
        hook = NeuGNReorder(teacher, space.q, rank_limit,
                            args.device, capture_state=capture_anchor, capture_scores=capture_scores,
                            domain_only=neugn_forward or pilot_switch or mode in ("neugn_domain", "neugn_full_domain"))

    def hidden(depth, frame):
        if frame.hidden is None:
            frame.hidden = shared.encode(representation.encode(depth, frame, graph=use_graph))
        return frame.hidden

    def rank(depth, u, mapping, local):
        if mode == "id_forward":
            return local
        began = perf_counter()
        if neugn_forward:
            ordered = hook(depth, u, mapping, local)
            if capture_scores and (rank_limit is None or depth < rank_limit):
                context_started = perf_counter()
                anchor, scores = hook.ranking_context[depth]
                frames[depth].ranking_context = (
                    anchor.numpy() if anchor is not None else None,
                    dict(zip(ordered, scores.tolist(), strict=True)),
                )
                stats["rank_context_seconds"] += perf_counter()-context_started
            return ordered
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

    def ancestor_closed(kind, depth, steps):
        stats["closed_" + kind] += 1
        stats["closure_steps_" + kind] += steps
        stats["closure_depth_" + kind] += depth

    def proof(depth, frame, action, propagate):
        if memo is not None:
            # The three head actions depend only on the residual domains within this query.
            tick = perf_counter()
            key = (action, tuple(sorted(frame.domains.items())))
            if key in memo:
                reason, matching = memo[key]
                memo.move_to_end(key)
                if action == "matching" and reason is None:
                    frame.matching = matching
                stats["cache_result_hits"] += 1
                if reason is not None:
                    stats["cache_result_cuts"] += 1
                stats["cache_seconds"] += perf_counter()-tick
                return reason
            stats["cache_result_misses"] += 1
            stats["cache_seconds"] += perf_counter()-tick
        began = perf_counter()
        stats["check_"+action] += 1
        parent = frames.get(depth-1) if reuse else None
        result = run_check(pruner, frame, action, parent=parent, cache=cache,
                           residues=residues, budget=args.fragment_budget, propagate=propagate,
                           reuse_matching=reuse, stats=stats)
        elapsed = perf_counter()-began
        stats["check_seconds"] += elapsed
        stats["seconds_"+action] += elapsed
        if result is not None:
            stats["cut_"+result] += 1
            stats["proved_"+action] += 1
        if memo is not None:
            tick = perf_counter()
            memo[key] = result, frame.matching if action == "matching" else None
            if len(memo) > 256:
                memo.popitem(last=False)
            stats["cache_seconds"] += perf_counter()-tick
        return result

    def decision(depth, frame, parent, image):
        if neugn_forward:
            features, anchor = neugn_action_inputs(depth, parent, frame, image)
            if anchor is not None:
                stats["anchor_decisions"] += 1
            predictions = neugn_heads[mode].costs(features, anchor)
            best = int(np.argmin(predictions))
            return NEUGN_ACTIONS[best] if predictions[best] < -margin else "skip"
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
        if neugn_forward and observe is not None and len(new) >= 2:
            frame.action_input = (parent, mapping[u])
        if triple_policy is not None and len(new) >= triple_policy.min_size:
            context = parent.ranking_context
            anchor = None if context is None else context[0]
            if triple_policy(space, new, anchor, stats):
                stats["closed_immediate"] += 1
                return None
        if sample_prefix is not None and len(new) >= 2:
            sample_prefix(depth, mapping)
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
        action = decision(depth, frame, parent, mapping[u])
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
            stats["closed_immediate"] += 1
            stats["closure_depth_immediate"] += depth
            return None
        if frame.domains is not new and frame.domains != new:
            started = perf_counter()
            stats["narrowed_"+action] += 1
            stats["removed_"+action] += sum(len(new[v])-len(values) for v, values in frame.domains.items())
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
        target = next((i for i in range(len(prefix), depth+1) if cache.conflict(frames[i].domains)), None)
        stats["cache_seconds"] += perf_counter()-started
        if target is not None:
            stats["backjumps"] += 1
            stats["levels_jumped"] += depth-target
        return target

    def switch_advance(depth, mapping, used, domains):
        started = perf_counter()
        u = space.order[depth-1]
        new = (pruner.build_domains(depth, mapping, used) if domains is None
               else pruner.extend_domains(domains, u, mapping[u]))
        stats["state_seconds"] += perf_counter()-started
        if new is None:
            stats["cut_empty"] += 1
        if new is None or depth < args.depth or len(new) < 2:
            return new
        began = perf_counter()
        action, fragment = select(space, new, "switch", args.threshold, 12)
        stats["decision_seconds"] += perf_counter()-began
        stats["decisions"] += 1
        if fragment:
            stats["check_"+action] += 1
            began = perf_counter()
            outcome = check_selected_fragment(pruner, depth, mapping, used, new, fragment,
                                              args.fragment_budget, incremental=True)
            elapsed = perf_counter()-began
            stats["check_seconds"] += elapsed
            stats["seconds_"+action] += elapsed
            if outcome is False:
                stats["cut_fragment"] += 1
                stats["proved_"+action] += 1
                stats["closed_immediate"] += 1
                stats["closure_depth_immediate"] += depth
                return None
            if outcome is None:
                stats["unknown_"+action] += 1
        else:
            stats["skips"] += 1
        return new

    callbacks = dict(ancestor_closed=ancestor_closed)
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
    if prefix:
        # Reconstruct the unpruned prefix outside the replay clock. Each policy
        # starts with identical domains and an empty proof cache.
        mapping = [-1] * space.q.n
        for depth, image in enumerate(prefix, 1):
            u = space.order[depth-1]
            mapping[u] = image
            if depth < len(prefix):
                parent = frames[depth-1]
                child = pruner.extend_domains(parent.domains, u, image)
                frames[depth] = representation.frame(child, parent, (u, image))
        start = perf_counter()
        deadline = start + args.seconds if args.seconds is not None else None
        domains = advance(len(prefix), mapping, set(prefix), frames[len(prefix)-1].domains)
        callbacks.update(prefix=prefix, prefix_domains=domains)
    root_cut = bool(prefix) and callbacks["prefix_domains"] is None
    if root_cut:
        expired = deadline is not None and perf_counter() >= deadline
        result = SearchResult(complete=not expired, timed_out=expired)
    else:
        enumeration_start = perf_counter()
        result = enumerate_space(space, rank if joint else hook, max_matches=None if count_matches else 1,
                                 max_steps=args.budget, deadline=deadline,
                                 record_all_match_steps=not count_matches,
                                 measure_first_match=count_matches,
                                 store_embeddings=store_embedding, **callbacks)
    if not joint or neugn_forward:
        stats.update(rank_seconds=hook.seconds, rank_calls=hook.calls)
    outcome = ("sat" if result.num_matches else "unsat" if result.complete else
               "timeout" if result.timed_out else "step_limit")
    stats["direct_cuts"] = sum(stats[key] for key in
        ("closed_immediate", "closed_deferred", "cut_empty", "cache_fragment_cuts", "cache_hall_cuts"))
    return dict(seconds=perf_counter()-start, solved=result.num_matches > 0, outcome=outcome,
                fms=result.first_match_step, steps=result.steps, timed_out=result.timed_out,
                num_matches=result.num_matches,
                first_match_seconds=(enumeration_start-start+result.first_match_seconds
                                     if result.first_match_seconds is not None else None),
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


ARC_ACTIONS = frozenset(("arc", "both", "cascade"))


def timed_arc(pruner, domains):
    started = perf_counter()
    reduced = domains.copy()
    consistent = pruner.check_arcs(reduced)
    return (reduced if consistent else None), (perf_counter()-started)*1e6


def arc_labels(pruner, frames, adjacency, batch_size, stride, rng):
    """Compute verdicts once and price each action using single-state CPU AC-3."""
    if adjacency is None:
        return [(*timed_arc(pruner, frame.domains), True) for frame in frames]

    label_rows = []
    graph = pruner.space.g
    query_size = pruner.space.q.n
    query_adj = pruner.space.q.adj_set
    for start in range(0, len(frames), batch_size):
        batch = frames[start:start+batch_size]
        packed = pack_states([(pruner, frame.domains) for frame in batch], query_size, graph.n)
        masks, rejected = propagate(adjacency, *(t.to(adjacency.device) for t in packed))
        reductions = unpack_domains((frame.domains for frame in batch), masks, rejected)
        contradictions = [reduced is None for reduced in reductions]
        measured = {}
        # One or more CPU measurements per batch. Use domain work and verdict
        # to pick a nearby measurement for unmeasured states.
        effort = [sum(len(d)*(len(query_adj[u] & frame.domains.keys()))
                      for u, d in frame.domains.items()) for frame in batch]
        for i in rng.sample(range(len(batch)), (len(batch)+stride-1)//stride):
            measured[i] = timed_arc(pruner, batch[i].domains)
        for i, reduced in enumerate(reductions):
            if i in measured:
                cpu_domains, cpu_us = measured[i]
                if reduced != cpu_domains:
                    raise AssertionError("Batched and CPU AC-3 disagree on a calibration state")
                # Keep the CPU iteration order for the sampled continuation.
                label_rows.append((cpu_domains, cpu_us, True))
            else:
                neighbor = min(measured, key=lambda j: (contradictions[j] != contradictions[i],
                                                        abs(effort[j]-effort[i])))
                label_rows.append((reduced, measured[neighbor][1], False))
    return label_rows


def collect(train, index, data, teacher, args, *, head_only=False):
    rank_rows, check_rows, diagnostics = [], [], []
    rng = Random(args.seed)
    calibration_rng = Random(args.seed)
    adjacency = (sparse_adjacency(index.graph).to(args.device)
                 if not head_only and getattr(args, "label_ac3", "cpu") == "cuda" else None)
    training_args = copy(args)
    training_args.seconds = None
    arc_label_seconds = 0.
    for query_id, query in enumerate(tqdm(train, desc="Label NeuGN checks" if head_only else "Label all DFS depths",
                                           disable=not args.progress)):
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

        observed = run(space, teacher, data, training_args,
                       "neugn_forward" if head_only else "joint_rank",
                       observe=observe, store_embedding=not head_only)
        if not head_only:
            embeddings = {tuple(query.origin)}
            if observed["embedding"] is not None:
                embeddings.add(observed["embedding"])
            for embedding in sorted(embeddings):
                rank_rows.extend(rank_examples(space, data, embedding, rng, args.negatives))

        pruner = ResidualPruner(space, "forward")
        entries = [(depth, row) for depth, sample in samples.items() for row in sample]
        arc_started = perf_counter()
        arcs = arc_labels(pruner, [row[1] for _, row in entries], adjacency,
                          getattr(args, "label_ac3_batch", 8),
                          getattr(args, "label_ac3_cpu_stride", 8), calibration_rng) if not head_only else [
                              (None, 0., False)] * len(entries)
        arc_label_seconds += perf_counter()-arc_started
        for (depth, row), (reduced, arc_us, measured) in zip(entries, arcs, strict=True):
            features, frame, start, remaining, micros, censored = row
            outcomes = {}
            menu = NEUGN_ACTIONS if head_only else CHECK_ACTIONS[1:]
            actions = list(menu)
            rng.shuffle(actions)
            for action in actions:
                if action == "arc" or (action in ARC_ACTIONS and reduced is None):
                    outcomes[action] = (arc_us, "arc" if reduced is None else None)
                    continue
                uses_arc = action in ARC_ACTIONS
                tick = perf_counter()
                reason = run_check(pruner, frame, action, propagate=False,
                                   budget=args.fragment_budget,
                                   arc_domains=reduced if uses_arc else None)
                outcomes[action] = ((perf_counter()-tick)*1e6 + (arc_us if uses_arc else 0.), reason)
            value = micros + args.step_value_us*remaining
            immediate = [outcomes[action][0] - value*(outcomes[action][1] is not None)
                         for action in menu]
            if head_only:
                parent, image = frame.action_input
                check_rows.append((*neugn_action_inputs(depth, parent, frame, image), immediate))
                diagnostics.append(dict(query=query_id, depth=depth, step=start, steps=remaining,
                                        remaining_us=micros, censored=censored, checks=outcomes))
                continue
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
                                    remaining_us=micros, censored=censored,
                                    arc_cpu_measured=measured, checks=outcomes))
    if not check_rows or (not head_only and not rank_rows):
        raise ValueError("Training split contains no rank pairs or eligible proof states")
    if not head_only:
        measured = sum(row["arc_cpu_measured"] for row in diagnostics)
        print(f"AC-3 labeling: {arc_label_seconds:.2f} s, CPU measurements {measured}/{len(diagnostics)}",
              flush=True)
    return rank_rows, check_rows, diagnostics


def paired_summary(records, modes):
    result = {}
    for mode in modes:
        trials = [r[mode] for r in records]
        solved = [ts[0]["fms"] for ts in trials if ts[0]["fms"] is not None]
        counters = {key for ts in trials for trial in ts for key in trial if
                    key.startswith(("check_", "cut_", "defer_", "cache_", "seconds_", "closed_", "closure_", "proved_", "unknown_", "narrowed_", "removed_"))
                    or key in ("rank_calls", "decisions", "anchor_decisions", "skips",
                               "backjumps", "levels_jumped", "direct_cuts")}
        phase_times = ("prepare_seconds", "state_seconds", "rank_seconds", "decision_seconds",
                       "rank_context_seconds")
        result[mode] = dict(
            outcomes_by_repeat=[dict(Counter(ts[i].get("outcome", "unknown") for ts in trials))
                                for i in range(len(trials[0]))],
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


def compare_pilot(records, modes, reference):
    """Pair time and first-repeat FMS on the same solved queries."""
    report = compare_paired(records, modes, reference)
    for mode, summary in report.items():
        common = [(r[mode][0]["fms"], r[reference][0]["fms"]) for r in records
                  if r[mode][0]["solved"] and r[reference][0]["solved"]]
        summary.update(
            common_mode_fms_median=median(a for a, _ in common) if common else None,
            common_reference_fms_median=median(b for _, b in common) if common else None,
            median_paired_fms_delta=median(a-b for a, b in common) if common else None,
        )
    return report


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
        return dict(seconds=perf_counter()-start, solved=False, outcome="timeout", timed_out=True, fms=None, steps=None)
    fields = {key.strip(): value.strip() for line in proc.stdout.splitlines() if "=" in line
              for key, value in (line.split("=", 1),)}
    status = fields.get("status", "").lower()
    if status not in ("true", "false", "aborted"):
        raise ValueError(f"Unrecognized Glasgow status: {proc.stdout[:500]!r}")
    return dict(seconds=perf_counter()-start, solved=status == "true",
                outcome={"true": "sat", "false": "unsat", "aborted": "timeout"}[status],
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
              external=None, *, solver=run, charge_setup=True):
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
                        trial_args.seconds = (max(0., common_deadline-(setup if charge_setup else 0.))
                                              if common_deadline is not None else None)
                        trial = solver(trial_space, teacher, data, trial_args, mode, **runtime, **options)
                        if trial.get("first_match_seconds") is not None:
                            trial["first_match_seconds"] += setup
                        trial["seconds"] += setup
                        trial["prepare_seconds"] = setup
                    trials[name].append(trial)
            record = dict(size=size, density=density, index=index, **trials)
            records.append(record)
            stream.write(json.dumps(record)+"\n")
    return records


def branch_benchmark(test, modes, index, teacher, data, args, runtime):
    """Compare policies from identical viable prefixes with the same learned ranker."""
    if getattr(args, "seven_mode_pilot", False):
        names = ("joint_graph_rank", "joint_graph")
        reference_names = ("joint_graph_rank",)
    else:
        names = ("joint_rank", "joint_fixed", "joint_tree", "joint_ensemble", "joint_mlp", "joint_adaptive")
        reference_names = ("joint_rank", "joint_fixed")
    rng = Random(args.seed)
    records = []
    groups = Counter()
    with (args.out / "branch_trials.jsonl").open("w", buffering=1) as stream:
        for size, density, query_id, query in tqdm(test, desc="Paired branches", disable=not args.progress):
            group = size, density
            if groups[group] >= args.branch_queries:
                continue
            groups[group] += 1
            space = indexed_space(query.graph, index)
            samples = []
            seen = 0

            def sample(depth, mapping):
                nonlocal seen
                seen += 1
                slot = seen-1 if seen <= args.branch_samples else rng.randrange(seen)
                if slot < args.branch_samples:
                    prefix = tuple(mapping[u] for u in space.order[:depth])
                    if slot < len(samples):
                        samples[slot] = prefix
                    else:
                        samples.append(prefix)

            run(space, teacher, data, args, reference_names[0], **runtime, sample_prefix=sample)
            for prefix in samples:
                trials = {name: [] for name in names}
                for _ in range(args.repeats):
                    for name in rng.sample(names, len(names)):
                        mode, options = modes[name]
                        trial = run(space, teacher, data, args, mode, **runtime, **options, prefix=prefix)
                        trials[name].append(trial)
                outcomes = {r["outcome"] for rows in trials.values() for r in rows}
                if {"sat", "unsat"} <= outcomes:
                    raise AssertionError(f"Conflicting proof for query {group}/{query_id} prefix {prefix}")
                record = dict(size=size, density=density, index=query_id, prefix=prefix,
                              depth=len(prefix), eligible_seen=seen, trials=trials)
                stream.write(json.dumps(record)+"\n")
                records.append(record)

    summary = {}
    for name in names:
        repeat_stats = []
        for i in range(args.repeats):
            rows = [record["trials"][name][i] for record in records]
            proved = [r for r in rows if r["outcome"] == "unsat"]
            comparisons = {}
            for baseline in reference_names:
                reference = [record["trials"][baseline][i] for record in records]
                common = [(r, b) for r, b in zip(rows, reference)
                          if r["outcome"] == b["outcome"] == "unsat"]
                comparisons[baseline] = dict(
                    common_unsat=len(common),
                    only_mode_unsat=len(proved)-len(common),
                    only_reference_unsat=sum(r["outcome"] == "unsat" for r in reference)-len(common),
                    paired_seconds=sum(r["seconds"] for r, _ in common),
                    paired_reference_seconds=sum(b["seconds"] for _, b in common),
                    paired_steps=sum(r["steps"] for r, _ in common),
                    paired_reference_steps=sum(b["steps"] for _, b in common),
                    median_paired_delta_ms=1000*median(
                        r["seconds"]-b["seconds"] for r, b in common
                    ) if common else None,
                )
            repeat_stats.append(dict(
                outcomes=dict(Counter(r["outcome"] for r in rows)),
                proof_steps_median=median(r["steps"] for r in proved) if proved else None,
                proof_ms_median=1000*median(r["seconds"] for r in proved) if proved else None,
                comparisons=comparisons,
            ))
        summary[name] = repeat_stats
    return dict(prefixes=len(records), by_repeat=summary)


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
    parser.add_argument("--branch-queries", type=int, default=8, help="Test queries per group for prefix replay; zero disables it.")
    parser.add_argument("--branch-samples", type=int, default=3, help="Maximum sampled viable prefixes per query.")
    parser.add_argument("--seven-mode-pilot", action="store_true",
                        help="Benchmark five NeuGN variants and two structural MLP variants.")
    parser.add_argument("--neugn-full-depth-pilot", action="store_true",
                        help="Train/evaluate full-depth NeuGN heads with per-query caches and paired controls.")
    parser.add_argument("--id-forward-control", action="store_true",
                        help="Add an increasing-ID forward-search control to the seven-mode pilot.")
    parser.add_argument("--neugn-heads", action="store_true",
                        help="Add frozen-NeuGN forward and two learned check policies.")
    parser.add_argument("--label-ac3", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--label-ac3-batch", type=int, default=8)
    parser.add_argument("--label-ac3-cpu-stride", type=int, default=8,
                        help="Measure about one single-state CPU AC-3 per this many states.")
    parser.add_argument("--step-value-us", type=float, default=10.)
    parser.add_argument("--glasgow-bin", help="Path to a Glasgow Subgraph Solver executable.")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.neugn_full_depth_pilot and not args.seven_mode_pilot:
        parser.error("--neugn-full-depth-pilot requires --seven-mode-pilot")
    if args.id_forward_control and not args.seven_mode_pilot:
        parser.error("--id-forward-control requires --seven-mode-pilot")
    if args.seven_mode_pilot:
        if args.glasgow_bin:
            parser.error("--seven-mode-pilot cannot be combined with --glasgow-bin")
        args.neugn_heads = True
    if min(args.depth, args.train_queries, args.validation_queries, args.queries,
           args.budget, args.repeats, args.samples, args.negatives, args.epochs,
           args.threshold, args.fragment_budget) <= 0 or args.seed < 0 or min(args.sizes) < 3:
        parser.error("use positive counts, sizes ≥3 and nonnegative seed")
    if min(args.label_ac3_batch, args.label_ac3_cpu_stride) < 1:
        parser.error("AC-3 batch and CPU calibration stride must be positive")
    if args.label_ac3 == "cuda" and (torch.device(args.device).type != "cuda" or not torch.cuda.is_available()):
        parser.error("--label-ac3 cuda requires an available CUDA --device")
    if args.branch_queries < 0 or args.branch_samples < 1:
        parser.error("invalid branch sampling counts")
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
    graph = load_dblp_proxy15() if args.graph == "dblp_proxy15" else load(args.graph)
    started = perf_counter()
    index = NLFIndex(graph)
    data = vertex_features(graph)
    graph_index_seconds = perf_counter()-started
    teacher = load_model(args.checkpoint, args.device)
    if (teacher.cfg.num_g, teacher.cfg.num_labels) != (graph.n, int(graph.labels.max()) + 1):
        parser.error("checkpoint graph vocabulary mismatch")
    train, validation, test = read_queries(args)
    started = perf_counter()
    with torch.no_grad():
        rank_rows, check_rows, labels = collect(train, index, data, teacher, args)
    with (args.out/"training_labels.jsonl").open("w") as stream:
        stream.writelines(json.dumps(row)+"\n" for row in labels)
    print(f"Rank pairs {len(rank_rows)}; proof states {len(check_rows)}; "
          f"labeling {perf_counter()-started:.2f} s (AC-3: {args.label_ac3})", flush=True)
    graph_model = train_joint(rank_rows, check_rows, graph=True, epochs=args.epochs,
                              seed=args.seed, device=args.device)
    saved = dict(graph=graph_model.state_dict(), proof_actions=MODEL_ACTIONS)
    runtime = dict(graph_model=Inference.from_model(graph_model))
    if args.seven_mode_pilot:
        modes = {name: (name, {}) for name in PILOT_MODES}
        if args.id_forward_control:
            modes["id_forward"] = ("id_forward", {})
        candidate_modes = {"joint_graph_rank": modes["joint_graph_rank"]}
        joint_reference = "joint_graph_rank"
        calibrated = ("joint_graph",)
    else:
        light = train_joint(rank_rows, check_rows, epochs=args.epochs, seed=args.seed, device=args.device)
        x = np.asarray([row[0] for row in check_rows])
        y = np.asarray([[0., *row[2][:len(CHECK_ACTIONS)-1]] for row in check_rows])
        tree = fit_tree(x, y, actions=CHECK_ACTIONS)
        ensemble = fit_cost_ensemble(x, y, rounds=16)
        saved.update(light=light.state_dict(), tree=tree, ensemble=ensemble)
        runtime.update(light=Inference.from_model(light),
                       costs={"joint_tree": tree, "joint_ensemble": ensemble})
        modes = {name: (name, {}) for name in (*BASELINES, *JOINT_MODES)}
        candidate_modes = {"joint_fixed": modes["joint_fixed"]}
        joint_reference = "joint_fixed"
        calibrated = ("joint_mlp", "joint_graph", "joint_adaptive")
    torch.save(saved, args.out/"joint_models.pt")
    if args.neugn_heads:
        with torch.no_grad():
            _, head_rows, head_labels = collect(train, index, data, teacher, args, head_only=True)
        heads = {mode: fit_head(head_rows, width, args.epochs, args.seed, args.device)
                 for mode, width in zip(NEUGN_HEAD_MODES, (0, teacher.cfg.d), strict=True)}
        runtime["neugn_heads"] = heads
        np.savez(args.out/"neugn_action_heads.npz", **{
            f"{mode}_{name}": value for mode, head in heads.items() for name, value in
            zip(("mean", "scale", "w1", "b1", "w2", "b2"), (head.mean, head.scale, *head.weights))
        })
        with (args.out/"neugn_head_labels.jsonl").open("w") as stream:
            stream.writelines(json.dumps(row)+"\n" for row in head_labels)
        candidate_modes["neugn_forward"] = ("neugn_forward", {})
        calibrated += NEUGN_HEAD_MODES
    for mode in calibrated:
        for margin in (0., .5, 1.):
            candidate_modes[f"{mode}_{margin:g}"] = mode, {"margin": margin}
    with torch.no_grad():
        validation_records = benchmark(
            validation, candidate_modes, index, teacher, data, args, runtime,
            args.out/"validation.jsonl", 1,
        )
    valve = paired_summary(validation_records, candidate_modes)
    pairs = compare_paired(validation_records, candidate_modes, joint_reference)
    head_pairs = (compare_paired(validation_records, candidate_modes, "neugn_forward")
                  if args.neugn_heads else {})
    selected = {}
    if args.neugn_heads:
        modes["neugn_forward"] = candidate_modes["neugn_forward"]
    for mode in calibrated:
        comparisons = head_pairs if mode in NEUGN_HEAD_MODES else pairs
        best = min((name for name in comparisons if candidate_modes[name][0] == mode),
                   key=lambda name: (len(comparisons[name]["only_reference"]), valve[name]["seconds"]))
        selected[mode] = candidate_modes[best][1]["margin"]
        modes[mode] = candidate_modes[best]
    (args.out/"calibration.json").write_text(json.dumps(dict(margins=selected, summary=valve,
                                                             paired=pairs, reference=joint_reference, paired_neugn_forward=head_pairs), indent=2)+"\n")
    if not args.seven_mode_pilot:
        for delay in DEFER_DELAYS:
            modes[f"joint_fixed_{delay}"] = "joint_rank", {"delay": delay}
        modes["joint_fixed_rebuild"] = "joint_fixed", {"reuse": False}
        modes["joint_fixed_no_cache"] = "joint_fixed", {"cache_proofs": False}
    if args.neugn_full_depth_pilot:
        modes["neugn_forward_limited"] = "neugn_forward", {"rank_all_depths": False}
        for name in NEUGN_HEAD_MODES:
            modes[name+"_no_cache"] = name, {**modes[name][1], "cache_proofs": False}
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
                  paired_vs_switch=compare_paired(records, modes, "switch"),
                  paired_graph_checks=compare_paired(records, ("joint_graph",), "joint_graph_rank"))
    if args.seven_mode_pilot:
        result["paired_vs_switch"] = compare_pilot(records, modes, "switch")
        result["paired_graph_checks"] = compare_pilot(records, ("joint_graph",), "joint_graph_rank")
        result["paired_neugn_forward"] = compare_pilot(records, ("neugn_forward",), "neugn_domain")
        result["paired_switch_checks"] = compare_pilot(records, ("switch",), "neugn_forward")
    else:
        result.update(
            paired_vs_neugn_original=compare_paired(records, modes, "neugn_original"),
            paired_vs_fixed=compare_paired(records, modes, "joint_fixed"),
            paired_vs_rank=compare_paired(records, ("joint_fixed", "joint_tree", "joint_ensemble",
                                                    "joint_mlp", "joint_adaptive"), "joint_rank"),
            paired_vs_adaptive=compare_paired(records, ("joint_mlp", "joint_fixed_64",
                                                        "joint_fixed_128", "joint_fixed_256"),
                                              "joint_adaptive"),
            paired_graph_rank=compare_paired(records, ("joint_graph_rank",), "joint_rank"),
        )
    if args.neugn_full_depth_pilot:
        result["paired_rank_depth"] = compare_pilot(records, ("neugn_forward",), "neugn_forward_limited")
        result["paired_head_cache"] = {
            name: compare_pilot(records, (name,), name+"_no_cache")[name] for name in NEUGN_HEAD_MODES
        }
    if args.id_forward_control:
        result["paired_vs_id_forward"] = compare_pilot(
            records, ("neugn_forward", "joint_graph_rank"), "id_forward",
        )
    if args.neugn_heads:
        compare_heads = compare_pilot if args.seven_mode_pilot else compare_paired
        result["paired_neugn_heads_vs_forward"] = compare_heads(records, NEUGN_HEAD_MODES, "neugn_forward")
        result["paired_neugn_anchor"] = compare_heads(records, ("neugn_head_anchor",), "neugn_head_domains")
    (args.out/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    if args.branch_queries:
        with torch.no_grad():
            branch_report = branch_benchmark(test, modes, index, teacher, data, args, runtime)
        (args.out/"branch_summary.json").write_text(json.dumps(branch_report, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
