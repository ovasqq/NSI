"""Compare a shallow tree policy with a single delayed strong check per active DFS branch."""
import argparse
import json
from collections import Counter
from math import isfinite
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter

import numpy as np
import torch
from tqdm import tqdm

from data.loaders import load
from experiments.check_latency import CheckEncoder, features
from experiments.deep_pruning import fingerprint
from experiments.fms import censored_median
from experiments.fragment_cost_oracle import continuation, query_run, reject
from experiments.fragment_pruning import search_space
from experiments.late_cost_policy import (
    LATE_ACTIONS, LATE_FEATURES, PROOF_ACTIONS, collect_late_costs, fit_action_costs, fit_late_costs, late_features,
    late_conflict_reason, learned_action,
)
from experiments.residual_policy import RESIDUAL_FEATURES, cost_action, fit_cost_ensemble, residual_features
from models.hook import NeuGNReorder
from models.pruning_controller import checkpoint_digest
from queries.random_walk import generate_query_set
from queries.storage import load_query_set, save_query_set
from search.enumerate import enumerate_space
from search.pruning import ResidualPruner
from search.residual_proofs import ProofCache
from train.train_neugn import load_model


TREE_ACTIONS = ("skip", "edge", "manual")


def cheap(depth, domains):
    sizes = [len(d) for d in domains.values()]
    return depth, min(sizes), max(sizes), sum(sizes), sizes.count(1)


def fit_tree(x, costs, depth=3, min_leaf=16, *, actions=TREE_ACTIONS):
    """Minimize estimated elapsed time over the given actions."""
    totals = costs.sum(axis=0)
    best_cost = totals.min()
    best = None
    if depth and len(costs) >= 2*min_leaf:
        for column in range(x.shape[1]):
            splits = np.unique(np.quantile(x[:, column], np.linspace(.1, .9, 9), method="nearest"))
            for pivot in splits:
                mask = x[:, column] <= pivot
                if min(int(mask.sum()), int((~mask).sum())) < min_leaf:
                    continue
                left_cost = costs[mask].sum(axis=0)
                score = left_cost.min() + (totals-left_cost).min()
                if score < best_cost:
                    best_cost = score
                    best = column, pivot.item(), mask
    if best is None:
        return actions[int(totals.argmin())]
    column, pivot, mask = best
    return column, pivot, fit_tree(x[mask], costs[mask], depth-1, min_leaf, actions=actions), fit_tree(
        x[~mask], costs[~mask], depth-1, min_leaf, actions=actions,
    )


def choose(tree, inputs):
    while isinstance(tree, tuple):
        column, pivot, left, right = tree
        tree = left if inputs[column] <= pivot else right
    return tree


def learn_tree(spaces, model, args, rng):
    inputs, residual_inputs, costs = [], [], []
    for space in tqdm(spaces, desc="Train tree", disable=not args.progress):
        seen, sample = 0, []

        def observe(prefix, domains):
            nonlocal seen
            seen += 1
            index = seen-1 if seen <= args.samples else rng.randrange(seen)
            if index == len(sample):
                sample.append((prefix, domains))
            elif index < len(sample):
                sample[index] = prefix, domains

        query_run(space, model, args, observe=observe)
        pruner = ResidualPruner(space, "forward")
        extract = residual_features(space) if args.early_ensemble else None
        for prefix, domains in sample:
            depth = len(prefix)
            mapping = [-1]*space.q.n
            for u, v in zip(space.order, prefix):
                mapping[u] = v
            used = set(prefix)
            timings = {action: [] for action in TREE_ACTIONS}
            cuts = {}
            for _ in range(3):
                order = list(TREE_ACTIONS)
                rng.shuffle(order)
                for action in order:
                    tick = perf_counter()
                    cuts[action] = reject(pruner, depth, mapping, used, domains, action, args)
                    timings[action].append((perf_counter()-tick)*1e6)
            remainder_us = 0.
            if any(cuts.values()):
                result = continuation(pruner, prefix, domains, args)
                if result.num_matches:
                    raise AssertionError("A fragment check discarded a valid embedding")
                remainder_us = min(5000., result.subtree_seconds*1e6) if result.complete else 5000.
            inputs.append(cheap(depth, domains))
            if extract is not None:
                residual_inputs.append(inputs[-1] + extract(depth, domains))
            costs.append([median(timings[action]) + (0. if cuts[action] else remainder_us)
                          for action in TREE_ACTIONS])
    if not inputs:
        raise ValueError("No eligible deep states in training")
    x, y = np.asarray(inputs), np.asarray(costs)
    trees = {depth: fit_tree(x, y, depth=depth) for depth in args.tree_depths}
    return trees, len(inputs), (np.asarray(residual_inputs), y) if residual_inputs else None


def search(space, model, args, tree, mode, *, encoder=None, observe_late=None, delay=None,
           logit=None, early_policy=None, late_observer=None, late_action="both", late_policy=None,
           proofs=False, min_savings_us=0.):
    pruner = ResidualPruner(space, "forward")
    counts = Counter()
    late_seconds = 0.
    delay = args.late_steps if delay is None else delay
    check_depth = 1 if delay == 0 else args.depth
    extract = residual_features(space) if early_policy is not None else None
    cache = ProofCache() if proofs else None
    save_fragment = cache.add_fragment if cache is not None else None

    def advance(depth, mapping, used, parent):
        u = space.order[depth-1]
        domains = (pruner.build_domains(depth, mapping, used) if parent is None
                   else pruner.extend_domains(parent, u, mapping[u]))
        if domains is None or depth < check_depth or len(domains) < 2:
            return domains
        if cache is not None and cache.entries:
            tick = perf_counter()
            conflict = cache.conflict(domains)
            counts["proof_cache_seconds"] += perf_counter() - tick
            counts["proof_cache_lookups"] += 1
            if conflict is not None:
                counts["proof_cache_"+conflict+"_cuts"] += 1
                return None
        if delay == 0:
            return domains
        if mode == "forward":
            action = "skip"
        elif mode == "switch":
            action = "switch"
        else:
            inputs = cheap(depth, domains)
            if early_policy is None:
                action = choose(tree, inputs)
            else:
                action = early_policy(inputs + extract(depth, domains))
        counts["early_"+action] += 1
        if reject(pruner, depth, mapping, used, domains, action, args, on_conflict=save_fragment):
            return None
        return domains

    def late(depth, domains):
        nonlocal late_seconds
        start = perf_counter()
        counts["late_states"] += 1
        if delay == 0:
            counts["checks_in_neugn_range"] += depth < args.depth
        if observe_late is not None:
            observe_late(domains)
            return False
        if encoder is not None:
            counts["model_calls"] += 1
            threshold = args.logit if logit is None else logit
            if encoder(*features(space, domains, True)).item() < threshold:
                elapsed = perf_counter()-start
                counts["late_action_skip"] += 1
                counts["late_decision_seconds"] += elapsed
                late_seconds += elapsed
                return False
        action = late_action
        if late_policy is not None:
            counts["model_calls"] += 1
            action = learned_action(late_policy, late_features(depth, domains, delay),
                                    min_savings_us)
        counts["late_action_"+action] += 1
        if action == "skip":
            elapsed = perf_counter()-start
            counts["late_decision_seconds"] += elapsed
            late_seconds += elapsed
            return False
        counts["strong_checks"] += 1
        check_start = perf_counter()
        reason = late_conflict_reason(pruner, domains, action, proof_cache=cache)
        end = perf_counter()
        counts["late_decision_seconds"] += check_start-start
        counts["late_"+action+"_seconds"] += end-check_start
        if reason is not None:
            counts["late_cuts"] += 1
            counts["late_cut_by_"+reason] += 1
        late_seconds += end-start
        return reason is not None

    start = perf_counter()
    hook = NeuGNReorder(model, space.q, args.depth, args.device)
    callbacks = {} if mode == "neugn" else {"branch_domains": advance}
    if mode in ("tree_late_always", "tree_late_nn") or observe_late is not None:
        callbacks.update(late_check=late, late_steps=delay, late_depth=check_depth)
    if late_observer is not None:
        callbacks.update(late_observer=late_observer, late_steps=delay, late_depth=check_depth)
    result = enumerate_space(space, hook, max_matches=1, max_steps=args.budget, **callbacks)
    return dict(seconds=perf_counter()-start, steps=result.steps, fms=result.first_match_step,
                solved=bool(result.num_matches),
                late_seconds=late_seconds, **counts)


def train_encoder(spaces, graph, model, tree, args, rng):
    samples = []
    positives = 0
    collection = []
    # Prefer the deployment trajectory. Use extra training trajectories only
    # when it fails to expose both classes. Never look at test queries here.
    passes = [("tree", args.late_steps), ("forward", args.late_steps)]
    earlier = max(1, args.late_steps//2)
    if earlier != args.late_steps:
        passes.append(("forward", earlier))
    for policy, delay in passes:
        start = len(samples)
        start_positives = positives
        for space in tqdm(spaces, desc=f"Collect {policy} at +{delay}", disable=not args.progress):
            reservoir = []
            seen = 0

            def observe(domains):
                nonlocal seen
                seen += 1
                slot = seen-1 if seen <= args.late_samples else rng.randrange(seen)
                if slot == len(reservoir):
                    reservoir.append(domains)
                elif slot < len(reservoir):
                    reservoir[slot] = domains

            search(space, model, args, tree, policy, observe_late=observe, delay=delay)
            pruner = ResidualPruner(space, "forward")
            for domains in reservoir:
                cut = not pruner.check_arcs(domains.copy()) or not pruner.covers_matching(domains)
                samples.append((features(space, domains, True), float(cut)))
                positives += cut
        collection.append(dict(policy=policy, delay=delay, states=len(samples)-start,
                               proofs=positives-start_positives))
        print(f"Late labels: {positives} proofs / {len(samples)} states", flush=True)
        if min(positives, len(samples)-positives) >= args.min_class:
            break
    if min(positives, len(samples)-positives) < args.min_class:
        (args.out/"collection.json").write_text(json.dumps(collection, indent=2)+"\n")
        raise ValueError("Insufficient late labels for the fourth mode. See collection.json")
    net = CheckEncoder(int(max(graph.labels))+1, 32)
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(samples)-positives)/positives))
    with torch.enable_grad():
        order = list(range(len(samples)))
        for _ in tqdm(range(args.epochs), desc="Train mini-transformer", disable=not args.progress):
            rng.shuffle(order)
            for start in range(0, len(order), 32):
                batch = [samples[i] for i in order[start:start+32]]
                logits = torch.cat([net(*x) for x,_ in batch])
                opt.zero_grad()
                loss(logits,torch.tensor([y for _,y in batch])).backward()
                opt.step()
    return net.eval(), collection


def summarize_trials(records, modes):
    timing_keys = ("late_seconds", "late_decision_seconds", "proof_cache_seconds",
                   *("late_"+action+"_seconds" for action in PROOF_ACTIONS[1:]))
    summary = {}
    for mode in modes:
        first = [r[mode][0] for r in records]
        fms = [r["fms"] for r in first]
        solved_fms = [f for f in fms if f is not None]
        median_fms = censored_median(fms)
        summary[mode] = dict(
            queries=len(records),
            seconds=sum(median(t["seconds"] for t in r[mode]) for r in records),
            solved=len(solved_fms),
            steps=sum(x["steps"] for x in first),
            median_fms=median_fms if isfinite(median_fms) else None,
            median_fms_solved=median(solved_fms) if solved_fms else None,
            median_ms=1000 * median(median(t["seconds"] for t in r[mode]) for r in records),
            **{key: sum(median(t.get(key, 0.) for t in r[mode]) for r in records)
               for key in timing_keys},
            **{key: sum(x.get(key, 0) for x in first) for key in
               ("early_skip", "early_edge", "early_manual", "early_switch",
                "late_states", "model_calls", "strong_checks", "late_cuts",
                "checks_in_neugn_range",
                "proof_cache_lookups", "proof_cache_hall_cuts", "proof_cache_fragment_cuts",
                "late_cut_by_arc", "late_cut_by_matching", "late_cut_by_lad",
                *("late_action_"+action for action in PROOF_ACTIONS))},
        )
    return summary


def select_tree(validation, trees, graph, model, args, rng):
    policies = {"switch": None, **{f"depth_{depth}": tree for depth, tree in trees.items()}}
    records = []
    for size, density, index, query in tqdm(validation, desc="Validate tree depth", disable=not args.progress):
        space = search_space(query.graph, graph)
        order = list(policies)
        rng.shuffle(order)
        record = dict(size=size, density=density, index=index)
        for name in order:
            mode = "switch" if name == "switch" else "tree"
            record[name] = [search(space, model, args, policies[name], mode)]
        records.append(record)

    summary = summarize_trials(records, policies)
    baseline_solved = summary["switch"]["solved"]

    def rank(depth):
        metrics = summary[f"depth_{depth}"]
        return max(0, baseline_solved - metrics["solved"]), metrics["seconds"], depth

    depth = min(trees, key=rank)
    return depth, dict(selected_depth=depth, matched_switch_coverage=rank(depth)[0] == 0,
                       summary=summary, records=records)


def compare_paired(records, modes, reference="switch"):
    comparisons = {}
    for mode in modes:
        if mode == reference:
            continue
        common, gains, losses, neither = [], [], [], []
        for record in records:
            trial, baseline = record[mode], record[reference]
            key = [record["size"], record["density"], record["index"]]
            trial_solved, baseline_solved = trial[0]["solved"], baseline[0]["solved"]
            if trial_solved and baseline_solved:
                common.append((median(t["seconds"] for t in trial),
                               median(t["seconds"] for t in baseline)))
            elif trial_solved:
                gains.append(key)
            elif baseline_solved:
                losses.append(key)
            else:
                neither.append((median(t["seconds"] for t in trial),
                                median(t["seconds"] for t in baseline)))
        comparisons[mode] = dict(
            common=len(common), only_mode=gains, only_reference=losses, neither=len(neither),
            common_mode_seconds=sum(a for a, _ in common),
            common_reference_seconds=sum(b for _, b in common),
            neither_mode_seconds=sum(a for a, _ in neither),
            neither_reference_seconds=sum(b for _, b in neither),
            median_paired_delta_ms=1000 * median(a-b for a, b in common) if common else None,
        )
    return comparisons


def select_logit(validation, graph, model, tree, encoder, args, rng, cost_variants,
                 full_depth_candidates, proof_reference):
    thresholds = args.logits if args.logits is not None else [args.logit]
    candidates = {f"logit_{threshold:g}": threshold for threshold in thresholds}
    check_variants = {**cost_variants, **full_depth_candidates}
    if full_depth_candidates and proof_reference is not None:
        check_variants["late_proof_fixed"] = proof_reference
    modes = ("switch", "tree", "tree_late_always", *candidates, *check_variants)
    records = []
    for size, density, index, query in tqdm(validation, desc="Calibrate late gate",
                                            disable=not args.progress):
        space = search_space(query.graph, graph)
        order = list(modes)
        rng.shuffle(order)
        record = dict(size=size, density=density, index=index)
        for name in order:
            if name in candidates:
                trial = search(space, model, args, tree, "tree_late_nn",
                               encoder=encoder, logit=candidates[name])
            elif name in check_variants:
                trial = search(space, model, args, tree, "tree_late_always", **check_variants[name])
            else:
                trial = search(space, model, args, tree, name)
            record[name] = [trial]
        records.append(record)
    summary = summarize_trials(records, modes)
    comparisons = compare_paired(records, modes)

    def rank(name):
        deficit = max(0, summary["switch"]["solved"] - summary[name]["solved"])
        return deficit, summary[name]["seconds"], len(comparisons[name]["only_reference"])

    selected = min(candidates, key=rank)
    result = dict(selected=selected, thresholds=candidates, summary=summary,
                  paired_vs_switch=comparisons, records=records)
    if cost_variants:
        paired = compare_paired(records, cost_variants, "tree_late_always")
        # Protect specific queries solved by the strongest fixed baseline.
        best = min(cost_variants, key=lambda name: (
            len(paired[name]["only_reference"]), summary[name]["seconds"],
        ))
        result.update(selected_cost_variant=best, cost_vs_always=paired)
    if full_depth_candidates:
        reference = "late_proof_fixed" if "late_proof_fixed" in check_variants else "tree_late_always"
        paired = compare_paired(records, full_depth_candidates, reference)
        best = min(full_depth_candidates, key=lambda name: (
            len(paired[name]["only_reference"]), summary[name]["seconds"],
        ))
        result.update(selected_full_depth_variant=best, full_depth_vs_reference=paired)
    return candidates[selected], result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint","query-dir","out"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--graph",default="wikics")
    p.add_argument("--device",default="cpu")
    p.add_argument("--sizes",type=int,nargs="+",default=[10,16,24,32])
    for key,value in dict(depth=10, threshold=256, proposals=12, fragment_budget=256,
                          train_offset=0,train_queries=100,test_offset=120,queries=40,samples=24,budget=10000,
                          late_samples=24,min_class=8,
                          branch_budget=2000,late_steps=128,repeats=3,epochs=6,seed=19).items():
        p.add_argument("--"+key.replace("_","-"),type=int,default=value)
    p.add_argument("--tree-depths", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--early-ensemble", action="store_true",
                   help="Also compare a tree and a cost ensemble with residual query features.")
    p.add_argument("--late-cost-study", action="store_true",
                   help="Compare exact check types, fixed delays and learned cost policies.")
    p.add_argument("--proof-study", action="store_true",
                   help="Add fixed and learned late checks with local LAD and reusable proofs.")
    p.add_argument("--full-depth-proof", action="store_true",
                   help="Train and test an exact-check controller at every DFS depth.")
    p.add_argument("--full-depth-samples", type=int, default=4,
                   help="Training states sampled per depth per query in the full-depth mode.")
    p.add_argument("--late-delays", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--step-value-us", type=float, default=10.)
    p.add_argument("--test-seed", type=int,
                   help="Generate disjoint test queries instead of taking the source suffix.")
    p.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--logit",type=float,default=0.)
    p.add_argument("--logits", type=float, nargs="+",
                   help="Thresholds to compare on validation. Otherwise use --logit.")
    p.add_argument("--branch-seconds",type=float,default=.05)
    args=p.parse_args()
    if min(args.train_queries, args.queries, args.samples, args.late_samples, args.min_class,
           args.budget, args.late_steps, args.depth,
           args.repeats, args.epochs, args.branch_budget, args.branch_seconds) <= 0:
        p.error("positive counts and budgets required")
    if min(args.sizes) < 2 or len(args.sizes) != len(set(args.sizes)):
        p.error("use distinct query sizes of at least two vertices")
    if max(args.sizes) < args.depth + 2:
        p.error("include a query size with at least two unassigned vertices after the check depth")
    if min(args.tree_depths) < 1 or len(args.tree_depths) != len(set(args.tree_depths)):
        p.error("use distinct positive tree depths")
    if min(args.late_delays) < 1 or len(args.late_delays) != len(set(args.late_delays)):
        p.error("use distinct positive late delays")
    if args.full_depth_samples < 1:
        p.error("full-depth samples must be positive")
    if not isfinite(args.step_value_us) or args.step_value_us <= 0:
        p.error("step value must be positive and finite")
    thresholds = args.logits if args.logits is not None else [args.logit]
    if not all(map(isfinite, thresholds)) or len({f"{v:g}" for v in thresholds}) != len(thresholds):
        p.error("use distinct finite logit thresholds")
    train_end = args.train_offset + args.train_queries
    if args.test_offset <= train_end or args.train_offset < 0:
        p.error("leave whole validation queries between training and test offsets")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng=Random(args.seed)
    graph=load(args.graph)
    model=load_model(args.checkpoint,args.device)
    if model.cfg.num_g != graph.n:
        p.error("checkpoint graph size mismatch")
    groups = {}
    query_hashes = {}
    for size in args.sizes:
        for density in ("dense", "sparse"):
            path = args.query_dir/f"size{size}_{density}.json"
            queries, metadata = load_query_set(path)
            required = args.test_offset + (args.queries if args.test_seed is None else 0)
            if metadata.get("graph", args.graph) != args.graph or len(queries) < required:
                p.error(f"incompatible query set: {path}")
            groups[size,density] = queries
            query_hashes[path.name] = checkpoint_digest(path)
    train, validation, test = [], [], []
    shallow_train, shallow_validation = [], []
    fresh_groups = {}
    if args.test_seed is None:
        for (size, density), queries in groups.items():
            for i in range(args.test_offset, args.test_offset+args.queries):
                test.append((size, density, i, queries[i]))
    else:
        used = {fingerprint(q) for queries in groups.values() for q in queries}
        for size, density in groups:
            print(f"Generate fresh test queries: {size} {density}", flush=True)
            seed = args.test_seed * 1_000_003 + size * 10 + (density == "sparse")
            generated = generate_query_set(graph, size, density, 2*args.queries, seed)
            selected = []
            for query in generated:
                key = fingerprint(query)
                if key not in used:
                    used.add(key)
                    selected.append(query)
                    if len(selected) == args.queries:
                        break
            if len(selected) != args.queries:
                p.error(f"not enough distinct fresh queries for {size} {density}")
            fresh_groups[size, density] = seed, selected
            test.extend((size, density, i, query) for i, query in enumerate(selected))
    seen = {fingerprint(q) for _,_,_,q in test}
    for (size, density), queries in groups.items():
        if size >= args.depth + 2 or args.full_depth_proof:
            target = validation if size >= args.depth + 2 else shallow_validation
            for i in range(train_end, args.test_offset):
                query = queries[i]
                key = fingerprint(query)
                if key not in seen:
                    seen.add(key)
                    target.append((size, density, i, query))
    for (size, density), queries in groups.items():
        if size < args.depth + 2 and not args.full_depth_proof:
            continue
        target = train if size >= args.depth + 2 else shallow_train
        for i in range(args.train_offset, args.train_offset+args.train_queries):
            query = queries[i]
            key = fingerprint(query)
            if key not in seen:
                seen.add(key)
                target.append(search_space(query.graph, graph))
    if not train or not validation:
        p.error("no training or validation queries remain after deduplication")
    args.out.mkdir(parents=True,exist_ok=True)
    if any(args.out.iterdir()):
        p.error("use an empty output directory")
    test_hashes = {}
    for (size, density), (seed, queries) in fresh_groups.items():
        path = args.out/"test_queries"/f"size{size}_{density}.json"
        save_query_set(path, queries, dict(graph=args.graph, size=size, density=density, seed=seed))
        test_hashes[path.name] = checkpoint_digest(path)
    with torch.no_grad():
        trees, tree_samples, residual_training = learn_tree(train, model, args, rng)
        tree_depth, validation_result = select_tree(validation, trees, graph, model, args, rng)
        tree = trees[tree_depth]
        (args.out/"validation.json").write_text(json.dumps(validation_result, indent=2)+"\n")
        print(f"Selected tree depth {tree_depth}; validation matches switch coverage: "
              f"{validation_result['matched_switch_coverage']}", flush=True)
        encoder, collection = train_encoder(train, graph, model, tree, args, rng)
        torch.save(encoder.state_dict(), args.out/"mini_transformer.pt")
        cost_variants = {}
        fixed_variants = {}
        if args.late_cost_study or args.proof_study:
            delays = sorted({args.late_steps, *args.late_delays}) if args.late_cost_study else [args.late_steps]
            actions = PROOF_ACTIONS if args.proof_study else LATE_ACTIONS
            rows = collect_late_costs(train, search, model, tree, args, rng, delays, actions)
            if not rows:
                p.error("no delayed training states; reduce the delay or increase the search budget")
            policies = fit_late_costs(rows, args.step_value_us, proof_study=args.proof_study)
            if args.late_cost_study:
                for name in ("late_binary_time", "late_multi_time", "late_multi_steps"):
                    cost_variants[name] = dict(late_policy=policies[name])
                for delay in delays:
                    if delay != args.late_steps:
                        fixed_variants[f"late_both_{delay}"] = dict(delay=delay)
                        cost_variants[f"late_multi_steps_{delay}"] = dict(
                            delay=delay, late_policy=policies["late_multi_steps"],
                        )
                for action in ("arc", "matching"):
                    fixed_variants[f"late_{action}_{args.late_steps}"] = dict(late_action=action)
            if args.proof_study:
                fixed_variants["late_proof_fixed"] = dict(late_action="cascade", proofs=True)
                cost_variants["late_proof_learned"] = dict(
                    late_policy=policies["late_proof_learned"], proofs=True,
                )
            (args.out/"late_cost_labels.jsonl").write_text(
                "".join(json.dumps(row)+"\n" for row in rows)
            )
            (args.out/"late_cost_policies.json").write_text(json.dumps(dict(
                features=LATE_FEATURES, check_delays=delays, step_value_us=args.step_value_us,
                policies=policies, samples=len(rows), censored=sum(r["censored"] for r in rows),
            ), indent=2)+"\n")
        full_depth_candidates = {}
        if args.full_depth_proof:
            rows = collect_late_costs(train + shallow_train, search, model, tree, args,
                                      Random(args.seed), (0,), PROOF_ACTIONS)
            if not rows:
                p.error("no eligible full-depth training states")
            policy = fit_action_costs(rows, PROOF_ACTIONS[1:], args.step_value_us)
            full_depth_candidates = {
                f"full_depth_margin_{margin}": dict(delay=0, late_policy=policy, proofs=True,
                                                   min_savings_us=margin)
                for margin in (0, 10, 50)
            }
            (args.out/"full_depth_labels.jsonl").write_text(
                "".join(json.dumps(row)+"\n" for row in rows)
            )
            (args.out/"full_depth_policy.json").write_text(json.dumps(dict(
                policy=policy, features=LATE_FEATURES, samples=len(rows),
                censored=sum(row["censored"] for row in rows),
                sample_depths=dict(Counter(row["depth"] for row in rows)),
                query_count=len(train)+len(shallow_train),
                step_value_us=args.step_value_us,
            ), indent=2)+"\n")
        logit, calibration = select_logit(
            validation + shallow_validation, graph, model, tree, encoder, args, rng, cost_variants,
            full_depth_candidates, fixed_variants.get("late_proof_fixed"),
        )
        (args.out/"late_calibration.json").write_text(json.dumps(calibration, indent=2)+"\n")
        print(f"Selected late gate logit {logit:g}", flush=True)
        modes = ("neugn", "switch", "tree", "tree_late_always", "tree_late_nn")
        late_variants = {**fixed_variants, **cost_variants}
        if full_depth_candidates:
            selected_full = calibration["selected_full_depth_variant"]
            late_variants["full_depth_proof_learned"] = full_depth_candidates[selected_full]
            print(f"Selected full-depth policy {selected_full}", flush=True)
        modes = (*modes, *late_variants)
        early_policies = {}
        if residual_training is not None:
            residual_tree = fit_tree(*residual_training, depth=tree_depth)
            residual_ensemble = fit_cost_ensemble(*residual_training)
            early_policies = {
                "residual_tree": lambda inputs: choose(residual_tree, inputs),
                "residual_ensemble": lambda inputs: cost_action(residual_ensemble, inputs),
            }
            modes = (*modes, *early_policies, *(name + "_late_nn" for name in early_policies))
            (args.out/"early_policies.json").write_text(json.dumps(dict(
                residual_features=RESIDUAL_FEATURES, tree_depth=tree_depth,
                residual_tree=residual_tree, residual_ensemble=residual_ensemble,
            ), indent=2)+"\n")
        records = []
        with (args.out/"test_records.jsonl").open("w", buffering=1) as test_log:
            for size, density, index, query in tqdm(test, desc=f"Test {len(modes)} modes",
                                                   disable=not args.progress):
                space = search_space(query.graph, graph)
                trials = {mode: [] for mode in modes}
                for _ in range(args.repeats):
                    order = list(modes)
                    rng.shuffle(order)
                    for mode in order:
                        if mode in late_variants:
                            trials[mode].append(search(space, model, args, tree, "tree_late_always",
                                                      **late_variants[mode]))
                            continue
                        late = mode.endswith("_late_nn")
                        policy_name = mode.removesuffix("_late_nn") if late else mode
                        trials[mode].append(search(
                            space, model, args, tree, "tree_late_nn" if late else mode,
                            encoder=encoder if late else None, logit=logit,
                            early_policy=early_policies.get(policy_name),
                        ))
                record = dict(size=size, density=density, index=index, **trials)
                records.append(record)
                test_log.write(json.dumps(record)+"\n")
    by_size = {
        n: summarize_trials([r for r in records if r["size"] == n], modes)
        for n in args.sizes
    }
    by_group = {
        f"{n}_{density}": summarize_trials(
            [r for r in records if r["size"] == n and r["density"] == density], modes
        )
        for n, density in groups
    }
    result=dict(params={k:str(v) for k,v in vars(args).items()},tree=tree,tree_samples=tree_samples,
                tree_depth=tree_depth, validation={k: v for k, v in validation_result.items() if k != "records"},
                late_logit=logit, paired_vs_switch=compare_paired(records, modes),
                paired_late_nn_vs_always=compare_paired(records, ("tree_late_nn",),
                                                       reference="tree_late_always")["tree_late_nn"],
                collection=collection, summary=summarize_trials(records, modes),
                by_size=by_size, by_group=by_group, records=records, train_queries=len(train),
                checkpoint_sha256=checkpoint_digest(args.checkpoint), query_sha256=query_hashes)
    if test_hashes:
        result["test_query_sha256"] = test_hashes
    if late_variants:
        result["paired_late_vs_always"] = compare_paired(records, late_variants, "tree_late_always")
    if cost_variants:
        result["selected_cost_variant"] = calibration["selected_cost_variant"]
    if full_depth_candidates:
        result["selected_full_depth_variant"] = calibration["selected_full_depth_variant"]
        references = ("tree_late_always", "late_proof_fixed", "late_proof_learned")
        result["paired_full_depth"] = {
            ref: compare_paired(records, ("full_depth_proof_learned",), ref)["full_depth_proof_learned"]
            for ref in references if ref in modes
        }
    if args.proof_study:
        result["paired_learned_vs_proof_fixed"] = compare_paired(
            records, ("late_proof_learned",), "late_proof_fixed",
        )["late_proof_learned"]
    if early_policies:
        result["paired_early_ablation"] = {
            mode: compare_paired(records, (mode,), reference)[mode]
            for mode, reference in (
                ("residual_tree", "tree"), ("residual_ensemble", "residual_tree"),
                ("residual_tree_late_nn", "tree_late_nn"),
                ("residual_ensemble_late_nn", "residual_tree_late_nn"),
            )
        }
    (args.out/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("records", "by_group")},indent=2))


if __name__=="__main__":
    main()
