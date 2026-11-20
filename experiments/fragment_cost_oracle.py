"""Measure local time headroom over switch and replay a sampled zero-cost oracle."""
import argparse
import json
from collections import Counter
from math import isfinite
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter, perf_counter_ns

import torch
from tqdm import tqdm

from data.loaders import load
from experiments.fragment_pruning import search_space
from experiments.fragment_switch import check_selected_fragment, select
from models.hook import NeuGNReorder
from models.pruning_controller import checkpoint_digest
from queries.storage import load_query_set
from search.enumerate import enumerate_space
from search.pruning import ResidualPruner
from train.train_neugn import load_model

ACTIONS = ("switch", "skip", "edge", "manual")


def reject(pruner, depth, mapping, used, domains, action, args):
    if action == "skip":
        return False
    _, fragment = select(pruner.space, domains, action, args.threshold, args.proposals)
    return bool(fragment) and check_selected_fragment(
        pruner, depth, mapping, used, domains, fragment, args.fragment_budget, incremental=True
    ) is False


def query_run(space, model, args, *, observe=None, choices=None, record_decision=None):
    start = perf_counter()
    hook = NeuGNReorder(model, space.q, args.depth, args.device)
    pruner = ResidualPruner(space, "forward")
    overrides = 0

    def advance(depth, mapping, used, parent):
        nonlocal overrides
        if parent is None:
            domains = pruner.build_domains(depth, mapping, used)
        else:
            vertex = space.order[depth - 1]
            domains = pruner.extend_domains(parent, vertex, mapping[vertex])
        if domains is None or depth < args.depth or len(domains) < 2:
            return domains
        if observe is not None or choices or record_decision is not None:
            prefix = tuple(mapping[u] for u in space.order[:depth])
            if observe is not None:
                observe(prefix, domains)
            action = choices.get(prefix, "switch") if choices else "switch"
            overrides += action != "switch"
        else:
            action = "switch"
        if record_decision is None:
            cut = reject(pruner, depth, mapping, used, domains, action, args)
        else:
            started = perf_counter_ns()
            cut = reject(pruner, depth, mapping, used, domains, action, args)
            record_decision(prefix, cut, perf_counter_ns()-started)
        if cut:
            return None
        return domains

    result = enumerate_space(space, hook, branch_domains=advance,
                             max_matches=1, max_steps=args.budget)
    return dict(seconds=perf_counter()-start, steps=result.steps,
                solved=bool(result.num_matches), complete=result.complete, overrides=overrides)


def continuation(pruner, prefix, domains, args):
    def advance(depth, mapping, used, parent):
        u = pruner.space.order[depth - 1]
        child = pruner.extend_domains(parent, u, mapping[u])
        if child is not None and len(child) >= 2:
            if reject(pruner, depth, mapping, used, child, "switch", args):
                return None
        return child

    return enumerate_space(pruner.space, prefix=prefix, prefix_domains=domains, branch_domains=advance,
                           max_matches=1, max_steps=args.branch_budget,
                           deadline=perf_counter()+args.branch_seconds)


def compare_state(pruner, prefix, domains, args, rng):
    mapping = [-1] * pruner.space.q.n
    for u, v in zip(pruner.space.order, prefix):
        mapping[u] = v
    used = set(prefix)
    depth = len(prefix)

    measures = {action: [] for action in ACTIONS}
    cuts = {}
    for _ in range(args.repeats):
        order = list(ACTIONS)
        rng.shuffle(order)
        for action in order:
            start = perf_counter_ns()
            cut = reject(pruner, depth, mapping, used, domains, action, args)
            measures[action].append((perf_counter_ns() - start)/1000)
            cuts[action] = cut
    # The root check either proves a cut or leaves exactly the same domains.
    # In the latter case all actions share one continuation under switch.
    trials = [continuation(pruner, prefix, domains, args) for _ in range(args.repeats)]
    closed = all(r.complete or r.num_matches for r in trials)
    if any(cuts.values()) and any(r.num_matches for r in trials):
        raise AssertionError("A successful fragment proof pruned a solution")
    # Censored trials are restricted-time measurements, not estimates of full runtime.
    continuation_us = median(r.subtree_seconds for r in trials) * 1e6
    cost = {action: median(measures[action]) + (0. if cuts[action] else continuation_us)
            for action in ACTIONS}
    best = min(ACTIONS, key=cost.__getitem__)
    gain = cost["switch"] - cost[best]
    return dict(prefix=prefix, closed=closed, steps=trials[0].steps,
                continuation_us=continuation_us, check_us={a: median(m) for a,m in measures.items()},
                cuts=cuts, switch_us=cost["switch"], gain_us=gain, best=best,
                chosen=best if gain > args.min_gain_us else "switch")


def evaluate(space, model, args, rng):
    sample = []
    seen = 0

    def capture(prefix, domains):
        nonlocal seen
        seen += 1
        slot = seen - 1 if seen <= args.samples else rng.randrange(seen)
        if slot == len(sample):
            sample.append((prefix, domains))
        elif slot < len(sample):
            sample[slot] = prefix, domains

    traced = query_run(space, model, args, observe=capture)
    pruner = ResidualPruner(space, "forward")
    records = [compare_state(pruner, prefix, domains, args, rng) for prefix, domains in sample]
    choices = {tuple(r["prefix"]): r["chosen"] for r in records if r["chosen"] != "switch"}
    runs = {"switch": [], "sampled_oracle": []}
    for _ in range(args.query_repeats):
        order = list(runs)
        rng.shuffle(order)
        for name in order:
            result = query_run(space, model, args, choices=choices if name == "sampled_oracle" else None)
            runs[name].append(result)
    for r in runs["switch"]:
        if (r["steps"], r["solved"], r["complete"]) != (traced["steps"], traced["solved"], traced["complete"]):
            raise AssertionError("Switch trace differs from the timed baseline")
    for r in runs["sampled_oracle"]:
        if r["complete"] and r["solved"] != traced["solved"] and traced["solved"]:
            raise AssertionError("A replay completed without the baseline solution")
    return dict(seen=seen, states=records, runs=runs)


def gate_headroom(space, model, args, rng):
    misses = {}
    counts = Counter()

    def record(prefix, cut, elapsed_ns):
        counts["eligible"] += 1
        if not cut:
            misses[prefix] = "skip"
            counts["miss_ns"] += elapsed_ns

    traced = query_run(space, model, args, record_decision=record)
    runs = {"switch": [], "perfect_gate": []}
    for _ in range(args.query_repeats):
        order = list(runs)
        rng.shuffle(order)
        for name in order:
            result = query_run(space, model, args, choices=misses if name=="perfect_gate" else None)
            for key in ("steps", "solved", "complete"):
                if result[key] != traced[key]:
                    raise AssertionError("Skipping a noncutting action changed the search")
            runs[name].append(result)
    if any(r["overrides"] != len(misses) for r in runs["perfect_gate"]):
        raise AssertionError("Replay skipped a different number of states")
    return dict(eligible=counts["eligible"], skipped=len(misses),
                skipped_checks_seconds=counts["miss_ns"]/1e9, runs=runs)


def summarize_gate(rows):
    baseline = [median(r["seconds"] for r in x["runs"]["switch"]) for x in rows]
    oracle = [median(r["seconds"] for r in x["runs"]["perfect_gate"]) for x in rows]
    return dict(queries=len(rows), solved=sum(r["runs"]["switch"][0]["solved"] for r in rows),
                eligible=sum(r["eligible"] for r in rows), skipped=sum(r["skipped"] for r in rows),
                skipped_checks_seconds=sum(r["skipped_checks_seconds"] for r in rows),
                switch_seconds=sum(baseline), perfect_gate_seconds=sum(oracle),
                paired_median_delta_ms=median((x-y)*1000 for x,y in zip(baseline,oracle)))


def summarize(rows):
    states = [s for row in rows for s in row["states"]]
    weight_total = gain_total = weighted_gain = 0.
    substitutions = Counter()
    opportunity = Counter()
    for row in rows:
        if not row["states"]:
            continue
        weight = row["seen"] / len(row["states"])
        for s in row["states"]:
            weight_total += weight
            gain_total += s["gain_us"]
            weighted_gain += weight * s["gain_us"]
            substitutions[s["chosen"]] += 1
            if s["chosen"] != "switch":
                category = "new_cut" if s["cuts"][s["chosen"]] and not s["cuts"]["switch"] else "cheaper"
                opportunity[category] += 1
    query_stats = {}
    for name in ("switch", "sampled_oracle"):
        results = [row["runs"][name] for row in rows]
        query_stats[name] = dict(solved=sum(r[0]["solved"] for r in results),
                                 steps=sum(r[0]["steps"] for r in results),
                                 seconds=sum(median(x["seconds"] for x in r) for r in results),
                                 overrides=sum(r[0]["overrides"] for r in results))
    return dict(queries=len(rows), sampled_states=len(states),
                censored=sum(not s["closed"] for s in states), reachable_states=int(weight_total+0.5),
                substitutions=dict(substitutions), opportunities=dict(opportunity),
                mean_sample_gain_us=gain_total/len(states) if states else None,
                weighted_gain_us_per_state=weighted_gain/weight_total if weight_total else None,
                query_runs=query_stats)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "query-dir", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--graph", default="wikics")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sizes", nargs="+", type=int, default=[24, 32])
    for key, default in dict(offset=120, queries=12, seed=19, depth=10, budget=10000,
                             samples=48, threshold=256, proposals=12, fragment_budget=256,
                             branch_budget=2000, repeats=3, query_repeats=3).items():
        parser.add_argument("--"+key.replace("_","-"), type=int, default=default)
    parser.add_argument("--branch-seconds", type=float, default=.05)
    parser.add_argument("--min-gain-us", type=float, default=5.)
    parser.add_argument("--gate-only", action="store_true",
                        help="skip every switch decision that makes no cut, preserving the trace")
    args = parser.parse_args()
    positive = ("queries","depth","budget","samples","threshold","proposals","fragment_budget",
                "branch_budget","repeats","query_repeats","branch_seconds")
    if any(not isfinite(getattr(args,k)) or getattr(args,k) <= 0 for k in positive):
        parser.error("positive finite counts and budgets required")
    if not args.sizes or len(set(args.sizes)) != len(args.sizes) or min(args.sizes) < args.depth+2:
        parser.error("use unique query sizes with at least two vertices after the gate depth")
    if args.offset < 0 or not isfinite(args.min_gain_us) or args.min_gain_us < 0:
        parser.error("invalid offset or minimum gain")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("use an empty output directory")
    torch.set_num_threads(1)
    graph = load(args.graph)
    model = load_model(args.checkpoint, args.device)
    if model.cfg.num_g != graph.n:
        parser.error("graph size differs from checkpoint")
    rng = Random(args.seed)
    rows = []
    experiment, summarize_run = (gate_headroom, summarize_gate) if args.gate_only else (evaluate, summarize)
    with (args.output/"queries.jsonl").open("w") as out:
        groups = [(n,d) for n in args.sizes for d in ("dense","sparse")]
        for n,d in tqdm(groups,desc=args.graph):
            path = args.query_dir/f"size{n}_{d}.json"
            queries, info = load_query_set(path)
            if info.get("graph",args.graph) != args.graph or len(queries) < args.offset+args.queries:
                parser.error(f"invalid query split: {path}")
            # Exclude first-query Torch startup from paired measurements.
            query_run(search_space(queries[max(0, args.offset-1)].graph,graph), model, args)
            for index in range(args.offset,args.offset+args.queries):
                space = search_space(queries[index].graph,graph)
                row = dict(size=n,density=d,index=index,**experiment(space,model,args,rng))
                out.write(json.dumps(row)+"\n")
                out.flush()
                rows.append(row)
    summary = dict(all=summarize_run(rows), groups={
        f"{n}_{d}":summarize_run([r for r in rows if r["size"]==n and r["density"]==d]) for n,d in groups
    }, protocol={**vars(args), "checkpoint_sha256":checkpoint_digest(args.checkpoint)})
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,default=str)+"\n")
    print(json.dumps(summary["all"],indent=2))


if __name__ == "__main__":
    main()
