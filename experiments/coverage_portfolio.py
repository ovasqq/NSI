"""Calibrate a coverage-first WikiCS solver and evaluate it under a shared budget."""
import argparse
from copy import copy
import json
from math import isfinite
from pathlib import Path
from time import perf_counter

import torch

from data.loaders import load
from experiments.joint_search import benchmark, compare_pilot, collect, paired_summary, read_queries, run
from experiments.triple_search import Policy
from models.joint_dfs import Inference, train_joint, vertex_features
from search.nlf_index import NLFIndex
from search.pruning import ResidualPruner, check_fragment
from train.train_neugn import load_model


def dynamic_search(space, args, propagate, *, count_matches=False, first_match_seconds=None):
    started = perf_counter()
    deadline = started + args.seconds
    pruner = ResidualPruner(space, "forward")
    steps = 0
    matches = 0
    first_match_step = None
    first_match_seconds = None
    first_deadline = None if first_match_seconds is None else started + first_match_seconds
    mapping = [-1] * space.q.n
    outcome = "unsat"

    def visit(domains):
        nonlocal steps, outcome, matches, first_match_step, first_match_seconds
        vertex = min(domains, key=lambda u: (
            len(domains[u]), -len(space.q.adj_set[u] & domains.keys()), u))
        neighbors = space.q.adj_set[vertex] & (domains.keys() - {vertex})
        candidates = list(domains[vertex])
        if len(candidates) > 1 and neighbors:
            def priority(image):
                supports = [len(domains[u] & space.g.adj_set[image]) for u in neighbors]
                return (-min(supports), -sum(supports), image)

            candidates.sort(key=priority)
        else:
            candidates.sort()
        for image in candidates:
            active_deadline = first_deadline if first_deadline is not None and not matches else deadline
            if perf_counter() >= active_deadline:
                outcome = "timeout"
                return True
            if args.budget is not None and steps >= args.budget:
                outcome = "step_limit"
                return True
            steps += 1
            mapping[vertex] = image
            if len(domains) == 1:
                matches += 1
                if matches == 1:
                    first_match_step = steps
                    first_match_seconds = perf_counter() - started
                if not count_matches:
                    outcome = "sat"
                    return True
                continue
            child = pruner.extend_domains(domains, vertex, image)
            if child is not None and (not propagate or pruner.check_arcs(child)) and visit(child):
                return True
            mapping[vertex] = -1
        return False

    domains = dict(enumerate(space.cand_set))
    if all(domains.values()) and (not propagate or pruner.check_arcs(domains)):
        visit(domains)
    seconds = perf_counter() - started
    if outcome == "unsat" and seconds >= args.seconds:
        outcome = "timeout"
    return dict(seconds=seconds, solved=matches > 0, outcome="sat" if matches else outcome,
                timed_out=outcome == "timeout", steps=steps, num_matches=matches,
                fms=first_match_step, first_match_seconds=first_match_seconds,
                embedding=mapping if matches and not count_matches else None)


def solve(space, teacher, data, args, mode, *, graph_model, policy, schedule=(), first_share=0.):
    def one(name, limits):
        if name in ("dynamic", "dynamic_arc"):
            return dynamic_search(space, limits, name == "dynamic_arc")
        ranker = "neugn_forward" if name.startswith("neugn") else "joint_graph_rank"
        return run(space, teacher, data, limits, ranker, graph_model=graph_model,
                   rank_all_depths=True, triple_policy=policy if name.endswith("_triple") else None)

    if not schedule:
        return one(mode, args)

    started = perf_counter()
    first_limits = copy(args)
    first_limits.budget = max(1, int(args.budget * first_share))
    first_limits.seconds = args.seconds * first_share
    first = one(schedule[0], first_limits)
    phases = [dict(mode=schedule[0], **first)]
    if first["solved"] or first["outcome"] == "unsat":
        return dict(first, phases=phases)

    elapsed = perf_counter() - started
    remaining_steps = args.budget - first["steps"]
    remaining_seconds = args.seconds - elapsed
    if remaining_steps <= 0 or remaining_seconds <= 0:
        outcome = "timeout" if remaining_seconds <= 0 else "step_limit"
        return dict(first, outcome=outcome, timed_out=outcome == "timeout",
                    seconds=elapsed, phases=phases)

    second_limits = copy(args)
    second_limits.budget = remaining_steps
    second_limits.seconds = remaining_seconds
    second = one(schedule[1], second_limits)
    phases.append(dict(mode=schedule[1], **second))
    totals = {
        key: first.get(key, 0) + second.get(key, 0)
        for key in first.keys() | second.keys()
        if key in ("steps", "decisions", "skips", "direct_cuts", "rank_calls")
        or key.endswith("_seconds") or key.startswith(("check_", "proved_", "unknown_"))
    }
    return {
        **second, **totals, "seconds": perf_counter() - started, "phases": phases,
        "fms": None if second["fms"] is None else first["steps"] + second["fms"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reuse-model", type=Path, help="Reuse weights from the same training split.")
    parser.add_argument("--dynamic", action="store_true",
                        help="Include dynamic variable ordering and arc-consistency candidates.")
    parser.add_argument("--dynamic-graph-fallback", action="store_true",
                        help="Evaluate the fixed dynamic-to-structural schedule with one fifth of the budget in the first phase.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seconds", type=float, default=.5)
    for name, value in dict(budget=5000, fragment_budget=4096, samples=2, negatives=4,
                            train_queries=24, validation_queries=12, test_offset=126,
                            queries=30, epochs=6, seed=19).items():
        parser.add_argument("--"+name.replace("_", "-"), type=int, default=value)
    args = parser.parse_args()
    args.graph, args.sizes, args.test_query_dir = "wikics", (16, 24, 32), None
    args.depth, args.step_value_us, args.progress = 10, 10., True
    if not isfinite(args.seconds) or args.seconds <= 0 or args.budget < 2:
        parser.error("use a positive time budget and at least two DFS steps")
    if args.test_offset < args.train_queries + args.validation_queries:
        parser.error("training and validation must precede the held-out queries")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        parser.error("output directory must be empty")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    graph = load(args.graph)
    index, data = NLFIndex(graph), vertex_features(graph)
    teacher = load_model(args.checkpoint, args.device)
    train, validation, test = read_queries(args)
    if args.reuse_model is None:
        with torch.no_grad():
            rank_rows, check_rows, _ = collect(train, index, data, teacher, args)
        model = train_joint(rank_rows, check_rows, graph=True, epochs=args.epochs,
                            seed=args.seed, device=args.device)
    else:
        from models.joint_dfs import JointModel

        weights = torch.load(args.reuse_model, map_location="cpu", weights_only=True)
        model = JointModel(weights["state.weight"].shape[1], weights["rank.weight"].shape[0],
                           weights["check.weight"].shape[0], width=weights["state.weight"].shape[0])
        model.load_state_dict(weights)
    torch.save(model.state_dict(), args.out/"joint_graph.pt")

    def checker(space, domains, subsets):
        return [check_fragment(space, domains, subset, args.fragment_budget) for subset in subsets]

    runtime = dict(graph_model=Inference.from_model(model), policy=Policy(None, checker))
    bare = ("neugn", "graph")
    singles = {name: (name, {}) for name in (*bare, "neugn_triple", "graph_triple")}
    if args.dynamic:
        singles.update({name: (name, {}) for name in ("dynamic", "dynamic_arc")})
    candidates = dict(singles)
    pairs = (("graph", "neugn"), ("neugn", "graph"),
                          ("graph_triple", "neugn"), ("graph", "neugn_triple"),
                          ("neugn", "graph_triple"), ("neugn_triple", "graph"))
    if args.dynamic:
        pairs = (("dynamic", "neugn"), ("neugn", "dynamic"),
                 ("dynamic_arc", "neugn"), ("neugn", "dynamic_arc"),
                 ("dynamic", "dynamic_arc"), ("dynamic_arc", "dynamic"))
    for first, second in pairs:
        for share in (.2, .4, .6):
            name = f"{first}_then_{second}_{share:g}"
            candidates[name] = (name, dict(schedule=(first, second), first_share=share))
    if args.dynamic_graph_fallback:
        name = "dynamic_then_graph_triple_0.2"
        candidates = {name: (name, dict(schedule=("dynamic", "graph_triple"), first_share=.2))}
    with torch.no_grad():
        val_rows = benchmark(validation, candidates, index, teacher, data, args, runtime,
                             args.out/"validation.jsonl", 1, solver=solve)
    scores = paired_summary(val_rows, candidates)
    eligible = candidates if args.dynamic else [name for name in candidates if name not in singles]
    selected = min(eligible, key=lambda key: (-scores[key]["solved"], scores[key]["seconds"], key))
    print(f"Selected {selected}: {scores[selected]['solved']}/{len(validation)} validation queries", flush=True)

    if args.dynamic_graph_fallback:
        singles = {name: (name, {}) for name in ("dynamic", "graph_triple")}
    modes = {**singles, "selected": candidates[selected]}
    with torch.no_grad():
        rows = benchmark(test, modes, index, teacher, data, args, runtime,
                         args.out/"test_records.jsonl", 1, solver=solve)
    result = dict(params=vars(args), selected=selected, validation=scores, summary=paired_summary(rows, modes),
                  paired={base: compare_pilot(rows, modes, base) for base in singles},
                  reference_union=sum(any(row[name][0]["solved"] for name in singles) for row in rows),
                  unresolved=[[row["size"], row["density"], row["index"], row["selected"][0]["outcome"]]
                              for row in rows if not row["selected"][0]["solved"]])
    (args.out/"results.json").write_text(json.dumps(result, default=str, indent=2)+"\n")
    print({name: values["solved"] for name, values in result["summary"].items()}, flush=True)


if __name__ == "__main__":
    main()
