"""Benchmark six transformer-free CPU methods on labelled SNAP DBLP."""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from data.dblp import load_dblp_proxy15
from experiments.joint_search import benchmark, collect, read_queries
from experiments.triple_search import Policy, nell_suite_solver, throughput_summary
from models.joint_dfs import Inference, train_joint, vertex_features
from search.nlf_index import NLFIndex
from search.pruning import check_fragment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-setup", action="store_true",
                        help="Also benchmark one second including per-query setup.")
    for name, default in dict(queries=100, train_queries=24, validation_queries=8,
                              test_offset=32, seed=19, epochs=6).items():
        parser.add_argument("--"+name.replace("_", "-"), type=int, default=default)
    args = parser.parse_args()
    args.graph, args.sizes, args.test_query_dir = "dblp_proxy15", (32,), None
    args.device, args.depth, args.seconds, args.progress = "cpu", 16, 1., True
    args.fragment_budget, args.samples, args.negatives, args.step_value_us = 4096, 2, 4, 10.
    if min(args.queries, args.train_queries, args.validation_queries, args.epochs) < 1:
        parser.error("positive query counts and epochs required")
    if args.test_offset < args.train_queries + args.validation_queries:
        parser.error("test queries must follow the training and validation split")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        parser.error("output directory must be empty")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    graph = load_dblp_proxy15()
    started = perf_counter()
    index, data = NLFIndex(graph), vertex_features(graph)
    index_seconds = perf_counter() - started
    training, _, test = read_queries(args)
    args.budget = 2000
    with torch.no_grad():
        rank_rows, check_rows, _ = collect(training, index, data, None, args)
    model = train_joint(rank_rows, check_rows, graph=True, epochs=args.epochs, seed=args.seed)
    torch.save(model.state_dict(), args.out/"joint_graph.pt")
    args.budget = None

    def check(space, domains, subsets):
        return [check_fragment(space, domains, subset, args.fragment_budget) for subset in subsets]

    policy = Policy(None, check)
    modes = {
        "gql": ("id_forward", {}),
        "dynamic": ("dynamic", {}),
        "dynamic_arc": ("dynamic_arc", {}),
        "structural_mlp": ("joint_graph_rank", {}),
        "structural_mlp_triple": ("joint_graph_rank", {"triple_policy": policy}),
        "dynamic_fallback": ("dynamic_fallback", {"triple_policy": policy}),
    }
    modes = {name: (mode, {**options, "count_matches": True}) for name, (mode, options) in modes.items()}
    runtime = {"graph_model": Inference.from_model(model)}
    result = dict(index_seconds=index_seconds, params=vars(args), phases={})
    phases = (False, True) if args.include_setup else (False,)
    for include_setup in phases:
        stage = "end_to_end" if include_setup else "enumeration"
        with torch.no_grad():
            rows = benchmark(test, modes, index, None, data, args, runtime,
                             args.out/f"{stage}.jsonl", 1, solver=nell_suite_solver,
                             charge_setup=include_setup)
        summary = throughput_summary(rows, modes, args.seconds, include_setup=include_setup)
        for name, stats in summary.items():
            first = [row[name][0]["first_match_seconds"] for row in rows
                     if row[name][0]["first_match_seconds"] is not None]
            stats["median_first_match_seconds_with_setup"] = median(first) if first else None
        result["phases"][stage] = summary
        (args.out/"results.json").write_text(json.dumps(result, default=str, indent=2)+"\n")
        print(stage, summary, flush=True)


if __name__ == "__main__":
    main()
