"""Compare queue-based AC-3 with batched sparse matrix propagation on WikiCS."""
import argparse
import csv
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter

import numpy as np
import torch

from data.loaders import load
from experiments.domain_update_bench import snapshots
from search.nlf_index import NLFIndex


def sparse_adjacency(graph):
    columns = np.concatenate(graph.adj).astype(np.int64, copy=False)
    offsets = np.r_[0, np.cumsum([len(row) for row in graph.adj], dtype=np.int64)]
    return torch.sparse_csr_tensor(offsets, columns, torch.ones(len(columns)),
                                   size=(graph.n, graph.n))


def pack_states(states, size, graph_size):
    domains = torch.zeros((len(states), size, graph_size), dtype=torch.bool)
    edges = torch.zeros((len(states), size, size))
    active = torch.zeros((len(states), size), dtype=torch.bool)
    for i, (pruner, state) in enumerate(states):
        for u, candidates in state.items():
            domains[i, u, list(candidates)] = True
            active[i, u] = True
            for v in pruner.space.q.adj_set[u] & state.keys():
                edges[i, u, v] = 1
    return domains, edges, active


def propagate(graph, domains, query_edges, active):
    """Apply arc revisions in parallel until each state fails or stabilizes."""
    batch, size, graph_size = domains.shape
    rejected = torch.zeros(batch, dtype=torch.bool, device=domains.device)
    while True:
        columns = domains.reshape(batch * size, graph_size).T.contiguous().float()
        supports = torch.sparse.mm(graph, columns)
        unsupported = (supports.T.reshape(batch, size, graph_size) == 0).float()
        revised = domains & ((query_edges @ unsupported) == 0)
        rejected |= (active & ~revised.any(dim=-1)).any(dim=-1)
        revised = torch.where(rejected[:, None, None], domains, revised)
        if torch.equal(revised, domains):
            return revised, rejected
        domains = revised


def reference(states):
    result = []
    for pruner, domains in states:
        reduced = domains.copy()
        result.append(reduced if pruner.check_arcs(reduced) else None)
    return result


def verify(outputs, expected):
    offset = 0
    for masks, rejected in outputs:
        for local_index, rejected_state in enumerate(rejected.tolist()):
            actual_index = offset + local_index
            domains = expected[actual_index]
            if rejected_state != (domains is None):
                raise AssertionError(f"Different contradiction verdict for state {actual_index}")
            if domains is not None:
                for u, images in domains.items():
                    actual = set(masks[local_index, u].nonzero().flatten().tolist())
                    if actual != images:
                        raise AssertionError(f"Different domain for state {actual_index}, vertex {u}")
        offset += len(rejected)


def benchmark(states, expected, packed, adjacency, batch_size, repeats):
    device = adjacency.device
    host_batches = [tuple(t[i:i + batch_size] for t in packed)
                    for i in range(0, len(states), batch_size)]
    resident_batches = [tuple(t.to(device) for t in batch) for batch in host_batches]

    def resident():
        for batch in resident_batches:
            masks, rejected = propagate(adjacency, *batch)
            yield masks, rejected

    def roundtrip():
        for batch in host_batches:
            masks, rejected = propagate(adjacency, *(t.to(device) for t in batch))
            yield masks.cpu(), rejected.cpu()

    verify(((masks.cpu(), rejected.cpu()) for masks, rejected in resident()), expected)
    methods = {"cpu_queue": lambda: iter(reference(states)),
               f"{device.type}_sparse_resident": resident}
    if device.type == "cuda":
        methods["cuda_sparse_roundtrip"] = roundtrip
    samples = {name: [] for name in methods}
    rng = Random(42)
    for repeat in range(repeats + 1):
        for name in rng.sample(list(methods), len(methods)):
            started = perf_counter()
            for _ in methods[name]():
                pass
            if device.type == "cuda" and name != "cpu_queue":
                torch.cuda.synchronize(device)
            if repeat:
                samples[name].append(perf_counter() - started)
    return [dict(batch=batch_size, states=len(states), method=name,
                 median_seconds=median(values),
                 median_us_per_state=median(values) * 1e6 / len(states))
            for name, values in samples.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 16, 32])
    for name, default in dict(size=32, offset=120, queries=8, steps=1000, samples=512,
                              states=128, repeats=3, seed=19).items():
        parser.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args()
    if min(args.size, args.queries, args.steps, args.samples, args.states, args.repeats,
           *args.batches) < 1 or args.offset < 0:
        parser.error("use positive sizes and counts and a nonnegative offset")
    if args.output.exists():
        parser.error("output already exists")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(1)
    graph = load("wikics")
    saved, visited = snapshots(graph, NLFIndex(graph), args)
    states = [(pruner, child) for pruner, _, _, _, child in saved
              if child is not None and len(child) >= 2][:args.states]
    if not states:
        parser.error("no sampled states with at least two nonempty residual domains")
    expected = reference(states)
    print(f"Graph: WikiCS, query size: {args.size}, visited assignments: {visited}, "
          f"AC-3 states: {len(states)}, contradictions: {sum(d is None for d in expected)}", flush=True)
    if args.device == "cuda":
        print(torch.cuda.get_device_name(), flush=True)
    started = perf_counter()
    packed = pack_states(states, args.size, graph.n)
    print(f"Host tensor packing: {perf_counter()-started:.3f} s (outside timings)", flush=True)
    adjacency = sparse_adjacency(graph).to(args.device)
    results = []
    for batch_size in dict.fromkeys(min(batch, len(states)) for batch in args.batches):
        for result in benchmark(states, expected, packed, adjacency, batch_size, args.repeats):
            results.append(result)
            print(result, flush=True)
    if args.device == "cuda":
        print(f"Peak PyTorch CUDA allocation: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB",
              flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
