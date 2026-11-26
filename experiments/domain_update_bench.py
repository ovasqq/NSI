"""Benchmark exact forward-domain updates on snapshots from WikiCS DFS."""
import argparse
import csv
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter

import numpy as np
import torch

from data.loaders import load
from queries.storage import load_query_set
from search.enumerate import SearchSpace, enumerate_space
from search.nlf_index import NLFIndex
from search.order import gql_order
from search.pruning import ResidualPruner


def pack(vertices, words):
    ids = np.fromiter(vertices, dtype=np.int64)
    bits = np.zeros(words, dtype=np.uint64)
    np.bitwise_or.at(bits, ids // 64, np.uint64(1) << (ids % 64).astype(np.uint64))
    return bits.view(np.int64)


def snapshots(graph, index, args):
    rng = Random(args.seed)
    saved = []
    seen = 0
    for density in ("dense", "sparse"):
        queries, _ = load_query_set(args.query_dir / f"size{args.size}_{density}.json")
        selected = queries[args.offset:args.offset + args.queries]
        if len(selected) != args.queries:
            raise ValueError(f"Too few {density} queries at offset {args.offset}")
        for query in selected:
            q = query.graph
            candidates = index.filter(q)
            space = SearchSpace(q, graph, candidates, gql_order(q, candidates))
            pruner = ResidualPruner(space, "forward")
            root = dict(enumerate(space.cand_set))

            def advance(depth, mapping, used, domains):
                nonlocal seen
                parent = root if domains is None else domains
                vertex = space.order[depth - 1]
                image = mapping[vertex]
                child = pruner.extend_domains(parent, vertex, image)
                seen += 1
                slot = seen - 1 if seen <= args.samples else rng.randrange(seen)
                if slot < args.samples:
                    row = pruner, parent, vertex, image, child
                    if slot == len(saved):
                        saved.append(row)
                    else:
                        saved[slot] = row
                return child

            enumerate_space(space, branch_domains=advance, max_matches=1, max_steps=args.steps)
    rng.shuffle(saved)
    return saved, seen


def packed_inputs(rows, size, words):
    domains = np.zeros((len(rows), size, words), dtype=np.int64)
    active = np.zeros((len(rows), size), dtype=bool)
    adjacent = np.zeros_like(active)
    images = np.empty(len(rows), dtype=np.int64)
    for i, (pruner, parent, vertex, image, _) in enumerate(rows):
        images[i] = image
        for u, domain in parent.items():
            if u != vertex:
                domains[i, u] = pack(domain, words)
                active[i, u] = True
                adjacent[i, u] = u in pruner.space.q.adj_set[vertex]
    return domains, active, adjacent, images


def bitset_update(domains, active, adjacent, images, adjacency, word_ids):
    # Keep unrestricted rows intact. Then clear the assigned image in every row.
    neighbor_masks = adjacency[images]
    compatible = torch.where(adjacent[..., None], neighbor_masks[:, None, :], -1)
    assigned = torch.where(word_ids == (images // 64)[:, None], 1 << (images % 64)[:, None], 0)
    child = domains & compatible & ~assigned[:, None, :]
    rejected = (active & ~child.ne(0).any(dim=-1)).any(dim=-1)
    return child, rejected


def verify(rows, child, rejected):
    child, rejected = child.cpu().numpy(), rejected.cpu().numpy()
    for i, (_, _, _, _, expected) in enumerate(rows):
        if rejected[i] != (expected is None):
            raise AssertionError(f"Different pruning outcome at snapshot {i}")
        if expected is not None:
            for u, domain in expected.items():
                if not np.array_equal(child[i, u], pack(domain, child.shape[-1])):
                    raise AssertionError(f"Different domain at snapshot {i}, vertex {u}")


def timings(rows, inputs, adjacency_cpu, device, batch_size, repeats):
    word_ids = torch.arange(adjacency_cpu.shape[-1], device=device)
    adjacency = adjacency_cpu.to(device)
    batches = [tuple(t[i:i + batch_size] for t in inputs)
               for i in range(0, len(rows) - batch_size + 1, batch_size)]
    resident = [tuple(t.to(device) for t in batch) for batch in batches]
    tested = rows[:len(batches) * batch_size]

    def update(batch):
        return bitset_update(*batch, adjacency, word_ids)

    for i, batch in enumerate(resident):
        verify(tested[i * batch_size:(i + 1) * batch_size], *update(batch))

    def sets():
        for pruner, parent, vertex, image, _ in tested:
            pruner.extend_domains(parent, vertex, image)

    def packed():
        for batch in resident:
            update(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    def roundtrip():
        for batch in batches:
            child, rejected = update(tuple(t.to(device) for t in batch))
            child.cpu(), rejected.cpu()

    methods = ({"cpu_sets": sets, "cpu_packed_resident": packed} if device.type == "cpu"
               else {"cuda_packed_resident": packed, "cuda_packed_roundtrip": roundtrip})
    samples = {name: [] for name in methods}
    for fn in methods.values():
        fn()
    rng = Random(42)
    for _ in range(repeats):
        for name in rng.sample(list(methods), len(methods)):
            start = perf_counter()
            methods[name]()
            samples[name].append((perf_counter() - start) * 1e6 / len(tested))
    return [dict(batch=batch_size, snapshots=len(tested), method=name,
                 median_us_per_update=median(values)) for name, values in samples.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64, 256])
    for name, default in dict(size=32, offset=120, queries=8, steps=1000, samples=512,
                              repeats=5, seed=19).items():
        parser.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args()
    if min(args.size, args.queries, args.steps, args.samples, args.repeats, *args.batches) < 1 or args.offset < 0:
        parser.error("use positive sizes and counts and a nonnegative offset")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(1)
    graph = load("wikics")
    rows, seen = snapshots(graph, NLFIndex(graph), args)
    words = (graph.n + 63) // 64
    adjacency = torch.from_numpy(np.stack([pack(v, words) for v in graph.adj]))
    inputs = tuple(torch.from_numpy(x) for x in packed_inputs(rows, args.size, words))
    print(f"Sampled {len(rows)} of {seen} DFS updates, including "
          f"{sum(row[-1] is None for row in rows)} empty-domain cuts.", flush=True)
    results = []
    for batch in args.batches:
        if batch > len(rows):
            print(f"Skipping batch {batch}: only {len(rows)} snapshots.", flush=True)
            continue
        results.extend(timings(rows, inputs, adjacency, torch.device("cpu"), batch, args.repeats))
        if args.device == "cuda":
            results.extend(timings(rows, inputs, adjacency, torch.device("cuda"), batch, args.repeats))
    if not results:
        parser.error("no full batches in the sampled states")
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)
    for row in results:
        print(row)
    if args.device == "cuda":
        peak_mib = torch.cuda.max_memory_allocated() / 2**20
        print(f"Peak PyTorch CUDA allocation: {peak_mib:.1f} MiB")


if __name__ == "__main__":
    main()
