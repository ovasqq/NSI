"""Fresh WikiCS random-walk queries and candidate-conditioned residual states."""

import math
import pickle
from time import perf_counter

import numpy as np

from experiments.deep_pruning import fingerprint
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from types import SimpleNamespace

import torch
from models.encoding import encode_query
from queries.euler import euler_sequence
from queries.random_walk import Query, random_walk_vertices
from search.pruning import ResidualPruner
from search.nlf_index import NLFIndex
from search.enumerate import SearchSpace
from search.order import gql_order
from train.dataset import component_sizes
from train.stage_timer import StageTimer


class PairStateStream:
    @staticmethod
    def epoch_size(graph, args):
        if args.states_per_epoch:
            return args.states_per_epoch
        eligible = int((component_sizes(graph) >= 5).sum())
        return math.ceil(eligible / args.reference_batch) * args.reference_batch

    def __init__(self, graph, teacher, sets, args):
        self.graph, self.teacher, self.args = graph, teacher, args
        self.max_sequence_length = teacher.pos_emb.num_embeddings - 3
        self.nlf_index = NLFIndex(graph)
        self.components = component_sizes(graph)
        self.starts = np.flatnonzero(self.components >= args.depth + 2)
        if not len(self.starts):
            raise ValueError(
                "no connected component supports the requested residual depth"
            )
        self.count = self.epoch_size(graph, args)
        self.last_epoch_stats = {}
        self.excluded = {
            fingerprint(query)
            for split in sets.values()
            for name in ("validation", "test")
            for _, query in split[name]
        }
        print(
            f"Online training: {self.count} fresh states/epoch, {args.epochs} epochs, "
            f"{self.count * args.epochs:,} state presentations",
            flush=True,
        )

    def __len__(self):
        return self.count

    def __getstate__(self):
        state = self.__dict__.copy()
        state["teacher"] = None
        state.pop("pool", None)
        return state

    def epoch(self, epoch):
        self.last_epoch_stats = dict(
            sampling_seconds=0.0,
            profitable=0,
            censored=0,
            preparation_worker_seconds={},
        )
        workers = getattr(self.args, "workers", 0)
        if workers:
            if not hasattr(self, "pool"):
                self.pool = ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=get_context("spawn"),
                    initializer=initialize_worker,
                    initargs=(self,),
                )
            pending = deque()
            submitted = 0
            prefetch = max(workers * self.args.batch_size * 2, workers)
            for _ in range(min(prefetch, self.count)):
                pending.append(self.pool.submit(generate_worker, epoch, submitted))
                submitted += 1

            def ready():
                nonlocal submitted
                while pending:
                    row = pickle.loads(pending.popleft().result())
                    if submitted < self.count:
                        pending.append(
                            self.pool.submit(generate_worker, epoch, submitted)
                        )
                        submitted += 1
                    yield row

            generated = ready()
        else:
            generated = (self.make_state(epoch, index) for index in range(self.count))
        for row in generated:
            self.last_epoch_stats["sampling_seconds"] += row["sampling_seconds"]
            self.last_epoch_stats["censored"] += row["censored"]
            self.last_epoch_stats["profitable"] += int(
                len(row["gains"]) > 0 and row["gains"].max().item() > 0
            )
            stages = self.last_epoch_stats["preparation_worker_seconds"]
            measured = dict(row["label_stage_seconds"], **row["sampling_stage_seconds"])
            for key, seconds in measured.items():
                stages[key] = stages.get(key, 0.0) + seconds
            other = row["sampling_seconds"] - sum(measured.values())
            stages["other_generation"] = stages.get("other_generation", 0.0) + other
            yield row

    def close(self):
        if hasattr(self, "pool"):
            self.pool.shutdown(cancel_futures=True)
            del self.pool

    def make_state(self, epoch, index):
        from experiments.pair_search import label_state

        args, graph = self.args, self.graph
        rng = np.random.default_rng([args.seed, epoch, index, 49])
        preferred = int(self.starts[index % len(self.starts)])
        started = perf_counter()
        sampler = StageTimer("cpu", enabled=True)
        for attempt in range(200):
            start = preferred if attempt == 0 else int(rng.choice(self.starts))
            size = int(
                rng.integers(
                    args.depth + 2, min(max(args.sizes), self.components[start]) + 1
                )
            )
            with sampler.stage("walk_query"):
                origin = random_walk_vertices(graph, start, size, rng)
                query = (
                    Query(graph.induced_subgraph(origin), origin)
                    if origin is not None
                    else None
                )
            if query is None:
                continue
            if fingerprint(query) in self.excluded:
                continue
            with sampler.stage("nlf"):
                candidates = self.nlf_index.filter(query.graph)
            with sampler.stage("order_space"):
                order = gql_order(query.graph, candidates)
                space = SearchSpace(query.graph, graph, candidates, order)
            with sampler.stage("mapping_domains"):
                depth = int(rng.integers(args.depth, size - 1))
                mapping = [-1] * size
                for u in space.order[:depth]:
                    mapping[u] = origin[u]
                # Half the attempts test an alternative at the last assignment.
                # Its compatibility with the already fixed prefix is exact.
                last = space.order[depth - 1]
                if rng.random() < 0.5:
                    mapping[last] = -1
                    used = {v for v in mapping if v >= 0}
                    local = space.local_candidates(depth - 1, mapping, used)
                    mapping[last] = int(rng.choice(local))
                used = {v for v in mapping if v >= 0}
                pruner = ResidualPruner(space, "forward")
                domains = pruner.build_domains(depth, mapping, used)
            if domains is None:
                continue
            with sampler.stage("euler_encoding"):
                sequence = euler_sequence(query.graph)
            if len(sequence) > self.max_sequence_length:
                continue
            hook = SimpleNamespace(qenc=encode_query(query.graph, sequence))
            prefix = tuple(mapping[u] for u in space.order[:depth])
            row, censored = label_state(
                space,
                hook,
                mapping,
                domains,
                prefix,
                args.budget,
                args,
                teacher_scores=False,
            )
            if row is None:
                continue
            row.update(size=size, density=query.density, censored=censored)
            # Known valid prefixes provide direct ground-truth supervision.
            if all(mapping[u] == origin[u] for u in space.order[:depth]):
                row["target_image"] = origin[space.order[depth]]
            row["sampling_seconds"] = perf_counter() - started
            row["sampling_stage_seconds"] = sampler.seconds
            return row
        else:
            raise RuntimeError(
                f"failed to sample online state {index} at epoch {epoch + 1}"
            )


_worker = None


def initialize_worker(stream):
    global _worker
    torch.set_num_threads(1)
    _worker = stream


def generate_worker(epoch, index):
    # Transfer bounded CPU rows as bytes, avoiding shared-memory handles for
    # dozens of small tensors per state.
    return pickle.dumps(_worker.make_state(epoch, index), protocol=5)
