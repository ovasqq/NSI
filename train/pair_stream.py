"""Fresh WikiCS random-walk queries and candidate-conditioned residual states."""

import math
from time import perf_counter

import numpy as np

from experiments.deep_pruning import fingerprint
from models.hook import NeuGNReorder
from queries.random_walk import Query, random_walk_vertices
from search.pruning import ResidualPruner
from train.dataset import component_sizes


class PairStateStream:
    @staticmethod
    def epoch_size(graph, args):
        if args.states_per_epoch:
            return args.states_per_epoch
        eligible = int((component_sizes(graph) >= 5).sum())
        return math.ceil(eligible / args.reference_batch) * args.reference_batch

    def __init__(self, graph, teacher, sets, args):
        self.graph, self.teacher, self.args = graph, teacher, args
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

    def epoch(self, epoch):
        from experiments.fragment_pruning import search_space
        from experiments.pair_search import label_state

        args, graph = self.args, self.graph
        rng = np.random.default_rng([args.seed, epoch, 49])
        starts = rng.permutation(self.starts)
        self.last_epoch_stats = dict(sampling_seconds=0.0, profitable=0, censored=0)
        for index in range(self.count):
            started = perf_counter()
            preferred = int(starts[index % len(starts)])
            for attempt in range(200):
                start = preferred if attempt == 0 else int(rng.choice(self.starts))
                size = int(
                    rng.integers(
                        args.depth + 2, min(max(args.sizes), self.components[start]) + 1
                    )
                )
                origin = random_walk_vertices(graph, start, size, rng)
                if origin is None:
                    continue
                query = Query(graph.induced_subgraph(origin), origin)
                if fingerprint(query) in self.excluded:
                    continue
                space = search_space(query.graph, graph)
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
                hook = NeuGNReorder(self.teacher, query.graph, args.depth, args.device)
                prefix = tuple(mapping[u] for u in space.order[:depth])
                row, censored = label_state(
                    space, hook, mapping, domains, prefix, args.budget, args
                )
                if row is None:
                    continue
                row.update(size=size, density=query.density, censored=censored)
                # Known valid prefixes provide direct ground-truth supervision.
                if all(mapping[u] == origin[u] for u in space.order[:depth]):
                    row["target_image"] = origin[space.order[depth]]
                self.last_epoch_stats["sampling_seconds"] += perf_counter() - started
                self.last_epoch_stats["profitable"] += int(
                    len(row["gains"]) > 0 and row["gains"].max().item() > 0
                )
                self.last_epoch_stats["censored"] += censored
                yield row
                break
            else:
                raise RuntimeError(
                    f"failed to sample online state {index} at epoch {epoch + 1}"
                )
