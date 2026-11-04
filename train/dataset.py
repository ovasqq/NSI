"""Self-supervised NeuGN training examples, generated on the fly.

One epoch = one random-walk subgraph from every eligible start vertex of G
(NeuGN: "from every vertex"). For each subgraph:
  size s ~ U{min_size .. min(max_size, |component|)}; walk; induced subgraph;
  Euler sequence from a random start; cyclic shift r ~ U{0..63};
  k ~ U{1..s} hidden vertices chosen uniformly; one of them is the target
  (CLS), the other hidden ones are PAD, the rest carry their true G id.
Label = G id of the target vertex.

Vertices whose connected component has fewer than `min_size` vertices are not
used as start vertices (a walk cannot reach `min_size` distinct vertices).
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from models.encoding import NUM_QVERT, collate, encode_query, encode_state
from queries.euler import euler_sequence
from queries.random_walk import random_walk_vertices
from search.graph import Graph


def component_sizes(g: Graph) -> np.ndarray:
    sizes = np.zeros(g.n, dtype=np.int64)
    for comp in nx.connected_components(g.to_networkx()):
        for v in comp:
            sizes[v] = len(comp)
    return sizes


def make_example(
    g: Graph, start: int, comp_size: int, rng: np.random.Generator,
    min_size: int = 5, max_size: int = 19, walk_retries: int = 10,
) -> tuple[dict, int] | None:
    size = int(rng.integers(min_size, min(max_size, comp_size) + 1))
    for _ in range(walk_retries):
        walk = random_walk_vertices(g, start, size, rng)
        if walk is not None:
            break
    else:
        return None
    q = g.induced_subgraph(walk)
    seq = euler_sequence(q, rng)
    shift = int(rng.integers(NUM_QVERT))
    k = int(rng.integers(1, size + 1))
    hidden = rng.choice(size, k, replace=False)
    target = int(hidden[rng.integers(k)])
    mapping = list(walk)
    for w in hidden.tolist():
        mapping[w] = -1
    return encode_state(encode_query(q, seq), mapping, target, shift, g.n), walk[target]


class TrainingStream(IterableDataset):
    """Endless stream of examples, epoch after epoch, starting at
    `start_epoch`. Start vertices are permuted per epoch (same permutation in
    every worker) and split between DataLoader workers by stride. Batches near
    an epoch boundary may mix two epochs; the training loop only counts
    `steps_per_epoch(batch_size)` batches per epoch."""

    def __init__(self, g: Graph, seed: int, start_epoch: int = 0,
                 min_size: int = 5, max_size: int = 19):
        self.g, self.seed, self.start_epoch = g, seed, start_epoch
        self.min_size, self.max_size = min_size, max_size
        self.comp = component_sizes(g)
        self.starts = np.flatnonzero(self.comp >= min_size)

    def steps_per_epoch(self, batch_size: int) -> int:
        return math.ceil(len(self.starts) / batch_size)

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        epoch = self.start_epoch
        while True:
            perm = np.random.default_rng([self.seed, epoch]).permutation(self.starts)
            rng = np.random.default_rng([self.seed, epoch, wid, 1])
            for v in perm[wid::nw].tolist():
                ex = make_example(self.g, v, int(self.comp[v]), rng,
                                  self.min_size, self.max_size)
                if ex is not None:
                    yield ex
            epoch += 1


def collate_examples(items: list[tuple[dict, int]]):
    return collate([s for s, _ in items], [t for _, t in items])
