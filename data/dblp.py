"""SNAP DBLP topology with reproducible proxy labels."""
import gzip
import hashlib

import numpy as np

from data.loaders import _download
from search.graph import Graph

URL = "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz"
DIGEST = "9eb0bd30312ddd04e2624f7c36c0983a2e99b116f0385be5a7fce6d6170f4cb3"


def load_dblp_proxy15():
    path = _download(URL, "com-dblp.ungraph.txt.gz")
    if hashlib.sha256(path.read_bytes()).hexdigest() != DIGEST:
        raise ValueError("Unexpected SNAP archive digest")
    with gzip.open(path, "rt") as stream:
        edges = np.loadtxt(stream, dtype=np.int64)
    ids, indices = np.unique(edges.reshape(-1), return_inverse=True)
    labels = np.random.default_rng(19).integers(15, size=len(ids))
    graph = Graph.from_edges(len(ids), indices.reshape(-1, 2), labels)
    assert (graph.n, graph.m) == (317080, 1049866)
    return graph
