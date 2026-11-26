"""NELL from the Planetoid archive, with an explicit label for missing classes."""
from collections import defaultdict
import hashlib
import io
import pickle
import tarfile

import numpy as np

from data.loaders import _download
from search.graph import Graph

URL = "https://www.cs.cmu.edu/~zhiliny/data/nell_data.tar.gz"
SHA256 = "651438dc3efc8634ad6548cf7941bb9971d15d872a30b50076b677a4e7476528"


class _PlanetoidUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("collections", "defaultdict"): defaultdict,
            ("__builtin__", "list"): list,
            ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"unexpected Planetoid type: {module}.{name}")
        return allowed[module, name]


def load_nell_planetoid():
    path = _download(URL, "nell_data.tar.gz")
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("unexpected NELL archive digest")
    with tarfile.open(path, "r:gz") as archive:
        def member(name):
            stream = archive.extractfile(f"nell_data/ind.nell.0.001.{name}")
            with stream:
                return stream.read()

        def decode(name):
            return _PlanetoidUnpickler(io.BytesIO(member(name)), encoding="latin1").load()

        neighbors = decode("graph")
        known, test = decode("ally"), decode("ty")
        test_indices = np.fromstring(member("test.index").decode(), sep=" ", dtype=np.int64)

    # As in PyG, the raw ty rows follow the order in test.index.
    # Preserve missing labels as a separate class instead of merging them into class zero.
    labels = np.full(len(neighbors), -1, dtype=np.int64)
    for indices, rows in ((np.arange(len(known)), known), (test_indices, test)):
        valid = rows.any(axis=1)
        labels[indices[valid]] = rows[valid].argmax(axis=1)
    _, labels = np.unique(labels, return_inverse=True)
    graph = Graph.from_edges(len(neighbors),
                             ((u, v) for u, adjacent in neighbors.items() for v in adjacent),
                             labels)
    if (graph.n, graph.m) != (65_755, 125_775):
        raise ValueError(f"unexpected NELL topology: {graph.n} vertices, {graph.m} edges")
    return graph
