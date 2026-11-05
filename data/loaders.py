"""Loaders for the data graphs used in NeuGN. Raw archives are cached in data/raw/."""

from __future__ import annotations

import csv
import io
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from search.graph import Graph

RAW_DIR = Path(__file__).resolve().parent / "raw"

HAMSTER_URL = "http://konect.cc/files/download.tsv.petster-hamster.tar.bz2"
LASTFM_URL = "https://snap.stanford.edu/data/lastfm_asia.zip"


def _download(url: str, name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / name
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".part")
        # curl uses the system certificate store; python.org builds of Python
        # often don't, and urllib then fails with CERTIFICATE_VERIFY_FAILED.
        if shutil.which("curl"):
            subprocess.run(["curl", "-sSfL", "-o", str(tmp), url], check=True)
        else:
            urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def load_hamster() -> Graph:
    """KONECT petster-hamster (Hamsterster full).

    Label = `species` attribute (16 values). Only the 2 421 vertices that have a
    metadata row are kept; this gives exactly 2 421 vertices / 16 621 edges,
    matching the NeuGN paper. Vertex ids are renumbered in increasing KONECT id.
    """
    path = _download(HAMSTER_URL, "petster-hamster.tar.bz2")
    with tarfile.open(path, "r:bz2") as tar:
        ent = tar.extractfile("petster-hamster/ent.petster-hamster").read().decode("utf-8", "replace")
        out = tar.extractfile("petster-hamster/out.petster-hamster").read().decode()

    rows = csv.reader(
        (l for l in io.StringIO(ent) if not l.startswith("%")), delimiter=" ", quotechar='"'
    )
    species = {int(r[0]): r[3] for r in rows}
    ids = sorted(species)
    idx = {k: i for i, k in enumerate(ids)}
    names = sorted(set(species.values()))
    labels = np.array([names.index(species[k]) for k in ids])

    edges = []
    for line in out.splitlines():
        if line.startswith("%") or not line.strip():
            continue
        a, b = map(int, line.split()[:2])
        if a in idx and b in idx:
            edges.append((idx[a], idx[b]))
    return Graph.from_edges(len(ids), edges, labels)


def load_lastfm() -> Graph:
    """SNAP LastFM Asia: 7 624 vertices / 27 806 edges, label = country (18 classes)."""
    path = _download(LASTFM_URL, "lastfm_asia.zip")
    with zipfile.ZipFile(path) as zf:
        # The SNAP archive's folder name is misspelled ("lasftm_asia").
        edges_name = next(n for n in zf.namelist() if n.endswith("lastfm_asia_edges.csv"))
        target_name = next(n for n in zf.namelist() if n.endswith("lastfm_asia_target.csv"))
        edges_rows = list(csv.reader(io.TextIOWrapper(zf.open(edges_name))))[1:]
        target_rows = list(csv.reader(io.TextIOWrapper(zf.open(target_name))))[1:]

    target = dict((int(i), int(t)) for i, t in target_rows)
    n = len(target)
    assert sorted(target) == list(range(n))
    labels = np.array([target[i] for i in range(n)])
    return Graph.from_edges(n, ((int(a), int(b)) for a, b in edges_rows), labels)


def load_rewired() -> Graph:
    """Synthetic transfer target: Hamster with the same vertices, labels and
    degrees, but the giant component's edges randomized (connected double-edge
    swap, seed 0). The 16-label vocabulary is unchanged, so the Hamster-trained
    inductive model applies directly; ~91% of edges differ from Hamster."""
    import networkx as nx

    base = load_hamster()
    g = base.to_networkx()
    comps = sorted(nx.connected_components(g), key=len, reverse=True)
    giant = comps[0]
    sub = g.subgraph(giant).copy()
    nx.connected_double_edge_swap(sub, nswap=5 * sub.number_of_edges(), seed=0)
    new_edges = set(sub.edges())
    for a, b in g.edges():
        if a not in giant or b not in giant:
            new_edges.add((a, b))
    return Graph.from_edges(base.n, new_edges, base.labels)


LOADERS = {"hamster": load_hamster, "lastfm": load_lastfm, "rewired": load_rewired}


def load(name: str) -> Graph:
    return LOADERS[name]()
