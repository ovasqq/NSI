"""Data graphs must match the sizes reported in the NeuGN paper (Table 1)."""

import pytest

from data.loaders import load


@pytest.mark.parametrize(
    "name, n, m, labels",
    [("hamster", 2421, 16621, 16), ("lastfm", 7624, 27806, 18)],
)
def test_sizes(name, n, m, labels):
    g = load(name)
    assert (g.n, g.m, g.num_labels) == (n, m, labels)
    for v in range(g.n):
        assert v not in g.adj_set[v]
        assert all(v in g.adj_set[w] for w in g.adj[v].tolist())
