"""CLI: generate the fixed evaluation query sets.

    python -m queries.generate --graph hamster lastfm --sizes 8 16 20 24 32

Writes data/queries/<graph>/size<N>_<density>.json for dense and sparse.
"""

from __future__ import annotations

import argparse

from data.loaders import load
from queries.random_walk import generate_query_set
from queries.storage import query_set_path, save_query_set


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", nargs="+", default=["hamster", "lastfm"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[8, 16, 20, 24, 32])
    ap.add_argument("--densities", nargs="+", default=["dense", "sparse"])
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-attempts", type=int, default=200_000)
    args = ap.parse_args()

    for name in args.graph:
        g = load(name)
        for size in args.sizes:
            for density in args.densities:
                # Seed depends on the configuration so sets are independent
                # but reproducible.
                seed = args.seed * 1_000_003 + size * 10 + args.densities.index(density)
                qs = generate_query_set(g, size, density, args.count, seed, args.max_attempts)
                path = query_set_path(name, size, density)
                meta = dict(graph=name, size=size, density=density, seed=seed,
                            requested=args.count, max_attempts=args.max_attempts)
                save_query_set(path, qs, meta)
                status = "" if len(qs) == args.count else "  (attempt limit hit)"
                print(f"{name:8s} size={size:2d} {density:6s} -> {len(qs):3d} queries{status}")


if __name__ == "__main__":
    main()
