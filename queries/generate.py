"""CLI: generate the fixed evaluation query sets.

    python -m queries.generate --graph hamster lastfm --sizes 8 16 20 24 32

Writes data/queries/<graph>/size<N>_<density>.json for dense and sparse.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data.dblp import load_dblp_proxy15
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
    ap.add_argument("--output-dir", type=Path, help="Write query sets into this directory.")
    args = ap.parse_args()
    if args.output_dir is not None and len(args.graph) != 1:
        ap.error("--output-dir expects exactly one graph")

    for name in args.graph:
        g = load_dblp_proxy15() if name == "dblp_proxy15" else load(name)
        for size in args.sizes:
            for density in args.densities:
                # Seed depends on the configuration so sets are independent
                # but reproducible.
                seed = args.seed * 1_000_003 + size * 10 + args.densities.index(density)
                qs = generate_query_set(g, size, density, args.count, seed, args.max_attempts)
                path = query_set_path(name, size, density)
                if args.output_dir is not None:
                    path = args.output_dir / path.name
                meta = dict(graph=name, size=size, density=density, seed=seed,
                            requested=args.count, max_attempts=args.max_attempts)
                save_query_set(path, qs, meta)
                status = "" if len(qs) == args.count else "  (attempt limit hit)"
                print(f"{name:8s} size={size:2d} {density:6s} -> {len(qs):3d} queries{status}")
                if args.output_dir is not None and len(qs) != args.count:
                    ap.error(f"incomplete query set: {path}")


if __name__ == "__main__":
    main()
