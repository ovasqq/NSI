"""CLI: generate + save the labeled training set and its statistics.

    python -m train.generate_training_data --graph hamster --n 2000 --seed 0

Writes data/queries/<graph>/train_labeled_<seed>.json (queries + labels) and
prints the per-depth statistics (pos_rate, mean cardinality).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data.loaders import load
from train.training_data import generate_labeled_set, save_labeled_set

QUERY_DIR = Path(__file__).resolve().parent.parent / "data" / "queries"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="hamster")
    ap.add_argument("--n", type=int, default=2000, help="number of labeled queries")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-size", type=int, default=5)
    ap.add_argument("--max-size", type=int, default=15)
    ap.add_argument("--label-depth", type=int, default=3)
    ap.add_argument("--budget", type=int, default=5_000_000)
    ap.add_argument("--out", default=None, help="output path (default: data/queries/<graph>/train_labeled_<seed>.json)")
    args = ap.parse_args()

    g = load(args.graph)
    queries, stats = generate_labeled_set(
        g, args.seed, args.n, args.min_size, args.max_size, args.label_depth, args.budget
    )
    out = Path(args.out) if args.out else QUERY_DIR / args.graph / f"train_labeled_{args.seed}.json"
    save_labeled_set(out, args.graph, args.seed, queries, stats,
                     n_requested=args.n, min_size=args.min_size, max_size=args.max_size,
                     label_depth=args.label_depth, budget=args.budget)
    print(f"{args.graph}: {stats['n_queries']}/{args.n} labeled queries "
          f"({stats['n_attempts']} attempts) -> {out}")
    print(f"  states={stats['n_states']:,}  examples={stats['n_examples']:,}")
    print(f"  pos_rate_by_depth       = {stats['pos_rate_by_depth']}")
    print(f"  mean_cardinality_by_depth = {stats['mean_cardinality_by_depth']}")
    print(f"  cardinality_histogram   = {stats['cardinality_histogram']}")


if __name__ == "__main__":
    main()
