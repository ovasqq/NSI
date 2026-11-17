"""Collect branch timings with frozen NeuGN and train a CPU pruning readout."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from data.loaders import load
from models.hook import NeuGNReorder
from models.pruning_controller import FEATURE_COUNT, PruningHead, branch_features, checkpoint_digest, eligible
from queries.storage import load_query_set
from search.enumerate import BranchState, SearchSpace, enumerate_matches
from search.filter import nlf_filter
from search.order import gql_order
from search.pruning import ResidualPruner
from train.train_neugn import load_model


def collect(args) -> None:
    g = load(args.graph)
    model = load_model(args.backbone, args.device)
    rows = []
    for path in args.query_set:
        queries, _ = load_query_set(path)
        for query in queries[args.offset:args.offset + args.limit]:
            q = query.graph
            cand = nlf_filter(q, g)
            if not all(map(len, cand)):
                continue
            order = gql_order(q, cand)
            hook = NeuGNReorder(model, q, device=args.device, max_depth=args.max_depth,
                               capture_state=True)
            pruner = ResidualPruner(SearchSpace(q, g, cand, order), "domains")
            group = sha256(repr((q.labels.tolist(), list(q.edges()))).encode()).hexdigest()
            samples = []
            sampled = 0
            bookkeeping = 0.0

            def observe(branch: BranchState, mapping: list[int], used: set[int]):
                nonlocal sampled, bookkeeping
                if sampled >= args.samples or not eligible(hook, branch):
                    return None
                if branch.step > 4 and branch.step % args.stride:
                    return None
                begin = perf_counter()
                features = branch_features(hook, g, q, branch, use_anchor=True).numpy()
                snapshot = (branch.depth + 1, mapping.copy(), used.copy())
                sampled += 1
                start = perf_counter()
                bookkeeping += start - begin
                before = bookkeeping

                def finish(resolved: bool) -> None:
                    nonlocal bookkeeping
                    stop = perf_counter()
                    elapsed = max(0.0, stop - start - (bookkeeping - before))
                    samples.append((features, snapshot, elapsed * 1000, resolved))
                    bookkeeping += perf_counter() - stop

                return finish

            # The unpruned traversal measures the factual continuation of every sampled prefix.
            enumerate_matches(q, g, cand, order, hook, max_matches=1, max_steps=args.budget,
                              deadline=perf_counter() + args.trace_seconds, branch_observer=observe)
            # Probes run only after traversal, outside all measured DFS intervals.
            for features, snapshot, dfs_ms, resolved in samples:
                started = perf_counter()
                rejected = pruner.rejects(*snapshot)
                check_ms = (perf_counter() - started) * 1000
                rows.append((features, group, dfs_ms, resolved, check_ms, rejected))
        print(f"{path}: {len(rows)} cumulative sampled branches", flush=True)
    if not rows:
        raise ValueError("no branch samples collected")
    features, groups, dfs, resolved, checks, rejected = zip(*rows)
    with args.output.open("wb") as stream:
        np.savez_compressed(stream, features=np.stack(features), groups=groups,
                            dfs_ms=dfs, resolved=resolved, check_ms=checks, rejected=rejected,
                            backbone_digest=checkpoint_digest(args.backbone), graph=args.graph,
                            max_depth=args.max_depth, budget=args.budget)
    print(f"Saved {len(rows)} samples to {args.output}")


def fit(args) -> None:
    with np.load(args.data, allow_pickle=False) as data:
        x = data["features"]
        groups, dfs, checks = data["groups"], data["dfs_ms"], data["check_ms"]
        rejected, resolved = data["rejected"], data["resolved"]
        metadata = {k: data[k].item() for k in ("backbone_digest", "graph", "max_depth", "budget")}
    use_anchor = args.features == "anchor"
    if not use_anchor:
        x = x[:, :FEATURE_COUNT]
    # A censored branch supplies an exact clipped target after it reaches the horizon.
    valid = ~rejected | resolved | (dfs >= args.horizon_ms)
    gain = rejected * np.minimum(dfs, args.horizon_ms) - checks
    unique = np.unique(groups[valid])
    if len(unique) < 2:
        raise ValueError("need samples from at least two distinct queries")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique)
    validation_groups = unique[:max(1, len(unique) // 5)]
    validation = np.isin(groups, validation_groups) & valid
    training = ~np.isin(groups, validation_groups) & valid
    if not np.any(gain[training] > 0):
        raise ValueError("no useful checks in the training split; collect more queries")

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    features = torch.from_numpy(x)
    target = gain / args.horizon_ms
    if args.target == "signed-log":
        target = np.sign(gain) * np.log1p(np.abs(gain))
    y = torch.tensor(target, dtype=torch.float32)
    head = PruningHead(x.shape[1], args.hidden)
    head.center.copy_(features[training].mean(0))
    head.scale.copy_(features[training].std(0, unbiased=False).clamp_min(1e-5))
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    train_ids = np.flatnonzero(training)
    for _ in range(args.epochs):
        for batch in np.array_split(rng.permutation(train_ids), (len(train_ids) + 255) // 256):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(head(features[batch]), y[batch])
            loss.backward()
            optimizer.step()

    with torch.inference_mode():
        predictions = head(features[validation]).numpy()
    candidates = [0.0, *np.quantile(predictions, np.linspace(0, 1, 21))]
    threshold = max(candidates, key=lambda t: (gain[validation][predictions > t].sum(), t))
    selected = predictions > threshold
    torch.save(dict(**metadata, state_dict=head.state_dict(), hidden=args.hidden,
                    input_size=x.shape[1], use_anchor=use_anchor, threshold=float(threshold),
                    horizon_ms=args.horizon_ms, target=args.target), args.output)
    print(f"Train/validation states: {training.sum()}/{validation.sum()}; "
          f"censored and excluded: {(~valid).sum()}; validation checks: {selected.sum()}; "
          f"estimated validation gain: {gain[validation][selected].sum():.2f} ms")
    print(f"Saved controller to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_args = sub.add_parser("collect")
    collect_args.add_argument("--graph", default="hamster")
    collect_args.add_argument("--query-set", type=Path, action="append", required=True)
    collect_args.add_argument("--backbone", type=Path, required=True)
    collect_args.add_argument("--device", default="cpu")
    collect_args.add_argument("--offset", type=int, default=0)
    collect_args.add_argument("--limit", type=int, default=40, help="queries per input set")
    collect_args.add_argument("--budget", type=int, default=100000)
    collect_args.add_argument("--trace-seconds", type=float, default=0.5)
    collect_args.add_argument("--max-depth", type=int, default=10)
    collect_args.add_argument("--samples", type=int, default=128, help="max samples per query")
    collect_args.add_argument("--stride", type=int, default=16)
    collect_args.add_argument("--output", type=Path, required=True)
    fit_args = sub.add_parser("fit")
    fit_args.add_argument("--data", type=Path, required=True)
    fit_args.add_argument("--output", type=Path, required=True)
    fit_args.add_argument("--features", choices=("cheap", "anchor"), default="anchor")
    fit_args.add_argument("--target", choices=("milliseconds", "signed-log"), default="milliseconds")
    fit_args.add_argument("--horizon-ms", type=float, default=50.0)
    fit_args.add_argument("--hidden", type=int, default=0, help="0 selects a linear head")
    fit_args.add_argument("--epochs", type=int, default=30)
    fit_args.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "collect":
        if min(args.limit, args.budget, args.max_depth, args.samples, args.stride) < 1 or args.offset < 0:
            parser.error("collection limits must be positive and offset nonnegative")
        if not np.isfinite(args.trace_seconds) or args.trace_seconds <= 0:
            parser.error("--trace-seconds must be finite and positive")
        collect(args)
    else:
        if (not np.isfinite(args.horizon_ms) or args.horizon_ms <= 0
                or args.epochs < 1 or args.hidden < 0):
            parser.error("horizon and epochs must be positive and hidden size nonnegative")
        fit(args)


if __name__ == "__main__":
    main()
