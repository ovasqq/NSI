"""Collect marginal domains timings and train NeuGN's candidate check output."""

from __future__ import annotations

import argparse
import math
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from data.loaders import load
from models.encoding import collate, encode_state
from models.hook import NeuGNReorder
from models.neugn_gate import CandidateCheckHead
from models.pruning_controller import checkpoint_digest, eligible
from queries.storage import load_query_set
from search.enumerate import SearchSpace, enumerate_space
from search.filter import nlf_filter
from search.order import gql_order
from search.pruning import ResidualPruner
from train.train_neugn import load_model


def collect(args) -> None:
    g = load(args.graph)
    model = load_model(args.backbone, args.device)
    horizon = args.horizon_ms / 1000
    query_encodings = {}
    rows = []
    censored = 0
    for path in args.query_set:
        queries, _ = load_query_set(path)
        selected = queries[args.offset:args.offset + args.limit]
        if not selected:
            raise ValueError(f"no queries in the selected range of {path}")
        for query in selected:
            q = query.graph
            cand = nlf_filter(q, g)
            if not all(map(len, cand)):
                continue
            order = gql_order(q, cand)
            space = SearchSpace(q, g, cand, order)
            hook = NeuGNReorder(model, q, args.max_depth, args.device)
            pruner = ResidualPruner(space, "domains")
            group = sha256(repr((q.labels.tolist(), list(q.edges()))).encode()).hexdigest()
            samples = []
            visits = 0

            def sample(branch, mapping):
                nonlocal visits
                if len(samples) >= args.samples or not eligible(hook, branch):
                    return
                visits += 1
                if visits <= 4 or visits % args.stride == 0:
                    samples.append((branch, mapping.copy()))

            enumerate_space(
                space, hook, max_matches=1, max_steps=args.budget, pruning="domains",
                deadline=perf_counter() + args.trace_seconds, before_check=sample,
            )
            for branch, mapping in samples:
                depth = branch.depth + 1
                prefix = tuple(mapping[u] for u in order[:depth])
                used = set(prefix)
                start = perf_counter()
                rejected = pruner.rejects(depth, mapping, used)
                check_ms = (perf_counter() - start) * 1000
                continuation_ms = 0.0
                if rejected:
                    # Skipping a successful cut starts the same DFS one level deeper.
                    # A passing check preserves domains only inside the check, so both
                    # continuations coincide and their costs cancel in the label.
                    start = perf_counter()
                    replay = enumerate_space(
                        space, hook, max_matches=1, max_steps=branch.steps_remaining,
                        pruning="domains", deadline=start + horizon, prefix=prefix,
                    )
                    elapsed = (perf_counter() - start) * 1000
                    if not (replay.complete or replay.num_matches) and elapsed < args.horizon_ms:
                        censored += 1
                        continue
                    continuation_ms = min(elapsed, args.horizon_ms)
                if group not in query_encodings:
                    query_encodings[group] = {k: torch.from_numpy(v) for k, v in hook.qenc.items()}
                rows.append((group, mapping, branch.vertex, branch.candidate, continuation_ms - check_ms))
        print(f"{path}: {len(rows)} cumulative samples, {censored} censored", flush=True)
    if not rows:
        raise ValueError("no branch samples collected")
    torch.save(dict(
        queries=query_encodings, rows=rows, backbone_digest=checkpoint_digest(args.backbone),
        graph=args.graph, max_depth=args.max_depth, horizon_ms=args.horizon_ms,
    ), args.output)
    print(f"Saved {len(rows)} samples from {len(query_encodings)} queries to {args.output}")


def calibrate(scores: np.ndarray, gain: np.ndarray) -> float:
    """Maximize validation gain across all distinct decisions induced by a threshold."""
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)) + 1, len(scores)]
    removed_gain = np.r_[0.0, np.cumsum(gain[order])[ends - 1]]
    best = np.argmin(removed_gain)
    return float(sorted_scores[ends[best - 1] - 1] if best else sorted_scores[0] - 1.0)


def fit(args) -> None:
    data = torch.load(args.data, map_location="cpu", weights_only=True)
    if data["backbone_digest"] != checkpoint_digest(args.backbone):
        raise ValueError("traces were collected with a different NeuGN checkpoint")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = load_model(args.backbone, args.device).requires_grad_(False)
    teacher = None
    params = []
    if args.finetune_last:
        teacher = deepcopy(model)
        layer = model.encoder.layers[-1].requires_grad_(True)
        params.append(dict(params=layer.parameters(), lr=args.encoder_lr))
    head = CandidateCheckHead(model.cfg.d, args.hidden).to(args.device)
    params.append(dict(params=head.parameters(), lr=args.lr))
    opt = torch.optim.AdamW(params)

    qenc = {group: {k: v.numpy() for k, v in enc.items()} for group, enc in data["queries"].items()}
    groups = sorted(qenc)
    if len(groups) < 2:
        raise ValueError("need samples from at least two distinct queries")
    rng.shuffle(groups)
    val_groups = set(groups[:max(1, len(groups) // 5)])
    states, candidates, gains = [], [], []
    train_ids, val_ids = [], []
    for i, (group, mapping, vertex, candidate, gain) in enumerate(data["rows"]):
        states.append(encode_state(qenc[group], mapping, vertex, 0, model.cfg.num_g))
        candidates.append(candidate)
        gains.append(gain)
        (val_ids if group in val_groups else train_ids).append(i)
    gains = np.array(gains)
    targets = (np.sign(gains) * np.log1p(np.abs(gains)) if args.target == "signed-log"
               else gains / data["horizon_ms"])

    def batches(ids):
        for start in range(0, len(ids), args.batch):
            indices = ids[start:start + args.batch]
            batch = {k: v.to(args.device) for k, v in collate([states[i] for i in indices]).items()}
            ids_for_head = torch.tensor([[candidates[i]] for i in indices], device=args.device)
            yield indices, batch, ids_for_head

    for epoch in range(1, args.epochs + 1):
        total = torch.zeros((), device=args.device)
        for indices, batch, ids in batches(rng.permutation(train_ids)):
            opt.zero_grad()
            ranks, checks = model.rank_and_check(batch, ids, head)
            y = torch.tensor(targets[indices], device=args.device, dtype=checks.dtype)
            loss = F.mse_loss(checks[:, 0], y)
            if teacher is not None:
                with torch.no_grad():
                    old_ranks = teacher(batch)
                loss = loss + F.kl_div(ranks.log_softmax(-1), old_ranks.softmax(-1),
                                       reduction="batchmean")
            loss.backward()
            opt.step()
            total += loss.detach() * len(indices)
        print(f"epoch {epoch}/{args.epochs}: loss {total.item() / len(train_ids):.6f}", flush=True)

    with torch.inference_mode():
        scores = np.concatenate([
            model.rank_and_check(batch, ids, head)[1][:, 0].cpu().numpy()
            for _, batch, ids in batches(val_ids)
        ])
    val_gain = gains[val_ids]
    threshold = calibrate(scores, val_gain)
    keep = scores > threshold
    print(f"Train/validation states: {len(train_ids)}/{len(val_ids)}; "
          f"profitable skips: {(gains[train_ids] < 0).sum()}/{(val_gain < 0).sum()}")
    print(f"Validation skips: {(~keep).sum()}/{len(keep)}; "
          f"estimated change vs checking all: {-val_gain[~keep].sum():.2f} ms")
    torch.save(dict(
        backbone_digest=data["backbone_digest"], graph=data["graph"], max_depth=data["max_depth"],
        horizon_ms=data["horizon_ms"], target=args.target, threshold=threshold, hidden=args.hidden,
        head=head.state_dict(), encoder_layer=model.encoder.layers[-1].state_dict() if teacher else None,
    ), args.output)
    print(f"Saved gate to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_args = commands.add_parser("collect")
    fit_args = commands.add_parser("fit")
    for command in (collect_args, fit_args):
        command.add_argument("--backbone", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cpu")
    collect_args.add_argument("--query-set", type=Path, action="append", required=True)
    collect_args.add_argument("--graph", default="hamster")
    collect_args.add_argument("--offset", type=int, default=40)
    collect_args.add_argument("--limit", type=int, default=60)
    collect_args.add_argument("--samples", type=int, default=96)
    collect_args.add_argument("--stride", type=int, default=8)
    collect_args.add_argument("--budget", type=int, default=100_000)
    collect_args.add_argument("--max-depth", type=int, default=10)
    collect_args.add_argument("--trace-seconds", type=float, default=0.5)
    collect_args.add_argument("--horizon-ms", type=float, default=50.0)
    fit_args.add_argument("--data", type=Path, required=True)
    fit_args.add_argument("--hidden", type=int, default=64)
    fit_args.add_argument("--epochs", type=int, default=20)
    fit_args.add_argument("--batch", type=int, default=32)
    fit_args.add_argument("--lr", type=float, default=1e-3)
    fit_args.add_argument("--encoder-lr", type=float, default=1e-5)
    fit_args.add_argument("--finetune-last", action="store_true")
    fit_args.add_argument("--target", choices=("milliseconds", "signed-log"), default="signed-log")
    fit_args.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "collect":
        if (min(args.limit, args.samples, args.stride, args.budget, args.max_depth) < 1 or args.offset < 0
                or any(not math.isfinite(v) or v <= 0 for v in (args.trace_seconds, args.horizon_ms))):
            parser.error("collection limits must be positive and offset nonnegative")
        collect(args)
    else:
        if (min(args.hidden, args.epochs, args.batch) < 1 or args.seed < 0
                or any(not math.isfinite(v) or v <= 0 for v in (args.lr, args.encoder_lr))):
            parser.error("training options must be positive and seed nonnegative")
        fit(args)


if __name__ == "__main__":
    main()
