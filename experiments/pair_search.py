"""Fine-tune NeuGN on residual domain pairs, calibrate checks and benchmark DFS."""

import argparse
import csv
import json
import math
from itertools import islice
from models.pair_batch import collate_pairs
from train.pair_loss import pair_loss
from train.stage_timer import StageTimer
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from data.loaders import LOADERS, load
from experiments.deep_pruning import load_splits, write_json
from experiments.fms import summarize
from models.fragment import manual_fragment, proposals
from models.hook import NeuGNReorder
from models.pair_transformer import PairTransformer, encode_pairs
from models.pruning_controller import checkpoint_digest
from search.enumerate import enumerate_space
from search.pruning import ResidualPruner, check_fragment
from experiments.fragment_pruning import search_space
from train.train_neugn import load_model

MODES = ("neugn", "forward", "manual_deep", "pair_rank", "pair_joint")


class Search:

    def __init__(self, space, backbone, model, args, mode, threshold):
        self.space, self.model, self.args = (space, model, args)
        self.mode, self.threshold = (mode, threshold)
        self.hook = NeuGNReorder(backbone, space.q, args.depth, args.device)
        self.pruner = ResidualPruner(space, "forward")
        self.cache = {}
        self.metrics = dict(
            domain_seconds=0.0,
            domains=0,
            domain_cuts=0,
            encoding_seconds=0.0,
            transformer_seconds=0.0,
            transformer_calls=0,
            check_seconds=0.0,
            checks=0,
            fragment_cuts=0,
            unknown=0,
            skip=0,
        )

    def reorder(self, depth, u, mapping, local):
        if depth < self.args.depth:
            return self.hook(depth, u, mapping, local)
        scores = self.cache[depth]
        return sorted(local, key=lambda v: (-scores[v], v)) if scores else local

    def check(self, depth, mapping, used):
        start = perf_counter()
        domains = self.pruner.build_domains(depth, mapping, used)
        self.metrics["domain_seconds"] += perf_counter() - start
        self.metrics["domains"] += 1
        if domains is None:
            self.metrics["domain_cuts"] += 1
            return True
        if depth < self.args.depth:
            return False
        options = proposals(self.space, domains, self.args.proposals)
        chosen = ()
        if self.mode == "manual_deep" and options:
            chosen = manual_fragment(self.space, domains, options)
        if self.mode in ("pair_rank", "pair_joint"):
            start = perf_counter()
            row = encode_pairs(
                self.space,
                self.hook.qenc,
                mapping,
                domains,
                options,
                self.args.candidates,
                self.args.max_pairs,
                relations_on_device=getattr(self.args, "relations_on_device", False),
            )
            self.metrics["encoding_seconds"] += perf_counter() - start
            start = perf_counter()
            with torch.inference_mode():
                ranks, scores = self.model(row)
                local_scores = (
                    ranks[torch.tensor(row["local"], device=ranks.device)]
                    .cpu()
                    .tolist()
                )
                values = scores.cpu().tolist()
            self.cache[depth] = dict(zip(row["local"], local_scores))
            self.metrics["transformer_seconds"] += perf_counter() - start
            self.metrics["transformer_calls"] += 1
            if self.mode == "pair_joint" and values:
                best = max(range(len(values)), key=values.__getitem__)
                if self.threshold == "always" or (
                    self.threshold is not None and values[best] > self.threshold
                ):
                    chosen = options[best]
                else:
                    self.metrics["skip"] += 1
        if chosen:
            start = perf_counter()
            outcome = check_fragment(
                self.space, domains, chosen, self.args.fragment_budget
            )
            self.metrics["check_seconds"] += perf_counter() - start
            self.metrics["checks"] += 1
            self.metrics["unknown"] += int(outcome is None)
            if outcome is False:
                self.metrics["fragment_cuts"] += 1
                return True
        return False


def run(query, g, backbone, model, args, mode, threshold):
    start = perf_counter()
    space = search_space(query.graph, g)
    session = Search(space, backbone, model, args, mode, threshold)
    uses_pairs = mode in ("pair_rank", "pair_joint")
    result = enumerate_space(
        space,
        session.reorder if uses_pairs else session.hook,
        branch_check=session.check if mode != "neugn" else None,
        max_matches=1,
        max_steps=args.budget,
        deadline=start + args.time_limit,
    )
    end = perf_counter()
    return dict(
        fms=result.first_match_step,
        steps=result.steps,
        match_steps=result.match_steps,
        timed_out=result.timed_out,
        complete=result.complete,
        total_seconds=end - start,
        search_seconds=end - start,
        early_model_calls=session.hook.calls,
        early_model_seconds=session.hook.seconds,
        **session.metrics,
    )


def label_state(
    space, hook, mapping, domains, prefix, remaining, args, *, teacher_scores=True
):
    stats = {"censored": 0}
    depth = len(prefix)
    measured = perf_counter()
    options = proposals(space, domains, args.proposals)
    proposed = perf_counter()
    row = encode_pairs(
        space,
        hook.qenc,
        mapping,
        domains,
        options,
        args.candidates,
        args.max_pairs,
        relations_on_device=getattr(args, "relations_on_device", False),
    )
    encoded = perf_counter()
    if not row["local"]:
        return None, 0
    times, refuted = ([], [])
    for option in options:
        start = perf_counter()
        outcome = check_fragment(space, domains, option, args.fragment_budget)
        times.append((perf_counter() - start) * 1000)
        refuted.append(outcome is False)
    checked = perf_counter()
    continuation = 0.0
    if any(refuted):
        start = perf_counter()
        replay = enumerate_space(
            space,
            hook if callable(hook) else None,
            pruning="forward",
            prefix=prefix,
            max_steps=remaining,
            max_matches=1,
            deadline=start + args.horizon_ms / 1000,
        )
        continuation = min((perf_counter() - start) * 1000, args.horizon_ms)
        stats["censored"] += int(not replay.complete)
        if replay.num_matches:
            raise RuntimeError("unsound local contradiction")
    continued = perf_counter()
    gains = torch.tensor(
        np.asarray(refuted) * continuation - np.asarray(times),
        dtype=torch.float32,
    )
    # scores() uses the pretrained model even beyond the early
    # inference depth limit, providing a distillation target.
    row["gains"] = gains
    row["label_stage_seconds"] = dict(
        proposals=proposed - measured,
        encoding=encoded - proposed,
        exact_checks=checked - encoded,
        continuation=continued - checked,
    )
    if teacher_scores:
        row["teacher"] = hook.scores(mapping, space.order[depth], row["local"])
    return row, stats["censored"]


def collect(sets, g, backbone, args):
    rng = np.random.default_rng(args.seed)
    rows = []
    report = []
    for (size, density), split in sets.items():
        stats = dict(size=size, density=density, states=0, profitable=0, censored=0)
        for query_id, query in split["train"]:
            space = search_space(query.graph, g)
            hook = NeuGNReorder(backbone, space.q, args.depth, args.device)
            pruner = ResidualPruner(space, "forward")
            sample, visits = ([], 0)

            def observe(branch, mapping):
                nonlocal visits
                depth = branch.depth + 1
                if depth < args.depth or depth >= space.q.n:
                    return
                visits += 1
                index = (
                    visits - 1 if visits <= args.samples else int(rng.integers(visits))
                )
                if index < args.samples:
                    item = (
                        tuple((mapping[u] for u in space.order[:depth])),
                        branch.steps_remaining,
                    )
                    if index == len(sample):
                        sample.append(item)
                    else:
                        sample[index] = item

            enumerate_space(
                space,
                hook,
                pruning="forward",
                before_check=observe,
                max_matches=1,
                max_steps=args.budget,
                deadline=perf_counter() + args.trace_seconds,
            )
            for prefix, remaining in sample:
                mapping = [-1] * space.q.n
                for u, v in zip(space.order, prefix):
                    mapping[u] = v
                depth = len(prefix)
                domains = pruner.build_domains(depth, mapping, set(prefix))
                if domains is None:
                    continue
                row, censored = label_state(
                    space, hook, mapping, domains, prefix, remaining, args
                )
                stats["censored"] += censored
                if row is None:
                    continue
                row.update(query_id=query_id, size=size, density=density)
                gains = row["gains"]
                rows.append(row)
                stats["states"] += 1
                stats["profitable"] += int(bool(len(gains)) and gains.max().item() > 0)
        print(f"Collect {size}/{density}: {stats}", flush=True)
        report.append(stats)
    if not rows:
        raise ValueError(
            "no residual training states; increase collection budget or sizes"
        )
    write_json(args.output / "collection.json", report)
    torch.save(rows, args.output / "traces.pt")
    return rows


def fit(model, rows, args, teacher_model=None):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    start_epoch = 0
    progress = args.output / "training_progress.pt"
    if args.resume:
        saved = torch.load(progress, map_location=args.device, weights_only=True)
        # Gradient accumulation may change with the physical batch size.
        previous = {
            k: v
            for k, v in saved["protocol"].items()
            if k not in ("accumulate", "profile_training", "epochs")
        }
        current = {
            k: v
            for k, v in args.training_protocol.items()
            if k not in ("accumulate", "profile_training", "epochs")
        }
        if previous != current:
            raise ValueError("resume protocol differs from the saved training run")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = saved["epoch"] + 1
    for epoch in range(start_epoch, args.epochs):
        torch.manual_seed(args.seed + epoch)
        started = perf_counter()
        total = 0.0
        timer = StageTimer(args.device, args.profile_training)
        optimizer.zero_grad()
        epoch_rows = (
            rows.epoch(epoch)
            if hasattr(rows, "epoch")
            else (rows[i] for i in rng.permutation(len(rows)))
        )
        iterator = iter(epoch_rows)
        processed = 0
        padding = total_tokens = 0
        input_wait = 0.0
        step = 0
        batches = math.ceil(len(rows) / args.batch_size)
        while processed < len(rows):
            wait_started = perf_counter()
            batch_rows = list(islice(iterator, args.batch_size))
            input_wait += perf_counter() - wait_started
            if not batch_rows:
                raise RuntimeError(
                    "training stream ended before the epoch was complete"
                )
            with timer.stage("collate"):
                cpu_batch = collate_pairs(batch_rows)
            length = (
                3
                + cpu_batch["content"].shape[1]
                + cpu_batch["vertices"].shape[1]
                + cpu_batch["pair_ids"].shape[1]
            )
            total_tokens += length * len(batch_rows)
            valid = (
                3 * len(batch_rows)
                + cpu_batch["tok_mask"].sum()
                + cpu_batch["vertex_keep"].sum()
                + cpu_batch["pair_keep"].sum()
            )
            padding += length * len(batch_rows) - int(valid)
            ranks, scores, batch = model.forward_padded(
                cpu_batch, timer, return_inputs=True
            )
            with timer.stage("teacher"):
                with torch.no_grad():
                    teacher_ranks = teacher_model(batch).detach()
            with timer.stage("loss"):
                loss = pair_loss(ranks, scores, batch, teacher_ranks)
            # Weight each batch by its size within this optimizer update.
            group_start = (step // args.accumulate) * args.accumulate * args.batch_size
            group_size = min(args.accumulate * args.batch_size, len(rows) - group_start)
            with timer.stage("backward"):
                (loss * len(batch_rows) / group_size).backward()
            if (step + 1) % args.accumulate == 0 or step + 1 == batches:
                with timer.stage("optimizer"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            total += loss.item() * len(batch_rows)
            processed += len(batch_rows)
            step += 1
            if processed % 1024 < args.batch_size or processed == len(rows):
                print(f"Epoch {epoch + 1}: {processed}/{len(rows)} states", flush=True)
        record = dict(
            epoch=epoch + 1,
            states=len(rows),
            loss=total / len(rows),
            seconds=perf_counter() - started,
            input_wait_seconds=input_wait,
            batch_size=args.batch_size,
            accumulate=args.accumulate,
            effective_batch_size=args.batch_size * args.accumulate,
            workers=args.workers,
        )
        record["stage_seconds"] = timer.seconds
        record["synchronized_timing"] = args.profile_training
        record["padding_fraction"] = padding / max(total_tokens, 1)
        if hasattr(rows, "last_epoch_stats"):
            record.update(rows.last_epoch_stats)
        checkpoint_started = perf_counter()
        temporary = progress.with_suffix(".tmp")
        torch.save(
            dict(
                epoch=epoch,
                protocol=args.training_protocol,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
            ),
            temporary,
        )
        temporary.replace(progress)
        record["checkpoint_seconds"] = perf_counter() - checkpoint_started
        record["seconds_including_checkpoint"] = perf_counter() - started
        with (args.output / "training.jsonl").open("a") as log:
            log.write(json.dumps(record) + "\n")
        print(f"Epoch {epoch + 1}/{args.epochs}: {record}", flush=True)
    return model.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", choices=sorted(LOADERS), default="wikics")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--online-states",
        action="store_true",
        help="generate fresh random-walk queries and residual states every epoch",
    )
    p.add_argument("--resume", action="store_true", help="resume training_progress.pt")
    p.add_argument(
        "--states-per-epoch",
        type=int,
        default=0,
        help="0 matches the ordinary NeuGN batch-rounded epoch size",
    )
    p.add_argument("--reference-batch", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--profile-training",
        action="store_true",
        help="synchronize CUDA at stage boundaries for diagnostic timings",
    )

    p.add_argument(
        "--from-scratch",
        action="store_true",
        help="randomly initialize the residual Transformer, embeddings and heads; use the checkpoint as teacher",
    )
    p.add_argument(
        "--phase", choices=("all", "train", "test", "profile"), default="all"
    )
    p.add_argument("--sizes", nargs="+", type=int, default=[10, 16, 24, 32])
    for name, value in (
        ("depth", 10),
        ("candidates", 8),
        ("max-pairs", 192),
        ("proposals", 12),
        ("fragment-budget", 256),
        ("train-queries", 100),
        ("validation-queries", 20),
        ("test-queries", 80),
        ("samples", 16),
        ("epochs", 10),
        ("accumulate", 8),
        ("budget", 100000),
        ("repeats", 3),
        ("seed", 0),
    ):
        p.add_argument("--" + name, type=int, default=value)
    for name, value in (
        ("time-limit", 0.5),
        ("trace-seconds", 1.0),
        ("horizon-ms", 30.0),
        ("lr", 1e-05),
    ):
        p.add_argument("--" + name, type=float, default=value)
    args = p.parse_args()
    if (
        args.seed < 0
        or args.states_per_epoch < 0
        or args.workers < 0
        or min(args.sizes) < 3
        or len(set(args.sizes)) != len(args.sizes)
        or (
            min(
                (
                    v
                    for k, v in vars(args).items()
                    if type(v) in (int, float)
                    and k not in ("seed", "states_per_epoch", "workers")
                )
            )
            <= 0
        )
        or any(
            (
                not math.isfinite(v)
                for v in (args.time_limit, args.trace_seconds, args.horizon_ms, args.lr)
            )
        )
    ):
        p.error("invalid protocol or limits")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        k: v
        for k, v in vars(args).items()
        if k
        not in (
            "output",
            "checkpoint",
            "phase",
            "repeats",
            "resume",
            "batch_size",
            "workers",
            "profile_training",
        )
    }
    protocol["checkpoint_digest"] = checkpoint_digest(args.checkpoint)
    backbone = load_model(args.checkpoint, args.device).eval().requires_grad_(False)
    g = load(args.graph)
    if backbone.cfg.num_g != g.n or backbone.cfg.num_labels <= int(g.labels.max()):
        p.error("checkpoint incompatible with graph")
    sets = load_splits(g, args)
    model = (
        PairTransformer(backbone, from_scratch=args.from_scratch)
        .to(args.device)
        .requires_grad_(True)
    )
    if args.online_states:
        from train.pair_stream import PairStateStream

        protocol["resolved_states_per_epoch"] = PairStateStream.epoch_size(g, args)
    args.relations_on_device = model.cache_graph(g)
    path = args.output / "pair_transformer.pt"
    if args.phase == "test":
        saved = torch.load(path, map_location=args.device, weights_only=True)
        if saved["protocol"] != protocol:
            p.error("test protocol differs from training")
        model.load_state_dict(saved["model"])
        threshold = saved["threshold"]
    else:
        start = perf_counter()
        if args.online_states:
            from train.pair_stream import PairStateStream

            rows = PairStateStream(g, backbone, sets, args)
            protocol["resolved_states_per_epoch"] = len(rows)
        else:
            rows = collect(sets, g, backbone, args)
        args.training_protocol = protocol
        timing_key = (
            "stream_setup_seconds" if args.online_states else "collection_seconds"
        )
        timings = {timing_key: perf_counter() - start}
        start = perf_counter()
        try:
            fit(model, rows, args, backbone)
        finally:
            if hasattr(rows, "close"):
                rows.close()
        timings["training_seconds"] = perf_counter() - start
        if args.phase == "profile":
            write_json(args.output / "timings.json", timings)
            write_json(args.output / "protocol.json", protocol)
            print(
                "Training profile saved. Resume from training_progress.pt to continue.",
                flush=True,
            )
            return
        start = perf_counter()
        trials = []
        for threshold in (None, "always", 0.0, 0.1, 0.5):
            records = [
                run(q, g, backbone, model, args, "pair_joint", threshold)
                for split in sets.values()
                for _, q in split["validation"]
            ]
            if not records:
                raise ValueError("no held-out validation queries remain")
            trial = dict(
                threshold=threshold,
                solved=sum((r["fms"] is not None for r in records)),
                seconds=sum((r["total_seconds"] for r in records)),
                checks=sum((r["checks"] for r in records)),
            )
            print(f"Validation: {trial}", flush=True)
            trials.append(trial)
        selected = min(trials, key=lambda t: (-t["solved"], t["seconds"]))
        threshold = selected["threshold"]
        write_json(
            args.output / "calibration.json", dict(selected=selected, trials=trials)
        )
        timings["calibration_seconds"] = perf_counter() - start
        write_json(args.output / "timings.json", timings)
        write_json(args.output / "protocol.json", protocol)
        torch.save(
            dict(protocol=protocol, threshold=threshold, model=model.state_dict()), path
        )
    model.eval()
    if args.phase == "train":
        return
    for rep in range(1, args.repeats + 1):
        output = args.output / f"test_rep{rep}"
        output.mkdir(exist_ok=True)
        table = []
        rng = np.random.default_rng(args.seed + rep)
        for (size, density), split in sets.items():
            records = {m: [] for m in MODES}
            for m in MODES:
                if split["validation"]:
                    run(
                        split["validation"][0][1],
                        g,
                        backbone,
                        model,
                        args,
                        m,
                        threshold,
                    )
            for query_id, query in split["test"]:
                for m in rng.permutation(MODES):
                    records[m].append(
                        dict(
                            query=query_id,
                            **run(query, g, backbone, model, args, m, threshold),
                        )
                    )
            for mode, items in records.items():
                summary = summarize(items, 1, args.budget)
                for key in ("median_fms", "median_steps_to_k"):
                    if not math.isfinite(summary[key]):
                        summary[key] = None
                write_json(
                    output / f"size{size}_{density}_{mode}.json",
                    dict(
                        config=dict(protocol, mode=mode, threshold=threshold),
                        summary=summary,
                        records=items,
                    ),
                )
                table.append(
                    dict(
                        size=size,
                        density=density,
                        mode=mode,
                        solved=summary["solved"],
                        seconds=summary["total_seconds"],
                        **{
                            k: sum((r[k] for r in items))
                            for k in (
                                "transformer_calls",
                                "transformer_seconds",
                                "encoding_seconds",
                                "checks",
                                "fragment_cuts",
                                "check_seconds",
                                "domain_seconds",
                            )
                        },
                    )
                )
                print(
                    f"Test {rep}/{size}/{density}/{mode}: {summary['solved']}/{len(items)}, {summary['total_seconds']:.3f}s",
                    flush=True,
                )
        with (output / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)


if __name__ == "__main__":
    main()
