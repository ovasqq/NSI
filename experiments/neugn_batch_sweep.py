"""Measure NeuGN training throughput on a frozen checkpoint, optionally resume."""

import argparse
import gc
import json
import os
from pathlib import Path
import random
import shutil
from statistics import median
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from data.dblp import load_dblp_proxy15
from data.loaders import load
from models.neugn import NeuGN, NeuGNConfig
from train.dataset import TrainingStream, collate_examples
from train.train_neugn import CKPT_DIR, main as train_main, pick_device, train_step


def probe(snapshot, stream, batch_size, workers, device, warmup, examples, repeats):
    # Reload Adam and the model for every candidate so probes start from one state.
    ckpt = torch.load(snapshot, map_location="cpu", weights_only=False)
    torch.manual_seed(ckpt["args"]["seed"])
    model = NeuGN(NeuGNConfig(**ckpt["config"])).to(device).train()
    model.load_state_dict(ckpt["model"])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(ckpt["opt"])
    del ckpt

    loader = DataLoader(stream, batch_size=batch_size, collate_fn=collate_examples,
                        num_workers=workers)
    batches = iter(loader)
    for _ in range(warmup//batch_size):
        train_step(model, optimizer, next(batches), device)

    seconds = []
    for _ in range(repeats):
        started = perf_counter()
        for _ in range(examples//batch_size):
            train_step(model, optimizer, next(batches), device)
        seconds.append(perf_counter()-started)
    return dict(batch=batch_size, status="ok", seconds=seconds,
                median_examples_per_second=examples/median(seconds),
                peak_cuda_mib=torch.cuda.max_memory_allocated(device)/2**20
                if device.type == "cuda" else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batches", nargs="+", type=int, default=[64, 128, 256])
    p.add_argument("--warmup-examples", type=int, default=1024)
    p.add_argument("--examples", type=int, default=8192)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--continue-to", type=int, help="Total epoch target, including completed epochs.")
    p.add_argument("--run", help="Name of the new continuation run, required with --continue-to.")
    args = p.parse_args()
    if min(*args.batches, args.examples, args.repeats) <= 0 or min(args.workers, args.warmup_examples) < 0:
        p.error("invalid batch, example or repeat count")
    if len(set(args.batches)) != len(args.batches):
        p.error("batch sizes must be unique")
    if any(args.examples % b or args.warmup_examples % b for b in args.batches):
        p.error("example and warmup counts must be divisible by every batch size")
    if (args.continue_to is None) != (args.run is None):
        p.error("--continue-to and --run must be supplied together")
    if args.run is not None and (Path(args.run).name != args.run or args.run in (".", "..")):
        p.error("--run must be a single new directory name")
    device = pick_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA is unavailable")
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot = args.out / "source.pt"
    # The trainer publishes checkpoints with an atomic rename. A hard link
    # pins this published version even if another epoch is saved later.
    try:
        os.link(args.checkpoint.resolve(), snapshot)
    except OSError:
        with args.checkpoint.open("rb") as source, snapshot.open("xb") as target:
            shutil.copyfileobj(source, target)

    ckpt = torch.load(snapshot, map_location="cpu", weights_only=False)
    if not {"config", "model", "opt", "sched", "epoch", "args"} <= ckpt.keys():
        p.error("a training checkpoint with Adam and scheduler state is required")
    graph_name = ckpt["args"]["graph"]
    completed = ckpt["epoch"] + 1
    if args.continue_to is not None and args.continue_to <= completed:
        p.error(f"--continue-to must exceed the {completed} completed epochs")
    if args.run is not None and (CKPT_DIR / graph_name / args.run).exists():
        p.error("continuation run already exists")
    graph = load_dblp_proxy15() if graph_name == "dblp_proxy15" else load(graph_name)
    cfg = ckpt["config"]
    if (cfg["num_g"], cfg["num_labels"]) != (graph.n, int(graph.labels.max())+1):
        p.error("checkpoint graph vocabulary does not match")
    saved = ckpt["args"]
    stream = TrainingStream(graph, saved["seed"], completed, saved["min_size"], saved["max_size"])
    if not len(stream.starts):
        p.error("graph has no eligible training examples")
    del ckpt
    print(f"Source {args.checkpoint} | {completed} epochs | device {device}", flush=True)
    results = []
    order = random.Random(19).sample(args.batches, len(args.batches))
    for batch_size in order:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        try:
            result = probe(snapshot, stream, batch_size, args.workers, device,
                           args.warmup_examples, args.examples, args.repeats)
        except torch.OutOfMemoryError as exc:
            if device.type != "cuda":
                raise
            result = dict(batch=batch_size, status="oom", error=str(exc))
        results.append(result)
        print(json.dumps(result), flush=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    eligible = [r for r in results if r["status"] == "ok"]
    best = max(eligible, key=lambda r: (r["median_examples_per_second"], -r["batch"])) if eligible else None
    report = dict(checkpoint=str(args.checkpoint.resolve()), completed_epochs=completed,
                  checkpoint_batch=saved["batch"], device=str(device),
                  warmup_examples=args.warmup_examples, examples_per_repeat=args.examples,
                  repeats=args.repeats, workers=args.workers, results=results,
                  selected_batch=best["batch"] if best else None,
                  continuation_run=args.run, continue_to=args.continue_to)
    (args.out/"batch_sweep.json").write_text(json.dumps(report, indent=2)+"\n")
    if best is None:
        raise SystemExit("No candidate batch fits on the GPU")
    print(f"Selected batch {best['batch']} | report {args.out/'batch_sweep.json'}", flush=True)
    if args.continue_to is not None:
        del graph, stream
        train_main(["--graph", graph_name, "--device", str(device), "--run", args.run,
                    "--resume-from", str(snapshot), "--batch", str(best["batch"]),
                    "--workers", str(args.workers), "--save-every", "1",
                    "--epochs", str(args.continue_to)])


if __name__ == "__main__":
    main()
