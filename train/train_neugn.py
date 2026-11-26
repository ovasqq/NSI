"""CLI: train NeuGN on one data graph.

    python -m train.train_neugn --graph hamster [--epochs 1000] [--run NAME]

Writes checkpoints/<graph>/<run>/{config.json, log.jsonl, model.pt}.
`--resume` continues in the same directory. `--resume-from` starts a new
run directory from an existing training checkpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.loaders import load
from models.neugn import NeuGN, NeuGNConfig
from train.dataset import TrainingStream, collate_examples

CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path: Path, model, opt, sched, epoch: int, args) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(dict(config=model.cfg.to_dict(), model=model.state_dict(),
                    opt=opt.state_dict(), sched=sched.state_dict(),
                    epoch=epoch, args=vars(args)), tmp)
    tmp.replace(path)


def load_model(path: Path, device: str | torch.device = "cpu") -> NeuGN:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = NeuGN(NeuGNConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


def train_step(model, opt, batch, device):
    batch = {k: v.to(device) for k, v in batch.items()}
    logits = model(batch)
    loss = torch.nn.functional.cross_entropy(logits, batch["target"])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    with torch.no_grad():
        top = logits.topk(min(10, logits.shape[1]), dim=1).indices
        hit = top == batch["target"][:, None]
        count = len(batch["target"])
        return count, loss.item() * count, hit[:, 0].sum().item(), hit.any(1).sum().item()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="hamster")
    ap.add_argument("--run", default="default")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--gamma", type=float, default=0.999, help="ExponentialLR, per epoch")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ffn", type=int, default=1024)
    ap.add_argument("--gcn-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no-token-labels", action="store_true")
    ap.add_argument("--no-gcn", action="store_true", help="ablation: without QSExtractor")
    ap.add_argument("--min-size", type=int, default=5)
    ap.add_argument("--max-size", type=int, default=19)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=10)
    resume = ap.add_mutually_exclusive_group()
    resume.add_argument("--resume", action="store_true")
    resume.add_argument("--resume-from", help="Resume model, optimizer and scheduler into a new run.")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    if args.graph == "dblp_proxy15":
        from data.dblp import load_dblp_proxy15
        g = load_dblp_proxy15()
    else:
        g = load(args.graph)
    run_dir = CKPT_DIR / args.graph / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "model.pt"

    cfg = NeuGNConfig(
        num_g=g.n, num_labels=int(g.labels.max()) + 1, d=args.d, layers=args.layers,
        heads=args.heads, ffn=args.ffn, gcn_layers=args.gcn_layers, dropout=args.dropout,
        token_labels=not args.no_token_labels, use_gcn=not args.no_gcn,
    )
    source = Path(args.resume_from) if args.resume_from else ckpt_path
    if args.resume_from and (source.resolve() == ckpt_path.resolve() or ckpt_path.exists()
                             or (run_dir / "log.jsonl").exists()):
        ap.error("--resume-from needs a fresh output run directory")
    ck = None
    if args.resume_from or (args.resume and source.exists()):
        ck = torch.load(source, map_location="cpu", weights_only=False)
        if args.resume_from:
            saved = ck["args"]
            if saved["graph"] != args.graph:
                ap.error("source checkpoint belongs to another graph")
            cfg = NeuGNConfig(**ck["config"])
            if (cfg.num_g, cfg.num_labels) != (g.n, int(g.labels.max()) + 1):
                ap.error("source checkpoint graph vocabulary does not match")
            args.seed, args.min_size, args.max_size = (saved[k] for k in ("seed", "min_size", "max_size"))
            args.lr, args.gamma = saved["lr"], ck["sched"]["gamma"]
            for key in ("d", "layers", "heads", "ffn", "gcn_layers", "dropout"):
                setattr(args, key, getattr(cfg, key))
            args.no_token_labels, args.no_gcn = not cfg.token_labels, not cfg.use_gcn
            torch.manual_seed(args.seed)
    model = NeuGN(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.gamma)
    start_epoch = 0
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch = ck["epoch"] + 1
        del ck
        print(f"Resuming from {source}, {start_epoch} epochs already completed", flush=True)
    (run_dir / "config.json").write_text(json.dumps(dict(model=cfg.to_dict(), args=vars(args)), indent=1))

    stream = TrainingStream(g, args.seed, start_epoch, args.min_size, args.max_size)
    loader = DataLoader(stream, batch_size=args.batch, collate_fn=collate_examples,
                        num_workers=args.workers, persistent_workers=args.workers > 0)
    batches = iter(loader)
    steps = stream.steps_per_epoch(args.batch)
    print(f"{args.graph}: {g.n} vertices, {len(stream.starts)} start vertices, "
          f"{steps} steps/epoch, {sum(p.numel() for p in model.parameters()):,} params, {device}")

    with open(run_dir / "log.jsonl", "a") as log:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            t0 = time.time()
            tot_loss = tot_top1 = tot_top10 = tot = 0
            for _ in range(steps):
                count, loss, top1, top10 = train_step(model, opt, next(batches), device)
                tot += count
                tot_loss += loss
                tot_top1 += top1
                tot_top10 += top10
            sched.step()
            rec = dict(epoch=epoch, loss=tot_loss / tot, top1=tot_top1 / tot,
                       top10=tot_top10 / tot, lr=sched.get_last_lr()[0],
                       seconds=time.time() - t0)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"epoch {epoch:4d}  loss {rec['loss']:.3f}  top1 {rec['top1']:.3f}  "
                  f"top10 {rec['top10']:.3f}  {rec['seconds']:.1f}s", flush=True)
            if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
                save_checkpoint(ckpt_path, model, opt, sched, epoch, args)


if __name__ == "__main__":
    main()
