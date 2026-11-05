"""CLI: train the inductive candidate scorer on one data graph.

    python -m train.train_inductive --graph hamster --run inductive

Training data: random-walk queries (5–15 vertices) whose search tree is fully
enumerated (`search.oracle.label_subtrees`); for every state at depth <
label_depth and every local candidate the label is has_solution (0/1). Loss is
BCE on the bilinear per-candidate logit. The model has no parameters tied to G,
so the checkpoint can be applied to another graph with the same label set.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from data.loaders import load
from models.encoding import collate, encode_query, encode_state
from models.inductive import InductiveConfig, InductiveNeuGN, g_features
from queries.euler import euler_sequence
from queries.random_walk import random_walk_vertices
from search.enumerate import SearchSpace
from search.filter import nlf_filter
from search.oracle import label_subtrees
from search.order import gql_order

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


def load_model(path: Path, g, device="cpu") -> InductiveNeuGN:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = InductiveConfig(**ck["config"])
    model = InductiveNeuGN(cfg, *g_features(g, cfg.num_labels))
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


def make_examples(g, rng: np.random.Generator, min_size, max_size,
                  label_depth, budget) -> list[tuple[dict, list[tuple[int, int]]]]:
    """One random-walk query -> list of (encoded state, [(candidate, num_matches)])."""
    size = int(rng.integers(min_size, max_size + 1))
    start = int(rng.integers(g.n))
    walk = random_walk_vertices(g, start, size, rng)
    if walk is None:
        return []
    q = g.induced_subgraph(walk)
    cand = nlf_filter(q, g)
    if any(len(c) == 0 for c in cand):
        return []
    order = gql_order(q, cand)
    labels, _, complete = label_subtrees(SearchSpace(q, g, cand, order), label_depth, budget)
    if not complete:
        return []
    cand_count = [len(c) for c in cand]
    qenc = encode_query(q, euler_sequence(q), cand_count)
    out: list[tuple[dict, list[tuple[int, int]]]] = []
    for key, cand_list in labels.items():
        i = len(key)
        mapping = [-1] * q.n
        for j in range(i):
            mapping[order[j]] = key[j]
        state = encode_state(qenc, mapping, order[i], 0, g.n)
        out.append((state, [(v, int(matches)) for v, _, _, matches in cand_list]))
    return out


class InductiveStream(IterableDataset):
    """Endless stream of (state, [(candidate, num_matches)]) examples."""

    def __init__(self, g, seed: int, min_size=5, max_size=15,
                 label_depth=3, budget=5_000_000):
        self.g, self.seed = g, seed
        self.min_size, self.max_size = min_size, max_size
        self.label_depth, self.budget = label_depth, budget

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng([self.seed, wid])
        while True:
            for ex in make_examples(self.g, rng, self.min_size, self.max_size,
                                    self.label_depth, self.budget):
                yield ex


def collate_inductive(items: list[tuple[dict, list[tuple[int, int]]]]):
    batch = collate([s for s, _ in items])
    b = len(items)
    K = max(len(cs) for _, cs in items)
    cand = torch.zeros(b, K, dtype=torch.long)
    target = torch.zeros(b, K)  # log(1 + num_matches) regression target
    mask = torch.zeros(b, K, dtype=torch.bool)
    for i, (_, cs) in enumerate(items):
        for j, (v, matches) in enumerate(cs):
            cand[i, j] = v
            target[i, j] = float(np.log1p(matches))
            mask[i, j] = True
    batch["cand"], batch["cand_mask"], batch["target"] = cand, mask, target
    return batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="hamster")
    ap.add_argument("--run", default="inductive")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps-per-epoch", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ffn", type=int, default=1024)
    ap.add_argument("--gcn-layers", type=int, default=2)
    ap.add_argument("--feat-dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no-token-labels", action="store_true")
    ap.add_argument("--no-gcn", action="store_true")
    ap.add_argument("--no-degree", action="store_true")
    ap.add_argument("--no-nlf", action="store_true")
    ap.add_argument("--min-size", type=int, default=5)
    ap.add_argument("--max-size", type=int, default=15)
    ap.add_argument("--label-depth", type=int, default=3)
    ap.add_argument("--budget", type=int, default=5_000_000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=2)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    g = load(args.graph)
    num_labels = int(g.labels.max()) + 1
    cfg = InductiveConfig(
        num_labels=num_labels, d=args.d, layers=args.layers, heads=args.heads,
        ffn=args.ffn, gcn_layers=args.gcn_layers, feat_dim=args.feat_dim,
        dropout=args.dropout, token_labels=not args.no_token_labels,
        use_gcn=not args.no_gcn, use_degree=not args.no_degree, use_nlf=not args.no_nlf,
    )
    model = InductiveNeuGN(cfg, *g_features(g, num_labels)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.gamma)

    run_dir = CKPT_DIR / args.graph / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(dict(model=cfg.to_dict(), args=vars(args)), indent=1))

    stream = InductiveStream(g, args.seed, args.min_size, args.max_size,
                             args.label_depth, args.budget)
    loader = DataLoader(stream, batch_size=args.batch, collate_fn=collate_inductive,
                        num_workers=args.workers, persistent_workers=args.workers > 0)
    batches = iter(loader)
    print(f"{args.graph}: {g.n} vertices, {num_labels} labels, "
          f"{sum(p.numel() for p in model.parameters()):,} params, {device}")

    with open(run_dir / "log.jsonl", "a") as log:
        for epoch in range(args.epochs):
            model.train()
            t0 = time.time()
            tot_loss = tot = tot_pos = 0
            for _ in range(args.steps_per_epoch):
                batch = {k: v.to(device) for k, v in next(batches).items()}
                s = model(batch)
                scores = model.score(s, batch["cand"])
                m = batch["cand_mask"]
                target = batch["target"][m]  # log(1 + num_matches)
                loss = F.mse_loss(scores[m], target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot_loss += loss.item() * int(m.sum())
                tot += int(m.sum())
                tot_pos += int((target > 0).sum())  # solvable candidates
            sched.step()
            rec = dict(epoch=epoch, loss=tot_loss / max(tot, 1), pos_rate=tot_pos / max(tot, 1),
                       lr=sched.get_last_lr()[0], seconds=time.time() - t0)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"epoch {epoch:3d}  loss {rec['loss']:.4f}  pos_rate {rec['pos_rate']:.3f}  "
                  f"{rec['seconds']:.1f}s", flush=True)
            if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
                save_checkpoint(run_dir / "model.pt", model, opt, sched, epoch, args)


if __name__ == "__main__":
    main()
