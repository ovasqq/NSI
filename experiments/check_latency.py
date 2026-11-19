"""Batch-one latency of an independent check gate on real deep switch states."""

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean
from time import perf_counter, perf_counter_ns

import torch
from torch import nn
from tqdm import tqdm

from data.loaders import load
from experiments.fragment_pruning import search_space
from experiments.fragment_switch import select
from models.hook import NeuGNReorder
from models.pruning_controller import checkpoint_digest
from queries.storage import load_query_set
from search.enumerate import enumerate_space
from search.pruning import ResidualPruner, check_fragment
from train.train_neugn import load_model


def features(space, domains, transformer):
    q = space.q
    widths = {u: len(d) for u, d in domains.items()}
    n = len(widths)
    if not transformer:
        products = [w * widths[v] for u, w in widths.items()
                    for v in q.adj_set[u] if v > u and v in widths]
        values = list(widths.values())
        summary = [q.n - n, n, len(products), min(values), sum(values) / n,
                   max(values), min(products, default=0)]
        return (torch.tensor([math.log1p(v) for v in summary], dtype=torch.float32),)
    # Index 0 is CHECK. Boolean attention mask: True blocks a connection.
    blocked = [[False] * (n + 1)]
    labels, stats = [], []
    for u, w in widths.items():
        neighbors = q.adj_set[u]
        free = sum(v in widths for v in neighbors)
        labels.append(int(q.labels[u]))
        stats.append([math.log1p(w), free, len(neighbors) - free])
        blocked.append([False] + [v != u and v not in neighbors for v in widths])
    return (torch.tensor(labels, dtype=torch.long), torch.tensor(stats, dtype=torch.float32),
            torch.tensor(blocked, dtype=torch.bool))


class CheckEncoder(nn.Module):
    def __init__(self, labels, width):
        super().__init__()
        self.labels = nn.Embedding(labels, width)
        self.stats = nn.Linear(3, width)
        self.check = nn.Parameter(torch.zeros(1, width))
        layer = nn.TransformerEncoderLayer(width, 4, width * 4, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.out = nn.Linear(width, 1)

    def forward(self, labels, stats, blocked):
        nodes = self.labels(labels) + self.stats(stats)
        sequence = torch.cat((self.check, nodes)).unsqueeze(0)
        hidden = self.encoder(sequence, mask=blocked, is_causal=False)
        return self.out(hidden[0, 0])


def symbolic(space, domains):
    _, f = select(space, domains, "switch", 256, 12)
    return check_fragment(space, domains, f, 256) if f else True


def sample_query(space, hook, args, rng):
    pruner = ResidualPruner(space, "forward")
    visits, sample = 0, []

    def visit(depth, mapping, used):
        nonlocal visits
        domains = pruner.build_domains(depth, mapping, used)
        if domains is None:
            return True
        if depth < 10 or len(domains) < 2:
            return False
        visits += 1
        slot = visits - 1 if visits <= args.samples else rng.randrange(visits)
        if slot < args.samples:
            entry = (tuple(mapping[u] for u in space.order[:depth]), domains)
            if slot == len(sample):
                sample.append(entry)
            else:
                sample[slot] = entry
        return symbolic(space, domains) is False

    enumerate_space(space, hook, branch_check=visit, max_matches=1, max_steps=100000,
                    deadline=perf_counter() + args.trace_seconds)
    return visits, sample


def timed(space, domains, model, device, transformer, stages):
    sync = stages and device.startswith("cuda")
    start = perf_counter_ns()
    if model is None:
        if stages:
            _, f = select(space, domains, "switch", 256, 12)
            selected = perf_counter_ns()
            if f:
                check_fragment(space, domains, f, 256)
            return dict(select_us=(selected-start)/1000, check_us=(perf_counter_ns()-selected)/1000)
        symbolic(space, domains)
    else:
        inputs = features(space, domains, transformer)
        if stages:
            prepared = perf_counter_ns()
        inputs = tuple(x.to(device) for x in inputs)
        if stages:
            if sync:
                torch.cuda.synchronize(device)
            moved = perf_counter_ns()
        score = model(*inputs)
        if stages:
            if sync:
                torch.cuda.synchronize(device)
            ran = perf_counter_ns()
        decision = score.item() > 0  # Host receives its decision before the timer stops.
        if stages:
            end = perf_counter_ns()
            return dict(prepare_us=(prepared-start)/1000, transfer_us=(moved-prepared)/1000,
                        forward_us=(ran-moved)/1000, return_us=(end-ran)/1000)
    return dict(total_us=(perf_counter_ns()-start)/1000)


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        for group in ("all", r["residual_bin"]):
            groups[group, r["method"]].append(r)
    result = []
    for (group, method), items in groups.items():
        weights = [r["weight"] for r in items]
        total_weight = sum(weights)
        def avg(k):
            return sum(r[k]*w for r, w in zip(items, weights)) / total_weight
        ordered = sorted((us, r["weight"] / len(r["repeats_us"]))
                         for r in items for us in r["repeats_us"])
        def quantile(f):
            count = 0
            for us, weight in ordered:
                count += weight
                if count >= f*total_weight:
                    return us
        result.append(dict(group=group, method=method, samples=len(items),
                           mean_us=avg("total_us"), median_us=quantile(.5), p95_us=quantile(.95),
                           **{k: avg(k) for k in items[0] if k.endswith("_us") and k not in ("total_us", "repeats_us")}))
    reference = {r["group"]: r["mean_us"] for r in result if r["method"] == "switch"}
    for r in result:
        r["ratio_to_switch"] = r["mean_us"] / reference[r["group"]]
    return result


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "query-dir", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--devices", nargs="+", choices=["cpu", "cuda:0"], default=["cpu"])
    p.add_argument("--sizes", nargs="+", type=int, default=[16, 24, 32])
    p.add_argument("--queries", type=int, default=20, help="first training queries per size/density")
    p.add_argument("--samples", type=int, default=12, help="reservoir per query")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--trace-seconds", type=float, default=.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if not 1 <= args.queries <= 100 or min(args.samples,args.repeats,args.warmup) < 1:
        p.error("positive counts required; first 100 queries are training data")
    if not math.isfinite(args.trace_seconds) or args.trace_seconds <= 0 or min(args.sizes)<12:
        p.error("positive trace time and sizes >=12 required")
    if len(set(args.devices)) != len(args.devices) or len(set(args.sizes)) != len(args.sizes):
        p.error("duplicate device or size")
    args.output.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    trace_device = "cuda:0" if "cuda:0" in args.devices else "cpu"
    graph = load("wikics")
    backbone = load_model(args.checkpoint, trace_device)
    if backbone.cfg.num_g != graph.n:
        p.error("checkpoint incompatible with WikiCS")
    methods = {"switch": (None, "cpu", False)}
    for name in ("mlp", "transformer32", "transformer64"):
        net = (nn.Sequential(nn.Linear(7,32), nn.ReLU(), nn.Linear(32,1)) if name=="mlp"
               else CheckEncoder(backbone.cfg.num_labels, int(name.removeprefix("transformer"))))
        for dev in args.devices:
            methods[f"{name}/{dev}"] = (deepcopy(net).eval().to(dev), dev, name!="mlp")
    protocol = dict(vars(args), torch_version=str(torch.__version__), trace_device=trace_device,
                    gpu=torch.cuda.get_device_name(0) if trace_device!="cpu" else None,
                    checkpoint_sha256=checkpoint_digest(args.checkpoint), query_sha256={})
    rows, traces, warmed = [], [], set()
    order_rng = random.Random(args.seed)
    for size in args.sizes:
        for density in ("dense","sparse"):
            path = args.query_dir / f"size{size}_{density}.json"
            protocol["query_sha256"][path.name] = checkpoint_digest(path)
            queries, meta = load_query_set(path)
            if len(queries)<args.queries or meta.get("graph","wikics")!="wikics":
                raise ValueError(f"unexpected dataset: {path}")
            for qid,q in enumerate(tqdm(queries[:args.queries], desc=f"{size} {density}")):
                space = search_space(q.graph,graph)
                hook = NeuGNReorder(backbone,space.q,10,trace_device)
                rng = random.Random(f"{args.seed}/{size}/{density}/{qid}")
                visits, sample = sample_query(space,hook,args,rng)
                traces.append(dict(size=size,density=density,query=qid,visits=visits,
                                   prefixes=[prefix for prefix,_ in sample]))
                for sid,(_,domains) in enumerate(sample):
                    values = {m: [] for m in methods}
                    for name, spec in methods.items():
                        key = (name,len(domains))
                        if key not in warmed:
                            for _ in range(args.warmup):
                                timed(space,domains,*spec,False)
                            warmed.add(key)
                    for _ in range(args.repeats):
                        order = list(methods)
                        order_rng.shuffle(order)
                        for name in order:
                            values[name].append(timed(space,domains,*methods[name],False)["total_us"])
                    for name,spec in methods.items():
                        diagnostic = timed(space, domains, *spec, True)
                        rows.append(dict(size=size,density=density,query=qid,sample=sid,
                                         residual=len(domains),residual_bin="2–5" if len(domains)<=5
                                         else "6–12" if len(domains)<=12 else "13+",
                                         weight=visits/len(sample),method=name,
                                         total_us=mean(values[name]), repeats_us=values[name], **diagnostic))
    if not rows:
        raise RuntimeError("no eligible deep states collected")
    for filename,data in [("protocol.json",protocol),("traces.json",traces)]:
        (args.output/filename).write_text(json.dumps(data,default=str,indent=2))
    summary = summarize(rows)
    for filename,data in [("states.csv",rows),("summary.csv",summary)]:
        with (args.output/filename).open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in data for k in r)))
            writer.writeheader()
            writer.writerows(data)
    for r in summary:
        if r["group"] == "all":
            print(f"{r['method']:24} mean {r['mean_us']:8.1f} us  p95 {r['p95_us']:8.1f} us  "
                  f"{r['ratio_to_switch']:.2f} x switch")


if __name__=="__main__":
    main()
