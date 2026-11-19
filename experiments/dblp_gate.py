"""Train independent CPU gates on trajectories of a frozen DBLP NeuGN."""
import argparse
import csv
import json
import random
from pathlib import Path
from statistics import median

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from data.dblp import DIGEST, load_dblp_proxy15
from experiments.check_latency import CheckEncoder, features
from experiments.fragment_switch import run
from models.pruning_controller import checkpoint_digest
from queries.random_walk import generate_query_set
from train.train_neugn import load_model


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def collect(queries, graph, model, args, rng):
    states = []
    for _, _, q in tqdm(queries, desc="Collect NeuGN + switch"):
        reservoir = []
        seen = 0

        def record(space, domains, cut):
            nonlocal seen
            seen += 1
            slot = seen-1 if seen <= args.samples else rng.randrange(seen)
            if slot < args.samples:
                row = (features(space, domains, False), features(space, domains, True), float(cut))
                if slot == len(reservoir):
                    reservoir.append(row)
                else:
                    reservoir[slot] = row

        run(q, graph, model, args, "switch", 256, observe=record)
        states.extend(reservoir)
    return states


def train_heads(states, graph, args):
    positives = sum(int(r[-1]) for r in states)
    write_json(args.out/"training_counts.json", dict(states=len(states), cuts=positives))
    if not 0 < positives < len(states):
        raise ValueError(f"Sample has {positives} cuts in {len(states)} states; increase --train or --samples")
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(states)-positives)/positives))
    heads = {"mlp": nn.Sequential(nn.Linear(7,32), nn.ReLU(), nn.Linear(32,1)),
             "transformer": CheckEncoder(graph.num_labels, 32)}
    for index, (name, head) in enumerate(heads.items()):
        rng = random.Random(19)
        order = list(range(len(states)))
        opt = torch.optim.AdamW(head.parameters(), lr=.001)
        for _ in tqdm(range(args.epochs), desc="Train "+name):
            rng.shuffle(order)
            for start in range(0,len(order),32):
                batch = [states[i] for i in order[start:start+32]]
                logits = (head(torch.stack([r[0][0] for r in batch])).flatten() if name=="mlp"
                          else torch.cat([head(*r[index]) for r in batch]))
                opt.zero_grad()
                loss(logits, torch.tensor([r[-1] for r in batch])).backward()
                opt.step()
        head.eval()
        torch.save(head.state_dict(), args.out/(name+".pt"))
    return heads


def experiment(graph, model, args):
    splits = {}
    signatures = set()
    for k, split in enumerate(("train", "val", "test")):
        split_rows = []
        for size in (16,32):
            count = getattr(args,split)
            queries = generate_query_set(graph,size,None,count,1000*size+k)
            if len(queries) != count:
                raise ValueError("Incomplete query set")
            for i,q in enumerate(queries):
                key = (tuple(q.graph.labels), tuple(q.graph.edges()))
                if key in signatures:
                    raise ValueError("Duplicate labeled query")
                signatures.add(key)
                split_rows.append((size,i,q))
        splits[split] = split_rows
    write_json(args.out/"queries.json", {s:[dict(size=z,index=i,origin=q.origin,density=q.density)
               for z,i,q in qs] for s,qs in splits.items()})
    graph.neighbor_label_counts(0)
    rng = random.Random(19)
    heads = train_heads(collect(splits["train"], graph, model, args, rng), graph, args)
    thresholds = {}
    records = []
    for split in ("val","test"):
        for size,index,q in tqdm(splits[split], desc=split):
            jobs = ([(name,t) for name in heads for t in (-2.,0.,2.)] if split=="val"
                    else [(name,thresholds.get(name)) for name in ("none","forward","switch",*heads)])
            rng.shuffle(jobs)
            for name,t in jobs:
                gate = None
                if name in heads:
                    def gate(space, domains):
                        return heads[name](*features(space, domains, name=="transformer")).item() >= t
                with torch.inference_mode():
                    r = run(q,graph,model,args,name if gate is None else "switch",256,gate=gate)
                row = dict(r,mode=name,gate_threshold=t,split=split,size=size,index=index,density=q.density)
                records.append(row)
                with (args.out/"attempts.jsonl").open("a") as f:
                    f.write(json.dumps(row)+"\n")
        if split=="val":
            for name in heads:
                def criterion(t):
                    rr = [r for r in records if r["mode"]==name and r["gate_threshold"]==t]
                    return -sum(r["solved"] for r in rr), sum(r["seconds"] for r in rr)
                thresholds[name] = min((-2.,0.,2.), key=criterion)
            write_json(args.out/"thresholds.json", thresholds)
    summary = []
    for size in (0,16,32):
        for name in ("none","forward","switch",*heads):
            rr = [r for r in records if r["split"]=="test" and r["mode"]==name and (not size or r["size"]==size)]
            times = [r["seconds"] for r in rr]
            summary.append(dict(size=size or "all",mode=name,queries=len(rr),solved=sum(r["solved"] for r in rr),
                                total_s=sum(times),median_ms=median(times)*1000,p95_ms=np.percentile(times,95)*1000,
                                steps=sum(r["steps"] for r in rr),gate_s=sum(r["gate_seconds"] for r in rr),
                                skips=sum(r["counts"].get("gate_skips",0) for r in rr),
                                cuts=sum(r["counts"].get(k,0) for k in ("manual_cuts","edge_cuts") for r in rr)))
    with (args.out/"summary.csv").open("w") as f:
        writer = csv.DictWriter(f,fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary[:5],indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    for name, default in dict(train=48,val=12,test=24,epochs=4,samples=96,budget=5000).items():
        p.add_argument("--"+name,type=int,default=default)
    p.add_argument("--time-limit",type=float,default=10)
    args=p.parse_args()
    if min(args.train,args.val,args.test,args.epochs,args.samples,args.budget,args.time_limit)<=0:
        p.error("counts and limits must be positive")
    args.depth, args.proposals, args.fragment_budget, args.domain_strategy = 10,12,256,"incremental"
    args.out.mkdir(parents=True,exist_ok=True)
    if any(args.out.iterdir()):
        p.error("use an empty output directory")
    torch.set_num_threads(1)
    torch.manual_seed(19)
    meta = torch.load(args.checkpoint,map_location="cpu",mmap=True,weights_only=False)
    if meta["args"].get("graph") != "dblp_proxy15":
        p.error("expected ranker trained on dblp_proxy15")
    epoch = meta["epoch"]+1
    del meta
    graph = load_dblp_proxy15()
    model = load_model(args.checkpoint,"cpu").to(args.device).eval().requires_grad_(False)
    if (model.cfg.num_g,model.cfg.num_labels)!=(graph.n,graph.num_labels):
        p.error("graph and checkpoint vocabulary mismatch")
    write_json(args.out/"protocol.json",dict(args={k:str(v) for k,v in vars(args).items()},
               checkpoint_sha256=checkpoint_digest(args.checkpoint),ranker_epochs=epoch,
               graph_sha256=DIGEST,labels="uniform 15 seed 19",torch=str(torch.__version__)))
    experiment(graph, model,args)


if __name__=="__main__":
    main()
