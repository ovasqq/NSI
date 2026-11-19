"""Replay each proved deep cut using incremental forward DFS from that prefix."""
import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
from tqdm import tqdm

from data.dblp import load_dblp_proxy15
from experiments.check_latency import CheckEncoder, features
from experiments.dblp_gate import write_json
from experiments.fragment_switch import run
from models.pruning_controller import checkpoint_digest
from queries.random_walk import Query
from search.enumerate import enumerate_space
from search.pruning import ResidualPruner
from train.train_neugn import load_model


def replay(space, prefix, pruner, args):
    depth = len(prefix)
    mapping = [-1]*space.q.n
    for u,v in zip(space.order,prefix):
        mapping[u] = v
    domains = pruner.build_domains(depth,mapping,set(prefix))
    if domains is None:
        raise ValueError("Cut prefix failed the earlier forward check")
    domain_s = 0.
    counts = Counter()
    def extend(d, mapping, used, parent):
        nonlocal domain_s
        tick = perf_counter()
        u = space.order[d-1]
        child = pruner.extend_domains(parent,u,mapping[u])
        domain_s += perf_counter() - tick
        counts["domain_calls"] += 1
        if child is None:
            counts["domain_cuts"] += 1
        return child

    times = []
    for _ in range(args.repeats):
        r = enumerate_space(space, branch_domains=extend, prefix=prefix, prefix_domains=domains,
                            max_steps=args.replay_steps, deadline=perf_counter()+args.replay_seconds,
                            max_matches=1)
        if r.num_matches:
            raise AssertionError(f"Unsound cut with embedding below {prefix}")
        times.append(r.subtree_seconds)
        if not r.complete:
            break
    return dict(steps=r.steps,closed=r.complete,forward_us=1e6*(median(times) if r.complete else times[-1]),trials=len(times))


def trace(query, graph, model, args, gate, expected):
    cuts = []
    space = None
    def capture(s, mapping, depth, fragment, cost):
        nonlocal space
        space = s
        cuts.append((tuple(mapping[u] for u in s.order[:depth]),fragment,cost))
    r = run(query, graph, model, args, "switch", 256, gate=gate, on_cut=capture)
    for key in ("steps","counts","solved","fms","timed_out","complete"):
        if expected is not None and r[key] != expected[key]:
            raise ValueError(f"Source trace changed ({key}): {r[key]} vs {expected[key]}")
    if not cuts:
        return []
    pruner = ResidualPruner(space,"forward")
    rows = []
    for prefix,fragment,cost in cuts:
        rows.append(dict(prefix=prefix,fragment=fragment,switch_us=cost*1e6,
                         **replay(space,prefix,pruner,args)))
    return rows


def describe(rows, gate_us):
    closed = [r['forward_us'] for r in rows if r['closed']]
    # A censored run establishes a lower bound on time spent.
    wins = sum(r['forward_us'] > r['switch_us']+gate_us for r in rows)
    unresolved = sum(not r['closed'] and r['forward_us'] <= r['switch_us']+gate_us for r in rows)
    return dict(cuts=len(rows),closed=len(closed),censored=len(rows)-len(closed),
                mean_gate_us=gate_us,
                median_closed_us=median(closed) if closed else None,
                p95_closed_us=float(np.percentile(closed,95)) if closed else None,
                median_closed_steps=median(r['steps'] for r in rows if r['closed']) if closed else None,
                wins=wins,unresolved=unresolved,
                fraction_above_cost_bounds=[wins/len(rows),(wins+unresolved)/len(rows)] if rows else [],
                measured_forward_s=sum(r['forward_us'] for r in rows)/1e6)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True,help="original results/ directory")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--replay-steps",type=int,default=100000)
    p.add_argument("--replay-seconds",type=float,default=.25)
    p.add_argument("--repeats",type=int,default=3)
    args=p.parse_args()
    if min(args.replay_steps,args.replay_seconds,args.repeats)<=0:
        p.error("replay budgets and repeats must be positive")
    src=args.source
    protocol=json.loads((src/"protocol.json").read_text())
    ck=Path(protocol["args"]["checkpoint"])
    if checkpoint_digest(ck)!=protocol["checkpoint_sha256"]:
        p.error("ranker differs from source")
    for k in ("depth","proposals","fragment_budget","budget","time_limit"):
        setattr(args,k,float(protocol["args"][k]) if k=="time_limit" else int(protocol["args"][k]))
    args.domain_strategy=protocol["args"]["domain_strategy"]
    torch.set_num_threads(1)
    graph=load_dblp_proxy15()
    graph.neighbor_label_counts(0)
    model=load_model(ck,"cpu").eval().to(args.device)
    encoder=CheckEncoder(graph.num_labels,32).eval()
    encoder.load_state_dict(torch.load(src/"transformer.pt",weights_only=True,map_location="cpu"))
    threshold=json.loads((src/"thresholds.json").read_text())["transformer"]
    def gate(space, domains):
        return encoder(*features(space,domains,True)).item()>=threshold
    queries=json.loads((src/"queries.json").read_text())["test"]
    attempts=[json.loads(line) for line in (src/"attempts.jsonl").read_text().splitlines()]
    reference={(r["mode"],r["size"],r["index"]):r for r in attempts if r["split"]=="test"}
    modes=("switch","transformer")
    args.out.mkdir(parents=True,exist_ok=True)
    if any(args.out.iterdir()):
        p.error("use an empty output directory")
    write_json(args.out/"protocol.json",dict(source=str(src),checkpoint_sha256=protocol["checkpoint_sha256"],
               repeats=args.repeats,replay_steps=args.replay_steps,replay_seconds=args.replay_seconds))
    by_mode={m:[] for m in modes}
    with (args.out/"cuts.jsonl").open("w") as f:
        for q in tqdm(queries,desc="Reproduce cuts and replay forward"):
            query=Query(graph.induced_subgraph(q["origin"]),q["origin"])
            for mode in modes:
                old=reference[mode,q["size"],q["index"]]
                rows=trace(query,graph,model,args,gate if mode=="transformer" else None,old)
                forward = reference["forward",q["size"],q["index"]]
                if old["solved"] and forward["solved"] and all(r["closed"] for r in rows):
                    saved_steps = forward["steps"] - old["steps"]
                    if sum(r["steps"] for r in rows) != saved_steps:
                        raise AssertionError(f"Replay steps disagree with paired forward run: {mode}, {q['index']}")
                for r in rows:
                    r.update(mode=mode,size=q["size"],index=q["index"])
                    f.write(json.dumps(r)+"\n")
                f.flush()
                by_mode[mode].extend(rows)
    original_gates=[r for (m,_,_),r in reference.items() if m=="transformer"]
    calls=sum(sum(r["counts"].get(k,0) for k in ("gate_skips","skip","manual","edge")) for r in original_gates)
    gate_us=1e6*sum(r["gate_seconds"] for r in original_gates)/calls
    summary={m:describe(by_mode[m],gate_us if m=="transformer" else 0.) for m in modes}
    summary["switch_cuts_vs_transformer_cost"]=describe(by_mode["switch"],gate_us)
    for mode in modes:
        summary[mode]["original_all_decision_and_check_s"]=sum(
            r["gate_seconds"]+r["select_seconds"]+r["check_seconds"] for (m,_,_),r in reference.items() if m==mode)
    write_json(args.out/"summary.json",summary)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
