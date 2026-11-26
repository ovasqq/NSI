"""Imitate triple witnesses and improve the pointer with measured branch rewards."""
import argparse
from collections import Counter
from copy import copy, deepcopy
import json
from pathlib import Path
from random import Random
from time import perf_counter

import torch
from tqdm import tqdm

from data.loaders import load
from experiments.joint_search import (benchmark, collect, compare_pilot, indexed_space,
                                     paired_summary, read_queries, run)
from models.fragment import manual_fragment, proposals
from models.joint_dfs import Inference, train_joint, vertex_features
from models.triple_pointer import TriplePointer
from search.nlf_index import NLFIndex
from search.pruning import hall_conflict
from search.triple_check import TripleCheck
from train.train_neugn import load_model


def options(space, domains):
    triples = [p for p in proposals(space, domains, 12) if len(p) == 3]
    narrow = tuple(sorted(sorted(domains, key=lambda u: (len(domains[u]), u))[:3]))
    return list(dict.fromkeys([*triples, narrow]))


def teacher_target(space, domains, checker):
    candidates = options(space, domains)
    hall = hall_conflict(domains)
    if hall and len(hall[0]) <= 3:
        members = list(hall[0])
        members.extend(sorted(domains.keys()-set(members), key=lambda u: (len(domains[u]), u))[:3-len(members)])
        candidates.insert(0, tuple(sorted(members)))
    candidates = list(dict.fromkeys(candidates))
    outcomes = checker.many(space, domains, candidates)
    return next((t for t, o in zip(candidates, outcomes) if o is False), None)


def traces(items, index, teacher, data, args, ranker="neugn_forward", runtime=None):
    rng = Random(args.seed)
    rows = []
    for query_id, item in enumerate(tqdm(items, desc="Baseline traces", disable=not args.progress)):
        query = item[-1] if isinstance(item, tuple) else item
        space = indexed_space(query.graph, index)
        seen, pools = Counter(), {}

        def observe(rep, frame, depth, step):
            if len(frame.domains) < 3:
                return None
            seen[depth] += 1
            slot = rng.randrange(seen[depth]) if seen[depth] > args.samples else seen[depth]-1
            if slot >= args.samples:
                return None
            context = frame.action_input[0].ranking_context if frame.action_input else None
            row = dict(space=space, domains=frame.domains, anchor=context[0] if context else None,
                       depth=depth, query=query_id,
                       group=list(item[:3]) if isinstance(item, tuple) else None)
            pool = pools.setdefault(depth, [])
            if slot == len(pool):
                pool.append(row)
            else:
                pool[slot] = row
            tick = perf_counter()

            def finish(end, complete):
                row.update(steps=end-step, seconds=perf_counter()-tick, complete=complete)
            return finish

        with torch.no_grad():
            run(space, teacher, data, args, ranker, observe=observe,
                rank_all_depths=True, **(runtime or {}))
        rows.extend(row for pool in pools.values() for row in pool)
    return rows


class Policy:
    def __init__(self, model, checker):
        self.model, self.checker = model, checker
        self.uses_anchor = model is not None and model.anchor is not None

    def action(self, space, domains, anchor, sample=False):
        if self.model is None:
            return manual_fragment(space, domains, options(space, domains)), None
        return self.model(space, domains, anchor, sample=sample)

    def __call__(self, space, domains, anchor, stats):
        start = perf_counter()
        triple, _ = self.action(space, domains, anchor)
        middle = perf_counter()
        outcome, = self.checker.many(space, domains, [triple])
        stats["decision_seconds"] += middle-start
        stats["check_seconds"] += perf_counter()-middle
        stats["check_triple"] += 1
        if outcome is False:
            stats["proved_triple"] += 1
        if outcome is None:
            stats["unknown_triple"] += 1
        return outcome is False


def train(rows, args, checker, anchor_dim, tag, architecture):
    model = TriplePointer(args.device, anchor_dim, architecture=architecture)
    teacher_rows = [r for r in rows if r["witness"] is not None]
    if not teacher_rows:
        raise ValueError("No certified training triples; increase train-queries, samples or fragment-budget")
    rng = Random(args.seed)
    history = []
    modes = {}
    for stage in ("imitation", "reward"):
        opt = torch.optim.AdamW(model.parameters(), lr=.001 if stage == "imitation" else .0002)
        examples = teacher_rows[:] if stage == "imitation" else list(rows)
        # Without a proof only check cost is charged, including on paths ending at a match.
        # With a proof a censored trace supplies a lower bound on avoidable DFS time.
        for epoch in range(args.epochs if stage == "imitation" else args.reward_epochs):
            rng.shuffle(examples)
            total_reward, proofs = 0., 0
            for offset in range(0, len(examples), args.batch_size):
                losses = []
                for row in examples[offset:offset+args.batch_size]:
                    tick = perf_counter()
                    key = (row["space"], row["domains"], row["anchor"])
                    if stage == "imitation":
                        _, lp = model(*key, witness=row["witness"])
                        losses.append(-lp)
                    else:
                        # Measure deployment-like sampling and verification, independently of autograd.
                        with torch.no_grad():
                            triple, _ = model(*key, sample=True)
                            outcome, = checker.many(key[0], key[1], [triple])
                        cost = perf_counter()-tick
                        reward = (row["seconds"] if outcome is False else 0.) - cost
                        total_reward += reward
                        proofs += outcome is False
                        _, lp = model(*key, forced=triple)
                        losses.append(-lp * reward * 1000.)
                opt.zero_grad()
                torch.stack(losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
            report = dict(model=tag, stage=stage, epoch=epoch+1, examples=len(examples),
                          proofs=proofs, reward_seconds=total_reward)
            history.append(report)
            print(report, flush=True)
        modes[tag+"_"+stage] = deepcopy(model).eval()
        torch.save(model.state_dict(), args.out/(tag+"_"+stage+".pt"))
    return modes, history


def replay(rows, policies, path):
    totals = {}
    with torch.no_grad(), path.open("w") as stream:
        for row in rows:
            record = {k:row[k] for k in ("query","group","depth","seconds","steps","complete")}
            for name, policy in policies.items():
                stats = Counter()
                proved = policy(row["space"], row["domains"], row["anchor"], stats)
                cost = stats["decision_seconds"]+stats["check_seconds"]
                group = ("censored" if not row["complete"] else
                         "closed_1ms_or_more" if row["seconds"] >= .001 else "closed_under_1ms")
                bucket = totals.setdefault(name, {}).setdefault(group, Counter())
                bucket.update(states=1, proofs=int(proved), unknown=stats["unknown_triple"],
                              policy_seconds=cost, observed_saved_seconds=row["seconds"]*proved,
                              observed_saved_steps=row["steps"]*proved)
                record[name] = dict(proved=proved, **stats,
                                    gain=(row["seconds"]*proved-cost) if row["complete"] else None)
            print(json.dumps(record), file=stream)
    return totals


def joint_graph_controls(training, validation, index, teacher, data, args):
    with torch.no_grad():
        rank_rows, check_rows, _ = collect(training, index, data, teacher, args)
    model = train_joint(rank_rows, check_rows, graph=True, epochs=args.joint_epochs,
                        seed=args.seed, device=args.device)
    torch.save(model.state_dict(), args.out/"joint_graph.pt")
    runtime = {"graph_model": Inference.from_model(model)}
    candidates = {"joint_graph_rank": ("joint_graph_rank", {})}
    for margin in (0., .5, 1.):
        candidates[f"joint_graph_{margin:g}"] = "joint_graph", {"margin": margin}
    with torch.no_grad():
        records = benchmark(validation, candidates, index, teacher, data, args, runtime,
                            args.out/"joint_graph_validation.jsonl", 1)
    summary = paired_summary(records, candidates)
    pairs = compare_pilot(records, candidates, "joint_graph_rank")
    best = min((name for name in candidates if name != "joint_graph_rank"),
               key=lambda name: (len(pairs[name]["only_reference"]), summary[name]["seconds"]))
    return {"joint_graph_rank": candidates["joint_graph_rank"], "joint_graph": candidates[best]}, runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint","query-dir","out"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--joint-graph", action="store_true", help="Train and benchmark both structural MLP controls.")
    parser.add_argument("--structural-ranker", action="store_true",
                        help="Use joint_graph_rank for proof traces and all triple policies.")
    parser.add_argument("--rank-step-test", action="store_true",
                        help="Also compare the two bare rankers at the same step budget without a deadline.")
    parser.add_argument("--selectors", nargs="+", choices=("attention", "anchor", "mlp", "message"),
                        default=("attention", "anchor"))
    parser.add_argument("--graph", choices=("wikics", "hamster"), default="wikics")
    parser.add_argument("--sizes", nargs="+", type=int, default=[16,24,32])
    for name, value in dict(train_queries=40, validation_queries=8, queries=20, test_offset=90,
                            budget=30000, repeats=3, samples=2, epochs=4, reward_epochs=2,
                            batch_size=16, fragment_budget=4096, seed=19,
                            joint_epochs=6, negatives=4).items():
        parser.add_argument("--"+name.replace("_","-"), type=int, default=value)
    parser.add_argument("--seconds", type=float, default=1.)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.structural_ranker and (not args.joint_graph or "anchor" in args.selectors):
        parser.error("--structural-ranker requires --joint-graph and selectors without anchor")
    if args.rank_step_test and not args.joint_graph:
        parser.error("--rank-step-test requires --joint-graph")
    if len(set(args.selectors)) != len(args.selectors):
        parser.error("--selectors must be unique")
    args.depth, args.test_query_dir, args.step_value_us = 10, None, 10.
    if args.test_offset < args.train_queries+args.validation_queries or min(
        args.seconds,args.repeats,args.samples,args.epochs,args.reward_epochs,args.batch_size,
        args.fragment_budget,args.budget,args.train_queries,args.validation_queries,args.queries,
        args.joint_epochs,args.negatives) <= 0:
        parser.error("use positive budgets and disjoint query splits")
    args.out.mkdir(exist_ok=True, parents=True)
    if any(args.out.iterdir()):
        parser.error("output directory must be empty")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    # Keep additive relational attention biases identical during training and evaluation.
    torch.backends.mha.set_fastpath_enabled(False)
    graph = load(args.graph)
    index, data = NLFIndex(graph), vertex_features(graph)
    teacher = load_model(args.checkpoint,args.device)
    if teacher.cfg.num_g != graph.n or teacher.cfg.num_labels != int(graph.labels.max())+1:
        raise ValueError(f"checkpoint vocabulary differs from {args.graph}")
    checker = TripleCheck(graph, args.device, args.fragment_budget)
    training, validation, test = read_queries(args)
    controls, runtime = ({}, {})
    if args.joint_graph:
        controls, runtime = joint_graph_controls(training, validation, index, teacher, data, args)
    ranker = "joint_graph_rank" if args.structural_ranker else "neugn_forward"
    rows = traces(training,index,teacher,data,args,ranker,runtime)
    with (args.out/"training_traces.jsonl").open("w") as stream:
        for row in tqdm(rows, desc="Teacher witnesses", disable=not args.progress):
            row["witness"] = teacher_target(row["space"], row["domains"], checker)
            print(json.dumps({k:row[k] for k in ("query","depth","complete","steps","seconds","witness")}),file=stream)
    learned, history = {}, []
    for selector in args.selectors:
        tag = {"attention": "pointer", "anchor": "pointer_anchor"}.get(selector, selector)
        dim = teacher.cfg.d if selector == "anchor" else 0
        architecture = "attention" if selector == "anchor" else selector
        torch.manual_seed(args.seed)
        variants, logs = train(rows,args,checker,dim,tag,architecture)
        learned.update(variants)
        history.extend(logs)
    policies = {"heuristic":Policy(None,checker), **{k:Policy(v,checker) for k,v in learned.items()}}
    val_branches = replay(traces(validation,index,teacher,data,args,ranker,runtime), policies,
                          args.out/"validation_branches.jsonl")
    test_branches = replay(traces(test,index,teacher,data,args,ranker,runtime), policies,
                           args.out/"test_branches.jsonl")
    modes = {"forward":("neugn_forward",dict(rank_all_depths=True))}
    modes.update({k:(ranker,dict(rank_all_depths=True,triple_policy=p)) for k,p in policies.items()})
    modes.update(controls)
    with torch.no_grad():
        records = benchmark(test,modes,index,teacher,data,args,runtime,args.out/"test_records.jsonl",args.repeats)
    by_group = {f"{size}_{density}": paired_summary(
        [row for row in records if (row["size"], row["density"]) == (size, density)], modes)
        for size in args.sizes for density in ("dense", "sparse")}
    result = dict(params=vars(args), history=history, train_states=len(rows),
                  validation_branches=val_branches, test_branches=test_branches,
                  complete_train_states=sum(r["complete"] for r in rows),
                  teacher_witnesses=sum(r["witness"] is not None for r in rows),
                  summary=paired_summary(records,modes),
                  by_group=by_group,
                  joint_graph_margin=modes["joint_graph"][1]["margin"] if args.joint_graph else None,
                  paired={base:compare_pilot(records,modes,base) for base in modes})
    (args.out/"results.json").write_text(json.dumps(result,default=str,indent=2)+"\n")
    print({k:(v["solved_by_repeat"],round(v["seconds"],3)) for k,v in result["summary"].items()})
    if args.rank_step_test:
        step_args = copy(args)
        step_args.seconds = None
        rank_modes = {name: modes[name] for name in ("forward", "joint_graph_rank")}
        with torch.no_grad():
            step_records = benchmark(test, rank_modes, index, teacher, data, step_args, runtime,
                                     args.out/"rank_step_records.jsonl", 1)
        step_result = dict(budget=args.budget, summary=paired_summary(step_records, rank_modes),
                           paired=compare_pilot(step_records, rank_modes, "forward"))
        (args.out/"rank_step_results.json").write_text(json.dumps(step_result, indent=2)+"\n")


if __name__ == "__main__":
    main()
