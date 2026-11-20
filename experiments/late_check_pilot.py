"""Compare a shallow tree policy with a single delayed strong check per active DFS branch."""
import argparse
import json
from collections import Counter
from math import isfinite
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter

import numpy as np
import torch
from tqdm import tqdm

from data.loaders import load
from experiments.check_latency import CheckEncoder, features
from experiments.deep_pruning import fingerprint
from experiments.fms import censored_median
from experiments.fragment_cost_oracle import continuation, query_run, reject
from experiments.fragment_pruning import search_space
from models.hook import NeuGNReorder
from models.pruning_controller import checkpoint_digest
from queries.storage import load_query_set
from search.enumerate import enumerate_space
from search.pruning import ResidualPruner
from train.train_neugn import load_model


TREE_ACTIONS = ("skip", "edge", "manual")


def cheap(depth, domains):
    sizes = [len(d) for d in domains.values()]
    return depth, min(sizes), max(sizes), sum(sizes), sizes.count(1)


def fit_tree(x, costs, depth=3, min_leaf=16):
    """Minimize estimated elapsed time over skip, edge, and manual."""
    totals = costs.sum(axis=0)
    best_cost = totals.min()
    best = None
    if depth and len(costs) >= 2*min_leaf:
        for column in range(x.shape[1]):
            splits = np.unique(np.quantile(x[:, column], np.linspace(.1, .9, 9), method="nearest"))
            for pivot in splits:
                mask = x[:, column] <= pivot
                if min(int(mask.sum()), int((~mask).sum())) < min_leaf:
                    continue
                left_cost = costs[mask].sum(axis=0)
                score = left_cost.min() + (totals-left_cost).min()
                if score < best_cost:
                    best_cost = score
                    best = column, int(pivot), mask
    if best is None:
        return TREE_ACTIONS[int(totals.argmin())]
    column, pivot, mask = best
    return column, pivot, fit_tree(x[mask], costs[mask], depth-1, min_leaf), fit_tree(
        x[~mask], costs[~mask], depth-1, min_leaf
    )


def choose(tree, inputs):
    while isinstance(tree, tuple):
        column, pivot, left, right = tree
        tree = left if inputs[column] <= pivot else right
    return tree


def learn_tree(spaces, model, args, rng):
    inputs, costs = [], []
    for space in tqdm(spaces, desc="Train tree", disable=not args.progress):
        seen, sample = 0, []

        def observe(prefix, domains):
            nonlocal seen
            seen += 1
            index = seen-1 if seen <= args.samples else rng.randrange(seen)
            if index == len(sample):
                sample.append((prefix, domains))
            elif index < len(sample):
                sample[index] = prefix, domains

        query_run(space, model, args, observe=observe)
        pruner = ResidualPruner(space, "forward")
        for prefix, domains in sample:
            depth = len(prefix)
            mapping = [-1]*space.q.n
            for u, v in zip(space.order, prefix):
                mapping[u] = v
            used = set(prefix)
            timings = {action: [] for action in TREE_ACTIONS}
            cuts = {}
            for _ in range(3):
                order = list(TREE_ACTIONS)
                rng.shuffle(order)
                for action in order:
                    tick = perf_counter()
                    cuts[action] = reject(pruner, depth, mapping, used, domains, action, args)
                    timings[action].append((perf_counter()-tick)*1e6)
            remainder_us = 0.
            if any(cuts.values()):
                result = continuation(pruner, prefix, domains, args)
                if result.num_matches:
                    raise AssertionError("A fragment check discarded a valid embedding")
                remainder_us = min(5000., result.subtree_seconds*1e6) if result.complete else 5000.
            inputs.append(cheap(depth, domains))
            costs.append([median(timings[action]) + (0. if cuts[action] else remainder_us)
                          for action in TREE_ACTIONS])
    if not inputs:
        raise ValueError("No eligible deep states in training")
    return fit_tree(np.asarray(inputs), np.asarray(costs)), len(inputs)


def search(space, model, args, tree, mode, *, encoder=None, observe_late=None, delay=None):
    pruner = ResidualPruner(space, "forward")
    counts = Counter()
    late_seconds = 0.

    def advance(depth, mapping, used, parent):
        u = space.order[depth-1]
        domains = (pruner.build_domains(depth, mapping, used) if parent is None
                   else pruner.extend_domains(parent, u, mapping[u]))
        if domains is None or depth < args.depth or len(domains) < 2:
            return domains
        if mode == "forward":
            action = "skip"
        elif mode == "switch":
            action = "switch"
        else:
            action = choose(tree, cheap(depth, domains))
        counts["early_"+action] += 1
        if reject(pruner, depth, mapping, used, domains, action, args):
            return None
        return domains

    def late(depth, domains):
        nonlocal late_seconds
        start = perf_counter()
        counts["late_states"] += 1
        if observe_late is not None:
            observe_late(domains)
            return False
        if encoder is not None:
            counts["model_calls"] += 1
            if encoder(*features(space, domains, True)).item() < args.logit:
                late_seconds += perf_counter()-start
                return False
        counts["strong_checks"] += 1
        # AC-3 mutates its input. Keep incremental snapshots immutable on return.
        cut = not pruner.check_arcs(domains.copy()) or not pruner.covers_matching(domains)
        counts["late_cuts"] += cut
        late_seconds += perf_counter()-start
        return cut

    start = perf_counter()
    hook = NeuGNReorder(model, space.q, args.depth, args.device)
    callbacks = {} if mode == "neugn" else {"branch_domains": advance}
    if mode == "tree_late_nn" or observe_late is not None:
        callbacks.update(late_check=late, late_steps=args.late_steps if delay is None else delay,
                         late_depth=args.depth)
    result = enumerate_space(space, hook, max_matches=1, max_steps=args.budget, **callbacks)
    return dict(seconds=perf_counter()-start, steps=result.steps, fms=result.first_match_step,
                solved=bool(result.num_matches),
                late_seconds=late_seconds, **counts)


def train_encoder(spaces, graph, model, tree, args, rng):
    samples = []
    positives = 0
    collection = []
    # Prefer the deployment trajectory. Use extra training trajectories only
    # when it fails to expose both classes. Never look at test queries here.
    passes = [("tree", args.late_steps), ("forward", args.late_steps)]
    earlier = max(1, args.late_steps//2)
    if earlier != args.late_steps:
        passes.append(("forward", earlier))
    for policy, delay in passes:
        start = len(samples)
        start_positives = positives
        for space in tqdm(spaces, desc=f"Collect {policy} at +{delay}", disable=not args.progress):
            reservoir = []
            seen = 0

            def observe(domains):
                nonlocal seen
                seen += 1
                slot = seen-1 if seen <= args.late_samples else rng.randrange(seen)
                if slot == len(reservoir):
                    reservoir.append(domains)
                elif slot < len(reservoir):
                    reservoir[slot] = domains

            search(space, model, args, tree, policy, observe_late=observe, delay=delay)
            pruner = ResidualPruner(space, "forward")
            for domains in reservoir:
                cut = not pruner.check_arcs(domains.copy()) or not pruner.covers_matching(domains)
                samples.append((features(space, domains, True), float(cut)))
                positives += cut
        collection.append(dict(policy=policy, delay=delay, states=len(samples)-start,
                               proofs=positives-start_positives))
        print(f"Late labels: {positives} proofs / {len(samples)} states", flush=True)
        if min(positives, len(samples)-positives) >= args.min_class:
            break
    if min(positives, len(samples)-positives) < args.min_class:
        (args.out/"collection.json").write_text(json.dumps(collection, indent=2)+"\n")
        raise ValueError("Insufficient late labels for the fourth mode. See collection.json")
    net = CheckEncoder(int(max(graph.labels))+1, 32)
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(samples)-positives)/positives))
    with torch.enable_grad():
        order = list(range(len(samples)))
        for _ in tqdm(range(args.epochs), desc="Train mini-transformer", disable=not args.progress):
            rng.shuffle(order)
            for start in range(0, len(order), 32):
                batch = [samples[i] for i in order[start:start+32]]
                logits = torch.cat([net(*x) for x,_ in batch])
                opt.zero_grad()
                loss(logits,torch.tensor([y for _,y in batch])).backward()
                opt.step()
    return net.eval(), collection


def summarize_trials(records, modes):
    summary = {}
    for mode in modes:
        first = [r[mode][0] for r in records]
        fms = [r["fms"] for r in first]
        solved_fms = [f for f in fms if f is not None]
        median_fms = censored_median(fms)
        summary[mode] = dict(
            queries=len(records),
            seconds=sum(median(t["seconds"] for t in r[mode]) for r in records),
            solved=len(solved_fms),
            steps=sum(x["steps"] for x in first),
            median_fms=median_fms if isfinite(median_fms) else None,
            median_fms_solved=median(solved_fms) if solved_fms else None,
            median_ms=1000 * median(median(t["seconds"] for t in r[mode]) for r in records),
            late_seconds=sum(median(t["late_seconds"] for t in r[mode]) for r in records),
            **{key: sum(x.get(key, 0) for x in first) for key in
               ("early_skip", "early_edge", "early_manual", "early_switch",
                "late_states", "model_calls", "strong_checks", "late_cuts")},
        )
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint","query-dir","out"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--graph",default="wikics")
    p.add_argument("--device",default="cpu")
    p.add_argument("--sizes",type=int,nargs="+",default=[10,16,24,32])
    for key,value in dict(depth=10, threshold=256, proposals=12, fragment_budget=256,
                          train_offset=0,train_queries=100,test_offset=120,queries=40,samples=24,budget=10000,
                          late_samples=24,min_class=8,
                          branch_budget=2000,late_steps=128,repeats=3,epochs=6,seed=19).items():
        p.add_argument("--"+key.replace("_","-"),type=int,default=value)
    p.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--logit",type=float,default=0.)
    p.add_argument("--branch-seconds",type=float,default=.05)
    args=p.parse_args()
    if min(args.train_queries, args.queries, args.samples, args.late_samples, args.min_class,
           args.budget, args.late_steps, args.depth,
           args.repeats, args.epochs, args.branch_budget, args.branch_seconds) <= 0:
        p.error("positive counts and budgets required")
    if min(args.sizes) < 2 or len(args.sizes) != len(set(args.sizes)):
        p.error("use distinct query sizes of at least two vertices")
    if max(args.sizes) < args.depth + 2:
        p.error("include a query size with at least two unassigned vertices after the check depth")
    if args.test_offset < args.train_offset+args.train_queries or args.train_offset < 0:
        p.error("training and test queries must be disjoint")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng=Random(args.seed)
    graph=load(args.graph)
    model=load_model(args.checkpoint,args.device)
    if model.cfg.num_g != graph.n:
        p.error("checkpoint graph size mismatch")
    groups = {}
    query_hashes = {}
    for size in args.sizes:
        for density in ("dense", "sparse"):
            path = args.query_dir/f"size{size}_{density}.json"
            queries, metadata = load_query_set(path)
            if metadata.get("graph", args.graph) != args.graph or len(queries) < args.test_offset+args.queries:
                p.error(f"incompatible query set: {path}")
            groups[size,density] = queries
            query_hashes[path.name] = checkpoint_digest(path)
    train, test = [], []
    for (size, density), queries in groups.items():
        for i in range(args.test_offset, args.test_offset+args.queries):
            test.append((size, density, i, queries[i]))
    seen = {fingerprint(q) for _,_,_,q in test}
    for (size, density), queries in groups.items():
        if size < args.depth + 2:
            continue
        for i in range(args.train_offset, args.train_offset+args.train_queries):
            query = queries[i]
            key = fingerprint(query)
            if key not in seen:
                seen.add(key)
                train.append(search_space(query.graph, graph))
    if not train:
        p.error("no training queries remain after deduplication")
    args.out.mkdir(parents=True,exist_ok=True)
    if any(args.out.iterdir()):
        p.error("use an empty output directory")
    with torch.no_grad():
        tree,tree_samples=learn_tree(train,model,args,rng)
        encoder, collection = train_encoder(train, graph, model, tree, args, rng)
        torch.save(encoder.state_dict(), args.out/"mini_transformer.pt")
        modes = ("neugn", "switch", "tree", "tree_late_nn")
        records = []
        for size, density, index, query in tqdm(test, desc="Test four modes", disable=not args.progress):
            space = search_space(query.graph, graph)
            trials = {mode: [] for mode in modes}
            for _ in range(args.repeats):
                order = list(modes)
                rng.shuffle(order)
                for mode in order:
                    trials[mode].append(search(space, model, args, tree, mode,
                                              encoder=encoder if mode=="tree_late_nn" else None))
            records.append(dict(size=size,density=density,index=index,**trials))
    by_size = {
        n: summarize_trials([r for r in records if r["size"] == n], modes)
        for n in args.sizes
    }
    by_group = {
        f"{n}_{density}": summarize_trials(
            [r for r in records if r["size"] == n and r["density"] == density], modes
        )
        for n, density in groups
    }
    result=dict(params={k:str(v) for k,v in vars(args).items()},tree=tree,tree_samples=tree_samples,
                collection=collection, summary=summarize_trials(records, modes),
                by_size=by_size, by_group=by_group, records=records, train_queries=len(train),
                checkpoint_sha256=checkpoint_digest(args.checkpoint), query_sha256=query_hashes)
    (args.out/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("records", "by_group")},indent=2))


if __name__=="__main__":
    main()
