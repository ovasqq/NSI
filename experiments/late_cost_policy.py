"""Measure and learn the cost of exact checks on delayed DFS frames."""

from time import perf_counter

import numpy as np
from tqdm import tqdm

from experiments.residual_policy import fit_cost_ensemble, predict_cost_deltas
from search.pruning import ResidualPruner, hall_conflict
from search.residual_proofs import local_lad_conflict


LATE_ACTIONS = ("skip", "arc", "matching", "both")
PROOF_ACTIONS = (*LATE_ACTIONS, "lad", "cascade")
LATE_FEATURES = (
    "depth", "remaining_vertices", "delay", "min_width", "max_width",
    "sum_widths", "singleton_domains", "repeated_singleton_images",
)


def late_features(depth, domains, delay):
    widths = [len(domain) for domain in domains.values()]
    # Repeated singleton images are a cheap signal of an all-different conflict.
    singleton_images = [next(iter(domain)) for domain in domains.values() if len(domain) == 1]
    return (depth, len(domains), delay, min(widths), max(widths), sum(widths),
            len(singleton_images), len(singleton_images) - len(set(singleton_images)))


def late_conflict_reason(pruner, domains, action, proof_cache=None):
    reduced = domains
    if action in ("arc", "both", "cascade"):
        reduced = domains.copy()
        if not pruner.check_arcs(reduced):
            return "arc"
    if action in ("matching", "both", "cascade"):
        candidates = reduced if action == "cascade" else domains
        if proof_cache is None:
            if not pruner.covers_matching(candidates):
                return "matching"
        else:
            witness = hall_conflict(candidates)
            if witness is not None:
                proof_cache.add_hall(*witness)
                return "matching"
    if action in ("lad", "cascade") and local_lad_conflict(pruner, reduced):
        return "lad"
    return None


def collect_late_costs(spaces, search, model, tree, args, rng, delays, actions=LATE_ACTIONS):
    rows = []
    for delay in delays:
        begin = len(rows)
        sample_limit = args.full_depth_samples if delay == 0 else args.late_samples
        for index, space in enumerate(tqdm(spaces, desc=f"Measure late cost at +{delay}",
                                          disable=not args.progress)):
            samples = {}
            visits = {}

            def observe(depth, domains, step):
                # Early depths would be drowned out by deep states in a
                # single reservoir when collecting at branch entry.
                group = depth if delay == 0 else 0
                count = visits[group] = visits.get(group, 0) + 1
                slot = count - 1 if count <= sample_limit else rng.randrange(count)
                if slot >= sample_limit:
                    return None
                sample = samples.setdefault(group, [])
                row = dict(depth=depth, domains=domains, step=step)
                if slot == len(sample):
                    sample.append(row)
                else:
                    sample[slot] = row
                start = perf_counter()

                def finish(end_step, complete):
                    row.update(steps=end_step-step, remaining_us=(perf_counter()-start)*1e6,
                               censored=not complete)

                return finish

            # The search continues unchanged after the observer fires. Proofs
            # are evaluated later, outside the observed continuation.
            search(space, model, args, tree, "tree", delay=delay, late_observer=observe)
            pruner = ResidualPruner(space, "forward")
            for state in (state for sample in samples.values() for state in sample):
                domains = state.pop("domains")
                ordered = list(actions[1:])
                rng.shuffle(ordered)
                checks = {}
                for action in ordered:
                    start = perf_counter()
                    reason = late_conflict_reason(pruner, domains, action)
                    checks[action] = dict(cut=reason is not None, reason=reason,
                                          cost_us=(perf_counter()-start)*1e6)
                rows.append(dict(query=index, delay=delay, **state, checks=checks,
                                 features=late_features(state["depth"], domains, delay)))
        print(f"Check cost labels at +{delay}: {len(rows)-begin}", flush=True)
    return rows


def fit_late_costs(rows, step_value_us, *, proof_study=False):
    features = np.asarray([r["features"] for r in rows])
    variants = [
        ("late_binary_time", ("both",), 0.),
        ("late_multi_time", LATE_ACTIONS[1:], 0.),
        ("late_multi_steps", LATE_ACTIONS[1:], step_value_us),
    ]
    if proof_study:
        variants.append(("late_proof_learned", PROOF_ACTIONS[1:], step_value_us))
    return {name: fit_action_costs(rows, actions, value, features)
            for name, actions, value in variants}


def fit_action_costs(rows, actions, step_value_us, features=None):
    if features is None:
        features = np.asarray([r["features"] for r in rows])
    costs = np.asarray([
        [0.] + [
            r["checks"][action]["cost_us"] -
            (r["remaining_us"] + step_value_us*r["steps"] if r["checks"][action]["cut"] else 0.)
            for action in actions
        ] for r in rows
    ])
    return actions, fit_cost_ensemble(features, costs, rounds=16)


def learned_action(policy, features, min_savings_us=0.):
    actions, ensemble = policy
    deltas = predict_cost_deltas(ensemble, features)
    best = min(range(len(actions)), key=deltas.__getitem__)
    return actions[best] if deltas[best] < -min_savings_us else "skip"
