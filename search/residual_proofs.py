"""Bounded exact pruning checks and reusable proofs for one DFS query."""

from collections import deque

from search.pruning import Domains, ResidualPruner, hall_conflict


class ProofCache:
    def __init__(self, limit=32):
        self.entries = deque(maxlen=limit)

    def add_fragment(self, vertices, domains):
        bounds = tuple((u, domains[u]) for u in sorted(vertices, key=lambda u: len(domains[u])))
        if sum(len(values) for _, values in bounds) <= 256:
            self.entries.append(("fragment", bounds))

    def add_hall(self, variables, images):
        self.entries.append(("hall", tuple((u, images) for u in variables)))

    def conflict(self, domains: Domains):
        for kind, bounds in reversed(self.entries):
            if all(u in domains and domains[u].issubset(bound) for u, bound in bounds):
                return kind
        return None


def local_lad_conflict(pruner: ResidualPruner, domains: Domains, centers=2, max_width=8,
                       offset=0):
    """A bounded LAD check. Reject only if a full center domain is ruled out."""
    q_adj, g_adj = pruner.space.q.adj_set, pruner.space.g.adj_set
    options = []
    for u, domain in domains.items():
        if len(domain) <= max_width:
            neighbors = tuple(v for v in q_adj[u] if v in domains)
            if len(neighbors) >= 2:
                options.append((len(domain), -len(neighbors), u, neighbors))
    for _, _, u, neighbors in sorted(options)[offset:offset+centers]:
        for image in domains[u]:
            local = {v: domains[v].intersection(g_adj[image]).difference((image,))
                     for v in neighbors}
            if hall_conflict(local) is None:
                break
        else:
            return True
    return False


def forest_conflict(space, domains: Domains):
    """Solve a spanning-forest relaxation with bottom-up support propagation."""
    unseen = set(domains)
    for root in sorted(domains, key=lambda u: (len(domains[u]), u)):
        if root not in unseen:
            continue
        unseen.remove(root)
        forest = [(root, None)]
        for u, _ in forest:
            children = space.q.adj_set[u] & unseen
            unseen.difference_update(children)
            forest.extend((v, u) for v in sorted(children))
        viable = {u: domains[u] for u, _ in forest}
        for u, parent in reversed(forest):
            if parent is not None:
                viable[parent] = frozenset(
                    v for v in viable[parent] if not space.g.adj_set[v].isdisjoint(viable[u])
                )
                if not viable[parent]:
                    return True
    return False


def triangle_edge_conflict(space, domains: Domains, triangle, budget):
    """Find a triangle by joining compatible pairs along its three edges."""
    u, v, w = sorted(triangle, key=lambda x: (len(domains[x]), x))
    neighbors = space.g.adj_set
    for a in domains[u]:
        left = domains[v] & neighbors[a]
        right = domains[w] & neighbors[a]
        for b in left:
            if budget == 0:
                return None
            budget -= 1
            if not right.isdisjoint(neighbors[b]):
                return False
        # Charge the two neighborhood intersections even in empty rows.
        if budget == 0:
            return None
        budget -= 1
    return True
