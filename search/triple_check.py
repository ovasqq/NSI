"""Budgeted exact three-variable non-induced CSP on a resident WikiCS adjacency."""
from math import prod
import torch


class TripleCheck:
    def __init__(self, graph, device, budget):
        if budget < 1:
            raise ValueError("triple budget must be positive")
        self.budget = budget
        self.device = device
        self.adj = torch.zeros((graph.n, graph.n), dtype=torch.bool, device=device)
        edges = [(u, v) for u, neighbors in enumerate(graph.adj) for v in neighbors]
        if edges:
            indices = torch.tensor(edges, dtype=torch.long, device=device)
            self.adj[indices[:, 0], indices[:, 1]] = True

    @torch.no_grad()
    def many(self, space, domains, triples):
        """Batch candidate triples for one DFS snapshot, with an independent tuple budget."""
        if not triples:
            return []
        orders = [sorted(triple, key=lambda u: (len(domains[u]), u)) for triple in triples]
        if any(len(set(row)) != 3 or len(row) != 3 for row in orders):
            raise ValueError("expected three distinct query vertices")
        widths = [[len(domains[u]) for u in row] for row in orders]
        if any(min(w) == 0 for w in widths):
            return [False if min(w) == 0 else self.many(space, domains, [row])[0]
                    for w, row in zip(widths, orders)]
        totals = [prod(w) for w in widths]
        count = min(self.budget, max(totals))
        idx = torch.arange(count, device=self.device)[None,:]
        w = torch.tensor(widths, device=self.device)
        offsets = (idx // (w[:,1:2]*w[:,2:3]), idx // w[:,2:3] % w[:,1:2], idx % w[:,2:3])
        images = []
        for axis, indices in enumerate(offsets):
            # Only the first count values of any domain can be visited by this prefix.
            values = [sorted(domains[row[axis]])[:count] for row in orders]
            width = max(map(len, values))
            matrix = torch.tensor([v+[0]*(width-len(v)) for v in values], device=self.device)
            images.append(matrix.gather(1, indices.clamp(max=width-1)))
        valid = idx < torch.tensor(totals, device=self.device)[:,None]
        for i,j in ((0,1),(0,2),(1,2)):
            edge = torch.tensor([row[i] in space.q.adj_set[row[j]] for row in orders],
                                device=self.device)[:,None]
            valid = valid & (images[i] != images[j]) & (~edge | self.adj[images[i],images[j]])
        return [True if found else False if total <= count else None
                for found,total in zip(valid.any(1).tolist(), totals)]
