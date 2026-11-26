"""Synchronous batched arc consistency for sparse graph adjacency."""

import numpy as np
import torch


def sparse_adjacency(graph):
    columns = np.concatenate(graph.adj).astype(np.int64, copy=False)
    offsets = np.r_[0, np.cumsum([len(row) for row in graph.adj], dtype=np.int64)]
    return torch.sparse_csr_tensor(offsets, columns, torch.ones(len(columns)),
                                   size=(graph.n, graph.n))


def pack_states(states, size, graph_size):
    domains = torch.zeros((len(states), size, graph_size), dtype=torch.bool)
    edges = torch.zeros((len(states), size, size))
    active = torch.zeros((len(states), size), dtype=torch.bool)
    for i, (pruner, state) in enumerate(states):
        for u, candidates in state.items():
            domains[i, u, list(candidates)] = True
            active[i, u] = True
            for v in pruner.space.q.adj_set[u] & state.keys():
                edges[i, u, v] = 1
    return domains, edges, active


def propagate(graph, domains, query_edges, active):
    """Apply arc revisions in parallel until each state fails or stabilizes."""
    batch, size, graph_size = domains.shape
    rejected = torch.zeros(batch, dtype=torch.bool, device=domains.device)
    while True:
        columns = domains.reshape(batch * size, graph_size).T.contiguous().float()
        supports = torch.sparse.mm(graph, columns)
        unsupported = (supports.T.reshape(batch, size, graph_size) == 0).float()
        revised = domains & ((query_edges @ unsupported) == 0)
        rejected |= (active & ~revised.any(dim=-1)).any(dim=-1)
        revised = torch.where(rejected[:, None, None], domains, revised)
        if torch.equal(revised, domains):
            return revised, rejected
        domains = revised



def unpack_domains(states, masks, rejected):
    """Copy masks to CPU and restore domains for states that reached a fixpoint."""
    masks = masks.cpu().numpy()
    domains_by_state = []
    for state, state_masks, cut in zip(states, masks, rejected.cpu().tolist(), strict=True):
        if cut:
            domains_by_state.append(None)
            continue
        reduced = {}
        for u, old in state.items():
            images = np.flatnonzero(state_masks[u])
            reduced[u] = old if len(images) == len(old) else frozenset(images.tolist())
        domains_by_state.append(reduced)
    return domains_by_state
