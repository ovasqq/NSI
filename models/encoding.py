"""Search state -> model input. The single place that defines NeuGN's token
format, shared by training (train/dataset.py) and inference (models/hook.py),
so the two cannot drift apart.

A state is (query Q, its Euler sequence, partial mapping, target vertex):
- content token of position j, for w = seq[j]:
    mapping[w]   if w is mapped (a vertex id of G),
    CLS          if w is the target (all occurrences),
    PAD          otherwise;
- query-vertex token: (w + shift) mod NUM_QVERT (cyclic shift, NeuGN);
- label token: label of w (our addition, switchable in the model).
"""

from __future__ import annotations

import numpy as np
import torch

from search.graph import Graph

NUM_QVERT = 64  # N in NeuGN's cyclic shift i' = (i + r) mod N


def pad_id(num_g: int) -> int:
    return num_g


def cls_id(num_g: int) -> int:
    return num_g + 1


def gcn_adjacency(q: Graph) -> np.ndarray:
    """Â = D^{-1/2} (A + I) D^{-1/2} (Kipf & Welling)."""
    a = np.eye(q.n, dtype=np.float32)
    for x, y in q.edges():
        a[x, y] = a[y, x] = 1.0
    d = a.sum(1) ** -0.5
    return a * d[:, None] * d[None, :]


def encode_query(q: Graph, seq: list[int], cand_count: np.ndarray | None = None) -> dict:
    """Parts of the input that do not depend on the mapping. `cand_count` (if
    given) is per query vertex |C(u)| — used by the inductive model only."""
    seq_arr = np.asarray(seq, dtype=np.int64)
    out = dict(
        seq=seq_arr,
        tok_label=q.labels[seq_arr],
        q_labels=q.labels.copy(),
        q_adj=gcn_adjacency(q),
    )
    if cand_count is not None:
        out["cand_count"] = np.asarray(cand_count, dtype=np.float32)
    return out


def encode_state(
    qenc: dict, mapping: list[int], target: int, shift: int, num_g: int
) -> dict:
    """`mapping[w]` = image of query vertex w or -1; `target` is a query vertex
    (its own mapping entry is ignored)."""
    seq = qenc["seq"]
    m = np.asarray(mapping, dtype=np.int64)
    content = np.where(m[seq] >= 0, m[seq], pad_id(num_g))
    content[seq == target] = cls_id(num_g)
    state = dict(
        qenc,
        content=content,
        qvert=(seq + shift) % NUM_QVERT,
    )
    if "cand_count" in qenc:
        state["cand_count"] = qenc["cand_count"][seq]  # per token
    return state


def collate(states: list[dict], targets: list[int] | None = None) -> dict[str, torch.Tensor]:
    """Pads a list of encoded states into a batch."""
    b = len(states)
    L = max(len(s["content"]) for s in states)
    n = max(len(s["q_labels"]) for s in states)
    has_cand = all("cand_count" in s for s in states)
    content = torch.zeros(b, L, dtype=torch.long)
    qvert = torch.zeros(b, L, dtype=torch.long)
    tok_label = torch.zeros(b, L, dtype=torch.long)
    tok_mask = torch.zeros(b, L, dtype=torch.bool)  # True = real token
    q_labels = torch.zeros(b, n, dtype=torch.long)
    q_adj = torch.zeros(b, n, n)
    q_mask = torch.zeros(b, n, dtype=torch.bool)
    cand_count = torch.zeros(b, L) if has_cand else None
    for i, s in enumerate(states):
        l, k = len(s["content"]), len(s["q_labels"])
        content[i, :l] = torch.from_numpy(s["content"])
        qvert[i, :l] = torch.from_numpy(s["qvert"])
        tok_label[i, :l] = torch.from_numpy(s["tok_label"])
        tok_mask[i, :l] = True
        q_labels[i, :k] = torch.from_numpy(s["q_labels"])
        q_adj[i, :k, :k] = torch.from_numpy(s["q_adj"])
        q_mask[i, :k] = True
        if has_cand:
            cand_count[i, :l] = torch.from_numpy(s["cand_count"])
    batch = dict(content=content, qvert=qvert, tok_label=tok_label, tok_mask=tok_mask,
                 q_labels=q_labels, q_adj=q_adj, q_mask=q_mask)
    if has_cand:
        batch["cand_count"] = cand_count
    if targets is not None:
        batch["target"] = torch.tensor(targets, dtype=torch.long)
    return batch
