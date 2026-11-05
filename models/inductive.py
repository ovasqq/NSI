"""Inductive candidate scorer (idea 2).

Same query-state encoding as NeuGN (the `content` field still carries a G
vertex id, or PAD/CLS), but the model interprets `content` through a
*feature encoder* instead of an id-embedding table, and the output is a
per-candidate bilinear score instead of a softmax over V(G). Nothing is tied
to a specific data graph G except the (cheap, recomputable) raw features:
label, log-degree and the neighbour-label histogram.

Outputs:
  * `forward(batch, h_q)` -> s (B, d)  : query-state vector (anchor output);
  * `encode_vertex(vids)` -> f (…, d)  : feature embedding of G vertices;
  * `score(s, vids)` -> (B, K)         : logit of p(has_solution) per candidate.

`p̂ = sigmoid(score)`. At inference candidates are sorted by `score` (sigmoid is
monotone), so completeness is untouched (permutation only).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from models.encoding import NUM_QVERT
from models.neugn import QSExtractor
from search.graph import Graph


@dataclass
class InductiveConfig:
    num_labels: int
    d: int = 256
    layers: int = 2
    heads: int = 8
    ffn: int = 1024
    gcn_layers: int = 2
    dropout: float = 0.1
    max_len: int = 512
    feat_dim: int = 64
    token_labels: bool = True
    use_gcn: bool = True
    use_degree: bool = True
    use_nlf: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def g_features(g: Graph, num_labels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw, graph-independent features of every vertex: (labels, log-degree,
    neighbour-label histogram normalized by degree). Recomputed on demand, so
    no vertex-level parameters are stored."""
    labels = g.labels.astype(np.int64)
    deg = g.degree.astype(np.float32)
    logdeg = np.log1p(deg)
    nlf = np.zeros((g.n, num_labels), dtype=np.float32)
    for v in range(g.n):
        counts = np.bincount(labels[g.adj[v]], minlength=num_labels)
        nlf[v] = counts / max(float(deg[v]), 1.0)
    return labels, logdeg, nlf


class FeatureEncoder(nn.Module):
    """Raw G-vertex features -> f(v) in R^d. Shared by the token content and
    the candidate-scoring head, so it is the single inductive interface to G."""

    def __init__(self, num_labels: int, d: int, feat_dim: int = 64,
                 use_degree: bool = True, use_nlf: bool = True):
        super().__init__()
        self.label_emb = nn.Embedding(num_labels, feat_dim)
        self.use_degree = use_degree
        self.nlf_proj = nn.Linear(num_labels, feat_dim) if use_nlf else None
        in_dim = feat_dim + (1 if use_degree else 0) + (feat_dim if use_nlf else 0)
        self.mlp = nn.Sequential(nn.Linear(in_dim, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, label_idx: torch.Tensor, logdeg: torch.Tensor,
                nlf: torch.Tensor) -> torch.Tensor:
        parts = [self.label_emb(label_idx)]
        if self.use_degree:
            parts.append(logdeg.unsqueeze(-1))
        if self.nlf_proj is not None:
            parts.append(self.nlf_proj(nlf))
        return self.mlp(torch.cat(parts, dim=-1))


class InductiveNeuGN(nn.Module):
    def __init__(self, cfg: InductiveConfig, g_label: np.ndarray,
                 g_logdeg: np.ndarray, g_nlf: np.ndarray):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.num_g = int(g_label.shape[0])
        self.qs = QSExtractor(cfg) if cfg.use_gcn else None
        self.feat = FeatureEncoder(cfg.num_labels, d, cfg.feat_dim,
                                   cfg.use_degree, cfg.use_nlf)
        self.pad_vec = nn.Parameter(torch.zeros(d))
        self.cls_vec = nn.Parameter(torch.zeros(d))
        self.qvert_emb = nn.Embedding(NUM_QVERT, d)
        self.pos_emb = nn.Embedding(cfg.max_len + 2, d)
        self.label_emb = nn.Embedding(cfg.num_labels, d) if cfg.token_labels else None
        self.cand_proj = nn.Linear(1, d)  # log1p(|C(u)|) scalar -> d
        self.anchor = nn.Parameter(torch.randn(d) * 0.02)
        self.no_query = nn.Parameter(torch.zeros(d))
        self.norm_in = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(
            d, cfg.heads, cfg.ffn, cfg.dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        # bilinear candidate head: score(s, f) = (s W) · f + b
        self.W = nn.Parameter(torch.randn(d, d) * 0.02)
        self.bias = nn.Parameter(torch.zeros(1))

        self.register_buffer("g_label", torch.from_numpy(g_label))
        self.register_buffer("g_logdeg", torch.from_numpy(g_logdeg))
        self.register_buffer("g_nlf", torch.from_numpy(g_nlf))

    def query_vector(self, batch) -> torch.Tensor:
        b = batch["content"].shape[0]
        if self.qs is None:
            return self.no_query.expand(b, -1)
        return self.qs(batch["q_labels"], batch["q_adj"], batch["q_mask"])

    def _content_features(self, content: torch.Tensor) -> torch.Tensor:
        """content (B, L): G vertex ids, PAD = num_g, CLS = num_g + 1 -> f (B, L, d)."""
        is_pad = content == self.num_g
        is_cls = content == self.num_g + 1
        vid = content.clamp(max=self.num_g - 1)
        f = self.feat(self.g_label[vid], self.g_logdeg[vid], self.g_nlf[vid])
        f = torch.where(is_pad[..., None], self.pad_vec.expand_as(f), f)
        f = torch.where(is_cls[..., None], self.cls_vec.expand_as(f), f)
        return f

    def forward(self, batch, h_q: torch.Tensor | None = None) -> torch.Tensor:
        """Query-state vector s = (B, d) at the anchor position."""
        content = batch["content"]
        b, L = content.shape
        if L + 2 > self.pos_emb.num_embeddings:
            raise ValueError(f"sequence of length {L} exceeds max_len={self.cfg.max_len}")
        if h_q is None:
            h_q = self.query_vector(batch)
        tok = self._content_features(content) + self.qvert_emb(batch["qvert"])
        if self.label_emb is not None:
            tok = tok + self.label_emb(batch["tok_label"])
        if batch.get("cand_count") is not None:
            c = batch["cand_count"].float().log1p().unsqueeze(-1)
            tok = tok + self.cand_proj(c)
        x = torch.cat([h_q[:, None], self.anchor.expand(b, 1, -1), tok], dim=1)
        x = x + self.pos_emb(torch.arange(L + 2, device=x.device))[None]
        x = self.norm_in(x)
        keep = torch.cat(
            [torch.ones(b, 2, dtype=torch.bool, device=x.device), batch["tok_mask"]], dim=1
        )
        x = self.encoder(x, src_key_padding_mask=~keep)
        return x[:, 1]

    def encode_vertex(self, vids: torch.Tensor) -> torch.Tensor:
        """Feature embedding f(v) for G vertex ids `vids` (any shape) -> (..., d)."""
        return self.feat(self.g_label[vids], self.g_logdeg[vids], self.g_nlf[vids])

    def score(self, s: torch.Tensor, vids: torch.Tensor) -> torch.Tensor:
        """Logits p(has_solution) for candidate vertices `vids` (B, K) given
        state vectors `s` (B, d)."""
        f = self.encode_vertex(vids)
        return torch.einsum("bd,bkd->bk", s @ self.W, f) + self.bias
