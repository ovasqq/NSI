"""NeuGN: QSExtractor (GCN over Q) + GGNavigator (bidirectional Transformer
over the Euler sequence). Output: logits over all vertices of G for the
vertex marked CLS.

Input sequence: [h_Q] [anchor] tok_1 … tok_L, where
tok_j = content_emb + qvert_emb + pos_emb (+ label_emb if token_labels).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from models.encoding import NUM_QVERT


@dataclass
class NeuGNConfig:
    num_g: int  # |V(G)|: size of the ID vocabulary and of the output layer
    num_labels: int
    d: int = 256
    layers: int = 2  # paper: 2 for Hamster/LastFM, 4 otherwise
    heads: int = 8
    ffn: int = 1024
    gcn_layers: int = 2
    dropout: float = 0.1
    max_len: int = 512  # max Euler-sequence length (+2 for h_Q and anchor)
    token_labels: bool = True
    use_gcn: bool = True  # False = ablation "without GCN"

    def to_dict(self) -> dict:
        return asdict(self)


class QSExtractor(nn.Module):
    """Label embedding -> GCN layers -> max-pooling -> h_Q."""

    def __init__(self, cfg: NeuGNConfig):
        super().__init__()
        self.label_emb = nn.Embedding(cfg.num_labels, cfg.d)
        self.layers = nn.ModuleList(nn.Linear(cfg.d, cfg.d) for _ in range(cfg.gcn_layers))

    def forward(self, q_labels, q_adj, q_mask):
        h = self.label_emb(q_labels)  # (B, n, d)
        for i, lin in enumerate(self.layers):
            h = q_adj @ lin(h)
            if i + 1 < len(self.layers):
                h = torch.relu(h)
        h = h.masked_fill(~q_mask[..., None], float("-inf"))
        return h.max(dim=1).values  # (B, d)


class NeuGN(nn.Module):
    def __init__(self, cfg: NeuGNConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.qs = QSExtractor(cfg) if cfg.use_gcn else None
        self.content_emb = nn.Embedding(cfg.num_g + 2, d)  # + PAD, CLS
        self.qvert_emb = nn.Embedding(NUM_QVERT, d)
        self.pos_emb = nn.Embedding(cfg.max_len + 2, d)
        self.label_emb = nn.Embedding(cfg.num_labels, d) if cfg.token_labels else None
        self.anchor = nn.Parameter(torch.randn(d) * 0.02)
        self.no_query = nn.Parameter(torch.zeros(d))  # stands in for h_Q without GCN
        self.norm_in = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(
            d, cfg.heads, cfg.ffn, cfg.dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, cfg.num_g)

    def query_vector(self, batch) -> torch.Tensor:
        b = batch["content"].shape[0]
        if self.qs is None:
            return self.no_query.expand(b, -1)
        return self.qs(batch["q_labels"], batch["q_adj"], batch["q_mask"])

    def forward(self, batch, h_q: torch.Tensor | None = None) -> torch.Tensor:
        """Returns logits (B, |V(G)|). `h_q` may be passed in precomputed
        (it depends only on Q, so search computes it once per query)."""
        return self.out(self.state_vector(batch, h_q))

    def rank_and_check(
        self, batch, candidates: torch.Tensor, check_head: nn.Module,
        h_q: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Ranking logits over G and check logits for the requested (B, C) IDs."""
        state = self.state_vector(batch, h_q)
        ranks = self.out(state)
        checks = check_head(state, self.content_emb(candidates))
        return ranks, checks

    def state_vector(self, batch, h_q: torch.Tensor | None = None) -> torch.Tensor:
        """Transformer anchor before the vertex-ranking projection."""
        content = batch["content"]
        b, L = content.shape
        if L + 2 > self.pos_emb.num_embeddings:
            raise ValueError(f"sequence of length {L} exceeds max_len={self.cfg.max_len}")
        if h_q is None:
            h_q = self.query_vector(batch)

        tok = self.content_emb(content) + self.qvert_emb(batch["qvert"])
        if self.label_emb is not None:
            tok = tok + self.label_emb(batch["tok_label"])
        x = torch.cat([h_q[:, None], self.anchor.expand(b, 1, -1), tok], dim=1)
        x = x + self.pos_emb(torch.arange(L + 2, device=x.device))[None]
        x = self.norm_in(x)

        keep = torch.cat(
            [torch.ones(b, 2, dtype=torch.bool, device=x.device), batch["tok_mask"]], dim=1
        )
        x = self.encoder(x, src_key_padding_mask=~keep)
        return x[:, 1]
