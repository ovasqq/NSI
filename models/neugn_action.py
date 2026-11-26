"""Small CPU action head for a frozen NeuGN ranker."""

from dataclasses import dataclass
from math import log1p

import numpy as np
import torch
from torch import nn

ACTIONS = ("matching", "lad", "forest")


def inputs(depth, parent, child, image):
    """Use fresh forward domains and context computed by the parent rank call."""
    before = parent.domains
    widths = [len(domain) for domain in child.domains.values()]
    total_before = sum(len(before[u]) for u in child.domains)
    context = parent.ranking_context
    anchor, scores = context if context is not None else (None, {})
    score = scores.get(image)
    features = np.asarray([
        log1p(depth), log1p(len(widths)), log1p(min(widths)), log1p(max(widths)),
        log1p(sum(widths)), log1p(total_before-sum(widths)),
        sum(width == 1 for width in widths),
        sum(len(before[u]) > 1 and len(domain) == 1 for u, domain in child.domains.items()),
        np.arcsinh(score) if score is not None else 0., score is not None,
    ], dtype=np.float32)
    return features, anchor


def features_for_head(features, anchor, anchor_width):
    if not anchor_width:
        return features
    return np.concatenate((features, anchor if anchor is not None else
                           np.zeros(anchor_width, dtype=np.float32)))


@dataclass
class ActionHead:
    anchor_width: int
    mean: np.ndarray
    scale: np.ndarray
    weights: tuple

    def costs(self, features, anchor):
        x = features_for_head(features, anchor, self.anchor_width)
        first_weight, first_bias, last_weight, last_bias = self.weights
        hidden = np.maximum(first_weight @ ((x-self.mean)/self.scale) + first_bias, 0.)
        return last_weight @ hidden + last_bias


def fit_head(rows, anchor_width, epochs, seed, device):
    x = np.stack([features_for_head(features, anchor, anchor_width)
                  for features, anchor, _ in rows])
    y = np.asarray([costs for _, _, costs in rows], dtype=np.float32)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), .01)
    x = torch.tensor((x-mean)/scale, device=device)
    y = torch.tensor(np.sign(y)*np.log1p(np.abs(y)/10), device=device)
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(x.shape[1], 24), nn.ReLU(), nn.Linear(24, len(ACTIONS))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for batch in np.array_split(rng.permutation(len(x)), (len(x)+127)//128):
            loss = nn.functional.smooth_l1_loss(model(x[batch]), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    weights = tuple(p.detach().cpu().numpy().copy() for p in model.parameters())
    return ActionHead(anchor_width, mean, scale, weights)
