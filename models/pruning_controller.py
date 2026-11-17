"""CPU readout of expected savings from the exact domains check."""

from __future__ import annotations

from hashlib import sha256
from math import log1p
from pathlib import Path

import torch
from torch import nn

from models.hook import NeuGNReorder
from search.enumerate import BranchState
from search.graph import Graph

FEATURE_COUNT = 12


def checkpoint_digest(path: Path) -> str:
    with path.open("rb") as stream:
        digest = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def eligible(hook: NeuGNReorder, branch: BranchState) -> bool:
    # The enumerator calls the ranker at this parent only with multiple candidates.
    return branch.width > 1 and branch.depth < hook.max_depth


def branch_features(
    hook: NeuGNReorder, g: Graph, q: Graph, branch: BranchState, use_anchor: bool
) -> torch.Tensor:
    anchor, scores = hook.ranking_context[branch.depth]
    score = scores[branch.rank]
    values = torch.tensor([
        q.n, q.m / max(1, q.n * (q.n - 1) / 2),
        (branch.depth + 1) / q.n, log1p(q.degree[branch.vertex]),
        log1p(g.degree[branch.candidate]), log1p(branch.width),
        branch.rank / (branch.width - 1), log1p(branch.step),
        log1p(branch.steps_remaining or 0), log1p(branch.backtracks),
        log1p(branch.last_deadend_steps), float(score - scores[0]),
    ], dtype=torch.float32)
    if use_anchor:
        if anchor is None:
            raise ValueError("the NeuGN hook must capture the Transformer anchor")
        return torch.cat((values, anchor))
    return values


class PruningHead(nn.Module):
    def __init__(self, input_size: int, hidden: int):
        super().__init__()
        self.register_buffer("center", torch.zeros(input_size))
        self.register_buffer("scale", torch.ones(input_size))
        self.readout = (nn.Sequential(nn.Linear(input_size, hidden), nn.ReLU(), nn.Linear(hidden, 1))
                        if hidden else nn.Linear(input_size, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.readout((inputs - self.center) / self.scale).squeeze(-1)


class LearnedPruning:
    def __init__(self, checkpoint: Path, backbone: Path):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["backbone_digest"] != checkpoint_digest(backbone):
            raise ValueError("controller was trained with a different NeuGN checkpoint")
        self.head = PruningHead(saved["input_size"], saved["hidden"])
        self.head.load_state_dict(saved["state_dict"])
        self.head.eval()
        self.use_anchor = saved["use_anchor"]
        self.threshold = saved["threshold"]

    def for_query(self, hook: NeuGNReorder, g: Graph, q: Graph, *, check_unranked: bool = False):
        def should_check(branch: BranchState) -> bool:
            if not eligible(hook, branch):
                return check_unranked
            with torch.inference_mode():
                features = branch_features(hook, g, q, branch, self.use_anchor)
                return bool(self.head(features).item() > self.threshold)

        return should_check
