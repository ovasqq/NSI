"""Candidate-conditioned output for the decision to run exact pruning."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from models.pruning_controller import checkpoint_digest, eligible


class CandidateCheckHead(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.state = nn.Linear(width, hidden)
        self.candidate = nn.Linear(width, hidden)
        self.out = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, state: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        state = self.state(state)[:, None, :]
        candidate = self.candidate(candidates)
        inputs = torch.cat((state.expand_as(candidate), candidate, state * candidate), dim=-1)
        return self.out(inputs).squeeze(-1)


class CandidateGate:
    def __init__(self, path: Path, backbone: Path, graph: str, max_depth: int):
        self.saved = torch.load(path, map_location="cpu", weights_only=True)
        if self.saved["backbone_digest"] != checkpoint_digest(backbone):
            raise ValueError("pruning gate was trained with a different NeuGN checkpoint")
        if (self.saved["graph"], self.saved["max_depth"]) != (graph, max_depth):
            raise ValueError("pruning gate requires the graph and max depth used for training")
        self.threshold = self.saved["threshold"]

    def attach(self, model):
        if self.saved["encoder_layer"] is not None:
            model.encoder.layers[-1].load_state_dict(self.saved["encoder_layer"])
        head = CandidateCheckHead(model.cfg.d, self.saved["hidden"])
        head.load_state_dict(self.saved["head"])
        return head.to(model.out.weight.device).eval()

    @staticmethod
    def for_query(hook, _g, _q):
        def should_check(branch):
            if not eligible(hook, branch):
                return True
            decision = hook.check_decisions[branch.depth][branch.rank]
            hook.gate_decisions += 1
            hook.gate_skips += int(not decision)
            return decision

        return should_check
