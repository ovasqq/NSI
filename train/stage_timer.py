"""Opt-in synchronized stage timings for a short diagnostic training run."""

from contextlib import contextmanager
from time import perf_counter

import torch


class StageTimer:
    def __init__(self, device, enabled=False):
        self.device = torch.device(device)
        self.enabled = enabled
        self.seconds = {}

    @contextmanager
    def stage(self, name):
        if not self.enabled:
            yield
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = perf_counter()
        try:
            yield
        finally:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.seconds[name] = self.seconds.get(name, 0.0) + perf_counter() - start
