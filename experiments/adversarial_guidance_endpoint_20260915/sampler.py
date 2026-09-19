"""Differentiable versions of the existing SiT/JiT IG solver steps.

The half-window decision for both Heun evaluations uses the original left step
boundary, exactly as the deployed sampler does. The frozen strong model remains
in the input-gradient graph, including all steps after guidance switches off.
"""

import torch
from experiments.guidance_dynamic_50k_20260915 import sampling


class GuidanceSteps:
    def __init__(self, adapter, head, labels, coefficient, method="guided_weak",
                 start=0, stop=None):
        self.adapter, self.head = adapter, head
        self.labels = labels.long()
        self.coefficient, self.method = coefficient, method
        self.total = 64 if adapter.name == "sit_small" else 100
        self.start, self.stop = start, self.total if stop is None else stop
        if not 0 <= self.start < self.stop <= self.total:
            raise ValueError("Invalid sampler interval")
        spec = sampling.k.ARMS.get(method)
        if spec is None or spec["base"] != "strong" or spec["loss"] not in sampling.k.WEAK_LOSSES:
            raise ValueError("This prototype supports a weak head on the strong IG base")
        self.grid = torch.linspace(0, 1, self.total + 1, device=labels.device)

    def __len__(self):
        return self.stop - self.start

    def __call__(self, index, state):
        index += self.start
        if not self.start <= index < self.stop:
            raise IndexError(index)
        t, u = self.grid[index], self.grid[index + 1]
        self.adapter.labels = self.labels
        with self.adapter.autocast():
            velocity = sampling.field(self.adapter, self.head, self.method,
                                      self.coefficient, state, t, float(t))
            predicted = state + (u - t) * velocity
            if self.adapter.name == "sit_small":
                second = sampling.field(self.adapter, self.head, self.method,
                                        self.coefficient, predicted, u, float(t))
                return state + ((u - t) / 2) * (velocity + second)
            return predicted


class ReverseFrozenSuffix:
    """Backward Heun approximation, NOT the exact inverse of forward Heun.

    Use only for an inversion/cycle-error preflight. The exact pullback identity
    in the research note requires an invertible continuation and its inverse.
    """

    def __init__(self, adapter, labels, boundary=32):
        if adapter.name != "sit_small":
            raise ValueError("The current inversion preflight is SiT-only")
        self.adapter, self.labels = adapter, labels.long()
        self.boundary = boundary
        self.grid = torch.linspace(0, 1, 65, device=labels.device)

    def __len__(self):
        return 64 - self.boundary

    def __call__(self, index, state):
        index = 64 - index
        t, u = self.grid[index], self.grid[index - 1]
        self.adapter.labels = self.labels
        with self.adapter.autocast():
            velocity = sampling.field(self.adapter, None, "strong", 0., state, t, float(t))
            predicted = state + (u - t) * velocity
            second = sampling.field(self.adapter, None, "strong", 0., predicted, u, float(t))
            return state + ((u - t) / 2) * (velocity + second)
