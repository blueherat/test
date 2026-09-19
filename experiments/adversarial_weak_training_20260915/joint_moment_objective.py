"""Match image/condition joint moments, retaining the condition's fixed identity."""

import torch
from torch.nn import functional as F
from experiments.advfd_cleanroom.core import (
    batch_moments, fit_calibration_from_moments,
    frechet_from_moments,
)
from .moment_objective import MomentFeedback, gathered_detached


def joint_calibrated_frechet(real, generated, labels, classes):
    assert real.shape == generated.shape and len(labels) == len(real)
    assert labels.dtype == torch.long and 0 <= labels.min() and labels.max() < classes
    real_moments = batch_moments(real.double()).detached()
    calibration = fit_calibration_from_moments(real_moments, mode='real',
        epsilon=1e-3, detach_statistics=True)
    real = (real.double() - calibration.center) @ calibration.transform
    generated = (generated.double() - calibration.center) @ calibration.transform
    identity = F.one_hot(labels, classes).to(torch.float64)
    prior = identity.mean(0)
    # P/R have exactly the same paired labels in the memory. Each condition
    # coordinate has its natural inverse-frequency scale, with no tuned weight.
    identity = (identity - prior) / prior.clamp_min(1. / len(labels)).sqrt()
    p = torch.cat((real, identity), dim=1)
    r = torch.cat((generated, identity), dim=1)
    return frechet_from_moments(batch_moments(p).detached(), batch_moments(r),
        covariance_jitter=1e-6)


class JointMomentFeedback(MomentFeedback):
    def __init__(self, capacity=2048, classes=100):
        super().__init__(capacity)
        self.classes = classes
        self.labels = None

    def append(self, real, generated, labels=None):
        assert labels is not None and len(labels) == len(real)
        super().append(real, generated)
        labels = gathered_detached(labels)
        if self.labels is not None:
            labels = torch.cat((self.labels, labels))
        self.labels = labels[-self.capacity:].detach()
        assert len(self.labels) == self.size

    def distance(self, real, generated):
        return joint_calibrated_frechet(real, generated, self.labels, self.classes)

    def state_dict(self):
        return dict(super().state_dict(), labels=self.labels.cpu(), classes=self.classes,
                    objective='joint_image_condition_frechet')

    def load_state_dict(self, state, device='cuda'):
        assert state['classes'] == self.classes and 'labels' in state
        super().load_state_dict(state, device)
        self.labels = state['labels'].to(device)
        assert len(self.labels) == self.size
