"""Distribution feedback in the current source critic's learned representation.

This is second-order feature matching, not a reproduction of AdvFD: the source
critic is still trained by CE. Only the weak generator objective changes.
Cache fixed Inception features and re-encode the whole cache with the CURRENT
critic, so historical trainable representations never mix coordinate systems.
"""

import torch
import torch.distributed as dist
from torch.nn import functional as F
from experiments.advfd_cleanroom.core import (
    batch_moments, fit_calibration_from_moments, calibrate_moments,
    frechet_from_moments,
)


def gathered_detached(value):
    if not dist.is_initialized():
        return value.detach()
    parts = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, value.detach().contiguous())
    return torch.cat(parts)


def representation(critic, features):
    """Exactly the existing critic's penultimate representation, in eval mode."""
    assert not critic.training
    values = (features - critic.feature_mean) / critic.feature_std
    values = F.leaky_relu(critic.first(values), .2)
    return F.leaky_relu(critic.second(values), .2)


def calibrated_frechet(real, generated):
    """Literal common real calibration; no identity-target substitution."""
    real_moments = batch_moments(real.double()).detached()
    fake_moments = batch_moments(generated.double())
    calibration = fit_calibration_from_moments(real_moments, mode="real",
        epsilon=1e-3, detach_statistics=True)
    components = frechet_from_moments(calibrate_moments(real_moments, calibration),
        calibrate_moments(fake_moments, calibration), covariance_jitter=1e-6)
    return components


class MomentFeedback:
    def __init__(self, capacity=2048):
        self.capacity = capacity
        self.real = None
        self.generated = None

    @property
    def size(self):
        return 0 if self.real is None else len(self.real)

    def append(self, real, generated, labels=None):
        real = gathered_detached(real)
        generated = gathered_detached(generated)
        assert real.shape == generated.shape
        if self.real is not None:
            real = torch.cat((self.real, real))
            generated = torch.cat((self.generated, generated))
        self.real = real[-self.capacity:].detach()
        self.generated = generated[-self.capacity:].detach()

    def loss(self, critic, fresh_local_generated):
        """Call after appending this step's P/R batch, with critic frozen/eval.

        Only this rank's fresh rows retain a graph. Manual averaging of head
        gradients needs a world-size factor. Multiplying by memory/global-batch
        prevents the surrogate gradient from shrinking with cache length. The
        reported value remains the FD of the current rolling memory, not a
        claim of an unbiased FD estimate of the current full distribution.
        """
        assert self.size == self.capacity
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        count = len(fresh_local_generated)
        assert count * world <= self.capacity
        start = self.capacity - count * world + count * rank
        cached = torch.cat((self.generated[:start], fresh_local_generated,
                            self.generated[start + count:]))
        with torch.no_grad():
            real = representation(critic, self.real)
        fake = representation(critic, cached)
        components = self.distance(real, fake)
        value = components.total
        gradient_scale = self.capacity / count
        surrogate = value.detach() + gradient_scale * (value - value.detach())
        return surrogate, dict(moment_fd=float(value.detach()),
            moment_mean=float(components.mean.detach()),
            moment_covariance=float(components.covariance.detach()),
            moment_memory=self.size, moment_gradient_scale=gradient_scale)

    def distance(self, real, generated):
        return calibrated_frechet(real, generated)

    def state_dict(self):
        return dict(capacity=self.capacity, real=self.real.cpu(), generated=self.generated.cpu())

    def load_state_dict(self, state, device="cuda"):
        assert state["capacity"] == self.capacity
        self.real = state["real"].to(device)
        self.generated = state["generated"].to(device)
