"""Fresh paired conditional energy score; no Gaussian or replay approximation."""

import torch
from .moment_objective import representation


def conditional_energy_score(real, generated, labels):
    """Unbiased fixed-feature score from two independent draws per class.

    The value omits the P-only constant. It therefore need not be nonnegative
    or zero when P=R. The R-R term has its proper-score coefficient, 1/2.
    Both generated members retain gradients; P is treated as fixed data.
    """
    assert real.shape == generated.shape and real.ndim == 2
    assert len(real) == len(labels) and len(real) % 2 == 0
    assert torch.equal(labels[::2], labels[1::2])
    p = real.detach().reshape(-1, 2, real.shape[-1])
    r = generated.reshape(-1, 2, generated.shape[-1])
    cross = torch.linalg.vector_norm(r[:, :, None] - p[:, None], dim=-1).mean()
    within = torch.linalg.vector_norm(r[:, 0] - r[:, 1], dim=-1).mean()
    return cross - .5 * within, dict(energy_cross=float(cross.detach()),
                                   energy_within=float(within.detach()))


def critic_energy_score(critic, real_features, generated_features, labels):
    assert not critic.training and not any(p.requires_grad for p in critic.parameters())
    with torch.no_grad():
        real = representation(critic, real_features)
    generated = representation(critic, generated_features)
    return conditional_energy_score(real, generated, labels)
