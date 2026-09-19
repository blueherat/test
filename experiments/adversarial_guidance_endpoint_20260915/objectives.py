"""Auxiliary source classification without changing the real/guided target.

Classes: real endpoint P, strong endpoint G, standalone weak endpoint Q, actual
guided endpoint R. G/Q are auxiliary classification data, not pooled negatives
in the weak generator's loss. Use a common class-conditioning prior throughout.
"""

import torch
from torch.nn import functional as F

REAL, STRONG, WEAK, GUIDED = range(4)


def source_discriminator_loss(logits, source_labels):
    if logits.ndim != 2 or logits.shape[1] != 4:
        raise ValueError("Expected four source logits per sample")
    return F.cross_entropy(logits, source_labels.long())


def pair_logits(logits, source_priors=None):
    """At the source-classification optimum, pair odds are p(x)/r(x).

    Priors are the effective source frequencies used by the CE objective, not
    image-class priors. Equal source minibatches imply equal source priors.
    """
    if logits.ndim != 2 or logits.shape[1] != 4:
        raise ValueError("Expected four source logits per sample")
    selected = logits[:, (REAL, GUIDED)]
    if source_priors is not None:
        priors = torch.as_tensor(source_priors, device=logits.device, dtype=logits.dtype)
        if priors.shape != (4,) or not bool((priors > 0).all()):
            raise ValueError("Four positive source priors are required")
        selected = selected - priors[[REAL, GUIDED]].log()
    return selected


def guided_generator_loss(logits, source_priors=None):
    """Non-saturating P-versus-R loss; call ONLY on differentiable R samples.

    Critic parameters should be frozen for this update; critic input gradients
    must remain enabled. Never detach the guided sample or the strong suffix.
    """
    selected = pair_logits(logits, source_priors)
    return F.softplus(selected[:, 1] - selected[:, 0]).mean()
