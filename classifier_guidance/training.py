"""The same alternating binary GAN update, with bounded-memory feedback."""
import torch
import torch.distributed as dist
from experiments.adversarial_weak_training_20260915.binary_critic import discriminator_loss, weak_loss
from .features import chunked_features


def average_gradients(module):
    parameters = tuple(module.parameters())
    if any(p.grad is None for p in parameters):
        raise RuntimeError('A trainable parameter has no gradient')
    if dist.is_initialized():
        # One collective instead of one per parameter; preserve average-before-clip.
        flat = torch.cat([p.grad.reshape(-1) for p in parameters])
        dist.all_reduce(flat)
        flat.div_(dist.get_world_size())
        offset = 0
        for p in parameters:
            p.grad.copy_(flat[offset:offset+p.numel()].view_as(p))
            offset += p.numel()


def step(*, head, critic, optimizer_w, optimizer_d, sample, decode, feature,
         real, noise, labels, r1=1., feature_chunk=1, update_head=True):
    """Exactly one D update, then one W update against the updated D.

    Features, decoder and frozen generator keep input derivatives, not weight
    gradients. The critic still sees only real RGB and guided final images.
    ``feature_chunk=0`` selects the historical full-batch autograd reference.
    """
    with torch.no_grad():
        real_features = torch.cat([feature(x) for x in real.split(feature_chunk or len(real))])
    with torch.set_grad_enabled(update_head):
        generated = sample(noise, labels)
        generated_features = (chunked_features(decode, feature, generated, feature_chunk)
                              if feature_chunk else feature(decode(generated)))

    critic.train().requires_grad_(True)
    optimizer_d.zero_grad(set_to_none=True)
    inputs = torch.cat([real_features, generated_features.detach()]).detach().requires_grad_(True)
    logits = critic(inputs, labels.repeat(2))
    n = len(labels)
    classification = discriminator_loss(logits[:n], logits[n:])
    input_gradient = torch.autograd.grad(logits[:n].sum(), inputs, create_graph=True)[0][:n]
    penalty = input_gradient.square().sum(1).mean()
    d_loss = classification + .5*r1*penalty
    d_loss.backward()
    average_gradients(critic)
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 10., error_if_nonfinite=True)
    optimizer_d.step()
    # Release critic training tape before rendering backward.
    real_accuracy = (logits[:n] > 0).float().mean().detach()
    fake_accuracy = (logits[n:] < 0).float().mean().detach()
    d_ce, r1_value = classification.detach(), penalty.detach()
    del inputs, logits, classification, input_gradient, penalty, d_loss

    critic.eval().requires_grad_(False)
    optimizer_w.zero_grad(set_to_none=True)
    generator_loss = weak_loss(critic(generated_features, labels))
    gradient_norm = torch.zeros((), device=noise.device)
    if update_head:
        generator_loss.backward()
        average_gradients(head)
        gradient_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        optimizer_w.step()
    metrics = torch.stack([d_ce, r1_value, generator_loss.detach(), gradient_norm,
                           real_accuracy, fake_accuracy, generated.detach().square().mean().sqrt()])
    if not bool(torch.isfinite(metrics).all()):
        raise FloatingPointError('Nonfinite binary GAN step')
    return metrics
