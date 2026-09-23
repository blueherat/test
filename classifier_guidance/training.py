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
         real, noise, labels, r1=1., feature_chunk=0, update_head=True, diagnostics=None):
    """Exactly one D update, then one W update against the updated D.

    Features, decoder and frozen generator keep input derivatives, not weight
    gradients. The critic still sees only real RGB and guided final images.
    ``feature_chunk=0`` selects the historical full-batch autograd reference.
    Optional diagnostics observe this update without extra model forwards or
    backward calls. Values remain detached device tensors for caller reduction.
    """
    observing = diagnostics is not None
    if observing:
        before_d = torch.cat([p.detach().reshape(-1) for p in critic.parameters()]).clone()
        before_w = torch.cat([p.detach().reshape(-1) for p in head.parameters()]).clone()
    with torch.no_grad():
        real_features = torch.cat([feature(x) for x in real.split(feature_chunk or len(real))])
    with torch.set_grad_enabled(update_head):
        generated = sample(noise, labels)
        generated_features = (chunked_features(decode, feature, generated, feature_chunk)
                              if feature_chunk else feature(decode(generated)))
    if observing and update_head:
        generated.retain_grad()
        generated_features.retain_grad()

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
    d_gradient_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), 10., error_if_nonfinite=True)
    optimizer_d.step()
    if observing:
        with torch.no_grad():
            delta_d = torch.cat([p.detach().reshape(-1) for p in critic.parameters()])-before_d
            diagnostics.update(d_gradient_norm=d_gradient_norm.detach(),
                d_update_norm=delta_d.norm(), d_relative_update_norm=delta_d.norm()/before_d.norm().clamp_min(1e-30),
                d_clip_applied=(d_gradient_norm>10).float(),
                real_logit_mean=logits[:n].mean().detach(), fake_logit_before_d_mean=logits[n:].mean().detach(),
                real_probability_mean=logits[:n].sigmoid().mean().detach(),
                fake_probability_before_d_mean=logits[n:].sigmoid().mean().detach(),
                d_real_saturated_fraction=(logits[:n].sigmoid()>.99).float().mean(),
                d_fake_saturated_fraction=(logits[n:].sigmoid()<.01).float().mean())
    # Release critic training tape before rendering backward.
    real_accuracy = (logits[:n] > 0).float().mean().detach()
    fake_accuracy = (logits[n:] < 0).float().mean().detach()
    d_ce, r1_value = classification.detach(), penalty.detach()
    del inputs, logits, classification, input_gradient, penalty, d_loss

    critic.eval().requires_grad_(False)
    optimizer_w.zero_grad(set_to_none=True)
    generator_logits = critic(generated_features, labels)
    generator_loss = weak_loss(generator_logits)
    gradient_norm = torch.zeros((), device=noise.device)
    if update_head:
        generator_loss.backward()
        average_gradients(head)
        if observing:
            raw_gradient = torch.cat([p.grad.detach().reshape(-1) for p in head.parameters()]).clone()
        gradient_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        optimizer_w.step()
    if observing:
        with torch.no_grad():
            diagnostics.update(fake_logit_after_d_mean=generator_logits.mean().detach(),
                fake_probability_after_d_mean=generator_logits.sigmoid().mean().detach(),
                # Magnitude of d softplus(-logit)/d logit per example, BEFORE
                # local-batch averaging. Small values mean loss-level saturation.
                g_logit_gradient_mean=(-generator_logits).sigmoid().mean().detach(),
                g_loss_saturated_fraction=((-generator_logits).sigmoid()<1e-4).float().mean())
            if update_head:
                delta_w = torch.cat([p.detach().reshape(-1) for p in head.parameters()])-before_w
                diagnostics.update(head_gradient=raw_gradient,
                    head_update=delta_w, head_update_norm=delta_w.norm(),
                    head_zero_gradient_fraction=(raw_gradient==0).float().mean(),
                    head_near_zero_gradient_fraction=(raw_gradient.abs()<1e-10).float().mean(),
                    head_clip_applied=(gradient_norm>1).float(),
                    # Undo local mean-loss scaling for interpretable per-image
                    # gradients. These are NOT parameter-gradient magnitudes.
                    g_feature_gradient_norm_mean=(generated_features.grad*n).flatten(1).norm(dim=1).mean(),
                    g_endpoint_gradient_norm_mean=(generated.grad*n).flatten(1).norm(dim=1).mean())
        if not bool(torch.isfinite(torch.cat([v.reshape(-1) for v in diagnostics.values()])).all()):
            raise FloatingPointError('Nonfinite GAN diagnostics')
    metrics = torch.stack([d_ce, r1_value, generator_loss.detach(), gradient_norm,
                           real_accuracy, fake_accuracy, generated.detach().square().mean().sqrt()])
    if not bool(torch.isfinite(metrics).all()):
        raise FloatingPointError('Nonfinite binary GAN step')
    return metrics
