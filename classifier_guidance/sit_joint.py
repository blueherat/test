"""Signed guidance and a native SiT adapter, jointly fitted to endpoint feedback.

The soft gap-energy anchor selects a scale on an independent probe distribution;
it does not establish identifiability of either neural parameters or dynamics.
"""
import torch
import torch.distributed as dist
from torch import nn

from .sampler import Sampler, prepare_adapter
from .schedules import GuidanceSchedule
from .training import average_gradients


class JointGuidance(nn.Module):
    def __init__(self, weak, steps=64, initial=.75):
        super().__init__()
        self.weak = weak
        self.schedule = GuidanceSchedule(steps, initial)
        with torch.no_grad():
            self.schedule.coefficients.fill_(initial)


class JointField:
    def __init__(self, adapter, joint):
        self.adapter, self.joint = adapter, joint

    def __call__(self, state, time, labels, index, active):
        a = self.adapter
        strong, _ = a.full(state, time, labels)
        weak = a.unpatch(self.joint.weak(a.values['context'], a.values['condition'])).float()
        return strong + self.joint.schedule(index)*(strong-weak)


def make_sampler(adapter, joint, example, labels, *, graphs=True):
    prepare_adapter(adapter, joint, example.device)
    steps = joint.schedule.coefficients.numel()
    grid = torch.linspace(0, 1, steps+1, device=example.device)
    return Sampler(JointField(adapter, joint), joint, example, labels, grid,
                   torch.arange(steps, device=example.device, dtype=example.dtype),
                   [True]*steps, heun=True, graphs=graphs)


def gap_anchor(gap, reference, epsilon=1e-8):
    energy, initial_energy = gap.square().mean(), reference.detach().square().mean()
    return torch.log((energy+epsilon)/(initial_energy+epsilon)).square()


class GapProbe:
    """Fresh real posterior interpolants, independent of the optimized dynamics.

    Each batch uses one uniform random t. The objective is explicitly an
    expectation of a minibatch log-energy ratio, not an unbiased estimator of
    the corresponding nonlinear population-moment objective.
    """
    def __init__(self, adapter, joint, reference, seed):
        self.adapter, self.joint, self.reference = adapter, joint, reference
        self.generator = torch.Generator(device='cuda').manual_seed(seed)

    @torch.no_grad()
    def states(self, real):
        from experiments.adversarial_weak_training_20260915.rendering import base
        posterior = self.adapter.rt.vae.encode(real*2-1).latent_dist
        clean = posterior.sample(generator=self.generator)*base.SD_VAE_SCALING_FACTOR
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=self.generator)
        t = torch.rand((), device=clean.device, generator=self.generator)
        return clean, noise, t

    def gap(self, x, t, labels):
        # x is independent of learned parameters. This no_grad is only for
        # the auxiliary probe; JointField retains the full state Jacobian.
        with torch.no_grad():
            strong, _ = self.adapter.full(x, t, labels)
            context = self.adapter.values['context'].detach()
            condition = self.adapter.values['condition'].detach()
            reference = strong-self.adapter.unpatch(self.reference(context, condition)).float()
        gap = strong-self.adapter.unpatch(self.joint.weak(context, condition)).float()
        return gap, reference

    def __call__(self, real, labels):
        clean, noise, t = self.states(real)
        gap, reference = self.gap((1-t)*noise+t*clean, t, labels)
        loss = gap_anchor(gap, reference)
        with torch.no_grad():
            rms, rms0 = gap.square().mean().sqrt(), reference.square().mean().sqrt()
            cosine = (gap*reference).mean()/(rms*rms0).clamp_min(1e-12)
        return loss, dict(probe_time=t, gap_rms=rms, reference_gap_rms=rms0,
                          gap_ratio=rms/rms0.clamp_min(1e-12), gap_cosine=cosine)

    @torch.no_grad()
    def profile(self, real, labels, count=16):
        clean, noise, _ = self.states(real)
        rows = []
        steps = self.joint.schedule.coefficients.numel()
        # Interval midpoints are paired with the corresponding actual scale.
        indices = torch.linspace(0, steps-1, count).round().long().tolist()
        for index in indices:
            t = torch.tensor((index+.5)/steps, device=clean.device)
            gap, reference = self.gap((1-t)*noise+t*clean, t, labels)
            rms, rms0 = gap.square().mean().sqrt(), reference.square().mean().sqrt()
            a = self.joint.schedule.coefficients[index]
            rows.append(dict(index=index, time=t.item(), coefficient=a.item(), gap_rms=rms.item(),
                reference_gap_rms=rms0.item(), correction_rms=(a.abs()*rms).item(),
                gap_ratio=(rms/rms0.clamp_min(1e-12)).item(),
                cosine=((gap*reference).mean()/(rms*rms0).clamp_min(1e-12)).item()))
        return rows


def joint_step(*, joint, critic, optimizer, optimizer_d, sample, decode, feature,
               real, noise, labels, microbatch, probe, norm_weight=.1, r1=1., update=True,
               probe_real=None, probe_labels=None, diagnostics=None):
    """One D step and one accumulated joint step; separate group clipping.

    All generated endpoints use the same pre-update joint parameters. The
    small discriminator sees the full effective batch in one forward so its
    spectral normalization is independent of the microbatch count.
    """
    from experiments.adversarial_weak_training_20260915.binary_critic import discriminator_loss, weak_loss
    count = len(labels)
    world = dist.get_world_size() if dist.is_initialized() else 1
    if world>1 and (probe_real is None or probe_labels is None):
        raise ValueError('All ranks must use the SAME global probe minibatch, not per-rank log moments')
    if count % microbatch:
        raise ValueError('Batch must be divisible by microbatch')
    flat = lambda module: torch.cat([p.detach().reshape(-1) for p in module.parameters()])
    before_d = flat(critic)
    before_w = flat(joint.weak) if update else None
    before_a = joint.schedule.coefficients.detach().clone()
    endpoints, reals, fakes = [], [], []
    endpoint_square = noise.new_zeros(())
    for start in range(0, count, microbatch):
        sl = slice(start, start+microbatch)
        with torch.no_grad():
            reals.append(feature(real[sl]))
        with torch.set_grad_enabled(update):
            generated = sample(noise[sl], labels[sl])
        with torch.no_grad():
            fakes.append(feature(decode(generated)))
            endpoint_square += generated.square().sum()
        endpoints.append(generated)
    real_features, fake_features, critic_labels = torch.cat(reals), torch.cat(fakes), labels
    if world>1:
        def gather(value):
            parts = [torch.empty_like(value) for _ in range(world)]
            dist.all_gather(parts,value.contiguous())
            return torch.cat(parts)
        real_features,fake_features,critic_labels = map(gather,(real_features,fake_features,labels))
    d_count = len(critic_labels)
    inputs = torch.cat([real_features,fake_features]).detach().requires_grad_(True)
    critic.train().requires_grad_(True)
    optimizer_d.zero_grad(set_to_none=True)
    logits = critic(inputs, critic_labels.repeat(2))
    ce = discriminator_loss(logits[:d_count], logits[d_count:])
    dx = torch.autograd.grad(logits[:d_count].sum(), inputs, create_graph=True)[0][:d_count]
    r1_loss = dx.square().sum(1).mean()
    (ce+.5*r1*r1_loss).backward()
    average_gradients(critic)
    d_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), 10., error_if_nonfinite=True)
    optimizer_d.step()
    metrics = dict(d_ce=ce.detach(), r1=r1_loss.detach(), d_gradient_norm=d_norm,
        d_update_norm=(flat(critic)-before_d).norm(),
        real_accuracy=(logits[:d_count]>0).float().mean().detach(),
        fake_accuracy=(logits[d_count:]<0).float().mean().detach(),
        real_probability=logits[:d_count].sigmoid().mean().detach(),
        fake_probability_before_d=logits[d_count:].sigmoid().mean().detach(),
        d_real_saturation=(logits[:d_count].sigmoid()>.99).float().mean().detach(),
        d_fake_saturation=(logits[d_count:].sigmoid()<.01).float().mean().detach(),
        endpoint_rms=(endpoint_square/noise.numel()).sqrt())
    del inputs, logits, ce, dx, r1_loss, reals, before_d,real_features,fake_features,critic_labels
    critic.eval().requires_grad_(False)
    optimizer.zero_grad(set_to_none=True)
    g_loss = noise.new_zeros(())
    endpoint_gradient = noise.new_zeros(())
    g_logits = []
    for i, generated in enumerate(endpoints):
        sl = slice(i*microbatch, (i+1)*microbatch)
        with torch.set_grad_enabled(update):
            features = feature(decode(generated)) if update else fakes[i]
            logits = critic(features, labels[sl])
            loss = weak_loss(logits)*(microbatch/count)
        g_loss += loss.detach()
        g_logits.append(logits.detach())
        if update:
            generated.retain_grad()
            loss.backward()
            endpoint_gradient += (generated.grad*count).flatten(1).norm(dim=1).sum()/count
        endpoints[i] = None
        del generated, features, logits, loss
    logits = torch.cat(g_logits)
    metrics.update(g_loss=g_loss, fake_probability_after_d=logits.sigmoid().mean(),
        g_logit_sensitivity=(-logits).sigmoid().mean(),
        g_loss_saturation=((-logits).sigmoid()<1e-4).float().mean(),
        endpoint_gradient_norm=endpoint_gradient)
    if update:
        # Average the local-mean endpoint gradient BEFORE adding the replicated
        # global probe loss and BEFORE either group's clipping. Never average
        # independent nonlinear per-rank gap-energy ratios.
        average_gradients(joint)
        weak_params = list(joint.weak.parameters())
        gan_gradient = torch.cat([p.grad.detach().reshape(-1) for p in weak_params])
        a_gradient = joint.schedule.coefficients.grad.detach().clone()
        norm_loss, probe_metrics = probe(real[:microbatch] if probe_real is None else probe_real,
                                         labels[:microbatch] if probe_labels is None else probe_labels)
        (norm_weight*norm_loss).backward()
        average_gradients(joint)
        total_gradient = torch.cat([p.grad.detach().reshape(-1) for p in weak_params])
        regularizer_gradient = total_gradient-gan_gradient
        if diagnostics is not None:
            diagnostics.update(weak_gradient=total_gradient.clone(),coefficient_gradient=a_gradient.clone())
        w_norm = torch.nn.utils.clip_grad_norm_(weak_params, 1., error_if_nonfinite=True)
        a_norm = torch.nn.utils.clip_grad_norm_(joint.schedule.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        metrics.update(probe_metrics, norm_loss=norm_loss.detach(),
            weak_gan_gradient_norm=gan_gradient.norm(), norm_gradient_norm=regularizer_gradient.norm(),
            weak_gradient_norm=w_norm, coefficient_gradient_norm=a_norm,
            coefficient_gradient=a_gradient, coefficient_nonzero_gradients=(a_gradient!=0).sum(),
            weak_update_norm=(flat(joint.weak)-before_w).norm(),
            coefficient_update_norm=(joint.schedule.coefficients.detach()-before_a).norm(),
            weak_clip=(w_norm>1).float(), coefficient_clip=(a_norm>1).float())
    detached = {k:v.detach() for k,v in metrics.items()}
    if world>1:
        # Local scalar feedback becomes the global mean. Gradients, updates,
        # classifier metrics and replicated probe metrics already agree.
        keys=('g_loss','fake_probability_after_d','g_logit_sensitivity','g_loss_saturation',
              'endpoint_gradient_norm','endpoint_rms')
        values=torch.stack([detached[k] for k in keys])
        values[-1].square_();dist.all_reduce(values);values/=world;values[-1].sqrt_()
        detached.update(zip(keys,values.unbind()))
    if not all(torch.isfinite(v).all().item() for v in detached.values()):
        raise FloatingPointError('Nonfinite joint GAN diagnostics')
    return {k:v.cpu().tolist() for k,v in detached.items()}
