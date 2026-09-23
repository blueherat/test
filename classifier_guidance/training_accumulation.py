"""One global D update then one global generator update with microbatch feedback."""
import torch
import torch.distributed as dist

from experiments.adversarial_weak_training_20260915.binary_critic import discriminator_loss, weak_loss
from .training import average_gradients, step as unchunked_step


def step(*, head, critic, optimizer_w, optimizer_d, sample, decode, feature,
         real, noise, labels, microbatch, r1=1., update_head=True, diagnostics=None):
    count = len(labels)
    if microbatch < 1 or count%microbatch:
        raise ValueError('The global batch must be divisible by the captured microbatch')
    distributed = dist.is_initialized() and dist.get_world_size()>1
    if microbatch==count and not distributed:
        return unchunked_step(head=head, critic=critic, optimizer_w=optimizer_w, optimizer_d=optimizer_d,
            sample=sample, decode=decode, feature=feature, real=real, noise=noise, labels=labels,
            r1=r1, update_head=update_head, diagnostics=diagnostics)
    observing = diagnostics is not None
    if observing:
        before_d = torch.cat([p.detach().flatten() for p in critic.parameters()]).clone()
        before_w = torch.cat([p.detach().flatten() for p in head.parameters()]).clone()
    endpoints, real_features, fake_features = [], [], []
    endpoint_square_sum = torch.zeros((), device=noise.device, dtype=noise.dtype)
    for start in range(0,count,microbatch):
        sl = slice(start,start+microbatch)
        with torch.no_grad():
            real_features.append(feature(real[sl]))
        with torch.set_grad_enabled(update_head):
            generated = sample(noise[sl],labels[sl])
        with torch.no_grad():
            fake_features.append(feature(decode(generated)))
            endpoint_square_sum += generated.square().sum()
        endpoints.append(generated)
    # The small classifier sees the full feature batch in ONE call. In
    # particular spectral-norm power iteration must not run four times per D
    # update. Its loss/R1/gradient clipping are the original global averages.
    real_batch, fake_batch, critic_labels = torch.cat(real_features), torch.cat(fake_features), labels
    if distributed:
        # Replicate only the small detached feature batch. D keeps the same
        # global batch shape and one spectral-norm update as the single GPU.
        def gather(value):
            parts = [torch.empty_like(value) for _ in range(dist.get_world_size())]
            dist.all_gather(parts, value.contiguous())
            return torch.cat(parts)
        real_batch, fake_batch, critic_labels = map(gather, (real_batch, fake_batch, labels))
    d_count = len(critic_labels)
    inputs = torch.cat([real_batch,fake_batch]).detach().requires_grad_(True)
    critic.train().requires_grad_(True); optimizer_d.zero_grad(set_to_none=True)
    logits = critic(inputs,critic_labels.repeat(2))
    classification = discriminator_loss(logits[:d_count],logits[d_count:])
    input_gradient = torch.autograd.grad(logits[:d_count].sum(),inputs,create_graph=True)[0][:d_count]
    penalty = input_gradient.square().sum(1).mean()
    (classification+.5*r1*penalty).backward()
    average_gradients(critic)
    d_gradient_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(),10.,error_if_nonfinite=True)
    optimizer_d.step()
    if observing:
        with torch.no_grad():
            delta = torch.cat([p.detach().flatten() for p in critic.parameters()])-before_d
            diagnostics.update(d_gradient_norm=d_gradient_norm.detach(),d_update_norm=delta.norm(),
                d_relative_update_norm=delta.norm()/before_d.norm().clamp_min(1e-30),
                d_clip_applied=(d_gradient_norm>10).float(),
                real_logit_mean=logits[:d_count].mean().detach(),fake_logit_before_d_mean=logits[d_count:].mean().detach(),
                real_probability_mean=logits[:d_count].sigmoid().mean().detach(),
                fake_probability_before_d_mean=logits[d_count:].sigmoid().mean().detach(),
                d_real_saturated_fraction=(logits[:d_count].sigmoid()>.99).float().mean(),
                d_fake_saturated_fraction=(logits[d_count:].sigmoid()<.01).float().mean())
    real_accuracy=(logits[:d_count]>0).float().mean().detach()
    fake_accuracy=(logits[d_count:]<0).float().mean().detach()
    d_ce,r1_value=classification.detach(),penalty.detach()
    del inputs,logits,classification,input_gradient,penalty,real_features,real_batch,fake_batch,critic_labels

    critic.eval().requires_grad_(False); optimizer_w.zero_grad(set_to_none=True)
    generator_loss = torch.zeros((),device=noise.device,dtype=noise.dtype)
    feature_gradient_sum=torch.zeros_like(generator_loss); endpoint_gradient_sum=torch.zeros_like(generator_loss)
    generator_logits=[]
    for index,generated in enumerate(endpoints):
        sl=slice(index*microbatch,(index+1)*microbatch)
        with torch.set_grad_enabled(update_head):
            # Recompute only the cheap frozen feature network; reuse the exact
            # generated trajectory. At most one Inception tape remains live.
            features=feature(decode(generated)) if update_head else fake_features[index]
            if observing and update_head:
                generated.retain_grad(); features.retain_grad()
            logits=critic(features,labels[sl])
            loss=weak_loss(logits)*(microbatch/count)
        generator_loss+=loss.detach(); generator_logits.append(logits.detach())
        if update_head:
            loss.backward()
            if observing:
                feature_gradient_sum+=(features.grad*count).flatten(1).norm(dim=1).sum()
                endpoint_gradient_sum+=(generated.grad*count).flatten(1).norm(dim=1).sum()
        endpoints[index]=None
        del generated,features,logits,loss
    gradient_norm=torch.zeros_like(generator_loss)
    if update_head:
        average_gradients(head)
        if observing:
            raw_gradient=torch.cat([p.grad.detach().flatten() for p in head.parameters()]).clone()
        gradient_norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
        optimizer_w.step()
    if observing:
        with torch.no_grad():
            logits=torch.cat(generator_logits)
            diagnostics.update(fake_logit_after_d_mean=logits.mean(),
                fake_probability_after_d_mean=logits.sigmoid().mean(),
                g_logit_gradient_mean=(-logits).sigmoid().mean(),
                g_loss_saturated_fraction=((-logits).sigmoid()<1e-4).float().mean())
            if update_head:
                delta=torch.cat([p.detach().flatten() for p in head.parameters()])-before_w
                diagnostics.update(head_gradient=raw_gradient,head_update=delta,head_update_norm=delta.norm(),
                    head_zero_gradient_fraction=(raw_gradient==0).float().mean(),
                    head_near_zero_gradient_fraction=(raw_gradient.abs()<1e-10).float().mean(),
                    head_clip_applied=(gradient_norm>1).float(),
                    g_feature_gradient_norm_mean=feature_gradient_sum/count,
                    g_endpoint_gradient_norm_mean=endpoint_gradient_sum/count)
        if not torch.isfinite(torch.cat([v.reshape(-1) for v in diagnostics.values()])).all():
            raise FloatingPointError('Nonfinite accumulated GAN diagnostics')
    metrics=torch.stack([d_ce,r1_value,generator_loss,gradient_norm,real_accuracy,fake_accuracy,
                         (endpoint_square_sum/noise.numel()).sqrt()])
    if not torch.isfinite(metrics).all():raise FloatingPointError('Nonfinite accumulated GAN step')
    return metrics
