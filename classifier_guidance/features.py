"""Bound image-feedback memory independently of the sampler batch size."""
import torch
from torch.autograd.function import once_differentiable
from torch.utils.checkpoint import checkpoint


class _ChunkedFeatures(torch.autograd.Function):
    @staticmethod
    def forward(ctx, images_from_latents, feature, chunk, latents):
        ctx.decode, ctx.feature, ctx.chunk = images_from_latents, feature, chunk
        ctx.save_for_backward(latents)
        return torch.cat([feature(images_from_latents(x)) for x in latents.split(chunk)])

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        (latents,) = ctx.saved_tensors
        result = []
        for x, g in zip(latents.split(ctx.chunk), gradient.split(ctx.chunk)):
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                f = ctx.feature(ctx.decode(x))
                result.append(torch.autograd.grad(f, x, g)[0])
        return None, None, None, torch.cat(result)


def chunked_features(decode, feature, latents, chunk=1):
    """Exact independent-example feature VJP; decoder/feature weights frozen.

    Both modules must be deterministic and in eval mode (no training BatchNorm).
    Avoids retaining a full-batch VAE + Inception tape while training the critic.
    """
    if chunk < 1:
        raise ValueError('feature chunk must be positive')
    return _ChunkedFeatures.apply(decode, feature, chunk, latents)


def enable_feedback_checkpointing(adapter, feature):
    """Keep the full batch and eval behavior, recomputing frozen block activations.

    Unlike microbatching this preserves convolution batch shapes. Wrapper
    installation changes no parameters or state_dict keys. Do not wrap a parent
    and its child: each selected block owns a single checkpoint boundary.
    """
    modules = []
    if adapter.name == 'sit_small':
        from diffusers.models.resnet import ResnetBlock2D
        from diffusers.models.attention_processor import Attention
        modules += [m for m in adapter.rt.vae.decoder.modules() if isinstance(m, (ResnetBlock2D, Attention))]
    elif adapter.name == 'raev2':
        # Decoder is a ViT: each layer is a natural checkpoint boundary.
        modules += list(adapter.rt.decoder.decoder.decoder_layers)
    model = feature.extractor
    modules += [getattr(model, name) for name in (
        'Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
        'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'Mixed_5b', 'Mixed_5c', 'Mixed_5d',
        'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d', 'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c')]
    for module in modules:
        if hasattr(module, '_classifier_original_forward'):
            continue
        original = module.forward
        module._classifier_original_forward = original

        def forward(*args, _original=original, **kwargs):
            if not torch.is_grad_enabled():
                return _original(*args, **kwargs)
            return checkpoint(_original, *args, use_reentrant=False, preserve_rng_state=False, **kwargs)
        module.forward = forward
    return len(modules)


def enable_backbone_checkpointing(adapter):
    """Optional lower-memory tradeoff, also captured inside field VJP graphs."""
    for module in adapter.model.blocks:
        if hasattr(module, '_classifier_original_forward'):
            continue
        original = module.forward
        module._classifier_original_forward = original

        def forward(*args, _original=original, **kwargs):
            if not torch.is_grad_enabled():
                return _original(*args, **kwargs)
            return checkpoint(_original, *args, use_reentrant=False, preserve_rng_state=False, **kwargs)
        module.forward = forward
