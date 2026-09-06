"""Fixed native base-prefix statistics for a low-dimensional ratio model.

The depth is the pretrained model's base_model_depth, never a searched layer.
Mean and second raw moment define a diagonal quadratic exponential family in
RMS-normalized pretrained token coordinates.
"""
import torch


def token_statistics(tokens):
    normalized = tokens / tokens.square().mean(-1, keepdim=True).add(1e-8).sqrt()
    return torch.cat((normalized.mean(1), normalized.square().mean(1)), dim=-1)


def prefix_features(model, state, times, labels):
    kwargs = {'context': labels, 'attn_mask': None}
    sequence, _ = model._build_sequence(state, times, kwargs)
    mask = model._build_attn_mask(sequence, kwargs)
    for i in range(model.base_model_depth):
        sequence = model.blocks[i](sequence, model.enc_rope, mask)
    return token_statistics(sequence[:, :model.s_embedder.num_patches])
