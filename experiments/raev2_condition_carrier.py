"""Shared DDT readout for guidance in spatial modulation conditions."""
import torch
from torch.nn import functional as F


def representations(model, z, t, labels):
    kw = dict(context=labels, attn_mask=None)
    seq, tb = model._build_sequence(z, t, kw)
    mask = model._build_attn_mask(seq, kw)
    n = model.s_embedder.num_patches
    shallow = None
    for i in range(model.num_enc_blocks):
        seq = model.blocks[i](seq, model.enc_rope, mask)
        if i + 1 == model.base_model_depth:
            shallow = seq[:, :n]
    assert shallow is not None
    return F.silu(tb + shallow), model.s_projector(F.silu(tb + seq[:, :n]))


def readout(model, z, condition):
    x = model.x_embedder(z)
    for block in model.blocks[model.num_enc_blocks:]:
        x = block(x, condition, model.dec_rope)
    return model.unpatchify(model.final_layer(x, condition), model.x_patch_size)


def adapted_condition(shallow, adapter):
    return (shallow - adapter['input_mean']) @ adapter['weight'] + adapter['output_mean']


def guided_outputs(model, z, shallow, strong_condition, adapter, gamma=.78):
    weak_condition = adapted_condition(shallow, adapter)
    full = readout(model, z, strong_condition)
    weak = readout(model, z, weak_condition)
    carrier = readout(model, z, strong_condition + gamma * (strong_condition - weak_condition))
    output = weak + (1+gamma) * (full-weak)
    return full, weak, output, carrier, weak_condition
