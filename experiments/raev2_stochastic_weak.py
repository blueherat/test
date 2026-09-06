"""S²-style weak reference with the native base prefix and full readout.

One uniformly sampled encoder block after base depth is omitted. The first
eight encoder blocks and both decoder blocks remain intact. No layer search.
"""
import torch.nn.functional as F


class SharedPrefixWeak:
    def __init__(self, model):
        self.model = model
        self.seq = None
        self.hook = model.blocks[model.base_model_depth-1].register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self.seq = output

    def weak(self, state, times, *, skip=None, attenuation=None):
        m = self.model
        if self.seq is None:
            raise RuntimeError('native full forward must precede weak continuation')
        if skip is not None and not m.base_model_depth <= skip < m.num_enc_blocks:
            raise ValueError('only post-base encoder blocks may be skipped')
        seq = self.seq
        # Native attn_mask=None gives an all-zero additive attention mask.
        mask = m._build_attn_mask(seq, {'attn_mask':None})
        t_emb_base, _ = m.t_embedder(times, return_base_embed=True)
        for i in range(m.base_model_depth, m.num_enc_blocks):
            if i == skip:
                continue
            result = m.blocks[i](seq, m.enc_rope, mask)
            seq = result if attenuation is None else (seq + attenuation*(result-seq))
        seq = m.s_projector(F.silu(t_emb_base+seq[:, :m.s_embedder.num_patches, :]))
        x = m.x_embedder(state)
        for i in range(m.num_dec_blocks):
            x = m.blocks[m.num_enc_blocks+i](x, seq, m.dec_rope)
        return m.unpatchify(m.final_layer(x, seq), m.x_patch_size)

    def close(self):
        self.hook.remove()
        self.seq = None
