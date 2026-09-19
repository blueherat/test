"""Preserve each model's deployed time, prediction and image conventions."""
import torch
from experiments.guidance_dynamic_50k_20260915.models import Adapter


def materialize_autocast_weights(adapter):
    """Store frozen Linear/Conv weights in their existing BF16 compute dtype.

    JiT/RAEv2 already autocast these operators to BF16. Norms, embeddings and
    arbitrary parameters stay FP32; the trainable head is never converted.
    Must run before capture. No SiT FP32 operator is changed.
    """
    if adapter.name not in ('jit', 'raev2'):
        return 0
    saved = 0
    roots = [adapter.model]
    if adapter.name == 'raev2': roots.append(adapter.rt.decoder)
    for root in roots:
        for module in root.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                for p in module.parameters(recurse=False):
                    if p.requires_grad:
                        raise ValueError('Only frozen weights can be materialized')
                    if p.dtype == torch.float32:
                        saved += p.numel()*2
                        p.data = p.data.to(torch.bfloat16)
    return saved


class RAEAdapter:
    def __init__(self, depth=8):
        from experiments.guidance_pasted_20260912.common import runtime
        from experiments.raev2_shallow_ig_20260914.core import Capture
        self.name, self.depth = 'raev2', depth
        self.rt = runtime('raev2')
        self.model = self.rt.model
        self.cfg = dict(shape=(1024, 16, 16), classes=1000)
        self.capture = Capture(self.rt, depth=depth)
        self.values = self.capture.values
        self.handles = self.capture.handles
        self.head_kind = 'mlp'

    def autocast(self):
        return self.rt.context()

    def unpatch(self, x):
        return self.model.unpatchify(x, self.model.s_patch_size)

    def clean_to_velocity(self, clean, state, time):
        return self.rt.native.clean_to_velocity(clean, state, time.expand(len(state)),
                    denominator_floor=float(self.rt.cfg.transport.t_eps))

    def full(self, state, time, labels):
        clean, _ = self.model(state, time.expand(len(state)), context=labels, attn_mask=None)
        # Match the deployed RAE sampler: convert both predictions before mixing.
        return self.clean_to_velocity(clean, state, time), None

    def native_to_velocity(self, raw, state, time):
        return raw.float()

    def make_head(self):
        from experiments.guidance_distribution_20260912.local_head import make_head
        return make_head(self.rt)


def load(model, head_checkpoint=None, head_key='ema'):
    adapter = RAEAdapter() if model == 'raev2' else Adapter(model)
    if model == 'sit_small':
        head = adapter.loaded_head('guided_weak')
        provenance = dict(kind='50k_full_data', steps=50000,
            path='/home/zhoushunyu/data/eqvae/experiments/guidance_dynamic_50k_20260915/sit_small/training/guided_weak/head.pt')
    else:
        head = adapter.make_head()
        if model == 'jit':
            path = '/home/zhoushunyu/data/eqvae/experiments/jit_readout_transfer_20260913/training/head.pt'
            state = torch.load(path, map_location='cpu', weights_only=False)
            head.load_state_dict(state['ema']['mlp'])
            provenance = dict(kind='engineering_fixture', steps=3000, path=path)
        else:
            path = '/home/zhoushunyu/data/eqvae/experiments/raev2_shallow_ig_20260914/depth8_continuation/raev2/training/head.pt'
            state = torch.load(path, map_location='cpu', weights_only=False)
            head.load_state_dict(state['ema']['context'])
            provenance = dict(kind='50k_depth8', steps=50000, path=path)
    if head_checkpoint is not None:
        state = torch.load(head_checkpoint, map_location='cpu', weights_only=False)
        weights = state
        for key in head_key.split('.'):
            weights = weights[key]
        head.load_state_dict(weights, strict=True)
        provenance = dict(kind='explicit_checkpoint', path=str(head_checkpoint), key=head_key,
                          steps=state.get('steps', state.get('step', 0)))
    from experiments.guidance_loss_50k_20260914.config import sha
    provenance['sha256'] = sha(provenance['path'])
    head.eval().requires_grad_(True)
    return adapter, head, provenance


def image_decoder(adapter):
    if adapter.name == 'sit_small':
        from experiments.adversarial_weak_training_20260915.rendering import sit_images
        return lambda x: sit_images(adapter, x)
    if adapter.name == 'jit':
        return lambda x: ((x.float()+1)/2).clamp(0, 1)

    def decode(x):
        with adapter.autocast():
            return adapter.rt.decoder.decode(x).float().clamp(0, 1)
    return decode
