"""Wide IG/lifting strength curves; preserve each model's existing conventions."""
from __future__ import annotations

import hashlib
import inspect
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from experiments.raev2_capacity_lifting import heun_flow

WORK = Path(__file__).resolve().parents[1]
DATA = Path('/home/zhoushunyu/data/eqvae')
EXPS = DATA / 'experiments'
ROOT = EXPS / 'lifting_wide_scale_20260909'
MODELS = ('sit_xl', 'raev2', 'jit', 'sit_small')
ALPHAS = (0., .25, .5, 1., 1.5, 2.)
SMALL_DATA = DATA / 'imagenet_sit_flow'
SMALL_REF = SMALL_DATA / 'pfr_query_controls_v1/fid1k_seed0/ordinary_ig'
XL_REPO = WORK / 'research_repos/internal_guidance_study/Internal-Guidance/SiT'
XL_CKPT = DATA / 'models/Internal-Guidance/official/SiT/SiT-XL-IG-ImageNet256-800EP.pt'
RAE_CKPT = DATA / 'models/RAEv2/stage2/imagenet/dinov3l-k7/checkpoint.pt'
JIT_CKPT = DATA / 'models/JiT/jit-b-16/checkpoint-last.pth'
JIT_HEAD = EXPS / 'jit_internal_readouts_20260908/last.pt'
SMALL_CKPT = SMALL_DATA / 'runs/sit-s-2_seed0/checkpoints/step_00800000.pt'
SMALL_HEAD = SMALL_DATA / 'multiscale_guidance_study_v1/runs/depth4_v/checkpoints/step_00050000.pt'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def arm_name(method, alpha):
    return f'{method}_a{alpha:.3f}'


def asset_paths(name):
    if name == 'sit_xl':
        return [XL_CKPT]
    if name == 'raev2':
        base = DATA / 'models/RAEv2/stage1/imagenet/dinov3l-k7'
        return [RAE_CKPT, base / 'decoder.pt', base / 'stats.pt']
    if name == 'jit':
        return [JIT_CKPT, JIT_HEAD]
    return [SMALL_CKPT, SMALL_HEAD]


def source_paths(name):
    common = [Path(__file__), WORK / 'experiments/raev2_capacity_lifting.py']
    relative = {
        'sit_xl': ['experiments/run_internal_guidance_sit_audit.py',
                   'experiments/audit_official_sit_pfr_interface.py',
                   'research_repos/internal_guidance_study/Internal-Guidance/SiT/models/sit.py'],
        'raev2': ['experiments/sample_raev2_pfr_retiming.py',
                  'experiments/raev2_pfr_retiming.py',
                  'experiments/raev2_stage1_compat.py',
                  'experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml'],
        'jit': ['experiments/jit_internal_guidance.py'],
        'sit_small': ['experiments/sample_imagenet100_sit_foresight_fixed_point.py',
                      'experiments/imagenet100_sit_multiscale_models.py'],
    }[name]
    return common + [WORK / p for p in relative]


class Runtime:
    def __init__(self, name):
        self.name = name
        self.batch = 8 if name == 'sit_small' else 4
        self.labels = None
        self.counts = dict(full=0, prefix=0)
        torch.cuda.set_device(0)
        torch.set_num_threads(2 if name in ('jit', 'sit_small') else 4)
        torch.backends.cuda.matmul.allow_tf32 = name != 'sit_xl'
        torch.backends.cudnn.allow_tf32 = name != 'sit_xl'
        self.autocast = name in ('raev2', 'jit')
        self.metadata = {}
        if name == 'sit_xl':
            from experiments.run_internal_guidance_sit_audit import load_model
            from experiments.audit_official_sit_pfr_interface import prefix
            self.model, self.metadata = load_model(
                repo=XL_REPO, checkpoint_path=XL_CKPT, model_name='SiT-XL/2',
                encoder_depth=8, state_key='ema', device=torch.device('cuda'))
            self.prefix = prefix
            self.grid = torch.linspace(1, 0, 101, dtype=torch.float64)
            self.full_blocks, self.prefix_blocks = 28, 8
        elif name == 'raev2':
            from experiments import sample_raev2_pfr_retiming as native
            self.native = native
            self.cfg = native.load_config(native.DEFAULT_CONFIG)
            native.install_raev2_decoder_config_compat()
            self.model = native.instantiate_from_config(self.cfg.stage_2).cuda().eval().requires_grad_(False)
            state = torch.load(RAE_CKPT, map_location='cpu', mmap=True, weights_only=False)
            self.model.load_state_dict(state['ema'], strict=True)
            del state
            self.decoder = native.instantiate_from_config(self.cfg.stage_1).cuda().eval().requires_grad_(False)
            del self.decoder.encoder
            self.grid = native.shifted_time_grid(100, 8., torch.device('cuda'))
            self.full_blocks, self.prefix_blocks = 30, 8
        elif name == 'jit':
            from experiments import jit_internal_guidance as jig
            self.jig = jig
            self.model = jig.load_source('cuda')
            self.heads = jig.Readouts().cuda().eval().requires_grad_(False)
            state = torch.load(JIT_HEAD, map_location='cpu', weights_only=False)
            assert state['step'] == 50000
            self.heads.load_state_dict(state['ema'], strict=True)
            del state
            self.grid = torch.linspace(0, 1, 101, device='cuda')
            self.full_blocks, self.prefix_blocks = 12, 4
        else:
            from experiments import sample_imagenet100_sit_foresight_fixed_point as s
            from experiments.imagenet100_sit_multiscale_models import evaluate_internal_head_only
            self.small = s
            torch.set_float32_matmul_precision('high')
            module, source = s.load_official_sit_module(s.DEFAULT_OFFICIAL_SIT_REPO, verify_source=True)
            self.model, self.semantics, self.metadata, payload = s._load_field_model(
                checkpoint_path=SMALL_CKPT, requested_field='auto', weights='ema',
                sit_module=module, source_metadata=source, device=torch.device('cuda'))
            del payload
            self.head = s.load_internal_head_for_source(
                checkpoint_path=SMALL_HEAD, name='depth4_v', head_weights='ema',
                model=self.model, sit_module=module, source_checkpoint_path=SMALL_CKPT,
                source_metadata=source, device=torch.device('cuda'))
            self.prefix = evaluate_internal_head_only
            self.grid = torch.linspace(0, 1, 101, device='cuda')
            self.full_blocks, self.prefix_blocks = len(self.model.blocks), 4
        if name in ('sit_xl', 'sit_small'):
            from diffusers.models import AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained(
                'stabilityai/sd-vae-ft-mse', local_files_only=True).cuda().eval().requires_grad_(False)
        actual_sources = source_paths(name) + [Path(inspect.getfile(type(self.model)))]
        if name == 'jit':
            actual_sources += [self.jig.REPO / 'model_jit.py', self.jig.REPO / 'util/model_util.py']
        self.sources = {str(p): sha(p) for p in actual_sources}

    def context(self):
        return torch.autocast('cuda', dtype=torch.bfloat16) if self.autocast else nullcontext()

    def times(self, z, t):
        if self.name == 'sit_xl':
            return (torch.ones(len(z), device='cuda', dtype=torch.float64) * t).float()
        if self.name == 'raev2':
            return torch.full((len(z),), float(t), device='cuda')
        return t.expand(len(z))

    def pair(self, z, t):
        ts = self.times(z, t)
        self.counts['full'] += 1
        if self.name == 'sit_xl':
            full, weak, _ = self.model(z.float(), ts, self.labels)
            return full, weak
        if self.name == 'raev2':
            full, weak = self.model(z, ts, context=self.labels, attn_mask=None)
            convert = lambda x: self.native.clean_to_velocity(x, z, ts, denominator_floor=float(self.cfg.transport.t_eps))
            return convert(full), convert(weak)
        if self.name == 'jit':
            feats, c = self.jig.features(self.model, z, ts, self.labels, depths=(4, 12))
            full = self.jig.unpatchify(self.model.final_layer(feats['12'], c))
            weak = self.jig.unpatchify(self.heads.layers['4'](feats['4'], c))
            return self.jig.velocity(full, z, ts), self.jig.velocity(weak, z, ts)
        full, weak, _ = self.small.evaluate_source_with_heads(
            self.model, z, ts, self.labels, heads={'depth4_v': self.head}, source_semantics=self.semantics)
        return full, weak['depth4_v']

    def field(self, z, t, kind):
        ts = self.times(z, t)
        if kind == 'base':
            self.counts['prefix'] += 1
            if self.name == 'sit_xl':
                return self.prefix(self.model, z.float(), ts, self.labels).double()
            if self.name == 'raev2':
                clean = self.native.evaluate_base_head_only(self.model, z, ts, context=self.labels, attn_mask=None)
                return self.native.clean_to_velocity(clean, z, ts, denominator_floor=float(self.cfg.transport.t_eps))
            if self.name == 'jit':
                feats, c = self.jig.features(self.model, z, ts, self.labels, depths=(4,))
                clean = self.jig.unpatchify(self.heads.layers['4'](feats['4'], c))
                return self.jig.velocity(clean, z, ts)
            return self.prefix(self.model, z, ts, self.labels, spec=self.head)
        if self.name in ('sit_xl', 'raev2'):
            value, _ = self.pair(z, t)
            return value.double() if self.name == 'sit_xl' else value
        self.counts['full'] += 1
        if self.name == 'jit':
            return self.jig.velocity(self.model(z, ts, self.labels), z, ts)
        return self.small._model_velocity(self.model, self.semantics, z, t, self.labels, autocast_dtype=None)

    def guided(self, z, t, alpha):
        if alpha == 0:
            return self.field(z, t, 'full')
        full, weak = self.pair(z, t)
        if self.name in ('sit_xl', 'raev2'):
            value = weak + (1. + alpha) * (full - weak)
        else:
            value = full + alpha * (full - weak)
        return value.double() if self.name == 'sit_xl' else value

    def segments(self, alpha, grid):
        if self.name == 'raev2':
            from experiments.raev2_capacity_lifting import blocks
            return [(i, j, alpha * a) for i, j, a in blocks(grid, 1., 'constant')]
        steps = len(grid) - 1
        if self.name == 'sit_small':
            result = []
            k = 0
            while k < 50:
                stop = min(k + 4, 25 if k < 25 else 50)
                result.append((k, stop, alpha / .7 * .6 if k < 25 else alpha))
                k = stop
            return result + [(50, 100, 0.)]
        if self.name == 'jit':
            result = [(k, min(k + 4, 50), alpha) for k in range(0, 50, 4)]
            return result + [(k, k + 1, 0.) for k in range(50, 100)]
        return [(k, min(k + 4, steps), alpha) for k in range(0, steps, 4)]

    @torch.inference_mode()
    def sample(self, noise, labels, method, alpha, *, steps=100, diagnostics=False):
        assert method in ('ig', 'lifting') and alpha >= 0
        assert steps == 100 or self.name == 'sit_xl'
        self.labels = labels
        z = noise.double() if self.name == 'sit_xl' else noise.clone()
        grid = self.grid if steps == 100 else torch.linspace(1, 0, steps + 1, dtype=torch.float64)
        before = self.counts.copy()
        rows, aux_full, aux_prefix = [], 0, 0
        with self.context():
            for i, j, a in self.segments(alpha, grid):
                sub = grid[i:j+1]
                if method == 'lifting' and a:
                    target = heun_flow(z, sub, 'full', self.field)
                    lifted = heun_flow(target, sub.flip(0), 'base', self.field)
                    if diagnostics:
                        flat = lambda x: x.double().flatten(1).square().mean(1).sqrt()
                        rows.append(torch.stack((flat(z), flat(lifted-z), flat(a*(lifted-z))), 1))
                    z = z + a * (lifted - z)
                    aux_full += 2 * (j - i)
                    aux_prefix += 2 * (j - i)
                if self.name == 'sit_small':
                    from torchdiffeq import odeint
                    function = (lambda t, x: self.guided(x, t, a)) if method == 'ig' else (lambda t, x: self.field(x, t, 'full'))
                    z = odeint(function, z, grid[[i, j]], method='dopri5', rtol=.001, atol=1e-6)[-1]
                else:
                    for t, u in zip(sub[:-1], sub[1:]):
                        v = self.guided(z, t, a) if method == 'ig' else self.field(z, t, 'full')
                        z = z + (u - t) * v
        if not torch.isfinite(z).all():
            raise FloatingPointError(f'{self.name} {method} {alpha}: nonfinite endpoint')
        counts = {key: self.counts[key] - before[key] for key in before}
        if self.name != 'sit_small':
            assert counts == dict(full=steps + aux_full, prefix=aux_prefix), counts
        else:
            assert counts['prefix'] == aux_prefix and counts['full'] >= aux_full
        stats = dict(counts=counts, auxiliary_full=aux_full, auxiliary_prefix=aux_prefix,
                     endpoint_rms=z.double().flatten(1).square().mean(1).sqrt().cpu().tolist())
        if diagnostics:
            stats['lift_rms_columns'] = ['state', 'unscaled_lift', 'applied_lift']
            stats['lift_rms_by_block_and_image'] = torch.stack(rows).cpu().tolist() if rows else []
        return z, stats

    @torch.inference_mode()
    def decode(self, z):
        with self.context():
            if self.name == 'sit_small':
                return self.small.official_pixel_quantization(self.small.decode_latents_in_chunks(
                    self.vae, z, scaling_factor=self.small.SD_VAE_SCALING_FACTOR, chunk_size=2))
            if self.name == 'sit_xl':
                value = self.vae.decode(z.float() / .18215).sample
                return (255. * ((value + 1) / 2.)).clamp(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
            if self.name == 'raev2':
                return self.decoder.decode(z).clamp(0, 1).mul(255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
            return np.round(np.clip(((z + 1) / 2).cpu().numpy().transpose(0, 2, 3, 1) * 255, 0, 255)).astype(np.uint8)
