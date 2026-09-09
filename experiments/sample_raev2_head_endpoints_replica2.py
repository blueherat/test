"""Paired standalone Full/Base endpoints for an AG mechanism diagnostic."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch
from experiments import sample_raev2_pfr_retiming as native

ROOT = Path('/home/zhoushunyu/data/eqvae/experiments/ig_head_endpoint_distribution_replica2_20260908')
REFERENCE = ROOT.parent/'raev2_pfr_working_point_20260908/full/quality'


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['full', 'base'], required=True)
    args = parser.parse_args()
    out = ROOT/args.arm
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = native.load_config(native.DEFAULT_CONFIG)
    native.install_raev2_decoder_config_compat()
    checkpoint_hash = native.file_sha256(native.DEFAULT_CHECKPOINT)
    assert checkpoint_hash == '723c56d7fa77ace9613909f7e38cb2386b898608218dc9b52649bb373d513c9a'
    model = native.instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
    state = torch.load(native.DEFAULT_CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(state['ema'], strict=True)
    del state
    decoder = native.instantiate_from_config(cfg.stage_1).cuda().eval().requires_grad_(False)
    del decoder.encoder
    sources = [Path(__file__), Path(native.__file__), Path(inspect.getfile(native.clean_to_velocity)),
               Path(inspect.getfile(type(model))), native.DEFAULT_CONFIG]
    request = dict(arm=args.arm, samples=1000, batch=4, seed=202609492, steps=100, shift=8.,
                   precision='FP32 state, BF16 autocast, TF32 enabled',
                   checkpoint_sha256=checkpoint_hash,
                   protocol_sha256=native.file_sha256(Path('docs/IG_HEAD_ENDPOINT_REPLICA_20260908_ZH.md')),
                   sources={str(p): native.file_sha256(p) for p in sources},
                   purpose='Standalone source-distribution diagnostic; not a proposed guidance method.')
    (out/'request.json').write_text(json.dumps(request, indent=2)+'\n')
    latents = np.lib.format.open_memmap(out/'latents.npy', mode='w+', dtype=np.float32,
                                      shape=(1000, *cfg.misc.latent_size))
    grid = native.shifted_time_grid(100, 8., torch.device('cuda'))
    rng = torch.Generator(device='cuda').manual_seed(request['seed'])
    noise_hash, label_hash = hashlib.sha256(), hashlib.sha256()
    images = []
    started = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for start in range(0, 1000, 4):
            z = torch.randn(4, *cfg.misc.latent_size, generator=rng, device='cuda')
            labels = torch.arange(start, start+4, device='cuda')
            noise_hash.update(z.cpu().numpy().tobytes())
            label_hash.update(labels.cpu().numpy().tobytes())
            for t, s in zip(grid[:-1], grid[1:]):
                times = torch.full((4,), float(t), device='cuda')
                full, base = model(z, times, context=labels, attn_mask=None)
                prediction = full if args.arm == 'full' else base
                velocity = native.clean_to_velocity(prediction, z, times,
                                                    denominator_floor=float(cfg.transport.t_eps))
                z = z+(s-t)*velocity
            assert torch.isfinite(z).all()
            latents[start:start+4] = z.cpu().numpy()
            images.append(decoder.decode(z).clamp(0, 1).mul(255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy())
            if start == 0 or (start+4) % 100 == 0:
                progress = dict(arm=args.arm, done=start+4, seconds=time.perf_counter()-started)
                (out/'progress.json').write_text(json.dumps(progress)+'\n')
                print(json.dumps(progress), flush=True)
    latents.flush()
    pixels = np.concatenate(images)
    np.savez(out/'samples.npz', arr_0=pixels)
    reference = json.loads((REFERENCE/'summary.json').read_text())
    assert label_hash.hexdigest() == reference['label_sha256']
    parity = None
    result = dict(complete=True, arm=args.arm, samples=1000,
                  seconds=time.perf_counter()-started, full_model_batch_calls=25000,
                  encoder_sample_calls=100000, full_head_sample_calls=100000, base_head_sample_calls=100000,
                  noise_sha256=noise_hash.hexdigest(), label_sha256=label_hash.hexdigest(),
                  full_reference_pixel_parity=parity,
                  request_sha256=native.file_sha256(out/'request.json'),
                  files={p.name:native.file_sha256(p) for p in [out/'latents.npy', out/'samples.npz']})
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
