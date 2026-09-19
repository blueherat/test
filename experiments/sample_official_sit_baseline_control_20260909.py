"""Frozen ordinary-IG controls, preserving the historical B4 sampling bank."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.raev2_training_core import file_sha256
from experiments.run_internal_guidance_sit_audit import load_model

DATA = Path('/home/zhoushunyu/data/eqvae/experiments/official_sit_baseline_control_20260909')
OLD = Path('/home/zhoushunyu/data/eqvae/experiments/official_sit_pfr_20260908')
CKPT = Path('/home/zhoushunyu/data/eqvae/models/Internal-Guidance/official/SiT/SiT-XL-IG-ImageNet256-800EP.pt')
ARMS = {'ig140_all': (1.4, 1.0), 'ig150_all': (1.5, 1.0),
        'ig175_all': (1.75, 1.0), 'ig140_lig': (1.4, .7)}


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


@torch.inference_mode()
def sample_batch(model, vae, noise, labels, scale, high):
    z = noise.double()
    grid = torch.linspace(1, 0, 116, dtype=torch.float64)
    for t, s in zip(grid[:-1], grid[1:]):
        ts = (torch.ones(len(z), device='cuda', dtype=torch.float64) * t).float()
        full, base, _ = model(z.float(), ts, labels)
        drift = base + scale * (full - base) if float(t) <= high else full
        z = z + (s - t) * drift.double()
    if not torch.isfinite(z).all():
        raise RuntimeError('Nonfinite endpoint')
    decoded = vae.decode(z.float() / .18215).sample
    return (255. * ((decoded + 1) / 2.)).clamp(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()


def prepare():
    DATA.mkdir(parents=True, exist_ok=False)
    old = json.loads((OLD / 'ordinary115/quality/summary.json').read_text())
    torch.cuda.set_device(0)
    gen = torch.Generator(device='cuda').manual_seed(202609428)
    noise = np.concatenate([torch.randn(4, 4, 32, 32, generator=gen, device='cuda').cpu().numpy()
                            for _ in range(250)])
    labels = np.arange(1000, dtype=np.int64)
    assert array_sha(noise) == old['noise_sha256']
    assert array_sha(labels) == old['label_sha256']
    np.savez(DATA / 'inputs.npz', noise=noise, labels=labels)
    repo = ROOT / 'research_repos/internal_guidance_study/Internal-Guidance/SiT'
    sources = [Path(__file__), ROOT / 'experiments/run_official_sit_baseline_control_20260909.py',
               ROOT / 'experiments/run_internal_guidance_sit_audit.py',
               ROOT / 'experiments/evaluate_raev2_official_samples.py',
               repo / 'models/sit.py', repo / 'samplers.py',
               ROOT / 'docs/OFFICIAL_SIT_GUIDANCE_BASELINE_CONTROL_20260909_ZH.md']
    snapshot = DATA / 'sources'
    snapshot.mkdir()
    for index, path in enumerate(sources):
        (snapshot / (str(index) + '_' + path.name)).write_bytes(path.read_bytes())
    refs = {}
    for arm in ('ordinary115', 'pfr'):
        ref = OLD / arm / 'quality'
        summary = json.loads((ref / 'summary.json').read_text())
        assert file_sha256(ref / 'samples.npz') == summary['pixel_sha256']
        assert summary['noise_sha256'] == old['noise_sha256']
        assert summary['label_sha256'] == old['label_sha256']
        refs[arm] = {'directory': str(ref), 'summary': summary,
                     'metrics': json.loads((ref / 'fid.json').read_text())[0]}
    assert file_sha256(CKPT) == old['checkpoint_sha256']
    write_json(DATA / 'request.json', dict(arms=ARMS, samples=1000, batch_size=4,
        seed=202609428, steps=115, noise_sha256=array_sha(noise), label_sha256=array_sha(labels),
        input_file_sha256=file_sha256(DATA / 'inputs.npz'), sources={str(p): file_sha256(p) for p in sources},
        checkpoint_sha256=old['checkpoint_sha256'], vae_state_sha256=old['vae_state_sha256'], references=refs))
    print('Historical inputs and assets verified.', flush=True)


@torch.inference_mode()
def worker(rank):
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    request = json.loads((DATA / 'request.json').read_text())
    for path, digest in request['sources'].items():
        assert file_sha256(Path(path)) == digest, path
    assert file_sha256(DATA / 'inputs.npz') == request['input_file_sha256']
    with np.load(DATA / 'inputs.npz') as bank:
        noise, labels = bank['noise'], bank['labels']
    assert array_sha(noise) == request['noise_sha256']
    assert array_sha(labels) == request['label_sha256']
    repo = ROOT / 'research_repos/internal_guidance_study/Internal-Guidance/SiT'
    model, meta = load_model(repo=repo, checkpoint_path=CKPT, model_name='SiT-XL/2',
                            encoder_depth=8, state_key='ema', device=torch.device('cuda'))
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse', local_files_only=True).cuda().eval().requires_grad_(False)
    vh = hashlib.sha256()
    for name, value in sorted(vae.state_dict().items()):
        vh.update(name.encode())
        vh.update(value.cpu().contiguous().numpy().tobytes())
    assert vh.hexdigest() == request['vae_state_sha256']
    start = rank * 4
    parity = sample_batch(model, vae, torch.from_numpy(noise[start:start+4]).cuda(),
                          torch.from_numpy(labels[start:start+4]).cuda(), 1.35, 1.)
    with np.load(OLD / 'ordinary115/quality/samples.npz') as reference:
        expected = reference['arr_0'][start:start+4]
    assert np.array_equal(parity, expected), f'Rank {rank}: pixel parity failed'
    write_json(DATA / f'parity_rank{rank}.json', dict(passed=True, indices=list(range(start,start+4)), pixel_sha256=array_sha(parity)))
    while not (DATA / 'parity_passed.json').exists():
        time.sleep(.25)
    for arm, (scale, high) in ARMS.items():
        arm_dir = DATA / arm
        arm_dir.mkdir(exist_ok=True)
        samples, indices, batches = [], [], 0
        torch.cuda.synchronize()
        began = time.perf_counter()
        for batch in range(rank, 250, 4):
            start = batch * 4
            pixels = sample_batch(model, vae, torch.from_numpy(noise[start:start+4]).cuda(),
                                  torch.from_numpy(labels[start:start+4]).cuda(), scale, high)
            samples.append(pixels)
            indices.extend(range(start, start+4))
            batches += 1
            if batches % 16 == 0:
                print(json.dumps(dict(arm=arm, rank=rank, images=batches*4, seconds=time.perf_counter()-began)), flush=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-began
        path = arm_dir / f'rank{rank}.npz'
        np.savez(path, arr_0=np.concatenate(samples), indices=np.array(indices, dtype=np.int64))
        write_json(arm_dir / f'rank{rank}.json', dict(complete=True, rank=rank, arm=arm,
            scale=scale, guidance_high=high, steps=115, samples=len(indices), batch_size=4,
            full_batch_calls=115*batches, full_sample_calls=115*len(indices), prefix_calls=0,
            inference_seconds=elapsed, pixel_file_sha256=file_sha256(path), metadata=meta,
            noise_sha256=array_sha(noise[indices]), label_sha256=array_sha(labels[indices]),
            request_sha256=file_sha256(DATA / 'request.json')))
        print(json.dumps(dict(arm=arm, rank=rank, complete=True, seconds=elapsed)), flush=True)
        while not (arm_dir / 'advance.json').exists():
            time.sleep(.25)
    for path, digest in request['sources'].items():
        assert file_sha256(Path(path)) == digest, path


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--rank', type=int, choices=range(4))
    a = p.parse_args()
    if a.prepare:
        prepare()
    elif a.rank is not None:
        worker(a.rank)
    else:
        p.error('--prepare or --rank required')
