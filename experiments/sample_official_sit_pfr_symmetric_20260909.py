"""Symmetric XL PFR scale curves; immutable discovery bank and native parity."""
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
from experiments.audit_official_sit_pfr_interface import prefix
from experiments.internal_guidance_path_extrapolation import project_to_forward_ray

DATA = Path('/home/zhoushunyu/data/eqvae/experiments/official_sit_pfr_symmetric_20260909')
BASE = Path('/home/zhoushunyu/data/eqvae/experiments/official_sit_baseline_control_20260909')
OLD = Path('/home/zhoushunyu/data/eqvae/experiments/official_sit_pfr_20260908')
CKPT = Path('/home/zhoushunyu/data/eqvae/models/Internal-Guidance/official/SiT/SiT-XL-IG-ImageNet256-800EP.pt')
ARMS = {'projected135': ('projected', 1.35), 'time140': ('time_only', 1.4),
        'projected140': ('projected', 1.4), 'time150': ('time_only', 1.5),
        'projected150': ('projected', 1.5), 'time175': ('time_only', 1.75),
        'projected175': ('projected', 1.75)}
DIAGNOSTIC_STEPS = (0, 10, 25, 40, 49)


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def verify_time_reversal():
    gen = torch.Generator().manual_seed(202609509)
    z, full, base = (torch.randn(4, 4, 32, 32, generator=gen) for _ in range(3))
    scale, h = 1.75, 1/32
    guided = base + scale * (full-base)
    calibration = scale * (full-base)
    noise_projection = project_to_forward_ray(calibration, guided)
    data_projection = project_to_forward_ray(-calibration, -guided)
    assert torch.equal(noise_projection.coefficient, data_projection.coefficient)
    assert torch.equal(noise_projection.parallel, -data_projection.parallel)
    assert torch.equal(z-h*noise_projection.parallel, z+h*data_projection.parallel)
    zero = project_to_forward_ray(calibration, torch.zeros_like(guided))
    backwards = project_to_forward_ray(-guided, guided)
    assert torch.isfinite(zero.parallel).all() and torch.count_nonzero(zero.parallel) == 0
    assert torch.count_nonzero(backwards.parallel) == 0
    return dict(passed=True, simultaneous_velocity_sign_reversal=True,
                query_coordinates_equal=True, zero_and_reverse_ray_finite=True)


@torch.inference_mode()
def sample_batch(model, vae, noise, labels, scale, mode, *, check_prefix=False, diagnostics=True):
    z = noise.double()
    grid = torch.linspace(1, 0, 101, dtype=torch.float64)
    full_calls, prefix_calls, probe_calls, records = 0, 0, 0, []
    for step, (t, s) in enumerate(zip(grid[:-1], grid[1:])):
        ts = (torch.ones(len(z), device='cuda', dtype=torch.float64) * t).float()
        full, base, _ = model(z.float(), ts, labels)
        full_calls += 1
        if check_prefix and step in (0, 25, 50, 75):
            assert torch.equal(base, prefix(model, z.float(), ts, labels))
            probe_calls += 1
        drift = base + scale * (full-base)
        if float(t) > .5:
            h = min(1/32, float(t)-.5)
            tf = torch.full((len(z),), max(.5, float(t)-1/32), device='cuda')
            query = z.float()
            coefficient = torch.zeros(len(z), device='cuda')
            shift = torch.zeros_like(query)
            if mode == 'projected':
                projection = project_to_forward_ray(scale*(full-base), drift)
                coefficient = projection.coefficient
                shift = -h * projection.parallel
                query = query + shift
            elif mode != 'time_only':
                raise ValueError(mode)
            future = prefix(model, query, tf, labels)
            prefix_calls += 1
            correction = scale * (base-future)
            if diagnostics and step in DIAGNOSTIC_STEPS:
                rms = lambda x: x.float().flatten(1).square().mean(1).sqrt()
                records.append(torch.stack((coefficient, rms(shift), rms(correction), rms(drift)), dim=1))
            drift = drift + correction
        z = z + (s-t) * drift.double()
    assert full_calls == 100 and prefix_calls == 50
    if not torch.isfinite(z).all():
        raise RuntimeError('Nonfinite endpoint')
    decoded = vae.decode(z.float() / .18215).sample
    pixels = (255.*((decoded+1)/2.)).clamp(0,255).permute(0,2,3,1).to('cpu',torch.uint8).numpy()
    stats = torch.stack(records, dim=1).cpu().numpy() if records else None
    return pixels, z.float().cpu().numpy(), stats, (full_calls, prefix_calls, probe_calls)


def prepare():
    DATA.mkdir(parents=True, exist_ok=False)
    previous = json.loads((BASE/'request.json').read_text())
    assert json.loads((BASE/'status.json').read_text())['phase'] == 'complete'
    for path, digest in previous['sources'].items():
        assert file_sha256(Path(path)) == digest, path
    assert file_sha256(BASE/'inputs.npz') == previous['input_file_sha256']
    (DATA/'inputs.npz').write_bytes((BASE/'inputs.npz').read_bytes())
    with np.load(DATA/'inputs.npz') as bank:
        assert array_sha(bank['noise']) == previous['noise_sha256']
        assert array_sha(bank['labels']) == previous['label_sha256']
    assert file_sha256(CKPT) == previous['checkpoint_sha256']
    repo = ROOT/'research_repos/internal_guidance_study/Internal-Guidance/SiT'
    sources = [Path(__file__), ROOT/'experiments/run_official_sit_pfr_symmetric_20260909.py',
               ROOT/'experiments/run_internal_guidance_sit_audit.py',
               ROOT/'experiments/audit_official_sit_pfr_interface.py',
               ROOT/'experiments/internal_guidance_path_extrapolation.py',
               ROOT/'experiments/evaluate_raev2_official_samples.py',
               repo/'models/sit.py', repo/'samplers.py',
               ROOT/'docs/OFFICIAL_SIT_PFR_SYMMETRIC_PROTOCOL_20260909_ZH.md']
    snapshot = DATA/'sources'
    snapshot.mkdir()
    for i, path in enumerate(sources):
        (snapshot/(str(i)+'_'+path.name)).write_bytes(path.read_bytes())
    references = {}
    ref_dirs = [('ordinary135',OLD/'ordinary115/quality','ordinary',1.35),
                ('time135',OLD/'pfr/quality','time_only',1.35),
                ('ordinary140',BASE/'ig140_all','ordinary',1.4),
                ('ordinary150',BASE/'ig150_all','ordinary',1.5),
                ('ordinary175',BASE/'ig175_all','ordinary',1.75)]
    for name, directory, method, scale in ref_dirs:
        metrics = json.loads((directory/'fid.json').read_text())[0]
        assert file_sha256(directory/'samples.npz') == metrics['sample_sha256']
        references[name] = dict(directory=str(directory), method=method, scale=scale,
            pixel_sha256=metrics['sample_sha256'], metrics_sha256=file_sha256(directory/'fid.json'))
    write_json(DATA/'request.json',dict(arms=ARMS, samples=1000, batch_size=4, seed=202609428,
        steps=100, prefix_per_image=50, noise_sha256=previous['noise_sha256'],
        label_sha256=previous['label_sha256'], input_file_sha256=file_sha256(DATA/'inputs.npz'),
        checkpoint_sha256=previous['checkpoint_sha256'], vae_state_sha256=previous['vae_state_sha256'],
        baseline_request_sha256=file_sha256(BASE/'request.json'), sources={str(p):file_sha256(p) for p in sources},
        references=references, diagnostic_steps=DIAGNOSTIC_STEPS,
        diagnostic_fields=['alpha','query_shift_rms','correction_rms','ordinary_drift_rms'],
        sampling_bank_role='reused_1k_discovery', time_reversal=verify_time_reversal()))
    print('Inputs/assets and projection time reversal verified.',flush=True)


@torch.inference_mode()
def worker(rank):
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    request = json.loads((DATA/'request.json').read_text())
    for path, digest in request['sources'].items():
        assert file_sha256(Path(path)) == digest, path
    assert file_sha256(DATA/'inputs.npz') == request['input_file_sha256']
    with np.load(DATA/'inputs.npz') as bank:
        noise, labels = bank['noise'], bank['labels']
    repo = ROOT/'research_repos/internal_guidance_study/Internal-Guidance/SiT'
    model, meta = load_model(repo=repo,checkpoint_path=CKPT,model_name='SiT-XL/2',
        encoder_depth=8,state_key='ema',device=torch.device('cuda'))
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse',local_files_only=True).cuda().eval().requires_grad_(False)
    vh = hashlib.sha256()
    for name, value in sorted(vae.state_dict().items()):
        vh.update(name.encode()); vh.update(value.cpu().contiguous().numpy().tobytes())
    assert vh.hexdigest() == request['vae_state_sha256']
    start = rank*4
    parity, _, _, calls = sample_batch(model,vae,torch.from_numpy(noise[start:start+4]).cuda(),
        torch.from_numpy(labels[start:start+4]).cuda(),1.35,'time_only',check_prefix=True,diagnostics=False)
    with np.load(OLD/'pfr/quality/samples.npz') as reference:
        assert np.array_equal(parity, reference['arr_0'][start:start+4]), f'Rank {rank}: PFR parity failed'
    assert calls == (100,50,4)
    write_json(DATA/f'parity_rank{rank}.json',dict(passed=True,indices=list(range(start,start+4)),
        prefix_parity=True,pixel_sha256=array_sha(parity)))
    while not (DATA/'parity_passed.json').exists(): time.sleep(.25)
    for arm, (mode, scale) in ARMS.items():
        arm_dir = DATA/arm
        arm_dir.mkdir(exist_ok=True)
        samples, endpoints, stats, indices = [], [], [], []
        batch_calls = np.zeros(3,dtype=np.int64)
        batches = 0
        torch.cuda.synchronize(); began = time.perf_counter()
        for batch in range(rank,250,4):
            start = batch*4
            pixels, latent, diagnostics, calls = sample_batch(model,vae,
                torch.from_numpy(noise[start:start+4]).cuda(),torch.from_numpy(labels[start:start+4]).cuda(),scale,mode)
            samples.append(pixels); endpoints.append(latent); stats.append(diagnostics)
            indices.extend(range(start,start+4)); batch_calls += calls; batches += 1
            if batches % 16 == 0:
                print(json.dumps(dict(arm=arm,rank=rank,images=batches*4,seconds=time.perf_counter()-began)),flush=True)
        torch.cuda.synchronize(); elapsed = time.perf_counter()-began
        path = arm_dir/f'rank{rank}.npz'
        np.savez(path,arr_0=np.concatenate(samples),latents=np.concatenate(endpoints),
            diagnostics=np.concatenate(stats),indices=np.array(indices,dtype=np.int64))
        write_json(arm_dir/f'rank{rank}.json',dict(complete=True,rank=rank,arm=arm,mode=mode,scale=scale,
            guidance_high=1.,steps=100,samples=len(indices),batch_size=4,full_batch_calls=int(batch_calls[0]),
            full_sample_calls=int(batch_calls[0])*4,prefix_batch_calls=int(batch_calls[1]),
            prefix_sample_calls=int(batch_calls[1])*4,probe_batch_calls=int(batch_calls[2]),
            inference_seconds=elapsed,pixel_file_sha256=file_sha256(path),metadata=meta,
            noise_sha256=array_sha(noise[indices]),label_sha256=array_sha(labels[indices]),
            request_sha256=file_sha256(DATA/'request.json')))
        print(json.dumps(dict(arm=arm,rank=rank,complete=True,seconds=elapsed)),flush=True)
        while not (arm_dir/'advance.json').exists(): time.sleep(.25)
    for path, digest in request['sources'].items(): assert file_sha256(Path(path)) == digest, path


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--rank',type=int,choices=range(4))
    a = p.parse_args()
    if a.prepare: prepare()
    elif a.rank is not None: worker(a.rank)
    else: p.error('--prepare or --rank required')
