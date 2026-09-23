"""Paired 5K checkpoint evaluation of JiT scales and SiT joint head/scales."""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from experiments.adversarial_weak_training_20260915 import common as c
from experiments.guidance_dynamic_50k_20260915.models import fingerprint


class IndexedGraph:
    """Forward-only replay with live time, label and left-interval index inputs."""
    def __init__(self, field, noise, labels):
        self.x = noise.clone()
        self.y = labels.clone()
        self.t = noise.new_tensor(.2)
        self.index = noise.new_tensor(0.)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                field(self.x, self.t, self.y, self.index, True)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.out = field(self.x, self.t, self.y, self.index, True)
        torch.cuda.current_stream().wait_stream(stream)

    def __call__(self, x, t, y, index, active):
        assert active
        self.x.copy_(x); self.t.copy_(t); self.y.copy_(y); self.index.copy_(index)
        self.graph.replay()
        return self.out.clone()


@torch.no_grad()
def integrate(field, noise, labels, steps, final_euler=False):
    grid = torch.linspace(0, 1, steps+1, device=noise.device)
    indices = torch.arange(steps, device=noise.device, dtype=noise.dtype)
    state = noise.clone()
    for i in range(steps):
        t, u, index = grid[i], grid[i+1], indices[i]
        h = u-t
        first = field(state, t, labels, index, True)
        predictor = state+h*first
        if final_euler and i == steps-1:
            state = predictor
        else:
            second = field(predictor, u, labels, index, True)
            state = state+(h/2)*(first+second)
    return state


@torch.no_grad()
def load_and_audit(args, state, noise, labels):
    """Check initialization against prior evaluator and learned endpoint against training."""
    if args.model == 'jit':
        from .jit_schedule import load_runtime, constant_schedule, reference, optimize, sampler, signature, Field
        from .evaluate_jit_ssg import integrate as old_integrate, official_field
        assert state['objective'] == 'jit_block1_gan_signed_schedule'
        runtime, provenance = load_runtime(Path(state['provenance']['weak_checkpoint']))
        assert provenance == state['provenance']
        schedule = constant_schedule(state['config']['initial_extra_a'])
        initial = old_integrate(official_field(runtime, 1+state['config']['initial_extra_a'], 1.), noise, labels)
        torch.testing.assert_close(reference(runtime, schedule, noise, labels), initial, rtol=0, atol=0)
        schedule.load_state_dict(state['ema' if args.weights == 'ema' else 'schedule'], strict=True)
        expected = reference(runtime, schedule, noise, labels)
        optimize(runtime, precast=state['config']['precast'], checkpoint_backbone=False)
        assert signature(runtime.net) == state['frozen']
        engine = sampler(runtime, schedule, noise, labels, graphs=False)
        torch.testing.assert_close(engine(noise, labels), expected, rtol=0, atol=0)
        del engine
        field = Field(runtime, schedule)
        graph = IndexedGraph(field, noise, labels)
        torch.testing.assert_close(integrate(graph, noise, labels, 50, True), expected, rtol=0, atol=0)
        schedule.requires_grad_(False)
        initial_signature = (signature(runtime.net), signature(schedule))
        def verify():
            assert (signature(runtime.net), signature(schedule)) == initial_signature
        def pixels(endpoint):
            # Exact rounding/order used by the existing published-SSG evaluation.
            x = endpoint.float().cpu().numpy().transpose(0, 2, 3, 1)
            return np.round(np.clip((x+1)/2*255, 0, 255)).astype(np.uint8)
        metadata = dict(steps=50, nfe=99, cfg=1., solver='50 intervals: Heun except final Euler',
                        ssg_interval='full', precision='published BF16 autocast',
                        coefficients=schedule.coefficients.tolist(), provenance=provenance,
                        initial_coefficient=state['config']['initial_extra_a'])
    else:
        from experiments.guidance_dynamic_50k_20260915.models import Adapter
        from .evaluate_sit_transformer import load_head, Field as OldField, integrate as old_integrate
        from .sit_joint import JointGuidance, JointField, make_sampler
        assert state['objective'] == 'sit_joint_signed_schedule_gap_anchor_v1'
        adapter = Adapter('sit_small')
        source = Path(state['provenance']['path'])
        assert c.sha(source) == state['provenance']['sha256']
        weak, _ = load_head(adapter, source)
        assert state['frozen'] == dict(strong=fingerprint(adapter.model),
            vae=fingerprint(adapter.rt.vae), initial_weak=fingerprint(weak))
        initial_a = state['args']['coefficient']
        initial = old_integrate(OldField(adapter, weak, 'guided'), noise, labels, initial_a, 'full', 'guided')
        joint = JointGuidance(weak.requires_grad_(True), 64, initial_a).cuda().eval()
        torch.testing.assert_close(integrate(JointField(adapter, joint), noise, labels, 64), initial, rtol=0, atol=0)
        joint.load_state_dict(state['ema' if args.weights == 'ema' else 'joint'], strict=True)
        expected = integrate(JointField(adapter, joint), noise, labels, 64)
        engine = make_sampler(adapter, joint, noise, labels, graphs=False)
        torch.testing.assert_close(engine(noise, labels), expected, rtol=0, atol=0)
        del engine
        graph = IndexedGraph(JointField(adapter, joint), noise, labels)
        torch.testing.assert_close(integrate(graph, noise, labels, 64), expected, rtol=0, atol=0)
        joint.requires_grad_(False)
        initial_signature = (fingerprint(adapter.model), fingerprint(adapter.rt.vae), fingerprint(joint))
        def verify():
            assert (fingerprint(adapter.model), fingerprint(adapter.rt.vae), fingerprint(joint)) == initial_signature
        pixels = adapter.pixels
        metadata = dict(steps=64, nfe=128, cfg=1.,
                        solver='64-step Heun; shared left-step guidance amount', window='full',
                        precision='FP32/TF32 as previous SiT experiments',
                        coefficients=joint.schedule.coefficients.tolist(), provenance=state['provenance'],
                        initial_coefficient=initial_a, weak_and_schedule_weights_paired=True)
    c.atomic(args.output/'parity.json', dict(passed=True, batch=8,
        initial_matches_previous_evaluator_exactly=True, learned_matches_training_sampler_exactly=True,
        graph_matches_unoptimized_learned_endpoint_exactly=True, checkpoint_weights=args.weights))
    return graph, pixels, verify, metadata


@torch.no_grad()
def main(args):
    _, world = c.setup()
    assert world == 1
    request = c.read(args.output/'request.json')
    for name, digest in {**request['sources'], **request['assets']}.items():
        if c.sha(name) != digest:
            raise RuntimeError(f'Source/asset changed: {name}')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    checkpoint_sha = c.sha(args.checkpoint)
    bank = c.original.model_root(args.model)/'quality_inputs'
    noise = np.load(bank/'noise.npy', mmap_mode='r')
    labels = np.load(bank/'labels.npy')
    classes = 1000 if args.model == 'jit' else 100
    assert len(noise) == len(labels) == 5000
    assert np.all(np.bincount(labels, minlength=classes) == 5000//classes)
    z = torch.from_numpy(np.array(noise[:8])).cuda()
    y = torch.from_numpy(labels[:8].copy()).cuda()
    torch.set_float32_matmul_precision('high')
    graph, to_pixels, verify, metadata = load_and_audit(args, state, z, y)
    step = state['step']
    del state, z, y
    print(dict(phase='parity_passed', model=args.model, step=step, weights=args.weights), flush=True)
    if args.audit_only:
        c.atomic(args.output/'complete.json', dict(complete=True, audit_only=True))
        return
    output = np.lib.format.open_memmap(args.output/'pixels.npy', mode='w+', dtype=np.uint8,
                                     shape=(5000, 256, 256, 3))
    records = []
    begin = time.perf_counter()
    for start in range(0, 5000, 8):
        z = torch.from_numpy(np.array(noise[start:start+8])).cuda()
        y = torch.from_numpy(labels[start:start+8].copy()).cuda()
        endpoint = integrate(graph, z, y, metadata['steps'], args.model == 'jit')
        if not torch.isfinite(endpoint).all() or endpoint.abs().max() > 1e6:
            raise FloatingPointError(f'Divergent endpoint at {start}; checkpoint {args.checkpoint}')
        output[start:start+8] = to_pixels(endpoint)
        records.append(dict(start=start, samples=8, labels=labels[start:start+8].tolist(),
                            noise_sha256=c.original.array_sha(noise[start:start+8])))
        if start % 80 == 0:
            c.atomic(args.output/'progress.json', dict(phase='sampling', samples=start+8, total=5000,
                step=step, weights=args.weights, seconds=time.perf_counter()-begin, updated_utc=c.now()))
    output.flush()
    sample_path = args.output/'samples.npz'
    with sample_path.open('wb') as stream:
        np.savez(stream, arr_0=output)
    del output
    verify()
    summary = dict(complete=True, valid=True, n=5000, batch=8, model=args.model, step=step,
        weights=args.weights, checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_sha,
        samples_path=str(sample_path), samples_sha256=c.sha(sample_path), records=records,
        noise_sha256=c.sha(bank/'noise.npy'), labels_sha256=c.sha(bank/'labels.npy'),
        strong_unchanged=True, evaluated_weights_unchanged=True, sampling_seconds=time.perf_counter()-begin,
        **metadata)
    c.atomic(args.output/'summary.json', summary)
    c.atomic(args.output/'progress.json', dict(phase='scoring', samples=5000, total=5000,
        step=step, weights=args.weights, updated_utc=c.now()))
    del graph, to_pixels, verify, endpoint, z, y
    gc.collect(); torch.cuda.empty_cache()
    if args.model == 'jit':
        from .jit_ssg import ROOT
        reference = ROOT/'literature/fid_stats/jit_in256_stats.npz'
        cmd = [c.PYTHON, str(c.WORK/'experiments/evaluate_raev2_official_samples.py'),
            '--branch', f'learned={sample_path}', '--output', str(args.output/'official.csv'),
            '--batch-size', '64', '--device', 'cuda', '--fid-reference', str(reference),
            '--feature-cache-dir', str(args.output/'feature_cache')]
        with (args.output/'evaluation.log').open('w') as stream:
            subprocess.run(cmd, cwd=c.WORK, stdout=stream, stderr=subprocess.STDOUT, check=True)
        result = c.read(args.output/'official.json')[0]
        assert np.isfinite(result['fid']) and np.isfinite(result['inception_score'])
        c.atomic(args.output/'metrics.json', dict(summary, fid=result['fid'],
            inception_score=result['inception_score'], reference=str(reference), reference_sha256=c.sha(reference)))
    else:
        cmd = [c.PYTHON, '-u', '-m', 'experiments.adversarial_weak_training_20260915.score',
               '--stage', str(args.output), '--shared-gpu', os.environ['CUDA_VISIBLE_DEVICES']]
        with (args.output/'score_holder.log').open('w') as stream:
            subprocess.run(cmd, cwd=c.WORK, stdout=stream, stderr=subprocess.STDOUT, check=True)
    metric = c.read(args.output/'metrics.json')
    c.atomic(args.output/'complete.json', dict(complete=True, valid=True, fid=metric['fid']))
    c.atomic(args.output/'progress.json', dict(phase='complete', complete=True, samples=5000,
        total=5000, step=step, weights=args.weights, fid=metric['fid'], updated_utc=c.now()))
    print(dict(phase='complete', model=args.model, step=step, weights=args.weights, fid=metric['fid']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=('jit', 'sit_small'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--weights', choices=('raw', 'ema'), required=True)
    parser.add_argument('--audit-only', action='store_true')
    main(parser.parse_args())
