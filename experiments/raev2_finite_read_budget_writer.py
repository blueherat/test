#!/usr/bin/env python3
"""Nonlinear latent writer with native input and finite read-energy budgets.

Four conditional-gradient iterations propose candidates. Every accepted sample
is checked against actual native BF16 forward values, both finite budgets and
the finite message projection. Autograd is only a search heuristic through the
quantized model; no derivative approximation is used as an acceptance claim.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.raev2_finite_guidance_retention import CONFIG, CHECKPOINT, sha, put, read

SCRIPT = Path(__file__).resolve()
PLAN = {
    'protocol': 'raev2_finite_read_budget_writer_v1',
    'hypothesis': 'At unchanged native latent-write budget and no larger finite Full read response, orient more of that response along the frozen native guidance message.',
    'reader': 'Full at the original successor noise time s, unchanged class and B8.',
    'objective': 'Maximize dot(M/||M||, Full(z_none+delta,s)-Full(z_none,s)).',
    'constraints': '||delta||<=||z_IG-z_Full|| and ||Full(z_none+delta,s)-Full(z_none,s)||<=native finite response norm.',
    'start': 'Unchanged original native IG successor, feasible by construction.',
    'iterations': 4, 'line_search': [1., .5, .25, .125],
    'optimization': 'Conditional-gradient ball vertex at current native state, four finite trial mixtures, best feasible improving trial per sample, recomputed assembled B8 forward after each update. Stop if an iteration accepts no sample.',
    'numerical': 'Input ball vertices are inset by a FP32 addition-error envelope; actual FP64 norms enforce the original budget. No numerical allowance for a larger read-response energy.',
    'derivative_status': 'Native BF16 autograd is a proposal heuristic, not the exact derivative of a discontinuous quantized function or a first-order extrapolation in IG strength.',
    'cohort': 'Reuse all 16 frozen B8 batches and all eight write times from finite_retention_v1; this is an adaptive mechanism follow-up, not independent confirmation.',
    'continuation': 'After the optimized one-time write, Full alone on every remaining native query to zero.',
    'triage': 'For a possible quality trial require finite constraints/monotonicity, improved terminal message projection at >=6/8 fixed time groups, and no lower mean cosine to the complete remaining-IG terminal contrast. These are mechanism criteria, not quality guarantees; full online solver cost must also be assessed.',
    'no_fid': True, 'parameter_training': False, 'precision': 'Original native BF16/B8 with TF32 on; FP32 latent arithmetic.',
}


def initialize(source, out):
    out.mkdir(exist_ok=False, parents=True)
    assert read(source/'audit.json')['complete']
    files = [SCRIPT, ROOT/'experiments/raev2_finite_guidance_retention.py',
             ROOT/'experiments/raev2_guidance_quadrature.py',
             ROOT/'experiments/sample_raev2_pfr_retiming.py',
             ROOT/'external/RAEv2/src/stage2/models/DDT.py', CONFIG]
    plan = dict(PLAN, source=str(source), source_analysis_sha256=sha(source/'analysis.json'),
                source_audit_sha256=sha(source/'audit.json'),
                source_sha256={str(p): sha(p) for p in files})
    put(out/'plan.json', plan)


def run(source, out, rank):
    for path in (ROOT, ROOT/'external/RAEv2/src'):
        sys.path.insert(0, str(path))
    from experiments.sample_raev2_pfr_retiming import load_config
    from utils.model_utils import instantiate_from_config
    plan = read(out/'plan.json')
    assert plan['source_analysis_sha256'] == sha(source/'analysis.json')
    for path, expected in plan['source_sha256'].items():
        assert sha(path) == expected
    target = out/f'rank{rank}'
    target.mkdir(exist_ok=False)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = load_config(CONFIG)
    model = instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(checkpoint['ema'], strict=True)
    del checkpoint
    cohort = read(source/'cohort.json')
    grid, forwards, backwards = cohort['time_grid'], 0, 0

    def evaluate(x, t, labels):
        nonlocal forwards
        with torch.autocast('cuda', dtype=torch.bfloat16):
            full, _ = model(x, torch.full((8,), t, device='cuda'), context=labels, attn_mask=None)
        forwards += 1
        return full.float()

    def norm(x):
        return x.double().flatten(1).square().sum(1).sqrt()

    def dot(a, b):
        return (a.double().flatten(1)*b.double().flatten(1)).sum(1)

    def metric(a, b):
        aa, bb = norm(a), norm(b)
        ab = dot(a, b)
        return {'projection': (ab/bb.square().clamp_min(1e-300)).cpu().tolist(),
                'cosine': (ab/(aa*bb).clamp_min(1e-300)).cpu().tolist(),
                'rms': (aa/(a[0].numel()**.5)).cpu().tolist()}

    began, completed = time.perf_counter(), []
    for c in cohort['selected'][rank::4]:
        old_path = source/f"rank{c['ordinal']%4}"/f"batch{c['ordinal']:02d}"
        old_summary = read(old_path/'summary.json')
        assert sha(old_path/'snapshots.npz') == old_summary['files']['snapshots.npz']
        old = np.load(old_path/'snapshots.npz')
        labels = torch.from_numpy(old['labels']).cuda()
        base = torch.from_numpy(old['successor_none']).cuda()
        current = torch.from_numpy(old['successor_once']).cuda()
        reference = torch.from_numpy(old['first_read_none']).cuda()
        message = torch.from_numpy(old['source_guided']-old['source_full']).cuda()
        message_unit = (message.double()/norm(message).view(8, 1, 1, 1)).float()
        k, s = c['query_index'], grid[c['query_index']+1]
        batch_dir = target/f"batch{c['ordinal']:02d}"
        batch_dir.mkdir()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            actual_reference = evaluate(base, s, labels)
            original_read = evaluate(current, s, labels)
            assert torch.equal(reference, actual_reference)
            assert np.array_equal(original_read.cpu().numpy(), old['first_read_once'])
            radius, response_cap = norm(current-base), norm(original_read-reference)
            assert radius.gt(0).all() and response_cap.gt(0).all()
            projection_original = dot(original_read-reference, message_unit)
            projection_best = projection_original.clone()
            chosen_read = original_read.clone()
            inset = 2*torch.finfo(torch.float32).eps*(norm(base)+radius)
            vertex_radius = (radius-inset).clamp_min(0)
        records = []
        for iteration in range(PLAN['iterations']):
            # Re-evaluate the current finite state, never the unwritten origin.
            variable = current.detach().requires_grad_(True)
            predicted = evaluate(variable, s, labels)
            objective = (predicted*message_unit).sum()
            gradient, = torch.autograd.grad(objective, variable)
            backwards += 1
            gradient = gradient.detach()
            del predicted, objective, variable
            with torch.no_grad():
                gn = norm(gradient)
                vertex = (vertex_radius.view(8, 1, 1, 1)*gradient.double()/gn.clamp_min(1e-300).view(8, 1, 1, 1)).float()
                current_delta = current-base
                candidate_best = current.clone()
                score_before = projection_best.clone()
                alpha_best = torch.zeros(8, dtype=torch.float64, device='cuda')
                accepted_trials = torch.zeros(8, dtype=torch.int64, device='cuda')
                for alpha in PLAN['line_search']:
                    candidate = base+((1-alpha)*current_delta+alpha*vertex)
                    candidate_read = evaluate(candidate, s, labels)
                    candidate_projection = dot(candidate_read-reference, message_unit)
                    feasible = norm(candidate-base).le(radius) & norm(candidate_read-reference).le(response_cap) & gn.gt(0)
                    accept = feasible & candidate_projection.gt(projection_best)
                    candidate_best = torch.where(accept.view(8, 1, 1, 1), candidate, candidate_best)
                    projection_best = torch.where(accept, candidate_projection, projection_best)
                    alpha_best = torch.where(accept, alpha, alpha_best)
                    accepted_trials += accept.to(torch.int64)
                current = candidate_best
                chosen_read = evaluate(current, s, labels)
                actual_projection = dot(chosen_read-reference, message_unit)
                assert torch.equal(actual_projection, projection_best), 'assembled B8 readout parity'
                assert norm(current-base).le(radius).all() and norm(chosen_read-reference).le(response_cap).all()
                assert actual_projection.ge(score_before).all()
                records.append({'iteration': iteration, 'alpha': alpha_best.cpu().tolist(),
                    'accepted_trials': accepted_trials.cpu().tolist(),
                    'projection_before': score_before.cpu().tolist(), 'projection_after': actual_projection.cpu().tolist(),
                    'latent_radius_ratio': (norm(current-base)/radius).cpu().tolist(),
                    'read_response_ratio': (norm(chosen_read-reference)/response_cap).cpu().tolist()})
                if not accepted_trials.gt(0).any():
                    break
        with torch.no_grad():
            optimized_state = current.clone()
            first_read = chosen_read.clone()
            for j in range(k+1, 100):
                full = chosen_read if j == k+1 else evaluate(current, grid[j], labels)
                current = current-(grid[j]-grid[j+1])*((current-full)/max(grid[j], .05))
            endpoint_none = torch.from_numpy(old['endpoint_none']).cuda()
            endpoint_native = torch.from_numpy(old['endpoint_once']).cuda()
            endpoint_all = torch.from_numpy(old['endpoint_all']).cuda()
            assert torch.isfinite(current).all()
            terminal = metric(current-endpoint_none, message)
            terminal_native = metric(endpoint_native-endpoint_none, message)
            to_all = metric(current-endpoint_none, endpoint_all-endpoint_none)
            to_all_native = metric(endpoint_native-endpoint_none, endpoint_all-endpoint_none)
            np.savez(batch_dir/'snapshots.npz', ids=old['ids'], successor_writer=optimized_state.cpu().numpy(),
                     read_writer=first_read.cpu().numpy(), endpoint_writer=current.cpu().numpy())
            torch.cuda.synchronize()
            summary = {'complete': True, 'ordinal': c['ordinal'], 'write_query': k, 'samples': 8,
                'iterations': records, 'terminal': terminal, 'terminal_native': terminal_native,
                'to_all': to_all, 'to_all_native': to_all_native,
                'initial_projection': projection_original.cpu().tolist(), 'final_projection': projection_best.cpu().tolist(),
                'read_radius': response_cap.cpu().tolist(), 'latent_radius': radius.cpu().tolist(),
                'final_latent_radius_ratio': (norm(optimized_state-base)/radius).cpu().tolist(),
                'final_read_response_ratio': (norm(first_read-reference)/response_cap).cpu().tolist(),
                'snapshot_sha256': sha(batch_dir/'snapshots.npz'), 'seconds': time.perf_counter()-start}
            put(batch_dir/'summary.json', summary)
            completed.append(c['ordinal'])
            put(target/'progress.json', {'completed_ordinals': completed, 'forward_batch_calls': forwards,
                                        'backward_batch_calls': backwards, 'seconds': time.perf_counter()-began})
            print(json.dumps({'rank': rank, 'ordinal': c['ordinal'], 'seconds': time.perf_counter()-began}), flush=True)
    for path, expected in plan['source_sha256'].items():
        assert sha(path) == expected
    put(target/'complete.json', {'complete': True, 'samples': len(completed)*8,
        'forward_batch_calls': forwards, 'backward_batch_calls': backwards,
        'seconds': time.perf_counter()-began, 'max_memory_allocated': torch.cuda.max_memory_allocated()})


def analyze(source, out):
    cohort = read(source/'cohort.json')['selected']
    results = []
    for c in cohort:
        path = out/f"rank{c['ordinal']%4}"/f"batch{c['ordinal']:02d}"
        r = read(path/'summary.json')
        assert r['complete'] and sha(path/'snapshots.npz') == r['snapshot_sha256']
        results.append(r)
    groups = []
    for k in sorted({r['write_query'] for r in results}):
        rs = [r for r in results if r['write_query'] == k]
        mean = lambda group, metric: float(np.mean([x for r in rs for x in r[group][metric]]))
        groups.append({'write_query': k,
            'terminal_projection_native': mean('terminal_native', 'projection'),
            'terminal_projection_writer': mean('terminal', 'projection'),
            'terminal_cosine_native': mean('terminal_native', 'cosine'),
            'terminal_cosine_writer': mean('terminal', 'cosine'),
            'to_all_cosine_native': mean('to_all_native', 'cosine'),
            'to_all_cosine_writer': mean('to_all', 'cosine'),
            'mean_initial_read_projection': float(np.mean([x for r in rs for x in r['initial_projection']])),
            'mean_final_read_projection': float(np.mean([x for r in rs for x in r['final_projection']]))})
    improved_groups = sum(g['terminal_projection_writer'] > g['terminal_projection_native'] for g in groups)
    average_to_all_gain = float(np.mean([g['to_all_cosine_writer']-g['to_all_cosine_native'] for g in groups]))
    ranks = [read(out/f'rank{rank}'/'complete.json') for rank in range(4)]
    assert all(r['complete'] and r['samples'] == 32 for r in ranks)
    summary = {'complete': True, 'samples': 128, 'groups': groups,
        'terminal_projection_improved_groups': improved_groups, 'mean_to_all_cosine_gain': average_to_all_gain,
        'mechanism_triage_passed': improved_groups >= 6 and average_to_all_gain >= 0,
        'quality_evaluated': False, 'goal_complete': False,
        'forward_batch_calls': sum(r['forward_batch_calls'] for r in ranks),
        'backward_batch_calls': sum(r['backward_batch_calls'] for r in ranks),
        'worker_seconds': [r['seconds'] for r in ranks], 'max_memory_allocated': max(r['max_memory_allocated'] for r in ranks),
        'max_latent_budget_ratio': max(x for r in results for x in r['final_latent_radius_ratio']),
        'max_read_response_budget_ratio': max(x for r in results for x in r['final_read_response_ratio']),
        'accepted_sample_updates': sum(x > 0 for r in results for it in r['iterations'] for x in it['alpha'])}
    put(out/'analysis.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def coordinate(source, out):
    handles, jobs, start = [], [], time.perf_counter()
    for rank in range(4):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', HF_HUB_OFFLINE='1')
        handle = (out/f'worker{rank}.log').open('wb')
        job = subprocess.Popen([sys.executable, str(SCRIPT), '--stage', 'run', '--source', str(source), '--output', str(out), '--rank', str(rank)],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
        jobs.append(job)
        handles.append(handle)
    put(out/'live_workers.json', {'controller_pid': os.getpid(), 'workers': [
        {'rank': rank, 'pid': p.pid, 'starttime': Path(f'/proc/{p.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]}
        for rank, p in enumerate(jobs)]})
    try:
        while any(p.poll() is None for p in jobs):
            if any(p.poll() not in (None, 0) for p in jobs):
                raise RuntimeError(f'worker failure: {[(p.pid, p.poll()) for p in jobs]}')
            time.sleep(5)
        assert all(p.returncode == 0 for p in jobs)
        analyze(source, out)
        put(out/'completion.json', {'complete': True, 'seconds': time.perf_counter()-start, 'analysis_sha256': sha(out/'analysis.json')})
    finally:
        for p in jobs:
            if p.poll() is None:
                p.terminate()
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('initialize', 'run', 'coordinate', 'analyze'))
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--rank', type=int, default=0, choices=range(4))
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    {'initialize': lambda: initialize(source, out), 'run': lambda: run(source, out, args.rank),
     'coordinate': lambda: coordinate(source, out), 'analyze': lambda: analyze(source, out)}[args.stage]()
