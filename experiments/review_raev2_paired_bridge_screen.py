#!/usr/bin/env python3
"""Independently review the completed paired-bridge screen, using CPU only.

Reads frozen records, all pixel archives, saved features and inherited cost
records. Writes one exclusive review JSON. Never imports a sampler/evaluator,
executes a model, starts CUDA, resumes a job, or selects a method from quality.
The low-rank FID calculation reuses saved official Inception features; it does
not independently re-extract features or reconstruct unsaved full noise/latents.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PILOT = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/paired_bridge_v1')
STUDY = PILOT/'screen_v1'
GPU_UUID = 'GPU-7d3e4e7d-abfa-e06e-c264-796052797949'
PROTOCOL = 'raev2_paired_bridge_fixed_1k_v1'
SAMPLING_PROTOCOL = 'raev2_paired_bridge_sampling_v1'
TRAINING_PROTOCOL = 'raev2_paired_bridge_fixed_mechanism_pilot_v1'
SEED, COUNT, BATCH = 202609151, 1000, 8
BRIDGE_SHA = '5013cbf075ddffa5c2ae021fc916be0615ca41d9817d3af91e6b3e46c86e1983'
REFERENCE = Path('/home/zhoushunyu/.cache/nanogen-evals/stats/datasets--nanovisionx--nanogen-evals-stats/snapshots/0227134b29f25704c3856ec002ce4a2183cc7419/imagenet_256_fid_stats.npz')
REFERENCE_SHA = '925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac'
INCEPTION = Path('/home/zhoushunyu/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth')
INCEPTION_SHA = '6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2'
EVALUATOR_COMMIT = '19dfb4c2705333eb8b97e454fb354d47d1fe135b'
IDENTITY_KEYS = ('global_noise_sha256', 'noise_rng_state_sha256', 'global_labels_sha256')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def instant(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(result.tzinfo is not None, 'timestamp requires explicit timezone')
    return result


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def close(actual, expected, label, atol=1e-9):
    require(len(actual) == len(expected), label+' length')
    require(all(math.isfinite(a) and math.isfinite(b) and abs(a-b) <= atol
                for a, b in zip(actual, expected)), label+f': {actual} != {expected}')


def costs(summary):
    require(summary.get('complete') is True and summary.get('samples') == COUNT, 'complete 1K summary required')
    pair = (summary['trajectory_through_decode_wall_seconds'], summary['total_wall_seconds_before_summary'])
    require(all(positive(x) for x in pair) and pair[1] >= pair[0], 'invalid measured T/W')
    return pair


def select_steps(candidate, previous, previous_steps=None):
    """Independent exact binary-float cost rule; no producer helper imported."""
    require(all(positive(x) for x in (*candidate, *previous)), 'invalid selection costs')
    require(candidate[1] >= candidate[0] and previous[1] >= previous[0], 'invalid whole-run cost')
    k = 100 if previous_steps is None else previous_steps
    require(isinstance(k, int) and not isinstance(k, bool) and k >= 100, 'invalid previous K')
    lower = 100 if previous_steps is None else k+1
    return max(lower, *(math.ceil(Fraction(k)*Fraction(a)/Fraction(b)) for a, b in zip(candidate, previous)))


def counts(calls):
    return {key: value for name, number in calls.items()
            for key, value in ((name+'_forward_calls', number), (name+'_sample_forwards', number*BATCH))}


def add_counts(rows):
    keys = set().union(*(row.keys() for row in rows))
    return {key: sum(row.get(key, 0) for row in rows) for key in sorted(keys)}


def check_counts(record, expected, label):
    require(record['observed'] == expected and record['expected'] == expected, label+' actual/expected hooks differ')
    require(all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in record['observed'].values()),
            label+' invalid integer count')


def low_rank_fid(features, reference_mean, reference_covariance):
    """Sample covariance rank <= n-1 gives an independent n-by-n Bures trace."""
    x = np.asarray(features, dtype=np.float64)
    mu = np.asarray(reference_mean, dtype=np.float64)
    cov = np.asarray(reference_covariance, dtype=np.float64)
    require(x.ndim == 2 and len(x) > 1 and mu.shape == (x.shape[1],)
            and cov.shape == (x.shape[1], x.shape[1]), 'FID dimensions')
    require(np.isfinite(x).all() and np.isfinite(mu).all() and np.isfinite(cov).all(), 'nonfinite FID input')
    require(np.allclose(cov, cov.T, rtol=0, atol=1e-10), 'asymmetric reference covariance')
    mean = x.mean(axis=0)
    centered = x-mean
    gram = (centered @ cov @ centered.T)/(len(x)-1)
    eigenvalues = np.linalg.eigvalsh((gram+gram.T)*.5)
    require(eigenvalues.min() > -1e-8, 'materially negative Bures Gram eigenvalue')
    result = float(np.sum((mean-mu)**2)+np.sum(centered**2)/(len(x)-1)+np.trace(cov)
                   -2*np.sqrt(np.maximum(eigenvalues, 0)).sum())
    require(math.isfinite(result) and result >= -1e-8, 'invalid reconstructed FID')
    return result, float(eigenvalues.min())


class Review:
    def __init__(self, study):
        self.study = study.resolve()
        self.verified = {}
        self.frozen = {}

    def verify(self, path, expected=None, size=None):
        path = Path(path).resolve()
        key = str(path)
        if key not in self.verified:
            self.verified[key] = sha(path)
        if expected is not None:
            require(is_sha(expected) and self.verified[key] == expected, 'SHA differs: '+key)
        if size is not None:
            require(path.stat().st_size == size, 'byte size differs: '+key)
        return self.verified[key]

    def record(self, record, expected_path=None):
        require(isinstance(record, dict) and {'path', 'sha256'} <= record.keys(), 'invalid artifact record')
        path = Path(record['path']).resolve()
        if expected_path is not None:
            require(path == Path(expected_path).resolve(), 'unexpected artifact path: '+str(path))
        self.verify(path, record['sha256'], record.get('size_bytes'))
        return path

    def json_record(self, record, expected_path=None):
        return read(self.record(record, expected_path))

    def frozen_record(self, record):
        path = self.record(record)
        require(self.frozen.get(path) == record['sha256'], 'identity absent from plan freeze: '+str(path))

    def training(self, training, identities):
        checkpoint_path = self.record(training['checkpoint'], PILOT/'train/final.pt')
        require(training['checkpoint']['sha256'] == BRIDGE_SHA, 'different bridge checkpoint')
        for key in ('checkpoint', 'request', 'summary', 'supplemental_sources'):
            self.frozen_record(training[key])
        request = self.json_record(training['request'], checkpoint_path.parent/'request.json')
        summary = self.json_record(training['summary'], checkpoint_path.parent/'summary.json')
        require(summary.get('complete') is True and summary['protocol'] == TRAINING_PROTOCOL
                and summary['mode'] == 'train' and summary['updates'] == 2048, 'training completion/identity')
        require(summary['checkpoint'] == training['checkpoint'] and summary['request'] == training['request'],
                'checkpoint-training summary chain')
        require(request['protocol'] == TRAINING_PROTOCOL and request['mode'] == 'train'
                and request['config'] == identities['config']
                and request['baseline_checkpoint'] == identities['baseline_checkpoint']
                and request['baseline_checkpoint_step'] == 100080, 'training model mismatch')
        for relative, record in request['sources'].items():
            self.record(record)
            self.verify(ROOT/relative, record['sha256'])
        supplement = self.json_record(training['supplemental_sources'], PILOT/'supplemental_source_environment.json')
        for record in supplement['sources']:
            self.verify(ROOT/record['path'], record['sha256'])
        weights = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        require(weights['protocol'] == TRAINING_PROTOCOL and weights['updates'] == 2048
                and Path(weights['request']).resolve() == Path(training['request']['path']).resolve()
                and weights['request_sha256'] == training['request']['sha256'], 'checkpoint internal binding')
        for name in ('candidate', 'control'):
            require(weights[name]['time_frequencies'].shape == (16,), name+' fixed frequency buffer')
            require(sum(x.numel() for key, x in weights[name].items() if key != 'time_frequencies') == 3652224,
                    name+' parameter count excluding the fixed frequency buffer')
            require(all(x.device.type == 'cpu' and x.dtype == torch.float32 and bool(torch.isfinite(x).all())
                        for x in weights[name].values()), name+' CPU finite FP32 weights')

    def request(self, summary, name, mode, steps, *, parity=None):
        directory = self.study/name
        request = self.json_record(summary['request'], directory/'request.json')
        n = 16 if mode == 'parity' else COUNT
        command = 'parity' if mode == 'parity' else 'sample'
        for value in (summary, request):
            require(value['protocol'] == SAMPLING_PROTOCOL and value['command'] == command
                    and value['mode'] == mode and value['seed'] == SEED and value['num_steps'] == steps,
                    name+' protocol/cohort identity')
        require(summary['complete'] is True and request['sample_count'] == n and request['batch_size'] == BATCH
                and request['time_shift'] == 8 and request['transport_t_eps'] == .05
                and request['fid_performed'] is False and request['automatic_step_selection'] is False
                and request['cuda_visible_devices'] == GPU_UUID, name+' fixed request')
        require(request['auxiliary_checkpoint_deserialized'] == (mode != 'official'), name+' residency boundary')
        base = torch.linspace(1., 0., steps+1, dtype=torch.float32, device='cpu')
        grid = (8*base/(1+7*base)).tolist()
        require(request['time_grid'] == grid, name+' frozen CPU FP32 grid')
        for relative, record in request['sources'].items():
            self.record(record, directory/'sources'/relative)
            require(self.frozen.get((ROOT/relative).resolve()) == record['sha256'], name+' source freeze')
        for record in request['identities'].values():
            self.frozen_record(record)
        for key in ('checkpoint', 'request', 'summary', 'supplemental_sources'):
            self.frozen_record(request['training'][key])
        if parity is None:
            require(request['parity'] is None, 'parity must have no prior parity')
        else:
            require(request['identities'] == parity['identities'] and request['training'] == parity['training'],
                    name+' identities differ from parity')
            self.record(request['parity']['request'], self.study/'parity/request.json')
            self.record(request['parity']['summary'], self.study/'parity/summary.json')
            require({k: v['sha256'] for k, v in request['sources'].items()}
                    == {k: v['sha256'] for k, v in parity['sources'].items()}, name+' source keys differ')
        inputs = self.json_record(summary['sampling_input'], directory/'sampling_input.json')
        require(inputs['request'] == summary['request'] and inputs['seed'] == SEED
                and inputs['noise_shape'] == [n, 1024, 16, 16]
                and inputs['cuda_visible_devices'] == GPU_UUID
                and inputs['frozen_before_first_model_forward'] is True, name+' noise input record')
        for key in IDENTITY_KEYS:
            require(is_sha(summary[key]) and summary[key] == inputs[key], name+' input hash mismatch: '+key)
        require(inputs['global_labels_sha256'] == array_sha(np.arange(n, dtype=np.int64)), name+' global labels SHA')
        with np.load(self.record(inputs['paired_noise_audit'], directory/'paired_noise_audit.npz'), allow_pickle=False) as z:
            require(set(z.files) == {'first_noise', 'rng_state'} and z['first_noise'].shape == (1024, 16, 16)
                    and z['first_noise'].dtype == np.float32 and np.isfinite(z['first_noise']).all(), name+' saved noise audit')
            require(array_sha(z['rng_state']) == inputs['noise_rng_state_sha256'], name+' saved RNG bytes')
            first_noise_sha = array_sha(z['first_noise'])
        require(summary['loading']['checkpoint_step'] == 100080, name+' stage2 step')
        return request, first_noise_sha

    def parity(self, record):
        summary = self.json_record(record, self.study/'parity/summary.json')
        request, _ = self.request(summary, 'parity', 'parity', 100)
        self.training(request['training'], request['identities'])
        detail = self.json_record(summary['parity_result'], self.study/'parity/parity_result.json')
        response = self.json_record(summary['finite_response'], self.study/'parity/finite_response.json')
        for key, value in detail.items():
            require(summary[key] == value, 'parity detail/summary differs: '+key)
        modes = {'production', 'official', 'zero_candidate', 'zero_control'}
        require(summary['samples_per_loop'] == 16 and all(summary[key] is True for key in
                ('stepwise_bitwise', 'endpoint_bitwise', 'pixel_bitwise', 'parity_is_not_quality_or_cost_comparison')),
                'parity flags')
        for key in ('endpoint_sha256', 'pixels_sha256'):
            require(set(summary[key]) == modes and len(set(summary[key].values())) == 1
                    and all(is_sha(x) for x in summary[key].values()), 'four-loop parity digest differs')
        expected_batch = counts(dict(stage2=400, decoder=4, candidate=0, control=0, zero_candidate=200, zero_control=200))
        require(len(summary['batches']) == 2, 'parity B8 loops')
        for i, batch in enumerate(summary['batches']):
            require(batch['start'] == i*8 and batch['steps_compared'] == 100
                    and batch['stepwise_bitwise'] is True and batch['pixel_bitwise'] is True
                    and positive(batch['complete_loop_wall_seconds']), 'parity batch')
            check_counts(batch['forward_counts'], expected_batch, 'parity batch')
        require(response['complete'] is True and response['sample_ids'] == list(range(8))
                and response['t'] == request['time_grid'][0] and response['s'] == request['time_grid'][1]
                and set(response['responses']) == {'candidate', 'control'}, 'finite trained response')
        for value in response['responses'].values():
            require(all(math.isfinite(value[k]) and value[k] >= 0 for k in ('rms_change', 'max_abs_change'))
                    and is_sha(value['endpoint_sha256']), 'nonfinite response evidence')
        expected_response = counts(dict(stage2=1, decoder=0, candidate=2, control=2, zero_candidate=0, zero_control=0))
        check_counts(response['forward_counts'], expected_response, 'finite response')
        expected_total = add_counts([expected_batch, expected_batch, expected_response])
        require(summary['forward_counts_observed'] == expected_total, 'parity total forward hooks')
        require(positive(response['wall_seconds']) and positive(summary['total_wall_seconds_before_summary']), 'parity timings')
        return summary, request

    def sampling(self, name, record, parity_request):
        summary = self.json_record(record, self.study/name/'summary.json')
        mode = 'candidate' if name == 'candidate100' else ('control' if name == 'control100' else 'official')
        steps = int(name.removeprefix(mode))
        require(mode == 'official' or steps == 100, 'auxiliary step schedule changed')
        request, first_noise = self.request(summary, name, mode, steps, parity=parity_request)
        measured = costs(summary)
        require(summary['global_ids'] == list(range(COUNT)) and summary['global_cohort_size'] == COUNT
                and summary['fid_performed'] is False and summary['image_sampling_performed'] is True
                and summary['stage2_nfe_per_sample'] == steps, name+' cohort completion')
        expected_calls = dict(stage2=steps, decoder=1)
        if mode != 'official':
            expected_calls[mode] = 2*steps
        expected_batch = counts(expected_calls)
        expected_total = {key: value*125 for key, value in expected_batch.items()}
        require(summary['forward_counts_observed'] == expected_total
                and summary['forward_counts_expected'] == expected_total
                and all(summary[key] == value for key, value in expected_total.items()), name+' full hook totals')
        require(summary['auxiliary_forwards_per_sample'] == {k: 2*steps if mode == k else 0 for k in ('candidate', 'control')},
                name+' auxiliary per-sample counts')
        archive = summary['samples_npz']
        require(summary['sample_archive'] == archive and summary['archive_sha256'] == archive['sha256'], name+' archive aliases')
        with np.load(self.record(archive, self.study/name/'samples.npz'), allow_pickle=False) as z:
            require(set(z.files) == {'arr_0', 'ids', 'labels'}, name+' archive members')
            images, ids, labels = z['arr_0'], z['ids'], z['labels']
        require(images.shape == (COUNT, 256, 256, 3) and images.dtype == np.uint8
                and ids.dtype == labels.dtype == np.int64
                and np.array_equal(ids, np.arange(COUNT)) and np.array_equal(labels, ids), name+' image/label cohort')
        batches = self.json_record(summary['batch_manifest'], self.study/name/'batch_manifest.json')['batches']
        require(len(batches) == 125, name+' 125 B8 archives')
        paired = []
        for i, batch in enumerate(batches):
            selected = np.arange(i*8, i*8+8, dtype=np.int64)
            require(batch['global_ids'] == selected.tolist(), name+' batch IDs')
            check_counts(batch['forward_counts'], expected_batch, name+' batch hooks')
            require(batch['labels_sha256'] == array_sha(selected) and is_sha(batch['noise_sha256'])
                    and is_sha(batch['endpoint_sha256']), name+' batch identities')
            paired.append((batch['noise_sha256'], batch['labels_sha256']))
            expected_path = self.study/name/'batches'/f'{i*8:06d}_{i*8+8:06d}.npz'
            with np.load(self.record(batch['archive'], expected_path), allow_pickle=False) as z:
                require(set(z.files) == {'arr_0', 'ids', 'labels'} and np.array_equal(z['arr_0'], images[selected])
                        and np.array_equal(z['ids'], ids[selected]) and np.array_equal(z['labels'], labels[selected]),
                        name+' batch pixels differ from merged archive')
        keys = set(batches[0]['trajectory'])
        require(all(set(b['trajectory']) == keys for b in batches), name+' timing columns')
        timing_sums = {}
        for key in sorted(keys):
            values = [b['trajectory'][key] for b in batches]
            require(all(positive(value) for value in values), name+' invalid measured timing '+key)
            timing_sums[key] = sum(values)
            close([summary[key]], [timing_sums[key]], name+' timing sum '+key)
        close([measured[0]], [timing_sums['trajectory_through_decode_wall_seconds']], name+' T sum')
        close([summary['inference_trajectory_plus_decode_wall_seconds']],
              [timing_sums['trajectory_wall_seconds']+timing_sums['decode_and_uint8_wall_seconds']], name+' component timing sum')
        for b in batches:
            t = b['trajectory']
            require(t['trajectory_through_decode_wall_seconds'] >= t['trajectory_wall_seconds']+t['decode_and_uint8_wall_seconds'],
                    name+' nested wall timing containment')
        require(add_counts([b['forward_counts']['observed'] for b in batches]) == expected_total, name+' batch hook sum')
        self.record(read(self.study/name/'progress.json')['summary'], self.study/name/'summary.json')
        self.verify(self.study/name/'progress.json')
        require(read(self.study/name/'progress.json')['complete'] is True, name+' terminal progress')
        identity = tuple(summary[key] for key in IDENTITY_KEYS)+(first_noise,)
        return summary, identity, paired, timing_sums


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=STUDY)
    parser.add_argument('--output', type=Path, help='default: STUDY/independent_review.json; refuses overwrite')
    args = parser.parse_args()
    study = args.study.expanduser().resolve()
    output = (args.output or study/'independent_review.json').expanduser().resolve()
    require(not output.exists(), 'preserve previous review; output already exists')
    began, cpu_began = time.perf_counter(), time.process_time()
    execution = read(study/'screen_execution.json')
    # No reading of partial features/FID and no waiting/restarting a running job.
    require(execution.get('complete') is True and execution.get('fid_started') is True,
            'screen must be authoritatively complete; this tool never waits or resumes')
    torch.set_num_threads(4)
    review = Review(study)
    review.verify(study/'screen_execution.json')
    plan = read(study/'plan.json')
    review.verify(study/'plan.json', execution['plan_sha256'])
    require(plan['protocol'] == execution['protocol'] == PROTOCOL and plan['source_freeze_complete'] is True
            and plan['cohort'] == {'seed': SEED, 'n': COUNT, 'batch_size': BATCH}
            and plan['gpu_uuid'] == execution['gpu_uuid'] == GPU_UUID and execution['physical_gpu_index'] == 3,
            'plan protocol/GPU/cohort mismatch')
    require(plan['full_total_cost_closed'] is False and execution['total_cost_complete'] is False
            and execution['goal_complete'] is False, '1K screen cannot certify final total-cost goal')
    for raw_path, expected in plan['frozen_files'].items():
        path = Path(raw_path)
        path = (path if path.is_absolute() else ROOT/path).resolve()
        review.verify(path, expected)
        review.frozen[path] = expected
    for relative, record in plan['source_archives'].items():
        review.record(record, study/'sources'/relative)
        require(review.frozen[(ROOT/relative).resolve()] == record['sha256'], 'plan source archive mismatch')
    review.frozen_record(plan['protocol_document'])
    for path, expected in ((PILOT/'train/final.pt', BRIDGE_SHA), (REFERENCE, REFERENCE_SHA), (INCEPTION, INCEPTION_SHA)):
        review.verify(path, expected)
        require(review.frozen.get(path.resolve()) == expected, 'mandatory identity missing from freeze')
    evaluator_root = Path(plan['evaluator']['root']).resolve()
    require(plan['evaluator']['commit'] == EVALUATOR_COMMIT and plan['evaluator']['clean'] is True, 'evaluator plan identity')
    commit = subprocess.check_output(['git', '-C', str(evaluator_root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(evaluator_root), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    untracked = subprocess.check_output(['git', '-C', str(evaluator_root), 'ls-files', '--others', '--exclude-standard'], text=True).splitlines()
    require(commit == EVALUATOR_COMMIT and not dirty, 'evaluator tracked tree/commit changed')
    require(any(p.is_relative_to(evaluator_root) for p in review.frozen), 'evaluator sources not frozen')
    jobs = execution['jobs']
    require(len(jobs) >= 5 and [j['name'] for j in jobs[:4]] == ['parity', 'official100', 'candidate100', 'control100']
            and jobs[-1]['name'] == 'evaluation' and all(j['exit_code'] == 0 for j in jobs), 'job sequence/completion')
    require(len({j['name'] for j in jobs}) == len(jobs), 'duplicate/restarted job')
    require(instant(jobs[-2]['finished_utc']) <= instant(execution['fid_authorized_utc'])
            < instant(jobs[-1]['started_utc']) < instant(jobs[-1]['finished_utc'])
            <= instant(execution['finished_utc']), 'sampling/FID/terminal chronology')
    require(instant(plan['created_utc']) <= instant(execution['started_utc']) <= instant(jobs[0]['started_utc']), 'freeze/start chronology')
    require(all(instant(a['finished_utc']) <= instant(b['started_utc']) for a, b in zip(jobs[:-1], jobs[1:])), 'overlapping jobs')
    for job in jobs:
        require(isinstance(job['pid'], int) and job['pid'] > 0 and positive(job['outer_wall_seconds'])
                and instant(job['started_utc']) < instant(job['finished_utc']), 'job PID/time evidence')
        require(job['gpu_before']['fixed_gpu_identity_verified'] is True
                and GPU_UUID in job['gpu_before']['stdout'], 'job physical GPU identity')
        review.verify(job['log'])
    match = read(study/'cost_match.json')
    review.verify(study/'cost_match.json', execution['cost_match_sha256'])
    require(all(match[k] is True for k in ('complete', 'inference_cost_match_complete', 'frozen_before_fid',
            'control_did_not_select_cost', 'all_intermediate_official_outputs_retained'))
            and match['same_gpu_uuid'] == GPU_UUID and match['total_cost_complete'] is False
            and match['goal_complete'] is False, 'cost match state')
    require(match['selection_records'] == execution['cost_selections'] and match['selection_records'], 'cost selection chain')
    parity_summary, parity_request = review.parity(execution['parity'])
    sampling_jobs = jobs[1:-1]
    require(set(execution['sampling_results']) == {j['name'] for j in sampling_jobs}, 'sampling result/job bijection')
    completed_dirs = {p.parent.name for p in study.glob('*/summary.json')
                      if p.parent.name == 'candidate100' or p.parent.name == 'control100'
                      or (p.parent.name.startswith('official') and p.parent.name[8:].isdigit())}
    require(completed_dirs == set(execution['sampling_results']), 'unaccounted completed sampling directory')
    summaries, records, paired_identity, paired_batches = {}, {}, None, None
    arm_costs = []
    for job in sampling_jobs:
        name = job['name']
        entry = execution['sampling_results'][name]
        summary, identity, batch_identities, timing = review.sampling(name, entry['summary'], parity_request)
        require(entry['samples_npz'] == summary['samples_npz'], name+' execution archive binding')
        close(entry['T_W'], costs(summary), name+' execution costs')
        if paired_identity is None:
            paired_identity, paired_batches = identity, batch_identities
        require(identity == paired_identity and batch_identities == paired_batches, name+' full-arm input pairing')
        summaries[name], records[name] = summary, entry['summary']
        require(job['outer_wall_seconds'] >= summary['total_wall_seconds_before_summary'], name+' outer/main timing')
        arm_costs.append(dict(branch=name, T_W=list(costs(summary)), outer_wall_seconds=job['outer_wall_seconds'],
                              actual_forward_counts=summary['forward_counts_observed'], stage2_nfe_per_sample=summary['num_steps'],
                              auxiliary_forwards_per_sample=summary['auxiliary_forwards_per_sample'], timing_components=timing,
                              peak_sampling=summary['peak_sampling'], peak_loading=summary['peak_loading'],
                              source_hash_and_preprocessing_seconds=summary['preprocessing_source_and_weight_hash_wall_seconds'],
                              model_loading=summary['loading'], minimum_queried_time=parity_request['time_grid'][-2]
                              if summary['num_steps'] == 100 else read(study/name/'request.json')['time_grid'][-2]))
    for name in ('official100', 'candidate100', 'control100'):
        require(match[name] == records[name], name+' match summary binding')
    candidate_cost = costs(summaries['candidate100'])
    close(match['T_W_candidate'], candidate_cost, 'matched candidate costs')
    previous_name, selection_audit, matched_name = 'official100', [], None
    sampled_cost_jobs = iter(sampling_jobs[3:])
    previous_finished = jobs[3]['finished_utc']
    for i, record in enumerate(match['selection_records']):
        selection = review.json_record(record, study/f'cost_selection_{i:02d}.json')
        previous = summaries[previous_name]
        require(selection['fid_started'] is False and selection['candidate'] == records['candidate100']
                and selection['previous_official'] == records[previous_name], 'selection dependency binding')
        close(selection['candidate_T_W'], candidate_cost, 'selection candidate costs')
        close(selection['previous_official_T_W'], costs(previous), 'selection previous costs')
        if i:
            require(any(b < a for a, b in zip(candidate_cost, costs(previous))), 'unnecessary cost retry')
        k = select_steps(candidate_cost, costs(previous), None if i == 0 else previous['num_steps'])
        require(selection['selected_steps'] == k, 'Fraction cost rule differs')
        reuse = k == 100 and all(b >= a for a, b in zip(candidate_cost, costs(summaries['official100'])))
        require(selection['reuse_official100'] == reuse, 'incorrect official100 reuse')
        require(instant(previous_finished) <= instant(selection['selected_utc'])
                <= instant(execution['fid_authorized_utc']) < instant(jobs[-1]['started_utc']), 'cost selection occurred after FID authorization')
        if reuse:
            require(i == 0 and len(match['selection_records']) == 1, 'reuse must terminate initial selection')
            matched_name = 'official100'
        else:
            job = next(sampled_cost_jobs, None)
            require(job is not None and job['name'] == f'official{k}', 'selected K/job sequence differs')
            require(instant(selection['selected_utc']) <= instant(job['started_utc']), 'K selected after sampler start')
            matched_name = job['name']
            previous_finished = job['finished_utc']
        actual = costs(summaries[matched_name])
        selection_audit.append(dict(selected_steps=k, reused_official100=reuse, previous_T_W=list(costs(previous)),
                                    measured_T_W=list(actual), summary_sha256=records[matched_name]['sha256']))
        previous_name = matched_name
    require(next(sampled_cost_jobs, None) is None, 'unaccounted cost-only official jobs')
    require(match['official_cost'] == records[matched_name] and match['final_official_steps'] == summaries[matched_name]['num_steps']
            and match['official100_reused_for_cost_baseline'] == (matched_name == 'official100'), 'final matched identity')
    matched_cost = costs(summaries[matched_name])
    close(match['T_W_official_cost'], matched_cost, 'matched costs')
    require(all(b >= a for a, b in zip(candidate_cost, matched_cost)), 'final official does not cover candidate T/W')
    close(match['relative_budget_excess'], [b/a-1 for a, b in zip(candidate_cost, matched_cost)], 'budget excess', atol=1e-12)
    require(match['stage2_nfe_candidate'] == 100 and match['auxiliary_forwards_per_candidate_sample'] == 200
            and match['stage2_nfe_baseline'] == summaries[matched_name]['num_steps'], 'model NFE accounting')
    expected_arms = ['official100', 'candidate100', 'control100'] + ([] if matched_name == 'official100' else [matched_name])
    require(execution['evaluation_branches'] == expected_arms and execution['cost_baseline_evaluation_branch'] == matched_name,
            'pre-FID branch selection')
    argv = jobs[-1]['argv']
    expected_argv = [argv[0], str(ROOT/'experiments/evaluate_raev2_official_samples.py')]
    for name in expected_arms:
        expected_argv += ['--branch', name+'='+summaries[name]['samples_npz']['path']]
    expected_argv += ['--output', str(study/'official_evaluation.csv'), '--batch-size', '64', '--device', 'cuda',
                      '--seed', '2020', '--fid-reference', 'imagenet_256_fid_stats', '--evaluator-root',
                      str(plan['evaluator']['root']), '--feature-cache-dir', str(study/'official_feature_cache')]
    require(argv == expected_argv, 'uniform frozen evaluator invocation differs')
    metrics = review.json_record(execution['evaluation'], study/'official_evaluation.json')
    review.record(execution['evaluation_csv'], study/'official_evaluation.csv')
    require([row['branch'] for row in metrics] == expected_arms, 'FID output arms differ')
    with np.load(REFERENCE, allow_pickle=False) as ref:
        reference_mean, reference_covariance = ref['mu'].astype(np.float64), ref['sigma'].astype(np.float64)
    results = []
    for row in metrics:
        name = row['branch']
        archive = summaries[name]['samples_npz']
        require(row['fid_reference'] == 'imagenet_256_fid_stats' and row['evaluator_commit'] == EVALUATOR_COMMIT
                and Path(row['evaluator_root']).resolve() == evaluator_root
                and Path(row['sample_path']).resolve() == Path(archive['path']).resolve()
                and row['sample_sha256'] == archive['sha256'] and positive(row['fid']), name+' evaluator identity')
        feature = study/'official_feature_cache'/f'{name}-{archive["sha256"][:16]}-inception.features.pt'
        feature_sha = review.verify(feature)
        tensor = torch.load(feature, map_location='cpu', weights_only=False)
        require(isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu' and tuple(tensor.shape) == (1000, 2048), name+' feature shape')
        fid, eig_min = low_rank_fid(tensor.numpy(), reference_mean, reference_covariance)
        difference = fid-float(row['fid'])
        require(abs(difference) < 2e-4, name+' independent FID differs beyond fixed tolerance')
        results.append(dict(branch=name, reported_fid=row['fid'], independent_low_rank_fid=fid,
                            reconstruction_difference=difference, gram_min_eigenvalue=eig_min,
                            feature_sha256=feature_sha, samples_sha256=archive['sha256']))
    fid_by_name = {row['branch']: row['reported_fid'] for row in results}
    gains = {name: {'vs_official100': 1-fid_by_name[name]/fid_by_name['official100'],
                    'vs_cost_official': 1-fid_by_name[name]/fid_by_name[matched_name]}
             for name in ('candidate100', 'control100')}
    gains['candidate_vs_mean_control'] = 1-fid_by_name['candidate100']/fid_by_name['control100']
    inherited = []
    for name in ('pilot', 'train', 'validate', 'rollout'):
        path = PILOT/name/'summary.json'
        review.verify(path)
        value = read(path)
        require(value['complete'] is True and value['protocol'] == TRAINING_PROTOCOL, 'inherited pilot stage incomplete')
        wall = value['wall_seconds_from_first_time_import']
        require(positive(wall), 'inherited runner wall')
        inherited.append(dict(stage=name, runner_wall_seconds=wall, summary_sha256=review.verified[str(path.resolve())]))
    inherited_sum = sum(row['runner_wall_seconds'] for row in inherited)
    close([inherited_sum], [match['additional_costs']['pilot_train_validate_rollout_runner_wall_seconds']], 'inherited runner sum', atol=1e-6)
    require(match['additional_costs']['joint_training_not_divided_between_fields'] is True, 'training cost divided')
    all_hook_counts = add_counts([parity_summary['forward_counts_observed']]
                                + [summary['forward_counts_observed'] for summary in summaries.values()])
    job_outer_sum = sum(job['outer_wall_seconds'] for job in jobs)
    require(execution['driver_wall_seconds'] >= job_outer_sum, 'driver wall omits job costs')
    report = dict(complete=True, goal_complete=False, total_cost_complete=False, samples_per_arm=COUNT,
                  unique_quality_arms=len(results), all_complete_sampling_arms=len(summaries),
                  all_125_batch_pixel_archives_equal_merged=True, all_noise_rng_labels_and_saved_first_noise_paired=True,
                  parity_four_full_paths_and_actual_hooks_verified=True, checkpoint_training_source_chain_verified=True,
                  cost_rule_recomputed_with_fraction=True, cost_selected_before_fid=True,
                  same_physical_gpu_sequential_runs=True, final_cost_baseline=matched_name,
                  relative_fid_gains=gains, quality_results=results, cost_selections=selection_audit,
                  all_sampling_costs=arm_costs, all_sampling_and_parity_actual_forward_counts=all_hook_counts,
                  parity_main_wall_seconds=parity_summary['total_wall_seconds_before_summary'],
                  inherited_stage_costs=inherited, inherited_runner_wall_seconds=inherited_sum,
                  additional_cost_disclosures=match['additional_costs'],
                  all_current_jobs=[{k: j[k] for k in ('name', 'pid', 'started_utc', 'finished_utc', 'outer_wall_seconds', 'exit_code')} for j in jobs],
                  all_current_job_outer_wall_seconds=job_outer_sum,
                  driver_wall_seconds=execution['driver_wall_seconds'],
                  driver_verification_wall_seconds=execution['verification_wall_seconds'],
                  plan_freeze_wall_seconds=plan['freeze_wall_seconds'],
                  outer_cost_baseline_covers_candidate=next(j['outer_wall_seconds'] for j in jobs if j['name'] == matched_name)
                      >= next(j['outer_wall_seconds'] for j in jobs if j['name'] == 'candidate100'),
                  cost_scope='T/W match inference only. Historical source selection and exact old training-driver outer wall remain unclosed. Joint fit, all diagnostics, parity, every arm and intermediate K, evaluation, plan/source audit and this CPU review are additional; nested timings must not be summed twice.',
                  quality_scope='Fixed exploratory paired 1K, not independent-seed or sufficient-scale confirmation. No local moment proxy is substituted for FID. A 5% 1K gain alone would not complete the research goal.',
                  reconstruction_scope='CPU Bures low-rank reconstruction from saved official Inception features; no independent feature extraction, CUDA noise regeneration, or reproduction of unsaved latent/stepwise parity states. Pairing and parity use recorded hashes/hooks bound to frozen sources.',
                  evaluator_commit=EVALUATOR_COMMIT, reference_sha256=REFERENCE_SHA,
                  evaluator_tracked_tree_clean=True, evaluator_untracked_paths_observed=untracked,
                  verified_artifact_count=len(review.verified), verified_artifacts=review.verified,
                  review_script_sha256=sha(__file__), wall_seconds=time.perf_counter()-began,
                  cpu_seconds=time.process_time()-cpu_began, new_model_or_gpu_calls=0)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('complete', 'goal_complete', 'unique_quality_arms', 'all_complete_sampling_arms',
                      'relative_fid_gains', 'quality_results', 'verified_artifact_count', 'wall_seconds', 'cpu_seconds')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
