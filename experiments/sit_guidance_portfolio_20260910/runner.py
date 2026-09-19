"""Validate and launch the frozen 50-idea catalog using the paired FID pipeline."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments import small_sit_carrier_flow_20260909 as infrastructure
from experiments.lifting_scale_sweep_20260909 import WORK, array_sha, atomic, read, sha
from . import assets, catalog, operators

ROOT = assets.ROOT
BANK_ROOT = infrastructure.BANK_ROOT
BATCH, SAMPLES, RANKS = 8, 1000, 4
CONFIGS = catalog.configs()
SPECS = {config['arm']: config for config in CONFIGS}
ARMS = tuple(SPECS)
BASE_VERIFY = infrastructure.verify_request
MODULE = 'experiments.sit_guidance_portfolio_20260910.runner'
DIAGNOSTICS = ('correction_to_native_norm', 'off_affine_energy', 'clean_norm_ratio',
               'constraint_fraction', 'zero_gap_fraction')


def batch_seed(noise):
    # Identical noise batches use identical stochastic queries in every arm.
    return int(array_sha(noise.detach().cpu().numpy())[:15], 16)


def sample(rt, noise, labels, arm, *, zero=False):
    return operators.sample(rt, noise, labels, SPECS[arm],
                            batch_seed=batch_seed(noise), zero=zero)


def verify_request():
    request, digest = BASE_VERIFY()
    assert request['configs'] == CONFIGS
    assert request['samples'] == SAMPLES and request['batch'] == BATCH
    assert request['ranks'] == RANKS
    assets.verify()
    assert sha('/data/shared/adm_refs/classify_image_graph_def.pb') == request['inception_graph_sha256']
    return request, digest


def install_infrastructure():
    infrastructure.ROOT, infrastructure.BANK_ROOT = ROOT, BANK_ROOT
    infrastructure.ARMS, infrastructure.SAMPLES = ARMS, SAMPLES
    infrastructure.BATCH, infrastructure.RANKS = BATCH, RANKS
    infrastructure.Runtime = lambda name: operators.make_runtime()
    infrastructure.sample, infrastructure.preflight = sample, preflight
    infrastructure.verify_request = verify_request


def prepare():
    assert not (ROOT / 'request.json').exists(), 'An existing request must not be overwritten'
    fit_request, fit_result = assets.verify()
    parent, parent_hash = infrastructure.verify_request()
    # Both parents are verified before freezing their current sources in this run.
    old_fit = operators.residual.verify_fit()
    sources = set(parent['sources']) | set(old_fit['sources']) | set(fit_request['sources'])
    sources.update(str(path.resolve()) for path in Path(__file__).parent.glob('*.py'))
    sources.add(str(WORK / 'experiments/run_sit_guidance_50ideas_20260910.py'))
    sources.add(str(WORK / 'train_gen/evaluator.py'))
    sources = {path: sha(path) for path in sorted(sources)}
    source_dir = ROOT / 'sources'
    source_dir.mkdir(exist_ok=True)
    for index, path in enumerate(sources):
        (source_dir / f'{index:03d}_{Path(path).name}').write_bytes(Path(path).read_bytes())
    model_assets = dict(parent['assets'])
    for path in (*assets.EXTRA_HEADS.values(), ROOT / 'assets/fitted.pt',
                 ROOT / 'assets/fit.json', ROOT / 'assets/fit_request.json',
                 operators.residual.ROOT / 'projection.pt'):
        model_assets[str(path)] = sha(path)
    required_space = int(len(CONFIGS) * SAMPLES * (256 * 256 * 3 * 2 + 4 * 32 * 32 * 8 + 32768))
    free_space = shutil.disk_usage(ROOT).free
    assert free_space > required_space + (10 << 30), (free_space, required_space)
    request = dict(parent, sources=sources, assets=model_assets, configs=CONFIGS, arms=ARMS,
        ideas=catalog.records(), samples=SAMPLES, batch=BATCH, ranks=RANKS,
        parent_request_sha256=parent_hash, portfolio_fit_sha256=fit_result['fitted_sha256'],
        inception_graph_sha256=sha('/data/shared/adm_refs/classify_image_graph_def.pb'),
        python=sys.executable, versions={name: importlib.metadata.version(name)
            for name in ('torch', 'torchdiffeq', 'diffusers', 'numpy')},
        diagnostic_names=DIAGNOSTICS, stochastic_seed='first 15 hex digits of batch noise SHA256',
        preflight='all grid points at two times; one complete trajectory per idea; native golden check',
        estimated_output_bytes=required_space, free_bytes_at_prepare=free_space,
        independent_confirmation=False, research_goal_achieved=False)
    # These fields described the old finite-write experiment, not this catalog.
    for key in ('grid', 'alphas', 'heun_steps', 'main_solver'):
        request.pop(key, None)
    atomic(ROOT / 'request.json', request)
    atomic(ROOT / 'status.json', dict(phase='prepared', ideas=50, configurations=len(CONFIGS)))
    print(json.dumps(dict(prepared=True, root=str(ROOT), ideas=50,
        configurations=len(CONFIGS), samples_per_configuration=SAMPLES)), flush=True)


def hook_counts(rt):
    return [(len(module._forward_hooks), len(module._forward_pre_hooks))
            for module in rt.model.modules()]


@torch.inference_mode()
def preflight(rt, noise, labels, rank, request):
    begin = time.perf_counter()
    rt.labels = labels
    for path, digest in rt.sources.items():
        assert request['sources'].get(path) == digest, path
    assert not rt.model.training
    original_hooks = hook_counts(rt)
    original_forwards = [block.attn.forward for block in rt.model.blocks]
    for tv in (0., .375, .75):
        t = noise.new_tensor(tv)
        strong, weak = rt.pair(noise, t)
        assert torch.equal(strong, rt.field(noise, t, 'full'))
        assert torch.equal(weak, rt.field(noise, t, 'base'))
    baseline, stats = sample(rt, noise, labels, 'control_full')
    assert stats['full_calls'] == 128 and stats['prefix_calls'] == 0
    pixels = rt.decode(baseline)
    assert pixels.shape == (BATCH, 256, 256, 3) and pixels.dtype == np.uint8
    ctx = operators.StepContext(shadow=noise.clone())
    generator = torch.Generator(device=noise.device).manual_seed(batch_seed(noise))
    ctx.xi = torch.randn(noise.shape, device=noise.device, generator=generator)
    ctx.permutations = torch.argsort(torch.rand((BATCH, 256), device=noise.device, generator=generator), -1)
    heads = rt.model.blocks[0].attn.num_heads
    ctx.head_orders = torch.argsort(torch.rand((BATCH, 4, heads), device=noise.device, generator=generator), -1)
    offsets = torch.randint(1, 100, (BATCH, 4), device=noise.device, generator=generator)
    ctx.alternate_labels = (labels[:, None] + offsets) % 100
    checked, trajectories = [], []
    for config in CONFIGS[rank::RANKS]:
        if config['role'] != 'candidate':
            continue
        for tv in (0., config['cutoff'] - 1 / 64):
            amount = operators.amount_at(config, tv)
            if config['key'] == 'ig_strang_split':
                value = operators.strang_step(rt, noise, tv, 1 / 64, amount, int(config['theta']))
            else:
                value, info = operators.evaluate(rt, noise, tv, config, amount, ctx)
                assert torch.isfinite(info['diagnostics']).all(), config['arm']
            assert value.shape == noise.shape and torch.isfinite(value).all(), config['arm']
            assert rt.labels is labels and hook_counts(rt) == original_hooks, config['arm']
            assert [block.attn.forward for block in rt.model.blocks] == original_forwards, config['arm']
        checked.append(config['arm'])
    print(json.dumps(dict(preflight_grid_passed=True, rank=rank, configurations=len(checked))), flush=True)
    for idea in catalog.IDEAS[rank::RANKS]:
        config = next(config for config in CONFIGS
                      if config['idea'] == idea.id and config['arm'].endswith('_g3_p1'))
        rt.labels = labels
        zero, _ = operators.evaluate(rt, noise, 0., config, 0., ctx)
        assert torch.equal(zero, rt.field(noise, noise.new_tensor(0.), 'full')), config['arm']
        z, stats = sample(rt, noise, labels, config['arm'])
        assert torch.isfinite(z).all() and np.isfinite(stats['diagnostics']).all(), config['arm']
        trajectories.append(dict(idea=idea.id, arm=config['arm'],
            full_calls=stats['full_calls'], prefix_calls=stats['prefix_calls'],
            endpoint_max_abs=float(z.abs().max())))
        print(json.dumps(dict(preflight_trajectory_passed=True, rank=rank, **trajectories[-1])), flush=True)
    # Hook-based reference changes must leave ordinary predictions intact.
    repeated, _ = sample(rt, noise, labels, 'control_full')
    assert torch.equal(baseline, repeated), 'A candidate changed subsequent native predictions'
    golden_config = next(config for config in CONFIGS
        if config['solver'] == 'dopri5' and config['strength'] == .7)
    golden, _ = sample(rt, noise, labels, golden_config['arm'])
    golden_path = operators.residual.OLD_ROOT / f'ig_restarted/rank{rank}/batch{rank * BATCH:04d}.npz'
    with np.load(golden_path) as previous:
        np.testing.assert_array_equal(golden.cpu().numpy(), previous['latents'])
    return dict(passed=True, rank=rank, grid_arms=checked, trajectories=trajectories,
        native_pair_prefix_exact=True, hooks_restored=True, native_after_candidates_exact=True,
        zero_field_exact=True, decode_passed=True, golden_exact=True,
        golden_path=str(golden_path), golden_sha256=sha(golden_path), runtime_sources=rt.sources,
        cuda_device=torch.cuda.get_device_name(), seconds=time.perf_counter() - begin)


def run():
    with (ROOT / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request, digest = verify_request()
        assert read(ROOT / 'status.json')['phase'] == 'prepared'
        workers, streams, results = [], [], []
        begin = time.perf_counter()
        def interrupted(signum, frame):
            raise RuntimeError(f'Controller received signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        try:
            for rank in range(RANKS):
                stream = (ROOT / f'worker{rank}.log').open('w')
                streams.append(stream)
                workers.append(subprocess.Popen([sys.executable, '-u', '-m', MODULE, '--rank', str(rank)],
                    cwd=WORK, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank),
                        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'), stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT))
            status = dict(controller_pid=os.getpid(), worker_pids=[p.pid for p in workers],
                          ideas=50, configurations=len(CONFIGS), research_goal_achieved=False)
            atomic(ROOT / 'status.json', dict(phase='preflight', **status))
            infrastructure.wait_files([ROOT / f'preflight_rank{rank}.json' for rank in range(RANKS)], workers)
            checks = [read(ROOT / f'preflight_rank{rank}.json') for rank in range(RANKS)]
            assert all(check['passed'] for check in checks)
            assert {arm for check in checks for arm in check['grid_arms']} == {
                config['arm'] for config in CONFIGS if config['role'] == 'candidate'}
            assert {row['idea'] for check in checks for row in check['trajectories']} == set(range(1, 51))
            assert all(check['runtime_sources'] == checks[0]['runtime_sources'] for check in checks)
            atomic(ROOT / 'preflight_passed.json', dict(passed=True, checks=checks, request_sha256=digest))
            print(json.dumps(dict(preflight_passed=True, ideas=50, candidate_grid_points=600)), flush=True)
            for arm in ARMS:
                atomic(ROOT / 'status.json', dict(phase='sampling', arm=arm, completed=len(results), **status))
                infrastructure.wait_files([ROOT / arm / f'rank{rank}/summary.json'
                                           for rank in range(RANKS)], workers)
                atomic(ROOT / 'status.json', dict(phase='evaluating', arm=arm, completed=len(results), **status))
                result = infrastructure.evaluate(arm, request, digest)
                result.update(SPECS[arm])
                atomic(ROOT / arm / 'result.json', result)
                results.append(result)
                atomic(ROOT / 'results.json', results)
                print(json.dumps(dict(arm=arm, complete=result['complete'], fid=result['fid'],
                                      completed=len(results), total=len(ARMS))), flush=True)
                atomic(ROOT / arm / 'advance.json', dict(complete=True, request_sha256=digest))
            codes = [worker.wait() for worker in workers]
            assert codes == [0] * RANKS, codes
            verify_request()
            atomic(ROOT / 'status.json', dict(phase='complete', completed=len(results),
                wall_seconds=time.perf_counter() - begin,
                numerical_failures=sum(not result['complete'] for result in results),
                worker_exit_codes=codes, **status))
        except BaseException as error:
            atomic(ROOT / 'status.json', dict(phase='failed', error=repr(error),
                controller_pid=os.getpid(), worker_pids=[worker.pid for worker in workers]))
            raise
        finally:
            for worker in workers:
                if worker.poll() is None:
                    worker.terminate()
            for worker in workers:
                try:
                    worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
            for stream in streams:
                stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run-prepared', action='store_true')
    mode.add_argument('--rank', type=int, choices=range(RANKS))
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        install_infrastructure()
        run() if args.run_prepared else infrastructure.worker(args.rank)


if __name__ == '__main__':
    main()
