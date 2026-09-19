"""Train a real-anchor residual refiner. Production GPU execution is caller-owned.

Each step chooses 16 distinct fit sources and uses ALL M supplied corruptions,
forming 16*M repair inputs (currently M=2). Identity uses the 16 clean sources once.
Loss = mean repair MSE + 2 * mean clean identity MSE. No tau input or
idempotence loss. The initial identity checkpoint participates in selection.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import random
import time
import numpy as np
import torch
from experiments.inverse_copy_refiner_20260913.model import Refiner


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--pairs', type=Path, default=Path('/home/zhoushunyu/data/eqvae/experiments/inverse_copy_refiner_20260913/pairs.npz'))
    result.add_argument('--output', type=Path, default=Path('/home/zhoushunyu/data/eqvae/experiments/inverse_copy_refiner_20260913/train_seed2026091361'))
    result.add_argument('--device', default='cpu')
    result.add_argument('--steps', type=int, default=1500)
    result.add_argument('--batch-sources', type=int, default=16)
    result.add_argument('--eval-every', type=int, default=100)
    result.add_argument('--lr', type=float, default=2e-4)
    result.add_argument('--weight-decay', type=float, default=1e-4)
    result.add_argument('--seed', type=int, default=2026091361)
    result.add_argument('--width', type=int, default=32)
    result.add_argument('--context-dim', type=int, default=64)
    result.add_argument('--num-classes', type=int, default=100)
    result.add_argument('--cpu-threads', type=int, default=4)
    result.add_argument('--allow-small-data', action='store_true', help='CPU smoke only: waive the exact 400 fit / 100 heldout sizes.')
    return result


def load_pairs(path, num_classes, allow_small=False):
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.array(data[key], copy=True) for key in ('clean', 'corrupted', 'labels', 'split')}
        key = 'source_ids' if 'source_ids' in data else 'source_id'
        if key not in data:
            raise ValueError('pairs.npz requires source_ids or source_id')
        source_ids = np.array(data[key], copy=True)
        taus = np.array(data['taus'], dtype=np.float64).tolist() if 'taus' in data else None
        tau_source = 'pairs.npz' if 'taus' in data else 'not supplied'
        directions = np.array(data['directions'], copy=True) if 'directions' in data else None
    clean, corrupted, labels, split = [arrays[key] for key in ('clean', 'corrupted', 'labels', 'split')]
    n = len(clean)
    if clean.dtype != np.float32 or clean.shape != (n, 4, 32, 32):
        raise ValueError('clean must be float32 [N,4,32,32]')
    if (corrupted.dtype != np.float32 or corrupted.ndim != 5 or corrupted.shape[0] != n
            or corrupted.shape[1] < 1 or corrupted.shape[2:] != (4, 32, 32)):
        raise ValueError('corrupted must be float32 [N,M,4,32,32], M>=1')
    corruption_count = corrupted.shape[1]
    if labels.dtype != np.int64 or labels.shape != (n,):
        raise ValueError('labels must be int64 [N]')
    if split.dtype != np.int8 or split.shape != (n,) or np.any(~np.isin(split, [0, 1])):
        raise ValueError('split must be int8 [N] containing only 0 and 1')
    if source_ids.shape != (n,) or source_ids.dtype.kind not in 'iuUS':
        raise ValueError('source_ids must be a one-dimensional integer or string array')
    if np.any((labels < 0) | (labels >= num_classes)):
        raise ValueError('class labels outside model class range')
    if not np.all(np.isfinite(clean)) or not np.all(np.isfinite(corrupted)):
        raise ValueError('non-finite latent values')
    if taus is None:
        taus = [.75, .5] if corruption_count == 2 else [None]*corruption_count
        tau_source = 'current M=2 schema [.75,.5]' if corruption_count == 2 else 'unspecified; ordered corruption indices only'
    elif len(taus) != corruption_count or not np.all(np.isfinite(taus)) or np.any((np.asarray(taus) <= 0) | (np.asarray(taus) >= 1)):
        raise ValueError('If supplied, taus must give M finite entries strictly between 0 and 1')
    if directions is not None:
        if directions.shape != (corruption_count,) or np.any(~np.isin(directions, [-1, 1])):
            raise ValueError('If supplied, directions must give M entries in {-1,+1}')
        directions = directions.tolist()
    fit = np.flatnonzero(split == 0)
    heldout = np.flatnonzero(split == 1)
    identities = source_ids.astype(str)
    if set(identities[fit]) & set(identities[heldout]):
        raise ValueError('Source-ID leakage between fit and heldout')
    if len(set(identities)) != n:
        raise ValueError('Duplicate source IDs within the dataset')
    if not len(fit) or not len(heldout):
        raise ValueError('Both fit and heldout must be nonempty')
    if not allow_small and (len(fit), len(heldout)) != (400, 100):
        raise ValueError('Production expects exactly 400 fit sources and 100 heldout sources')
    metadata = {'sources': n, 'fit_sources': len(fit), 'heldout_sources': len(heldout),
                'corruptions_per_source': corruption_count,
                'directions': directions,
                'taus': taus, 'tau_metadata_source': tau_source,
                'fit_class_counts': np.bincount(labels[fit], minlength=num_classes).tolist(),
                'heldout_class_counts': np.bincount(labels[heldout], minlength=num_classes).tolist(),
                'fit_source_ids': identities[fit].tolist(),
                'heldout_source_ids': identities[heldout].tolist()}
    return {key: torch.from_numpy(arrays[key]) for key in ('clean', 'corrupted', 'labels')}, fit, heldout, metadata


def zero_baseline(data, indices):
    clean, corrupted = data['clean'][indices], data['corrupted'][indices]
    per_tau = ((corrupted-clean[:, None])**2).flatten(2).mean(2).mean(0)
    repair = float(per_tau.mean())
    return {'repair_mse': repair, 'identity_mse': 0., 'objective': repair,
            'repair_mse_by_tau': per_tau.tolist(), 'repair_mse_by_corruption': per_tau.tolist(), 'sources': len(indices)}


@torch.no_grad()
def evaluate(model, data, indices, device, batch_sources):
    model.eval()
    repairs, identities = [], []
    for start in range(0, len(indices), batch_sources):
        selected = indices[start:start+batch_sources]
        clean = data['clean'][selected].to(device)
        corrupted = data['corrupted'][selected].to(device)
        labels = data['labels'][selected].to(device)
        output = model(corrupted.flatten(0, 1), labels.repeat_interleave(corrupted.shape[1])).reshape_as(corrupted)
        repairs.append(((output-clean[:, None])**2).flatten(2).mean(2).cpu())
        identities.append(((model(clean, labels)-clean)**2).flatten(1).mean(1).cpu())
    repair_by_tau = torch.cat(repairs).mean(0)
    repair = float(repair_by_tau.mean())
    identity = float(torch.cat(identities).mean())
    return {'repair_mse': repair, 'identity_mse': identity,
            'objective': repair + 2*identity,
            'repair_mse_by_tau': repair_by_tau.tolist(), 'repair_mse_by_corruption': repair_by_tau.tolist(), 'sources': len(indices)}


def save_checkpoint(path, model, optimizer, step, model_config, heldout, metadata):
    temporary = Path(str(path)+'.tmp')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
                'model_config': model_config, 'heldout': heldout,
                'run_metadata': metadata}, temporary)
    os.replace(temporary, path)


def run(args):
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite existing output directory: {args.output}')
    if args.steps < 1 or args.batch_sources < 1 or args.eval_every < 1 or args.lr <= 0:
        raise ValueError('steps, batch-sources, eval-every and lr must be positive')
    data, fit, heldout, input_metadata = load_pairs(args.pairs, args.num_classes, args.allow_small_data)
    if args.batch_sources > len(fit):
        raise ValueError('batch-sources exceeds fit source count')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model_config = {'num_classes': args.num_classes, 'width': args.width, 'context_dim': args.context_dim}
    model = Refiner(**model_config).to(device=device, dtype=torch.float32)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    probe = data['clean'][heldout[:2]].to(device)
    probe_labels = data['labels'][heldout[:2]].to(device)
    with torch.no_grad():
        if not torch.equal(model(probe, probe_labels), probe):
            raise RuntimeError('Zero-initialized refiner is not exactly identity')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    source_files = [Path(__file__), Path(__file__).with_name('model.py')]
    metadata = {'args': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                'model_config': model_config, 'model_parameters': parameter_count,
                'input': input_metadata, 'pairs_sha256': sha256(args.pairs),
                'source_sha256': {str(path.resolve()): sha256(path) for path in source_files},
                'python_version': platform.python_version(), 'torch_version': str(torch.__version__),
                'numpy_version': np.__version__, 'dtype': 'float32; no autocast',
                'batch_definition': f'{args.batch_sources} distinct fit sources, all {input_metadata["corruptions_per_source"]} corruptions => {input_metadata["corruptions_per_source"]*args.batch_sources} repair inputs; same {args.batch_sources} identity inputs.',
                'loss': 'repair MSE + 2 * identity MSE; no idempotence term',
                'checkpoint_selection': 'Lowest heldout repair MSE + 2 * heldout identity MSE, including the initial identity model at step 0.',
                'initial_identity_bitwise_exact': True}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    baseline = {'fit': zero_baseline(data, fit), 'heldout': zero_baseline(data, heldout)}
    (args.output/'zero_p_baseline.json').write_text(json.dumps(baseline, indent=2)+'\n')
    current_heldout = evaluate(model, data, heldout, device, args.batch_sources)
    best = dict(current_heldout)
    best_step = 0
    save_checkpoint(args.output/'best.pt', model, optimizer, 0, model_config, current_heldout, metadata)
    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    with (args.output/'rawmetrics.jsonl').open('x', buffering=1) as log:
        log.write(json.dumps({'kind': 'heldout', 'step': 0, **current_heldout})+'\n')
        for step in range(1, args.steps+1):
            model.train()
            selected = rng.choice(fit, size=args.batch_sources, replace=False)
            clean = data['clean'][selected].to(device)
            corrupted = data['corrupted'][selected].to(device)
            labels = data['labels'][selected].to(device)
            optimizer.zero_grad(set_to_none=True)
            repaired = model(corrupted.flatten(0, 1), labels.repeat_interleave(corrupted.shape[1])).reshape_as(corrupted)
            repair_loss = ((repaired-clean[:, None])**2).mean()
            identity_loss = ((model(clean, labels)-clean)**2).mean()
            loss = repair_loss + 2*identity_loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'Non-finite objective at step {step}')
            loss.backward()
            optimizer.step()
            log.write(json.dumps({'kind': 'fit', 'step': step, 'repair_mse': float(repair_loss.detach()),
                                  'identity_mse': float(identity_loss.detach()), 'objective': float(loss.detach())})+'\n')
            if step % args.eval_every == 0 or step == args.steps:
                current_heldout = evaluate(model, data, heldout, device, args.batch_sources)
                if not np.isfinite(current_heldout['objective']):
                    raise FloatingPointError(f'Non-finite heldout objective at step {step}')
                if current_heldout['objective'] < best['objective']:
                    best, best_step = dict(current_heldout), step
                    save_checkpoint(args.output/'best.pt', model, optimizer, step, model_config, current_heldout, metadata)
                save_checkpoint(args.output/'last.pt', model, optimizer, step, model_config, current_heldout, metadata)
                event = {'kind': 'heldout', 'step': step, 'best_step': best_step,
                         'elapsed_seconds': time.monotonic()-started, **current_heldout}
                log.write(json.dumps(event)+'\n')
                print(json.dumps(event), flush=True)
    summary = {'complete': True, 'steps': args.steps, 'best_step': best_step,
               'best_heldout': best, 'last_heldout': current_heldout,
               'zero_p_baseline': baseline, 'model_parameters': parameter_count,
               'model_config': model_config, 'elapsed_seconds': time.monotonic()-started,
               'source_sha256': metadata['source_sha256'], 'pairs_sha256': metadata['pairs_sha256'],
               'limitation': 'Heldout reconstruction selects the refiner. It does not establish improved generation quality or FID.'}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    print(json.dumps(run(parser().parse_args()), indent=2))
