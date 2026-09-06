"""One fixed actual-law density-ratio fit, using the same critic and loss."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.raev2_actual_ratio_data import ActualPairs
from experiments.raev2_paired_ratio_loss import paired_scaled_logistic
from experiments.raev2_paired_ratio_model import PairedRatioCritic
from experiments.summarize_raev2_guidance_20260907 import DATA, sha


def pair_loss(model, batch):
    p, q, t, labels = batch
    values = model(torch.cat((p, q)), t.repeat(2), labels.repeat(2))
    fp, fq = values.chunk(2)
    return paired_scaled_logistic(fp, fq, 1-t), fp, fq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local = int(os.environ['LOCAL_RANK'])
    assert world == plan['world_size'] == 4
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    dist.init_process_group('nccl', device_id=device)
    torch.manual_seed(plan['model_seed'])
    source_paths = [Path(__file__).resolve(), ROOT / 'experiments/raev2_actual_ratio_data.py',
                    ROOT / 'experiments/raev2_paired_ratio_model.py', ROOT / 'experiments/raev2_paired_ratio_loss.py',
                    args.plan.resolve()]
    frozen = {str(p): sha(p) for p in source_paths}
    bank_root = DATA / 'actual_ratio_bank'
    bank_path = bank_root / 'execution.json'
    bank = json.loads(bank_path.read_text())
    assert bank['complete'] and sha(bank_path) == plan['actual_bank_execution_sha256']
    real_path = DATA / 'paired_ratio_data/manifest.json'
    real_manifest = json.loads(real_path.read_text())
    assert sha(real_path) == plan['real_manifest_sha256']
    features_path = DATA / 'paired_ratio_data/class_features.pt'
    assert sha(features_path) == real_manifest['class_features_sha256']
    old_path = DATA / 'paired_ratio_fit/critic.pt'
    assert sha(old_path) == plan['old_critic_sha256']
    out = args.output.resolve()
    started = time.time()
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
        for split, shards in bank['splits'].items():
            for shard in shards:
                for name, digest in shard['files'].items():
                    assert sha(bank_root / split / f"shard{shard['shard']}" / name) == digest
        for split in real_manifest['splits'].values():
            for path, digest in split['sources'].items():
                # Only the real sources are used by the new actual-law fit.
                if path == split['real_path'] or path.endswith('metadata.npz'):
                    assert sha(Path(path)) == digest
            assert sha(Path(split['real_lookup'])) == plan['lookup_sha256'][split['real_lookup']]
        (out / 'frozen_source').mkdir()
        for p in source_paths:
            (out / 'frozen_source' / p.name).write_bytes(p.read_bytes())
        state = {'complete': False, 'pid': os.getpid(), 'started_unix': started,
                 'plan': plan, 'sources': frozen, 'torch_version': torch.__version__,
                 'all_input_sources_verified': True, 'stage': 'training',
                 'data_preparation_model_calls': sum(s['sample_model_calls'] for parts in bank['splits'].values() for s in parts),
                 'data_preparation_worker_seconds': sum(s['seconds'] for parts in bank['splits'].values() for s in parts)}
        (out / 'execution.json').write_text(json.dumps(state, indent=2) + '\n')
    dist.barrier()
    features = torch.load(features_path, map_location='cpu', weights_only=True)
    model = PairedRatioCritic(features).to(device).train()
    ddp = DistributedDataParallel(model, device_ids=[local], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'],
                                 weight_decay=plan['weight_decay'], betas=(.9, .999))
    train = ActualPairs(bank_root / 'train', real_manifest['splits']['train'], 5000)
    valid = ActualPairs(bank_root / 'validation', real_manifest['splits']['validation'], 1000)
    rng = np.random.default_rng(plan['train_pair_seed'] + rank)
    batch_size = plan['global_batch'] // world
    assert batch_size*world == plan['global_batch']
    losses, clips = [], 0
    torch.cuda.synchronize()
    training_start = time.perf_counter()
    for step in range(plan['updates']):
        ids = rng.integers(0, 5000, batch_size)
        batch = train.batch(ids, rng.integers(0, 5, batch_size), device)
        optimizer.zero_grad(set_to_none=True)
        loss = pair_loss(ddp, batch)[0].mean()
        assert torch.isfinite(loss).item()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), plan['gradient_clip'], error_if_nonfinite=True)
        clips += int(norm.item() > plan['gradient_clip'])
        optimizer.step()
        losses.append(loss.detach())
        if (step+1) % 128 == 0:
            recent = torch.stack(losses[-128:]).mean()
            dist.all_reduce(recent)
            if rank == 0:
                progress = {'step': step+1, 'last_128_loss': recent.item()/world,
                            'elapsed_seconds': time.perf_counter()-training_start,
                            'rank0_gradient_clip_count': clips}
                (out / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
                print(json.dumps(progress), flush=True)
    torch.cuda.synchronize()
    timing = torch.tensor([time.perf_counter()-training_start], dtype=torch.float64, device=device)
    dist.all_reduce(timing)
    model.eval()
    old = PairedRatioCritic(features).to(device).eval().requires_grad_(False)
    old.load_state_dict(torch.load(old_path, map_location='cpu', weights_only=False)['state_dict'])
    validation = []
    indices = np.arange(rank, 1000, world)
    with torch.no_grad():
        for begin in range(0, len(indices), batch_size):
            ids = indices[begin:begin+batch_size]
            batch = valid.batch(ids, np.zeros(len(ids), dtype=int), device)
            loss, fp, fq = pair_loss(model, batch)
            old_loss = pair_loss(old, batch)[0]
            validation.extend(zip(ids.tolist(), batch[2].cpu().tolist(), loss.cpu().tolist(),
                                  old_loss.cpu().tolist(), fp.cpu().tolist(), fq.cpu().tolist()))
    gathered = [None]*world
    dist.all_gather_object(gathered, validation)
    for path, digest in frozen.items():
        assert sha(Path(path)) == digest, 'source changed during fit: ' + path
    if rank == 0:
        records = np.array(sorted(sum(gathered, []), key=lambda r: r[0]))
        assert np.array_equal(records[:, 0], np.arange(1000)) and np.isfinite(records).all()
        values, old_values = records[:, 2], records[:, 3]
        mean, se = float(values.mean()), float(values.std(ddof=1)/np.sqrt(1000))
        delta = values - old_values
        change, change_se = float(delta.mean()), float(delta.std(ddof=1)/np.sqrt(1000))
        validation_result = {'samples': 1000, 'mean_scaled_logistic': mean,
            'class_standard_error': se, 'upper_two_standard_errors': mean+2*se,
            'old_critic_mean_scaled_logistic': float(old_values.mean()),
            'change_vs_old_critic': change, 'change_vs_old_class_standard_error': change_se,
            'entry_condition_passed': mean+2*se < 0 and change+2*change_se < 0,
            'not_a_fid_or_input_gradient_generalization_claim': True}
        np.savez(out / 'validation_records.npz', records=records)
        checkpoint = out / 'critic.pt'
        torch.save({'state_dict': model.state_dict(), 'plan': plan, 'sources': frozen,
                    'validation': validation_result, 'updates': plan['updates']}, checkpoint)
        state.update(complete=True, stage='fit_and_validation_complete', training_gpu_seconds=timing.item(),
                     elapsed_seconds=time.time()-started, validation=validation_result,
                     checkpoint_sha256=sha(checkpoint), validation_sha256=sha(out / 'validation_records.npz'),
                     trainable_parameters=sum(p.numel() for p in model.parameters()))
        (out / 'execution.json').write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps(state, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
