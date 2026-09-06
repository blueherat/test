"""One fixed distributed fit of the boundary-consistent noisy ratio critic."""
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
from experiments.raev2_paired_ratio_loss import paired_scaled_logistic
from experiments.raev2_paired_ratio_model import PairedRatioCritic
from experiments.summarize_raev2_guidance_20260907 import DATA, sha


class Banks:
    def __init__(self, split):
        self.real = np.load(split['real_path'], mmap_mode='r')
        self.lookup = np.load(split['real_lookup'])
        self.fake = [np.load(p, mmap_mode='r') for p in split['fake_paths']]

    def batch(self, labels, real_k, fake_k, device):
        real = np.array(self.real[self.lookup[labels, real_k]], copy=True)
        ids = labels + 1000 * fake_k
        fake = np.empty_like(real)
        for rank, bank in enumerate(self.fake):
            mask = ids % 4 == rank
            fake[mask] = bank[ids[mask] // 4]
        return (torch.from_numpy(real).to(device=device, dtype=torch.float32),
                torch.from_numpy(fake).to(device=device, dtype=torch.float32))


def predict_pair(model, real, fake, t, labels, generator):
    noise = torch.randn(real.shape, device=real.device, dtype=torch.float32, generator=generator)
    shape = (-1, 1, 1, 1)
    signal = 1 - t
    p = signal.reshape(shape) * real + t.reshape(shape) * noise
    q = signal.reshape(shape) * fake + t.reshape(shape) * noise
    values = model(torch.cat((p, q)), torch.cat((t, t)), torch.cat((labels, labels)))
    fp, fq = values.chunk(2)
    return paired_scaled_logistic(fp, fq, signal), fp, fq


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
    source_paths = [Path(__file__).resolve(), ROOT / 'experiments/raev2_paired_ratio_model.py',
                    ROOT / 'experiments/raev2_paired_ratio_loss.py', args.plan.resolve()]
    frozen = {str(p): sha(p) for p in source_paths}
    manifest_path = DATA / 'paired_ratio_data/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert sha(manifest_path) == plan['data_manifest_sha256']
    features_path = DATA / 'paired_ratio_data/class_features.pt'
    assert sha(features_path) == manifest['class_features_sha256']
    for split in manifest['splits'].values():
        assert sha(Path(split['real_lookup'])) == plan['lookup_sha256'][split['real_lookup']]
    started = time.time()
    out = args.output.resolve()
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
        for split in manifest['splits'].values():
            for path, digest in split['sources'].items():
                assert sha(Path(path)) == digest, 'data changed: ' + path
        (out / 'frozen_source').mkdir()
        for p in source_paths:
            (out / 'frozen_source' / p.name).write_bytes(p.read_bytes())
        state = {'complete': False, 'pid': os.getpid(), 'started_unix': started,
                 'plan': plan, 'sources': frozen, 'data_manifest_sha256': sha(manifest_path),
                 'torch_version': torch.__version__, 'gpu': torch.cuda.get_device_name(),
                 'all_data_sources_verified': True, 'stage': 'training'}
        (out / 'execution.json').write_text(json.dumps(state, indent=2) + '\n')
    dist.barrier()
    model = PairedRatioCritic(torch.load(features_path, map_location='cpu', weights_only=True)).to(device)
    model.train()
    ddp = DistributedDataParallel(model, device_ids=[local], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'],
                                  weight_decay=plan['weight_decay'], betas=(.9, .999))
    train_bank = Banks(manifest['splits']['train'])
    validation_bank = Banks(manifest['splits']['validation'])
    generator = torch.Generator(device=device).manual_seed(plan['train_noise_seed'] + rank)
    rng = np.random.default_rng(plan['train_pair_seed'] + rank)
    grid = torch.linspace(1, 0, 101, device=device)
    grid = 8 * grid / (1 + 7 * grid)
    batch = plan['global_batch'] // world
    assert batch * world == plan['global_batch']
    losses, clips = [], 0
    torch.cuda.synchronize()
    training_start = time.perf_counter()
    for step in range(plan['updates']):
        labels_np = rng.integers(0, 1000, batch)
        real, fake = train_bank.batch(labels_np, rng.integers(0, 5, batch), rng.integers(0, 5, batch), device)
        labels = torch.from_numpy(labels_np).to(device)
        t = grid[torch.from_numpy(rng.integers(1, 100, batch)).to(device)]
        optimizer.zero_grad(set_to_none=True)
        per_pair, _, _ = predict_pair(ddp, real, fake, t, labels, generator)
        loss = per_pair.mean()
        assert torch.isfinite(loss).item()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), plan['gradient_clip'], error_if_nonfinite=True)
        clips += int(norm.item() > plan['gradient_clip'])
        optimizer.step()
        losses.append(loss.detach())
        if (step + 1) % 128 == 0:
            recent = torch.stack(losses[-128:]).mean()
            dist.all_reduce(recent)
            if rank == 0:
                progress = {'step': step + 1, 'last_128_loss': recent.item() / world,
                            'elapsed_seconds': time.perf_counter() - training_start,
                            'rank0_gradient_clip_count': clips}
                (out / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
                print(json.dumps(progress), flush=True)
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - training_start
    timing = torch.tensor([train_seconds], dtype=torch.float64, device=device)
    dist.all_reduce(timing)
    model.eval()
    # One fixed post-fit holdout. Every fake validation endpoint appears once.
    # Times are balanced over all 99 positive-signal native query times, with
    # a fixed permutation unrelated to labels. No validation checkpoint search.
    valid_times = np.tile(np.arange(1, 100), 51)[:5000]
    np.random.default_rng(plan['validation_time_seed']).shuffle(valid_times)
    valid_generator = torch.Generator(device=device).manual_seed(plan['validation_noise_seed'] + rank)
    index = np.arange(rank, 5000, world)
    validation = []
    with torch.no_grad():
        for begin in range(0, len(index), batch):
            ids = index[begin:begin + batch]
            labels_np = ids % 1000
            real, fake = validation_bank.batch(labels_np, np.zeros(len(ids), dtype=int), ids // 1000, device)
            labels = torch.from_numpy(labels_np).to(device)
            t = grid[torch.from_numpy(valid_times[ids]).to(device)]
            loss, fp, fq = predict_pair(model, real, fake, t, labels, valid_generator)
            validation.extend(zip(ids.tolist(), valid_times[ids].tolist(), loss.cpu().tolist(),
                                  fp.cpu().tolist(), fq.cpu().tolist()))
    gathered = [None] * world
    dist.all_gather_object(gathered, validation)
    for p, digest in frozen.items():
        assert sha(Path(p)) == digest, 'source changed during training: ' + p
    if rank == 0:
        records = np.array(sorted(sum(gathered, []), key=lambda r: r[0]))
        assert np.array_equal(records[:, 0], np.arange(5000)) and np.isfinite(records).all()
        values = records[:, 2]
        class_means = values.reshape(5, 1000).mean(0)
        mean = float(values.mean())
        se = float(class_means.std(ddof=1) / np.sqrt(1000))
        validation_result = {'samples': 5000, 'mean_scaled_logistic': mean,
                             'class_cluster_standard_error': se,
                             'upper_two_standard_errors': mean + 2 * se,
                             'zero_critic_loss': 0., 'entry_condition_passed': mean + 2 * se < 0,
                             'entry_is_not_a_fid_or_gradient_generalization_claim': True}
        np.savez(out / 'validation_records.npz', records=records)
        checkpoint = out / 'critic.pt'
        torch.save({'state_dict': model.state_dict(), 'plan': plan, 'sources': frozen,
                    'validation': validation_result, 'updates': plan['updates']}, checkpoint)
        state.update(complete=True, stage='fit_and_validation_complete', training_gpu_seconds=timing.item(),
                     elapsed_seconds=time.time() - started, validation=validation_result,
                     checkpoint_sha256=sha(checkpoint), validation_sha256=sha(out / 'validation_records.npz'),
                     trainable_parameters=sum(p.numel() for p in model.parameters()))
        (out / 'execution.json').write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps(state, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
