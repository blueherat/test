"""One fixed 64K/8K real-row expansion, disjoint from current six thousand."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.prepare_raev2_potential_clean_bank import checked_identity, load_config, DEFAULT_CONFIG
from experiments.summarize_raev2_guidance_20260907 import DATA, sha


def main():
    out = DATA/'real_ratio_bank64k'
    out.mkdir(exist_ok=False)
    packed = Path('/data/shared/imagenet-1k/random_access_v1')
    old = DATA.parent/'raev2_guidance_restart_20260906/potential_clean_bank_fp32_v1'
    excluded = np.concatenate([np.load(old/s/'metadata.npz')['rows'] for s in ('train', 'validation')])
    args = SimpleNamespace(config=Path(DEFAULT_CONFIG), packed_data_path=packed,
        dino_ckpt_dir=DATA.parent.parent/'models/RAEv2/encoders/dinov3',
        dino_repo_dir=DATA.parent.parent/'models/RAEv2/dinov3_repo',
        identity_reference=DATA.parent/'raev2_guidance_restart_20260906/heldout_clean_c_current_fp32/request.json')
    identity = checked_identity(args, load_config(DEFAULT_CONFIG))
    manifest = json.loads((packed/'manifest.json').read_text())
    all_labels = np.concatenate([np.load(packed/s['labels_file']) for s in manifest['shards']])
    eligible = np.ones(len(all_labels), dtype=bool)
    eligible[excluded] = False
    rng = np.random.default_rng(202609105)
    train, validation = np.empty(64000, dtype=np.int64), np.empty(8000, dtype=np.int64)
    for c in range(1000):
        candidates = np.flatnonzero((all_labels == c) & eligible)
        selected = rng.choice(candidates, size=72, replace=False)
        train[c::1000], validation[c::1000] = selected[:64], selected[64:]
    assert len(np.unique(np.r_[train, validation])) == 72000
    assert not np.intersect1d(np.r_[train, validation], excluded).size
    sources = {str(p): sha(p) for p in (Path(__file__).resolve(), ROOT/'experiments/encode_raev2_ratio_real64k.py',
               ROOT/'experiments/prepare_raev2_potential_clean_bank.py', ROOT/'experiments/raev2_training_core.py')}
    selection = {'seed': 202609105, 'packed_path': str(packed), 'train_count': 64000, 'validation_count': 8000,
                 'identity': identity, 'excluded_current_bank_rows': 6000, 'train_validation_rows_disjoint': True,
                 'limits': 'new fit holdout, not a claim that no older repository research ever used these public images',
                 'preprocessing': 'original packed bytes, deterministic ADM crop256, FP32/255, no flip, official K7 normalization'}
    (out/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    state = {'complete': False, 'pid': os.getpid(), 'started_unix': time.time(), 'sources': sources,
             'selection_sha256': sha(out/'selection.json'), 'splits': {},
             'concurrency_note': 'may share the four GPUs with native-state data preparation; worker time is inclusive, not exclusive compute time'}
    def save():
        temp = out/'execution.tmp'; temp.write_text(json.dumps(state, indent=2)+'\n'); temp.replace(out/'execution.json')
    save()
    (out/'frozen_source').mkdir()
    for path in sources:
        p = Path(path); (out/'frozen_source'/p.name).write_bytes(p.read_bytes())
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
           'HF_HUB_OFFLINE': '1', 'PYTHONUNBUFFERED': '1'}
    for split, rows in [('train', train), ('validation', validation)]:
        folder = out/split; folder.mkdir()
        ids = np.arange(len(rows), dtype=np.int64); labels = ids%1000
        assert np.array_equal(all_labels[rows], labels)
        np.savez(folder/'metadata.npz', ids=ids, rows=rows, labels=labels)
        np.save(folder/'real_lookup.npy', ids.reshape(-1, 1000).T)
        latents = np.lib.format.open_memmap(folder/'latents.npy', mode='w+', dtype=np.float16, shape=(len(rows), 1024, 16, 16))
        hashes = np.lib.format.open_memmap(folder/'image_hashes.npy', mode='w+', dtype='S64', shape=(len(rows),))
        del latents, hashes
        state.update(stage=split, jobs=[]); save()
        jobs = []
        for rank in range(4):
            command = [sys.executable, str(ROOT/'experiments/encode_raev2_ratio_real64k.py'), '--split', split, '--shard', str(rank)]
            log = (folder/f'shard{rank}.log').open('w')
            child = subprocess.Popen(command, cwd=ROOT, env={**env, 'CUDA_VISIBLE_DEVICES': str(rank)}, stdout=log, stderr=subprocess.STDOUT)
            jobs.append((child, log)); state['jobs'].append({'rank': rank, 'pid': child.pid, 'command': command}); save()
        while any(child.poll() is None for child, _ in jobs):
            failed = [child.returncode for child, _ in jobs if child.poll() not in (None, 0)]
            if failed:
                for child, _ in jobs:
                    if child.poll() is None: child.terminate()
                state.update(stage='failed_requires_inspection', exit_codes=failed); save()
                raise RuntimeError('real encoder failed; existing data retained')
            time.sleep(2)
        latents = np.load(folder/'latents.npy', mmap_mode='r')
        summaries = []
        for rank, (child, log) in enumerate(jobs):
            log.close(); assert child.returncode == 0
            summary = json.loads((folder/f'shard{rank}_summary.json').read_text())
            assert summary['complete'] and summary['original_encoder_anchor_bitwise_equal']
            digest = hashlib.sha256()
            for batch in range(rank, len(rows)//8, 4):
                digest.update(latents[batch*8:batch*8+8].tobytes())
            assert digest.hexdigest() == summary['owned_latent_slices_sha256']
            summaries.append(summary)
        assert sum(s['count'] for s in summaries) == len(rows)
        assert np.all(np.load(folder/'image_hashes.npy') != b'')
        for path, digest in sources.items(): assert sha(Path(path)) == digest
        state['splits'][split] = {'count': len(rows), 'real_path': str(folder/'latents.npy'),
            'real_lookup': str(folder/'real_lookup.npy'), 'worker_seconds_inclusive': sum(s['seconds'] for s in summaries),
            'files': {name: sha(folder/name) for name in ('latents.npy','real_lookup.npy','metadata.npz','image_hashes.npy')}}
        save()
    state.update(complete=True, stage='real_bank_complete', elapsed_seconds=time.time()-state['started_unix']); save()


if __name__ == '__main__':
    main()
