#!/usr/bin/env python3
"""Frozen two-witness test of the failed potential's omitted channel space."""
from __future__ import annotations
import time
START, CPU_START = time.perf_counter(), time.process_time()
import argparse
import gc
import json
import os
from pathlib import Path
import resource
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.train_raev2_observable_potential import (
    RESTART, DEFAULT_CONFIG, DEFAULT_CHECKPOINT, artifact, atomic_json,
    sha256_file, load_banks, batch_from_bank, clean_forward, load_config,
    instantiate_from_config, ScalarGuidancePotential,
)

PROTOCOL = 'raev2_potential_two_omitted_witnesses_v1'
BATCH = 32
BANK = RESTART/'potential_clean_bank_fp32_v1'
POTENTIAL = RESTART/'observable_potential_train_v1/potential_final.pt'
PREPARE = RESTART/'potential_omitted_mean_v1'
FIELDS = ('residual_dot_g1', 'residual_dot_g2', 'g1_energy', 'g1_dot_g2', 'g2_energy', 'residual_energy_fp64')


def record(path):
    result = artifact(Path(path))
    result['mtime_ns'] = Path(path).stat().st_mtime_ns
    return result


def verify(rec, *, content=True):
    path = Path(rec['path'])
    stat = path.stat()
    if stat.st_size != rec['size_bytes'] or ('mtime_ns' in rec and stat.st_mtime_ns != rec['mtime_ns']):
        raise ValueError(f'file identity changed: {path}')
    if content and sha256_file(path) != rec['sha256']:
        raise ValueError(f'file hash changed: {path}')
    return path


def cost():
    return {'wall_seconds_from_script_start': time.perf_counter()-START,
            'cpu_seconds_from_script_start': time.process_time()-CPU_START,
            'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'timing_excludes': 'interpreter startup before time import and final summary serialization/exit'}


def prepare(out):
    training_path = RESTART/'observable_potential_train_v1/request.json'
    training = json.loads(training_path.read_text())
    if record(POTENTIAL)['sha256'] != '495b3313e945f4b845307ae0d520c3c8d8fa70e60e3983002297cd3f7ccc632e':
        raise ValueError('not the fixed failed potential')
    for value in training['sources'].values():
        verify(value)
    request = {'protocol': PROTOCOL, 'mode': 'prepare', 'runner': record(Path(__file__)),
        'bank_dir': str(BANK.resolve()), 'potential': record(POTENTIAL),
        'training_request': record(training_path), 'baseline': record(DEFAULT_CHECKPOINT),
        'config': record(DEFAULT_CONFIG), 'time_grid': training['time_grid'],
        'time_probability': training['time_probability'], 'time_weight_normalizer': training['time_weight_normalizer'],
        'seeds': training['seeds'], 'batch_size': BATCH,
        'mean_rule': 'all five training images per class; full CHW mean in FP64 stored FP32; no validation fitting',
        'basis_rule': 'CPU FP64 SVD of frozen 128x1024 W; retain all 128 right singular vectors, no rank search',
        'witnesses': ['Qz', 'Q training_class_mean'], 'time_units': 'clean gradients; velocity witnesses g/t, weight h/t^2',
        'statistical_scope': 'previously viewed validation bank; paired mechanism diagnosis, not independent method confirmation',
        'device': 'cpu', 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'new_fid': False, 'training': False, 'model_calls': 0}
    if request['baseline']['sha256'] != training['baseline_checkpoint']['sha256'] or request['config']['sha256'] != training['config']['sha256']:
        raise ValueError('baseline differs from fixed training')
    atomic_json(out/'request.json', request)
    banks, bank_identity = load_banks(BANK)
    if bank_identity != training['bank']:
        raise ValueError('bank differs from failed potential training')
    request['bank'] = bank_identity
    atomic_json(out/'request.json', request)
    values, metadata = banks['train']
    path = out/'training_class_means.npy'
    means = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(1000,1024,16,16))
    for label in range(1000):
        ids = np.flatnonzero(metadata['labels'] == label)
        if len(ids) != 5:
            raise ValueError('expected five train images per class')
        means[label] = np.asarray(values[ids], dtype=np.float64).mean(axis=0).astype(np.float32)
    means.flush()
    del means
    checkpoint = torch.load(POTENTIAL, map_location='cpu', weights_only=False)
    if checkpoint['request_sha256'] != request['training_request']['sha256'] or checkpoint['updates'] != 2048:
        raise ValueError('potential training identity mismatch')
    weight = checkpoint['potential']['input_conv.weight'].double().reshape(128,1024)
    _, singular, basis = torch.linalg.svd(weight, full_matrices=False)
    rank = int(torch.linalg.matrix_rank(weight))
    if rank != 128:
        raise ValueError('frozen input matrix not full row rank; no implicit truncation')
    orth_error = float((basis@basis.T-torch.eye(128,dtype=torch.float64)).abs().max())
    residual = float(torch.linalg.norm(weight-(weight@basis.T)@basis)/torch.linalg.norm(weight))
    if orth_error > 1e-12 or residual > 1e-12:
        raise FloatingPointError('invalid SVD rowspace')
    np.savez(out/'rowspace.npz', basis=basis.numpy(), singular_values=singular.numpy())
    summary = {'protocol': PROTOCOL, 'complete': True, 'mode': 'prepare',
        'request': record(out/'request.json'), 'class_means': record(path), 'rowspace': record(out/'rowspace.npz'),
        'bank': bank_identity, 'rank': rank, 'svd_orthogonality_max_error': orth_error,
        'rowspace_relative_reconstruction_error': residual, 'training_images_read': 5000,
        'training_latent_payload_bytes': 5000*1024*16*16*2,
        'model_calls': 0, 'cost': cost()}
    atomic_json(out/'summary.json', summary)
    print(json.dumps({'complete': True, 'mode': 'prepare', 'rank': rank, 'cost': summary['cost']}), flush=True)


def validate(out, shard):
    prep = json.loads((PREPARE/'summary.json').read_text())
    if not prep['complete']:
        raise ValueError('mean prepare is incomplete')
    verify(prep['request'])
    request0 = json.loads((PREPARE/'request.json').read_text())
    for key in ('class_means','rowspace','bank'):
        verify(prep[key])
    for key in ('runner','potential','training_request','config'):
        verify(request0[key])
    # The 10.5GB checkpoint was just hashed by prepare; guard its unchanged
    # metadata through loading, disclose this rather than hash it four times.
    verify(request0['baseline'], content=False)
    bank_summary = json.loads(Path(prep['bank']['path']).read_text())
    validation = bank_summary['banks']['validation']
    for key in ('latents','metadata'):
        verify(validation[key])
    with np.load(validation['metadata']['path'], allow_pickle=False) as data:
        metadata = {key:data[key].copy() for key in data.files}
    latents = np.load(validation['latents']['path'], mmap_mode='r', allow_pickle=False)
    if latents.shape != (1000,1024,16,16) or latents.dtype != np.float16 or not np.array_equal(metadata['ids'],np.arange(1000)):
        raise ValueError('invalid heldout bank')
    means = np.load(prep['class_means']['path'], mmap_mode='r', allow_pickle=False)
    with np.load(prep['rowspace']['path'], allow_pickle=False) as data:
        basis_cpu = data['basis'].copy()
    if means.shape != (1000,1024,16,16) or basis_cpu.shape != (128,1024):
        raise ValueError('unexpected prepared shapes')
    indices = list(range(shard,100,4))
    old_root = RESTART/'observable_potential_validation_v1'
    request = {'protocol': PROTOCOL, 'mode': 'validate', 'shard_index': shard, 'num_shards': 4,
        'time_indices': indices, 'prepare_summary': record(PREPARE/'summary.json'),
        'prepare_request': prep['request'], 'runner': record(Path(__file__)),
        'time_grid': request0['time_grid'], 'time_probability': request0['time_probability'],
        'time_weight_normalizer': request0['time_weight_normalizer'],
        'validation_metadata': validation['metadata'], 'baseline': request0['baseline'],
        'potential': request0['potential'], 'batch_size': BATCH, 'noise_seed': 202609094,
        'precision': 'native BF16 heads, FP32 states/R, TF32 off; Q and new products/reductions FP64',
        'baseline_identity_scope': 'full SHA in prepare, unchanged size/mtime checked before and after loading',
        'old_validation_root': str(old_root.resolve()), 'old_step_artifacts': [],
        'witnesses': list(FIELDS), 'training': False, 'new_fid': False}
    for k in indices:
        path = old_root/f'shard{k%3}'/f'step{k:03d}.npz'
        request['old_step_artifacts'].append(record(path))
    atomic_json(out/'request.json',request)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = load_config(DEFAULT_CONFIG)
    model = instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    checkpoint = torch.load(DEFAULT_CHECKPOINT,map_location='cpu',mmap=True,weights_only=False)
    model.load_state_dict(checkpoint['ema'],strict=True)
    if int(checkpoint['step']) != 100080:
        raise ValueError('unexpected backbone step')
    del checkpoint
    verify(request0['baseline'],content=False)
    basis = torch.from_numpy(basis_cpu).to(device=device,dtype=torch.float64)
    grid = torch.tensor(request['time_grid'],dtype=torch.float32,device=device)
    generator = torch.Generator(device=device)
    bank = (latents,metadata)
    stage2_calls = potential_calls = sample_calls = 0
    orthogonality = None
    step_records = []
    torch.cuda.reset_peak_memory_stats()
    phase_start = time.perf_counter()

    def complement(value):
        flat = value.double().flatten(2)
        small = torch.einsum('kc,bcp->bkp',basis,flat)
        return (flat-torch.einsum('kc,bkp->bcp',basis,small)).reshape(value.shape)

    def inner(first,second):
        return (first*second).flatten(1).mean(1)

    with torch.no_grad():
        for k, old_record in zip(indices,request['old_step_artifacts'],strict=True):
            generator.manual_seed(202609094)
            blocks, old_mse = [], []
            for start in range(0,1000,BATCH):
                ids = np.arange(start,min(start+BATCH,1000))
                clean,labels = batch_from_bank(bank,ids,device)
                times = grid[k].expand(len(ids))
                noise = torch.randn(clean.shape,device=device,generator=generator)
                z = (1-times[:,None,None,None])*clean+times[:,None,None,None]*noise
                guided = clean_forward(model,z,times,labels)
                stage2_calls += 1
                sample_calls += len(ids)
                r = clean-guided
                old_mse.append(r.square().flatten(1).mean(1).double().cpu().numpy())
                mu = torch.from_numpy(np.array(means[metadata['labels'][ids]],copy=True)).to(device)
                g1,g2 = complement(z),complement(mu)
                r64 = r.double()
                blocks.append(torch.stack((inner(r64,g1),inner(r64,g2),inner(g1,g1),inner(g1,g2),inner(g2,g2),inner(r64,r64)),1).cpu().numpy())
                if orthogonality is None:
                    potential = ScalarGuidancePotential().to(device).eval().requires_grad_(False)
                    checkpoint = torch.load(POTENTIAL,map_location='cpu',weights_only=False)
                    potential.load_state_dict(checkpoint['potential'],strict=True)
                    c = potential.clean_correction(z,times,labels)
                    potential_calls += 1
                    qc = complement(c)
                    fraction = float(torch.linalg.vector_norm(qc)/torch.linalg.vector_norm(c.double()))
                    orthogonality = {'step':k,'sample_ids':ids.tolist(),'Qc_relative_norm':fraction,
                        'max_absolute_c_dot_g1_per_image':float(inner(c.double(),g1).abs().max()),
                        'max_absolute_c_dot_g2_per_image':float(inner(c.double(),g2).abs().max())}
                    if not np.isfinite(fraction) or fraction > 2e-5:
                        raise FloatingPointError('potential correction not numerically in frozen rowspace')
                    del potential,checkpoint,c,qc
                    gc.collect()
            values = np.concatenate(blocks)
            r2_old = np.concatenate(old_mse)
            verify(old_record)
            with np.load(old_record['path'],allow_pickle=False) as old:
                if not np.array_equal(old['ids'],metadata['ids']) or not np.array_equal(old['labels'],metadata['labels']):
                    raise ValueError('original validation ID/label mismatch')
                if not np.array_equal(r2_old,old['residual_mse']):
                    raise AssertionError(f'old R2 parity failed at step {k}; max error {np.max(np.abs(r2_old-old["residual_mse"]))}')
            if values.shape != (1000,6) or not np.isfinite(values).all():
                raise FloatingPointError('invalid witness values')
            path = out/f'step{k:03d}.npz'
            np.savez(path,ids=metadata['ids'],labels=metadata['labels'],step=np.int64(k),time=np.float64(float(grid[k])),
                residual_mse_fp32_parity=r2_old,**{key:values[:,j] for j,key in enumerate(FIELDS)})
            step_records.append(record(path))
            progress = {'step':k,'times_complete':len(step_records),'times_total':len(indices),
                        'stage2_sample_forwards':sample_calls,'phase_wall_seconds':time.perf_counter()-phase_start}
            atomic_json(out/'progress.json',progress)
            print(json.dumps(progress),flush=True)
    summary = {'protocol':PROTOCOL,'complete':True,'mode':'validate','request':record(out/'request.json'),
        'time_indices':indices,'step_artifacts':step_records,'orthogonality':orthogonality,
        'old_residual_mse_bitwise_parity':True,'stage2_forward_calls':stage2_calls,
        'stage2_sample_forwards':sample_calls,'potential_input_gradient_calls':potential_calls,
        'potential_input_gradient_sample_forwards':len(orthogonality['sample_ids']),
        'phase_wall_seconds':time.perf_counter()-phase_start,
        'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'cost':cost(),
        'new_images':0,'new_fid':False,'goal_achieved':False}
    atomic_json(out/'summary.json',summary)
    print(json.dumps({'complete':True,'shard':shard,'cost':summary['cost']}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prepare','validate'),required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--shard-index',type=int,default=0)
    args=parser.parse_args()
    if not 0<=args.shard_index<4:
        raise ValueError('invalid shard')
    out=args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    (out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    (out/'frozen_protocol.md').write_bytes((ROOT/'docs/RAEV2_POTENTIAL_OMITTED_WITNESS_20260906_ZH.md').read_bytes())
    torch.set_num_threads(4)
    if args.mode=='prepare': prepare(out)
    else: validate(out,args.shard_index)


if __name__=='__main__': main()
