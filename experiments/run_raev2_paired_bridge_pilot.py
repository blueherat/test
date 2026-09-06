#!/usr/bin/env python3
"""One fixed paired-bridge fit with a matched mean-field mechanism control.

No images, FID, checkpoint selection, gain search, or training of the RAE.
Validation labels are used only at teacher interpolants, never at model midpoints.
"""
from __future__ import annotations

import time
START_WALL, START_CPU = time.perf_counter(), time.process_time()
import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'external/RAEv2/src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.raev2_paired_bridge import (
    PairedBridgeField, bridge_midpoint, construct_bridge, native_euler,
)
from experiments.train_raev2_observable_potential import (
    artifact, atomic_json, atomic_torch_save, batch_from_bank, clean_forward,
    load_banks, sha256_file, write_csv,
)
from experiments.sample_raev2_pfr_retiming import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, load_config, shifted_time_grid,
)
from utils.model_utils import instantiate_from_config
from utils.guidance_utils import forward_with_internalguidance

PROTOCOL = 'raev2_paired_bridge_fixed_mechanism_pilot_v1'
R = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906')
PLAN = ROOT / 'docs/RAEV2_PAIRED_BRIDGE_PILOT_PROTOCOL_20260906_ZH.md'
BATCH, UPDATES, LR = 32, 2048, 1e-4
SEEDS = dict(model=202609141, data=202609142, noise=202609143,
             tau=202609144, validation=202609145, rollout=202609146)
SOURCES = (
    Path(__file__), ROOT / 'experiments/raev2_paired_bridge.py',
    ROOT / 'tests/test_raev2_paired_bridge.py', PLAN,
    ROOT / 'experiments/train_raev2_observable_potential.py',
    ROOT / 'experiments/sample_raev2_pfr_retiming.py',
    ROOT / 'external/RAEv2/src/utils/guidance_utils.py',
    ROOT / 'external/RAEv2/src/stage2/models/DDT.py',
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def sync():
    torch.cuda.synchronize()


class Calls:
    def __init__(self, modules):
        self.counts = {k: dict(calls=0, samples=0) for k in modules}
        self.handles = []
        for name, module in modules.items():
            def hook(_module, args, key=name):
                self.counts[key]['calls'] += 1
                self.counts[key]['samples'] += len(args[0])
            self.handles.append(module.register_forward_pre_hook(hook))

    def snapshot(self):
        return copy.deepcopy(self.counts)


def column(x):
    return x[:, None, None, None]


def baseline_forward(model,z,t,labels,*,heads=False):
    """All real RAE calls use B8, including charged padding of validation tails."""
    parts=[]
    for begin in range(0,len(z),8):
        state,times,classes=z[begin:begin+8],t[begin:begin+8],labels[begin:begin+8]
        size=len(state)
        if size<8:
            indices=torch.arange(8,device=z.device)%size
            state,times,classes=state[indices],times[indices],classes[indices]
        result=clean_forward(model,state,times,classes,heads=heads)
        parts.append(tuple(item[:size] for item in result) if heads else result[:size])
    if heads:
        return tuple(torch.cat([part[k] for part in parts]) for k in range(3))
    return torch.cat(parts)


def scalar_terms(pred, target):
    p, r = pred.float(), target.float()
    return dict(target_energy=r.square().flatten(1).mean(1),
                prediction_energy=p.square().flatten(1).mean(1),
                cross=(p*r).flatten(1).mean(1))


def regression_loss(pred, target):
    # Dropping the target-only constant retains the exact normalized MSE gradient.
    return (.5*pred.square()-pred*target.detach()).flatten(1).mean(1).mean()


def grad_norm(module):
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    norm = torch.stack([g.detach().square().sum() for g in grads]).sum().sqrt()
    if not bool(torch.isfinite(norm)) or float(norm) <= 0:
        raise FloatingPointError('nonfinite or zero total parameter gradient')
    return float(norm)


def field_statistics(pred, target):
    terms = scalar_terms(pred, target)
    terms['risk_gain'] = 2*terms['cross']-terms['prediction_energy']
    return {k: float(v.mean()) for k, v in terms.items()}


def moments(z):
    """Fixed 2048-vector: all channel spatial means and spatial second moments."""
    z = z.double()
    return torch.cat((z.mean((2, 3)), z.square().mean((2, 3))), dim=1)


def sources_to(output):
    records = {}
    directory = output / 'sources'
    directory.mkdir()
    for path in SOURCES:
        dest = directory / path.name
        shutil.copy2(path, dest)
        records[str(path.relative_to(ROOT))] = artifact(dest)
    return records


def check_previous(directory, request, required_mode):
    summary = json.loads((directory / 'summary.json').read_text())
    prior = json.loads((directory / 'request.json').read_text())
    if not summary.get('complete') or summary.get('mode') != required_mode:
        raise ValueError('prior stage not complete')
    for key in ('protocol', 'bank', 'config', 'baseline_checkpoint', 'architecture', 'seeds'):
        if prior[key] != request[key]:
            raise ValueError(f'prior stage identity differs: {key}')
    for key, item in request['sources'].items():
        if prior['sources'][key]['sha256'] != item['sha256']:
            raise ValueError(f'prior source changed: {key}')
    if sha256_file(directory / 'request.json') != summary['request']['sha256']:
        raise ValueError('prior request changed')
    return summary


def numerical_pilot(model, candidate, control, banks, grid, out, calls, device):
    clean, labels = batch_from_bank(banks['train'], np.arange(BATCH), device)
    steps = torch.tensor([0, 14, 28, 42, 57, 71, 85, 99]*4, device=device)
    t, s = grid[steps], grid[steps+1]
    gen = torch.Generator(device=device).manual_seed(SEEDS['noise']+1)
    noise = torch.randn(clean.shape, generator=gen, device=device)
    z = (1-column(t))*clean+column(t)*noise
    guided, full, base = baseline_forward(model, z, t, labels, heads=True)

    class Cached:
        in_channels = 1024
        def __call__(self, state, times, **kwargs):
            assert torch.equal(state, z) and torch.equal(times, t)
            return full, base

    ref = forward_with_internalguidance(
        Cached(), torch.cat([z,z]), torch.cat([t,t]), 1.78, (.1,1),
        context=torch.cat([labels,labels]), attn_mask=None,
    )[:BATCH].float()
    if not torch.equal(ref, guided):
        raise AssertionError('production native-head IG parity failed')
    bridge = construct_bridge(clean, noise, guided, t, s, torch.full_like(t,.5))
    y, target = bridge['Y'], bridge['target'].detach()
    if not torch.equal(bridge['z'], z):
        raise AssertionError('teacher bridge arithmetic changed')
    for field, locked in ((candidate,False), (control,True)):
        with torch.no_grad():
            zero = bridge_midpoint(field, y, t, s, labels, tau_locked_zero=locked)
        if not torch.equal(zero, y):
            raise AssertionError('zero-initialized field changed native successor')
    opts = {name: torch.optim.Adam(net.parameters(),lr=LR)
            for name,net in [('candidate',candidate),('control',control)]}
    history = []
    for step in range(8):
        row = {'update':step+1}
        for name,net,state,tau in (
            ('candidate',candidate,bridge['U'],torch.full_like(t,.5)),
            ('control',control,y,torch.zeros_like(t)),
        ):
            opts[name].zero_grad(set_to_none=True)
            pred = net(state,t,s,tau,labels)
            loss = regression_loss(pred,target)
            loss.backward()
            row[name+'_gradient_norm'] = grad_norm(net)
            row[name+'_loss'] = float(loss.detach())
            opts[name].step()
        history.append(row)
    for name in opts:
        if not history[-1][name+'_loss'] < history[0][name+'_loss']:
            raise AssertionError(f'{name}: fixed-batch numerical learning did not descend')
    # No weights from this numerical exercise are used by the actual fit.
    write_csv(out/'updates.csv',history)
    return dict(pilot_passed=True, native_head_parity=True, zero_midpoint_parity=True,
                numerical_only=True, learned_weights_saved=False,
                discarded_parameter_updates_per_model=8,
                parameter_backwards=dict(candidate=8,control=8),
                history=history, calls=calls.snapshot())


def train(model, candidate, control, banks, grid, out, calls, device):
    rng = np.random.default_rng(SEEDS['data'])
    noise_gen = torch.Generator(device=device).manual_seed(SEEDS['noise'])
    tau_gen = torch.Generator(device=device).manual_seed(SEEDS['tau'])
    nets = dict(candidate=candidate,control=control)
    opts = {name:torch.optim.Adam(net.parameters(),lr=LR) for name,net in nets.items()}
    counts = np.zeros((100,),dtype=np.int64)
    image_counts = np.zeros((5000,),dtype=np.int64)
    history = []
    started = time.perf_counter()
    for update in range(UPDATES):
        ids = rng.integers(5000,size=BATCH)
        steps = rng.integers(100,size=BATCH)
        np.add.at(counts,steps,1); np.add.at(image_counts,ids,1)
        clean, labels = batch_from_bank(banks['train'],ids,device)
        steps_gpu = torch.from_numpy(steps).to(device)
        t, s = grid[steps_gpu],grid[steps_gpu+1]
        noise = torch.randn(clean.shape,generator=noise_gen,device=device)
        tau = torch.rand((BATCH,),generator=tau_gen,device=device)
        z = (1-column(t))*clean+column(t)*noise
        guided = baseline_forward(model,z,t,labels)
        bridge = construct_bridge(clean,noise,guided,t,s,tau)
        target = bridge['target'].detach()
        row = {'update':update+1}
        for name,net,state,aux_time in (
            ('candidate',candidate,bridge['U'],tau),
            ('control',control,bridge['Y'],torch.zeros_like(tau)),
        ):
            opts[name].zero_grad(set_to_none=True)
            pred = net(state,t,s,aux_time,labels)
            loss = regression_loss(pred,target)
            loss.backward()
            gradient_norm = grad_norm(net)
            opts[name].step()
            row[name+'_loss'] = float(loss.detach())
            row[name+'_gradient_norm'] = gradient_norm
            if update == 0 or (update+1)%32 == 0:
                row.update({name+'_'+k:v for k,v in field_statistics(pred.detach(),target).items()})
        if not all(np.isfinite(v) for v in row.values()):
            raise FloatingPointError('nonfinite training statistic')
        row['elapsed_seconds'] = time.perf_counter()-started
        history.append(row)
        if update == 0 or (update+1)%32 == 0:
            atomic_json(out/'progress.json',dict(complete=False,mode='train',**row))
            print(json.dumps(row),flush=True)
        if (update+1)%256 == 0:
            # Consistent columns: all rows retain only the always-recorded fields.
            common = ('update','candidate_loss','candidate_gradient_norm',
                      'control_loss','control_gradient_norm','elapsed_seconds')
            write_csv(out/'training.csv',[{k:r[k] for k in common} for r in history])
    sync()
    elapsed = time.perf_counter()-started
    saved = out/'final.pt'
    states = {name:{k:v.detach().cpu() for k,v in net.state_dict().items()}
              for name,net in nets.items()}
    atomic_torch_save(saved,dict(protocol=PROTOCOL,updates=UPDATES,**states,
        request=str(out/'request.json'),request_sha256=sha256_file(out/'request.json'),
        data_rng_state=rng.bit_generator.state,noise_rng_state=noise_gen.get_state(),
        tau_rng_state=tau_gen.get_state()))
    np.savez(out/'training_counts.npz',step_counts=counts,image_counts=image_counts)
    atomic_json(out/'training_rows.json',history)
    return dict(updates=UPDATES,checkpoint=artifact(saved),
                checkpoint_selection='fixed final update, both arms',
                joint_training_seconds=elapsed,calls=calls.snapshot(),
                parameter_backwards=dict(candidate=UPDATES,control=UPDATES),
                actual_time_histogram=counts.tolist(),
                unique_training_images=int(np.count_nonzero(image_counts)),
                extra_control_training_charged=True,final_row=history[-1])


@torch.no_grad()
def validate(model, candidate, control, banks, grid, out, calls, device):
    # Ten assigned times per source, all 100 original query times equally covered.
    perm = np.random.default_rng(SEEDS['validation']).permutation(1000)
    groups = [perm[k*100:(k+1)*100] for k in range(10)]
    rows, moment_records = [], []
    noise_gen = torch.Generator(device=device).manual_seed(SEEDS['validation'])
    # One independent noise per held-out source; reused across its assigned times.
    noise_bank = torch.randn((1000,1024,16,16),generator=noise_gen,device=device).cpu()
    noise_identity = tensor_hash(noise_bank)
    sums = {name:np.zeros((100,2048),dtype=np.float64)
            for name in ('target','official','candidate','control')}
    squared_delta_sums = {name:np.zeros(100,dtype=np.float64)
                          for name in ('official','candidate','control')}
    counts = np.zeros(100,dtype=np.int64)
    for k in range(100):
        group = groups[k%10]
        for start in range(0,len(group),BATCH):
            ids = group[start:start+BATCH]
            clean, labels = batch_from_bank(banks['validation'],ids,device)
            noise = noise_bank[torch.from_numpy(ids)].to(device)
            t = grid[k].expand(len(ids)); s = grid[k+1].expand(len(ids))
            z = (1-column(t))*clean+column(t)*noise
            guided = baseline_forward(model,z,t,labels)
            bridge = construct_bridge(clean,noise,guided,t,s,torch.zeros_like(t))
            target, y = bridge['target'],bridge['Y']
            predictions = {}
            for aux in (0.,.5,1.):
                u = (1-aux)*y+aux*bridge['W']
                pred = candidate(u,t,s,torch.full_like(t,aux),labels)
                predictions['candidate_tau'+str(aux)] = scalar_terms(pred,target)
            pred0 = control(y,t,s,torch.zeros_like(t),labels)
            predictions['control_tau0.0'] = scalar_terms(pred0,target)
            outputs = dict(official=y,
                candidate=bridge_midpoint(candidate,y,t,s,labels),
                control=bridge_midpoint(control,y,t,s,labels,tau_locked_zero=True))
            features = {'target':moments(bridge['W']).cpu().numpy()}
            features.update({name:moments(value).cpu().numpy() for name,value in outputs.items()})
            for name,value in features.items(): sums[name][k] += value.sum(0)
            counts[k] += len(ids)
            for name in outputs:
                delta = features[name]-features['target']
                squared_delta_sums[name][k] += np.square(delta).sum()
            for i,sample_id in enumerate(ids):
                row = dict(sample_id=int(sample_id),label=int(labels[i]),step_index=k,
                           t=float(t[i]),s=float(s[i]),beta=float(bridge['beta'][i]))
                for name,terms in predictions.items():
                    for term,value in terms.items(): row[name+'_'+term]=float(value[i])
                for name,value in outputs.items():
                    if not bool(torch.isfinite(value).all()):
                        raise FloatingPointError('nonfinite heldout finite map')
                    row[name+'_shift_energy'] = float((value[i]-y[i]).square().mean())
                rows.append(row)
        if k%10 == 0 or k == 99:
            atomic_json(out/'progress.json',dict(complete=False,mode='validate',step=k,records=len(rows)))
            print(json.dumps(dict(mode='validate',step=k,records=len(rows))),flush=True)
    assert np.all(counts==100) and len(rows)==10000
    np.savez(out/'moments.npz',counts=counts,**{name+'_sum':value for name,value in sums.items()},
             **{name+'_squared_delta_sum':value for name,value in squared_delta_sums.items()})
    for k in range(100):
        rec = dict(step_index=k,t=float(grid[k]),s=float(grid[k+1]),samples=int(counts[k]))
        for name in outputs:
            delta_sum = sums[name][k]-sums['target'][k]
            rec[name+'_empirical_squared_mean_gap'] = float(np.square(delta_sum/counts[k]).mean())
            rec[name+'_offdiagonal_cross_statistic'] = float(
                (np.square(delta_sum).sum()-squared_delta_sums[name][k])/
                (counts[k]*(counts[k]-1)*2048))
        moment_records.append(rec)
    write_csv(out/'teacher_regression.csv',rows)
    write_csv(out/'finite_map_moments.csv',moment_records)
    return dict(records=len(rows),unique_validation_images=1000,records_per_time=100,
        noise_sha256=noise_identity,assignment=perm.tolist(),calls=calls.snapshot(),
        supervised_states='teacher Y+tau*R only; model midpoint has no paired velocity target',
        finite_map_metric='fixed channel first/second moments; empirical cohort quantities, not FID/KL',
        cross_statistic_scope='off-diagonal paired mean-gap statistic; fixed nonidentical classes, not claimed population-unbiased',
        source_reuse='each source and its independent noise appears at ten assigned times',
        endpoint_paired_mse_used=False)


@torch.no_grad()
def rollout(model,candidate,control,grid,out,calls,device):
    gen = torch.Generator(device=device).manual_seed(SEEDS['rollout'])
    noise = torch.randn((32,1024,16,16),generator=gen,device=device).cpu()
    noise_identity = tensor_hash(noise)
    all_moments, rows = {}, []
    hashes = {}
    for mode in ('official','candidate','control'):
        features = np.empty((101,32,2048),dtype=np.float64)
        endpoints = []
        for begin in range(0,32,8):
            labels = torch.arange(begin,begin+8,device=device)
            state = noise[begin:begin+8].to(device)
            features[0,begin:begin+8] = moments(state).cpu().numpy()
            sync(); started=time.perf_counter()
            for k in range(100):
                t,s = grid[k].expand(8),grid[k+1].expand(8)
                guided = baseline_forward(model,state,t,labels)
                successor = native_euler(state,guided,t,s)
                if mode=='candidate': state=bridge_midpoint(candidate,successor,t,s,labels)
                elif mode=='control': state=bridge_midpoint(control,successor,t,s,labels,tau_locked_zero=True)
                else: state=successor
                if not bool(torch.isfinite(state).all()):
                    raise FloatingPointError(f'nonfinite {mode} actual rollout at step{k}')
                features[k+1,begin:begin+8] = moments(state).cpu().numpy()
            sync()
            rows.append(dict(mode=mode,first_id=begin,count=8,
                seconds_including_per_step_diagnostics=time.perf_counter()-started))
            endpoints.append(state.cpu())
            print(json.dumps(rows[-1]),flush=True)
        all_moments[mode] = features
        joined=torch.cat(endpoints)
        hashes[mode]=tensor_hash(joined)
        np.save(out/(mode+'_endpoints.npy'),joined.numpy(),allow_pickle=False)
    np.savez(out/'rollout_moments.npz',**all_moments)
    write_csv(out/'diagnostic_costs.csv',rows)
    return dict(actual_rollouts=96,images_decoded=0,fid=False,calls=calls.snapshot(),
        noise_sha256=noise_identity,endpoint_sha256=hashes,labels=list(range(32)),
        timing_scope='includes per-step FP64 moments/copies; not a deployment speed benchmark',
        teacher_target_or_velocity_label_on_actual_state=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('pilot','train','validate','rollout'),required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--prior-dir',type=Path)
    parser.add_argument('--bank',type=Path,default=R/'potential_clean_bank_fp32_v1')
    args=parser.parse_args()
    output=args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'running.json',dict(pid=os.getpid(),created_utc=utc(),mode=args.mode))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    banks,bank_identity=load_banks(args.bank)
    config_id,checkpoint_id=artifact(DEFAULT_CONFIG),artifact(DEFAULT_CHECKPOINT)
    if config_id['sha256']!='3062762f2f0f12e0d4b64b074fc5b45628e5022937bbf7857cc6dc6e2720d342':
        raise ValueError('strict configuration mismatch')
    if checkpoint_id['sha256']!='723c56d7fa77ace9613909f7e38cb2386b898608218dc9b52649bb373d513c9a':
        raise ValueError('baseline identity mismatch')
    torch.manual_seed(SEEDS['model'])
    candidate=PairedBridgeField()
    control=copy.deepcopy(candidate)
    params=sum(p.numel() for p in candidate.parameters())
    request=dict(protocol=PROTOCOL,mode=args.mode,created_utc=utc(),sources=sources_to(output),
        bank=bank_identity,config=config_id,baseline_checkpoint=checkpoint_id,seeds=SEEDS,
        architecture=dict(parameters_each=params,full_channels=1024,blocks=2,
            input='current U,t,s,tau,class only',output='normalized vector field, full zero-initialized readout'),
        training=dict(batch_size=BATCH,baseline_microbatch_size=8,updates=UPDATES,optimizer='Adam',lr=LR,
            betas=[.9,.999],eps=1e-8,weight_decay=0,ema=False,lr_schedule=None,
            time_sampling='uniform100 query indices',aux_time='candidate independent Uniform[0,1]; control0',
            target='(W - actual_native_Euler(Z_t))/beta',beta='(t-s)/max(t,.05)',
            loss='.5*prediction**2-target*prediction, mean coordinates then mean samples',
            final_checkpoint_only=True,matched_initialization=True,shared_teacher_batches=True),
        finite_solver='two-stage explicit midpoint; control locks both tau inputs to0, keeps current midpoint state',
        baseline_batch_policy='all native forwards B8; pad validation remainder with repeated rows, score original rows only; charge actual padded calls',
        interpolation_arithmetic='FP32 (1-tau)*Y+tau*W in training and teacher validation',
        precision='native BF16 baseline heads/IG; FP32 state, auxiliary nets, training and Euler; TF32 off',
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        inherited_data_cost=dict(encoded_images=6000,reported_encoder_forward_calls=750,
            reported_bank_elapsed_seconds=113.86073904205114,
            boundary='existing bank creation summary; earlier source selection/history costs not fully timed; no total-cost success claim'),
        images_decoded=0,fid=False)
    previous=None
    if args.mode!='pilot':
        if args.prior_dir is None: raise ValueError('prior completed stage required')
        previous=check_previous(args.prior_dir,request,'pilot' if args.mode=='train' else 'train')
        request['prior_summary']=artifact(args.prior_dir/'summary.json')
    atomic_json(output/'request.json',request)
    device=torch.device('cuda:0'); torch.cuda.set_device(device)
    candidate=candidate.to(device);control=control.to(device)
    if args.mode in ('validate','rollout'):
        ck=previous['checkpoint']
        if sha256_file(Path(ck['path']))!=ck['sha256']: raise ValueError('final weights changed')
        weights=torch.load(ck['path'],map_location='cpu',weights_only=False)
        if weights['updates']!=UPDATES or weights['protocol']!=PROTOCOL: raise ValueError('wrong final checkpoint')
        prior_request=(args.prior_dir/'request.json').resolve()
        if (Path(weights['request']).resolve()!=prior_request or
                weights['request_sha256']!=sha256_file(prior_request)):
            raise ValueError('checkpoint internal training request binding mismatch')
        candidate.load_state_dict(weights['candidate']);control.load_state_dict(weights['control'])
        del weights
        candidate.eval().requires_grad_(False);control.eval().requires_grad_(False)
    config=load_config(DEFAULT_CONFIG)
    model=instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
    checkpoint=torch.load(DEFAULT_CHECKPOINT,map_location='cpu',mmap=True,weights_only=False)
    model.load_state_dict(checkpoint['ema'],strict=True)
    request['baseline_checkpoint_step']=int(checkpoint['step'])
    request['cuda_device']=torch.cuda.get_device_name(device)
    del checkpoint;gc.collect()
    atomic_json(output/'request.json',request)
    grid=shifted_time_grid(100,8,torch.device('cpu')).to(device)
    calls=Calls(dict(baseline=model,candidate=candidate,control=control))
    torch.cuda.reset_peak_memory_stats();sync()
    started=time.perf_counter()
    if args.mode=='pilot': result=numerical_pilot(model,candidate,control,banks,grid,output,calls,device)
    elif args.mode=='train': result=train(model,candidate,control,banks,grid,output,calls,device)
    elif args.mode=='validate': result=validate(model,candidate,control,banks,grid,output,calls,device)
    else: result=rollout(model,candidate,control,grid,output,calls,device)
    sync()
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise AssertionError('baseline optimizer boundary violated')
    result.update(protocol=PROTOCOL,mode=args.mode,complete=True,finished_utc=utc(),
        stage_seconds=time.perf_counter()-started,
        wall_seconds_from_first_time_import=time.perf_counter()-START_WALL,
        cpu_seconds_from_first_time_import=time.process_time()-START_CPU,
        peak_gpu_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
        request=artifact(output/'request.json'))
    atomic_json(output/'summary.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
