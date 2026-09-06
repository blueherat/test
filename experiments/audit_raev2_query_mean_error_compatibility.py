#!/usr/bin/env python3
"""Frozen teacher-only error witness for the already audited query-mean branch.

prepare/self-test are CPU-only. run requires explicit invocation on one GPU.
No sampler, gain selection, new noise, decoder images, or FID is implemented.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import audit_raev2_decoder_query_mean as helper

PROTOCOL = 'raev2_fixed_teacher_query_mean_error_compatibility_v1'
R = helper.RESTART
PROTOCOL_DOCUMENT = ROOT/'docs/RAEV2_QUERY_MEAN_ERROR_COMPATIBILITY_PROTOCOL_20260906_ZH.md'
SOURCE_REQUEST_SHA = 'd7ac7fb4a2bfdcda86244e474ac26e46d406d0073c3dd42e7b0229c606af2c17'
HELPER_SHA = '2f6313482f0752a5b8c5fb1e6bd0f24d852156f3007ab4e7c071b92472cc8296'
STEPS = [0,47,67,77,84,89,92,95,97,99]
LATENT_SHAPE = (1024,16,16)
DIM = math.prod(LATENT_SHAPE)
BRIDGE_EPS_MULTIPLIER = 8.0
NORM_ATOL, NORM_RTOL = 1e-10, 1e-12
IDENTITY_ATOL, IDENTITY_RTOL = 1e-12, 1e-11


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024*1024):
            h.update(block)
    return h.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}


def verify(record):
    path = Path(record['path'])
    if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
        raise ValueError(f'frozen input changed: {path}')


def read(path):
    return json.loads(Path(path).read_text())


def put(path, value):
    helper.atomic_json(Path(path), value)


def csv_rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def clean_join(bank_path, ids, labels, source_rows, *, expected_shape=(1000,*LATENT_SHAPE)):
    """Read small identity arrays and only selected latent rows from the NPZ."""
    with np.load(bank_path, allow_pickle=False) as archive:
        bank_ids, bank_labels, bank_rows = (archive[k] for k in ('ids','labels','rows'))
    if not (bank_ids.shape == bank_labels.shape == bank_rows.shape == (expected_shape[0],)):
        raise ValueError('unexpected clean bank identity arrays')
    if len(np.unique(bank_ids)) != len(bank_ids):
        raise ValueError('duplicate clean bank IDs')
    mapping = {int(value):index for index,value in enumerate(bank_ids)}
    indices = []
    for sample_id,label,source_row in zip(ids,labels,source_rows):
        if sample_id not in mapping:
            raise ValueError(f'clean ID missing: {sample_id}')
        index = mapping[sample_id]
        if int(bank_labels[index]) != label or int(bank_rows[index]) != source_row:
            raise ValueError(f'clean label/source_row mismatch for ID {sample_id}')
        indices.append(index)
    with zipfile.ZipFile(bank_path) as archive, archive.open('latents.npy') as stream:
        version = np.lib.format.read_magic(stream)
        shape,fortran,dtype = np.lib.format._read_array_header(stream,version)
        if tuple(shape) != tuple(expected_shape) or fortran or dtype != np.dtype('float16'):
            raise ValueError('expected original C-order FP16 clean latent bank')
        offset = stream.tell(); row_bytes = math.prod(shape[1:])*dtype.itemsize
        selected = np.empty((len(indices),*shape[1:]),dtype=np.float16)
        for slot,index in sorted(enumerate(indices),key=lambda item:item[1]):
            stream.seek(offset+index*row_bytes)
            data = stream.read(row_bytes)
            if len(data) != row_bytes:
                raise ValueError('truncated selected clean latent row')
            selected[slot] = np.frombuffer(data,dtype=dtype).reshape(shape[1:])
    if not np.isfinite(selected).all():
        raise ValueError('nonfinite paired clean latent')
    return selected,indices


def bridge_check(state, clean, noise, t):
    """Check the historical FP32 operation order; never replace the state."""
    if any(x.device.type!='cpu' or x.dtype!=torch.float32 for x in (state,clean,noise)):
        raise ValueError('bridge identity requires CPU FP32 tensors')
    if not (state.shape == clean.shape == noise.shape):
        raise ValueError('bridge shape mismatch')
    if not all(bool(torch.isfinite(x).all()) for x in (state,clean,noise)):
        raise ValueError('nonfinite bridge input')
    expected = (1-t)*clean+t*noise
    difference = state.double()-expected.double()
    # Conservative operation-rounding allowance tied to magnitudes, not an
    # empirical fitted threshold. It permits only FP32 arithmetic discrepancies.
    eps,tiny = torch.finfo(torch.float32).eps,torch.finfo(torch.float32).tiny
    scale = abs(1-t)*clean.double().abs()+abs(t)*noise.double().abs()+tiny
    bound = BRIDGE_EPS_MULTIPLIER*eps*scale
    violations = int((difference.abs()>bound).sum())
    result = {'bitwise_equal_to_cpu_original_order':torch.equal(state,expected),
              'max_abs_difference':float(difference.abs().max()),
              'rms_difference':float(difference.square().mean().sqrt()),
              'max_elementwise_allowed_bound':float(bound.max()),
              'max_abs_difference_over_bound':float((difference.abs()/bound).max()),
              'elements_outside_bound':violations}
    if violations:
        raise ValueError(f'teacher bridge differs from paired clean/noise: {result}')
    return result


def risk_records(full,base,weak,clean):
    """All main statistics first promote stored head/target values to FP64."""
    arrays = [np.asarray(a) for a in (full,base,weak,clean)]
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError('risk tensor shapes differ')
    f,b,w,x = (a.astype(np.float64).reshape(len(a),-1) for a in arrays)
    if not all(np.isfinite(a).all() for a in (f,b,w,x)):
        raise ValueError('nonfinite risk input')
    residual = x-f
    records = []
    for index in range(len(f)):
        r = residual[index]
        row = {'R_F':float(np.mean(r*r)),
               'R_W':float(np.mean((x[index]-w[index])**2)),
               'R_B':float(np.mean((x[index]-b[index])**2))}
        for suffix,other in [('W',w),('B',b)]:
            delta = f[index]-other[index]
            c = float(np.mean(r*delta)); q = float(np.mean(delta*delta))
            difference = row[f'R_{suffix}']-row['R_F']
            identity_residual = difference-2*c-q
            bound = IDENTITY_ATOL+IDENTITY_RTOL*(abs(difference)+abs(2*c)+q)
            if abs(identity_residual)>bound:
                raise AssertionError(f'FP64 risk identity failed: {suffix}/{index}')
            if q==0 and c!=0:
                raise AssertionError('zero direction energy with nonzero cross product')
            # The separately rounded FP32 subtraction is only the v2 parity
            # object. Retain its discrepancy from the main FP64 difference.
            delta32 = (f[index].astype(np.float32)-other[index].astype(np.float32)).astype(np.float64)
            roundoff = delta32-delta
            row.update({f'C_{suffix}':c,f'Q_{suffix}':q,f'Q_{suffix}_zero':q==0,
                        f'R_{suffix}_minus_R_F':difference,
                        f'identity_{suffix}_residual':identity_residual,
                        f'identity_{suffix}_allowed_bound':bound,
                        f'oracle_C_over_Q_{suffix}_descriptive':c/q if q>0 else None,
                        f'delta_{suffix}_fp32_minus_fp64_max_abs':float(np.max(np.abs(roundoff))),
                        f'delta_{suffix}_fp32_minus_fp64_rms':float(np.sqrt(np.mean(roundoff*roundoff)))})
        records.append(row)
    return records


def summarize(rows):
    primitive = ('R_F','R_W','R_B','C_W','Q_W','C_B','Q_B','R_W_minus_R_F','R_B_minus_R_F')
    result = {'rows':len(rows)}
    for name in primitive:
        values = [row[name] for row in rows]
        result[f'{name}_mean'] = math.fsum(values)/len(values)
        result[f'{name}_min'] = min(values); result[f'{name}_max'] = max(values)
    for suffix in ('W','B'):
        c,q = result[f'C_{suffix}_mean'],result[f'Q_{suffix}_mean']
        result[f'oracle_mean_C_over_mean_Q_{suffix}_descriptive'] = c/q if q>0 else None
        result[f'zero_Q_{suffix}_rows'] = sum(row[f'Q_{suffix}_zero'] for row in rows)
        result[f'positive_C_{suffix}_rows'] = sum(row[f'C_{suffix}']>0 for row in rows)
        result[f'negative_C_{suffix}_rows'] = sum(row[f'C_{suffix}']<0 for row in rows)
        result[f'identity_{suffix}_max_abs_residual'] = max(abs(row[f'identity_{suffix}_residual']) for row in rows)
    return result


def expected_events():
    return {key:value for key,value in helper.ForwardCallCounter.expected_events().items() if key!='native_reference'}


def prepare(args):
    started,cpu = time.perf_counter(),time.process_time()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('refusing to overwrite existing compatibility audit')
    source_request_path = args.source_dir/'request.json'
    if sha(source_request_path)!=SOURCE_REQUEST_SHA or sha(ROOT/'experiments/audit_raev2_decoder_query_mean.py')!=HELPER_SHA:
        raise ValueError('requires the frozen completed query_mean_v2 helper/request')
    source = read(source_request_path); source_summary = read(args.source_dir/'summary.json')
    if not source_summary['complete'] or not source_summary['native_Full_Base_bitwise_parity_all']:
        raise ValueError('source query-mean audit has not passed current native parity')
    verify(source_summary['request'])
    if source['sample_ids']!=list(range(8)) or source['labels']!=list(range(8)) or [s['step_index'] for s in source['snapshots']]!=STEPS:
        raise ValueError('source cohort changed')
    old_inputs = {name:source[name] for name in ('config','checkpoint','state_source_request','state_source_summary')}
    for record in old_inputs.values(): verify(record)
    source_outputs = {Path(v['path']).name:v for v in source_summary['outputs']}
    for name in ('F_minus_W.npy','F_minus_B.npy','per_image_directions.csv'): verify(source_outputs[name])
    bank_identity = artifact(args.clean_bank)
    normal_request = read(source['state_source_request']['path'])
    original_bank_hashes = {str(Path(k).resolve()):v for k,v in normal_request['bank_metadata']['source_sha256'].items()}
    if original_bank_hashes.get(str(args.clean_bank.resolve()))!=bank_identity['sha256']:
        raise ValueError('paired clean bank is not the exact bank used for the historical teacher')
    selected,bank_rows = clean_join(args.clean_bank,source['sample_ids'],source['labels'],source['source_rows'])
    clean32 = torch.from_numpy(selected.astype(np.float32))
    snapshots = []
    bridge_rows = []
    noise = None
    for old in source['snapshots']:
        verify(old)
        cache = torch.load(old['path'],map_location='cpu',weights_only=True,mmap=True)
        if cache['sample_ids'].tolist()!=source['sample_ids'] or cache['labels'].tolist()!=source['labels'] or cache['step_index']!=old['step_index'] or cache['t']!=old['t']:
            raise ValueError('snapshot identity mismatch')
        state = cache['teacher']['state']
        if state.shape!=(8,*LATENT_SHAPE): raise ValueError('unexpected teacher state shape')
        if noise is None:
            if old['step_index']!=0 or old['t']!=1.: raise ValueError('first teacher snapshot must be t=1')
            noise = state.clone()
        bridge_rows.append({'step_index':old['step_index'],'t':old['t'],**bridge_check(state,clean32,noise,old['t'])})
        snapshots.append(dict(old)); del cache
    reference_rows = csv_rows(source_outputs['per_image_directions.csv']['path'])
    selected_row_numbers = [16*k+i for k in range(10) for i in range(8)]
    for k,step in enumerate(STEPS):
        for i in range(8):
            row = reference_rows[16*k+i]
            if (int(row['step_index']),row['domain'],int(row['sample_id']),int(row['label']),int(row['source_row']))!=(step,'teacher',i,i,source['source_rows'][i]):
                raise ValueError('v2 teacher row order/source identity mismatch')
    out.mkdir(parents=True,exist_ok=True)
    np.save(out/'paired_X_fp16.npy',selected,allow_pickle=False)
    put(out/'paired_clean_identity.json',{'sample_ids':source['sample_ids'],'labels':source['labels'],
        'source_rows':source['source_rows'],'bank_array_rows':bank_rows,'dtype':'float16','shape':list(selected.shape)})
    put(out/'bridge_identity_prepare.json',{'complete':True,'rows':bridge_rows,'epsilon_source':'step000 teacher state; no new noise',
        'bound':'8*float32_eps*(abs((1-t)X)+abs(t*epsilon)+float32_tiny), elementwise; compare original CPU FP32 multiply/add order',
        'states_replaced':False})
    sources = {}
    for relative in dict.fromkeys(('experiments/audit_raev2_query_mean_error_compatibility.py',*helper.SOURCE_FILES)):
        target = out/'sources'/relative; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target); sources[relative]=artifact(target)
    shutil.copy2(args.protocol_plan,out/'frozen_protocol.md')
    totals = helper.ForwardCallCounter.totals(expected_events())
    request = {'protocol':PROTOCOL,'created_utc':stamp(),'protocol_document':artifact(args.protocol_plan),
        'source_request':artifact(source_request_path),'source_summary':artifact(args.source_dir/'summary.json'),
        'source_outputs':{k:source_outputs[k] for k in ('F_minus_W.npy','F_minus_B.npy','per_image_directions.csv')},
        **old_inputs,'sources':sources,'snapshots':snapshots,'clean_bank':bank_identity,
        'paired_clean':artifact(out/'paired_X_fp16.npy'),'paired_clean_identity':artifact(out/'paired_clean_identity.json'),
        'bridge_identity_prepare':artifact(out/'bridge_identity_prepare.json'),
        'sample_ids':source['sample_ids'],'labels':source['labels'],'source_rows':source['source_rows'],
        'bank_array_rows':bank_rows,'source_teacher_vector_rows':selected_row_numbers,
        'domain':'teacher','batches':10,'rows':80,'unique_ids':8,'latent_shape':list(LATENT_SHAPE),'coordinate_normalizer':DIM,
        'output_heads':{'F.npy':[80,*LATENT_SHAPE],'B.npy':[80,*LATENT_SHAPE],'W.npy':[80,*LATENT_SHAPE]},
        'output_head_dtype':'float32','output_order':'snapshot ascending original step, IDs0..7; teacher only',
        'precision':'FP32 weights/states; native BF16 CUDA autocast; TF32 off; helper capture=False; stored FP32 heads; main risk/C/Q promote all heads and paired FP16 X to FP64 before subtraction',
        'parity_arithmetic':'CPU FP32 F-W/F-B exactly as v2; both arrays must be bitwise equal to all 80 corresponding teacher rows; individual head norms checked against v2',
        'main_delta_arithmetic':'FP64 subtraction after promoting saved F/B/W; separately record FP32-difference roundoff, never alter direction',
        'native_parity_scope':'No new independent model.forward; v2 native Full/Base parity is prior evidence. Current audit checks both difference arrays and individual head norms, not individual old head tensor equality.',
        'norm_tolerance':{'absolute':NORM_ATOL,'relative':NORM_RTOL},
        'bridge_bound_float32_eps_multiplier':BRIDGE_EPS_MULTIPLIER,
        'risk_identity_tolerance':{'absolute':IDENTITY_ATOL,'relative_to_difference_plus_2C_plus_Q':IDENTITY_RTOL},
        'zero_Q_rule':'If FP64 Q=0 retain row, require C=0, oracle C/Q is null. No filtering.',
        'oracle_scope':'C/Q is descriptive only, unconstrained; no deployment coefficient, fitted curve, window or sampling',
        'aggregation':'8 images per original time: coordinate-mean R/C/Q mean/min/max; any all80 summary is equal-time descriptive, not trajectory/training weighting',
        'expected_module_events_per_batch':expected_events(),'expected_forward_counts':{k:10*v for k,v in totals.items()},
        'data_boundary':'Already inspected historical states. Only8 related source IDs/classes, one original epsilon each; no population sign proof, covariance-compatibility proof or FID inference.',
        'no_new_noise':True,'no_rollout_witness':True,'no_sampling':True,'no_decoder_images':True,'no_fid':True,
        'torch_version':str(torch.__version__)}
    put(out/'request.json',request)
    put(out/'prepare_cost.json',{'wall_seconds':time.perf_counter()-started,'cpu_seconds':time.process_time()-cpu,
        'request_sha256':sha(out/'request.json'),'gpu_model_calls':0,'gpu_calls':0,'imports_excluded':True})
    put(out/'progress.json',{'complete':False,'status':'prepared_no_gpu'})
    print(json.dumps({'prepared':True,'request':artifact(out/'request.json')}),flush=True)


def run(args):
    out = args.output_dir
    if (out/'summary.json').exists() or (out/'run_started.json').exists() or (out/'failure.json').exists():
        raise FileExistsError('refusing to restart or overwrite an attempted run')
    started,cpu = time.perf_counter(),time.process_time(); run_utc=stamp(); counter=None; costs={}
    try:
        request=read(out/'request.json')
        if request['protocol']!=PROTOCOL: raise ValueError('wrong protocol')
        with helper.timed_stage(costs,'input_verification_cpu','cpu'):
            for name,rec in request['sources'].items():
                verify(rec)
                if sha(ROOT/name)!=rec['sha256']: raise ValueError(f'current source changed: {name}')
            for key in ('protocol_document','source_request','source_summary','config','checkpoint','state_source_request',
                        'state_source_summary','clean_bank','paired_clean','paired_clean_identity','bridge_identity_prepare'):
                verify(request[key])
            for rec in [*request['snapshots'],*request['source_outputs'].values()]: verify(rec)
        put(out/'run_started.json',{'run_started_utc':run_utc,'request':artifact(out/'request.json')})
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise RuntimeError('run requires exactly one visible GPU')
        device=torch.device('cuda:0');torch.cuda.set_device(device);torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.cuda.reset_peak_memory_stats(device)
        from utils.model_utils import instantiate_from_config
        with helper.timed_stage(costs,'model_load',device):
            config=helper.load_config(Path(request['config']['path']))
            model=instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
            checkpoint=torch.load(request['checkpoint']['path'],map_location='cpu',mmap=True,weights_only=False)
            model.load_state_dict(checkpoint['ema'],strict=True);checkpoint_step=int(checkpoint['step']);del checkpoint
            if checkpoint_step!=100080: raise ValueError('unexpected EMA step')
        counter=helper.ForwardCallCounter(model)
        with helper.timed_stage(costs,'paired_and_reference_load_cpu','cpu'):
            clean16=np.load(request['paired_clean']['path'],allow_pickle=False)
            if clean16.shape!=(8,*LATENT_SHAPE) or clean16.dtype!=np.float16: raise ValueError('paired target changed')
            clean32=torch.from_numpy(clean16.astype(np.float32))
            reference_w=np.load(request['source_outputs']['F_minus_W.npy']['path'],mmap_mode='r',allow_pickle=False)
            reference_b=np.load(request['source_outputs']['F_minus_B.npy']['path'],mmap_mode='r',allow_pickle=False)
            if any(a.shape!=(160,*LATENT_SHAPE) or a.dtype!=np.float32 for a in (reference_w,reference_b)):
                raise ValueError('v2 reference vector shape/dtype changed')
            norm_rows=csv_rows(request['source_outputs']['per_image_directions.csv']['path'])
            noise=torch.load(request['snapshots'][0]['path'],map_location='cpu',weights_only=True,mmap=True)['teacher']['state'].clone()
            maps={name:np.lib.format.open_memmap(out/name,mode='w+',dtype=np.float32,shape=(80,*LATENT_SHAPE)) for name in ('F.npy','B.npy','W.npy')}
        rows=[];parity_rows=[];batch_costs=[];all_events=Counter()
        for k,snapshot in enumerate(request['snapshots']):
            batch={};identity={'step_index':snapshot['step_index'],'t':snapshot['t'],'domain':'teacher'}
            with helper.timed_stage(batch,'snapshot_and_bridge_check_cpu','cpu'):
                cache=torch.load(snapshot['path'],map_location='cpu',weights_only=True,mmap=True)
                if cache['sample_ids'].tolist()!=request['sample_ids'] or cache['labels'].tolist()!=request['labels'] or cache['step_index']!=snapshot['step_index'] or cache['t']!=snapshot['t']:
                    raise ValueError('teacher snapshot identity changed')
                bridge=bridge_check(cache['teacher']['state'],clean32,noise,snapshot['t'])
            with helper.timed_stage(batch,'state_transfer',device):
                state=cache['teacher']['state'].to(device)
                labels=cache['labels'].to(device);times=torch.full((8,),snapshot['t'],dtype=torch.float32,device=device)
            counter.reset()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                heads,_=helper.shared_full_base_weak(model,state,times,labels,capture=False,costs=batch,counter=counter)
            with helper.timed_stage(batch,'head_transfer_and_counts',device):
                if any(v.dtype!=torch.bfloat16 for v in heads.values()): raise ValueError('head dtype is not native BF16')
                head_cpu={key:value.float().cpu() for key,value in heads.items()}
                if counter.events!=request['expected_module_events_per_batch']: raise AssertionError('actual module events differ')
                observed=helper.ForwardCallCounter.totals(counter.events);all_events.update(observed)
                events={phase:dict(values) for phase,values in counter.events.items()}
            del heads
            with helper.timed_stage(batch,'v2_difference_and_norm_parity_cpu','cpu'):
                dw=head_cpu['full']-head_cpu['weak'];db=head_cpu['full']-head_cpu['base']
                source_slice=slice(16*k,16*k+8)
                if not np.array_equal(dw.numpy(),reference_w[source_slice]) or not np.array_equal(db.numpy(),reference_b[source_slice]):
                    raise AssertionError('FP32 difference parity with v2 failed; stop without changing the branch')
                norm_delta={}
                for head in ('full','base','weak'):
                    norms=head_cpu[head].double().flatten(1).square().sum(1).sqrt().tolist()
                    differences=[]
                    for i,value in enumerate(norms):
                        target=float(norm_rows[16*k+i][head+'_norm'])
                        if not math.isclose(value,target,rel_tol=NORM_RTOL,abs_tol=NORM_ATOL):
                            raise AssertionError(f'{head} norm differs from v2')
                        differences.append(abs(value-target))
                    norm_delta[head+'_norm_max_abs_difference']=max(differences)
                parity_rows.append({**identity,'F_minus_W_fp32_bitwise':True,'F_minus_B_fp32_bitwise':True,
                                    'head_dtype':'torch.bfloat16',**norm_delta})
            with helper.timed_stage(batch,'risk_statistics_cpu','cpu'):
                per_image=risk_records(head_cpu['full'].numpy(),head_cpu['base'].numpy(),head_cpu['weak'].numpy(),clean16)
                for i,row in enumerate(per_image):
                    rows.append({**identity,'output_row':8*k+i,'sample_id':i,'label':i,'source_row':request['source_rows'][i],
                                 'paired_X_row':i,'v2_vector_row':16*k+i,**row})
            with helper.timed_stage(batch,'output_write_cpu','cpu'):
                for name,head in [('F.npy','full'),('B.npy','base'),('W.npy','weak')]:
                    maps[name][8*k:8*k+8]=head_cpu[head].numpy();maps[name].flush()
                helper.write_csv(out/'per_image_risk.csv',rows);helper.write_csv(out/'v2_parity.csv',parity_rows)
                put(out/'progress.json',{'complete':False,'batches':k+1,'rows':len(rows),'last':identity})
            batch_costs.append({**identity,'bridge_identity':bridge,'stages':batch,
                                'observed_module_events':events,'observed_forward_counts':observed})
            del head_cpu,dw,db,state,cache
            print(json.dumps({'batches':k+1,'rows':len(rows),'last':identity}),flush=True)
        if len(rows)!=80 or len(parity_rows)!=10 or dict(all_events)!=request['expected_forward_counts']:
            raise AssertionError('fixed cohort or successful forward count incomplete')
        for mmap in maps.values(): mmap.flush()
        maps.clear()
        per_time=[{'step_index':step,'t':request['snapshots'][k]['t'],**summarize(rows[8*k:8*k+8])} for k,step in enumerate(STEPS)]
        helper.write_csv(out/'per_time_risk.csv',per_time)
        put(out/'batch_costs.json',batch_costs)
        totals={name:math.fsum(b['stages'][name]['wall_seconds'] for b in batch_costs) for name in batch_costs[0]['stages']}
        with helper.timed_stage(costs,'final_output_hash_cpu','cpu'):
            outputs=[artifact(out/name) for name in ('F.npy','B.npy','W.npy','paired_X_fp16.npy','paired_clean_identity.json',
                'per_image_risk.csv','per_time_risk.csv','v2_parity.csv','batch_costs.json','bridge_identity_prepare.json')]
        summary={'complete':True,'protocol':PROTOCOL,'run_started_utc':run_utc,'request':artifact(out/'request.json'),
            'rows':80,'unique_ids':8,'teacher_times':10,'t1_retained_for_clean_risk_only':True,
            'all_difference_parity_bitwise':True,'all_head_norm_parity_within_fixed_tolerance':True,
            'checkpoint_step':checkpoint_step,'observed_forward_counts':dict(all_events),
            'expected_forward_counts':request['expected_forward_counts'],'all_module_event_checks_passed':True,
            'per_time':per_time,'all80_equal_time_descriptive':summarize(rows),'outputs':outputs,
            'setup_and_final_stages':costs,'batch_stage_wall_seconds':totals,
            'wall_seconds_before_summary':time.perf_counter()-started,'cpu_seconds_before_summary':time.process_time()-cpu,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),
            'cuda_device':torch.cuda.get_device_name(device),'no_sampling':True,'no_new_noise':True,'no_rollout_witness':True,
            'no_decoder_images':True,'no_fid':True,'oracle_coefficients_deployed':False,
            'scope':'Frozen empirical teacher clean-risk witness on 8 related IDs. No population sign proof, error-collinearity conclusion, sampler, gain/window choice or FID inference.',
            'timing_boundary':'Includes preflight and all recomputation/weak replay/CPU statistics/IO/hash stages; imports and final summary write/exit excluded. No cost-matched sampler conclusion.'}
        put(out/'summary.json',summary);put(out/'progress.json',{'complete':True,'summary':artifact(out/'summary.json')})
        print(json.dumps({'complete':True,'summary':artifact(out/'summary.json'),'rows':80}),flush=True)
    except BaseException as error:
        put(out/'failure.json',{'complete':False,'run_started_utc':run_utc,'error':f'{type(error).__name__}: {error}',
            'wall_seconds':time.perf_counter()-started,'cpu_seconds':time.process_time()-cpu,
            'cuda_initialized':torch.cuda.is_initialized(),'current_batch_observed_events':counter.events if counter else {}})
        raise
    finally:
        if counter is not None: counter.close()


def self_test():
    """Small deterministic CPU identities; never accesses the research banks."""
    f=np.array([[[[1.,2.,3.,4.]]],[[[1.,0.,-1.,2.]]]],dtype=np.float32)
    x=f+np.float32(.25);w=f-np.float32(.5);b=f.copy()
    rows=risk_records(f,b,w,x)
    assert all(r['C_W']>0 and r['Q_B_zero'] and r['oracle_C_over_Q_B_descriptive'] is None for r in rows)
    assert all(abs(r['identity_W_residual'])<1e-14 for r in rows)
    assert risk_records(f,b,f+np.float32(.5),x)[0]['C_W']<0
    stats=summarize(rows);assert stats['oracle_mean_C_over_mean_Q_W_descriptive']==.5
    clean=torch.from_numpy(x);noise=torch.from_numpy(f);t=.5
    bridge_check((1-t)*clean+t*noise,clean,noise,t)
    try: bridge_check((1-t)*clean+t*noise+1,clean,noise,t)
    except ValueError: pass
    else: raise AssertionError('bad bridge was accepted')
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'toy.npz';latents=np.arange(12,dtype=np.float16).reshape(3,1,2,2)
        np.savez(path,ids=np.array([7,2,9]),labels=np.array([4,1,8]),rows=np.array([30,20,10]),latents=latents)
        selected,indices=clean_join(path,[9,7],[8,4],[10,30],expected_shape=(3,1,2,2))
        assert indices==[2,0] and np.array_equal(selected,latents[[2,0]])
        try: clean_join(path,[9],[8],[999],expected_shape=(3,1,2,2))
        except ValueError: pass
        else: raise AssertionError('wrong source row accepted')
    totals=helper.ForwardCallCounter.totals(expected_events())
    assert totals['shared_encoder_passes']==1 and totals['native_model_forward_calls']==0
    assert totals['encoder_block_calls']==28 and totals['weak_decoder_block_calls']==2
    if torch.cuda.is_initialized(): raise AssertionError('CPU self-test initialized CUDA')
    print(json.dumps({'cpu_self_test_passed':True,'gpu_calls':0,'research_payload_read':False}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prepare','run','self-test'),required=True)
    parser.add_argument('--output-dir',type=Path,default=R/'query_mean_error_compatibility_v1')
    parser.add_argument('--source-dir',type=Path,default=R/'decoder_query_mean_v2')
    parser.add_argument('--clean-bank',type=Path,default=R/'heldout_clean_c_current_fp32/clean_rank00.npz')
    parser.add_argument('--protocol-plan',type=Path,default=PROTOCOL_DOCUMENT)
    args=parser.parse_args()
    for name in ('output_dir','source_dir','clean_bank','protocol_plan'):
        setattr(args,name,getattr(args,name).expanduser().resolve())
    if args.mode=='self-test': self_test()
    elif args.mode=='prepare': prepare(args)
    else: run(args)


if __name__=='__main__': main()
