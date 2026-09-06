"""Independent completed-output review; never imports the GPU runner.

Execute only after root confirms completion. F-W/F-B are independently
reduced while streaming and hashing each NPY once. Attention probabilities
and individual F/B/W vectors are not stored, so their source statistics are
checked for identity/consistency only, never claimed independently rebuilt.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

RESTART = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906')
RUNNER_SHA = '2f6313482f0752a5b8c5fb1e6bd0f24d852156f3007ab4e7c071b92472cc8296'
SHAPE = (160, 1024, 16, 16)
D = math.prod(SHAPE[1:])
ATOL, RTOL = 1e-10, 2e-11

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1024 * 1024): h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def csvrows(path):
    with Path(path).open() as f: return list(csv.DictReader(f))

def flag(value):
    if isinstance(value, bool): return value
    assert value in ('True', 'False'), value
    return value == 'True'

def num(row, key):
    value = row[key]
    if value in ('', None): return None
    value = float(value)
    assert math.isfinite(value), (key, value)
    return value

def match(actual, expected, key):
    if actual is None or expected is None:
        assert actual is expected, (key, actual, expected)
        return 0.0
    assert math.isfinite(actual) and math.isfinite(expected)
    assert math.isclose(actual, expected, abs_tol=ATOL, rel_tol=RTOL), (key, actual, expected)
    return abs(actual - expected)

def aggregate(rows, keys):
    result = {'rows': len(rows)}
    for key in keys:
        vals = [r[key] for r in rows if r[key] is not None]
        result[key] = {'count_defined': len(vals), 'null_count': len(rows)-len(vals),
                       'nonzero_count':sum(v!=0 for v in vals),'positive_count':sum(v>0 for v in vals),'negative_count':sum(v<0 for v in vals),
                       'mean': math.fsum(vals)/len(vals) if vals else None,
                       'min': min(vals) if vals else None, 'max': max(vals) if vals else None}
    return result

def batch_identity(row):
    return int(row['step_index']), row['domain']

def expected_events():
    enc = {'s_embedder': 1, **{f'encoder_block_{i:02d}': 1 for i in range(28)}}
    dec = {f'decoder_{i:02d}_{part}': 1 for i in (28,29) for part in ('q','mlp')}
    return {'shared_encoder': enc, 'full_decoder_and_readout': {**dec,'full_readout':1},
            'base_readout': {'base_readout':1}, 'weak_decoder_and_readout': {**dec,'full_readout':1},
            'native_reference': {'model':1,**enc,**dec,'full_readout':1,'base_readout':1}}

def event_counts(events):
    out = Counter()
    for phase, group in events.items():
        for event, count in group.items():
            if event.startswith('encoder_block_'): out['encoder_block_calls'] += count
            if event == 'full_readout': out['full_final_readout_calls_including_weak_and_reference'] += count
            if event == 'base_readout': out['base_readout_calls_including_reference'] += count
            if phase == 'shared_encoder' and event == 's_embedder': out['shared_encoder_passes'] += count
            if phase == 'native_reference' and event == 'model': out['native_model_forward_calls'] += count
            if event.startswith('decoder_') and event.endswith('_q'):
                key = {'full_decoder_and_readout':'explicit_full_decoder_block_calls',
                       'weak_decoder_and_readout':'weak_decoder_block_calls',
                       'native_reference':'native_reference_decoder_block_calls'}[phase]
                out[key] += count
    return dict(out)

def check_record(record):
    p = Path(record['path'])
    assert p.stat().st_size == record['bytes'] and sha(p) == record['sha256'], p

def open_vector(path):
    f = path.open('rb')
    version = np.lib.format.read_magic(f)
    shape, fortran, dtype = np.lib.format._read_array_header(f, version)
    assert tuple(shape) == SHAPE and not fortran and dtype == np.dtype('float32')
    header_end = f.tell(); f.seek(0)
    h = hashlib.sha256(f.read(header_end))
    return f, h

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir', type=Path, default=RESTART/'decoder_query_mean_v2')
    ap.add_argument('--request-sha256', required=True)
    args = ap.parse_args()
    started, cpu = time.perf_counter(), time.process_time()
    source, out = args.input_dir.resolve(), Path(__file__).resolve().parent
    assert source != out and not (out/'review.json').exists()
    request = read(source/'request.json'); summary = read(source/'summary.json')
    assert summary['complete'] and sha(source/'request.json') == args.request_sha256
    assert not (source/'failure.json').exists()
    initial_hashes = {name:sha(source/name) for name in ['request.json','summary.json','batch_costs.json','run_started.json','prepare_cost.json','progress.json']}
    check_record(summary['request']); check_record(summary['batch_costs'])
    progress, run_started = read(source/'progress.json'), read(source/'run_started.json')
    assert progress['complete']; check_record(progress['summary']); check_record(run_started['request'])
    assert run_started['run_started_utc'] == summary['run_started_utc']
    assert request['rows'] == summary['rows'] == 160 and request['batches'] == 20 and summary['unique_ids'] == 8
    assert request['sample_ids'] == request['labels'] == list(range(8))
    assert request['domains'] == ['teacher','rollout'] and tuple(request['output_shape']) == SHAPE
    assert request['output_dtype'] == 'float32'
    assert request['sources']['experiments/audit_raev2_decoder_query_mean.py']['sha256'] == RUNNER_SHA
    for rec in request['sources'].values(): check_record(rec)
    for key in ['root_protocol_plan','protocol_document','config','state_source_request','state_source_summary']:
        check_record(request[key])
    historical = read(request['state_source_summary']['path'])
    old_by_step = {s['step_index']:s for s in historical['snapshots']}
    snapshots = request['snapshots']
    assert len(snapshots) == 10 and [s['step_index'] for s in snapshots] == sorted(old_by_step)
    for s in snapshots:
        assert s['sha256'] == old_by_step[s['step_index']]['sha256'] and s['t'] == old_by_step[s['step_index']]['t']
    steps = [s['step_index'] for s in snapshots]; times = {s['step_index']:s['t'] for s in snapshots}
    batches = [(k, d) for k in steps for d in request['domains']]
    expected_images = [(k,d,i) for k,d in batches for i in range(8)]
    expected_attention = [(k,d,b,l,i,h) for k,d in batches for b in ['full','weak'] for l in range(2) for i in range(8) for h in range(16)]
    records = {Path(r['path']).name:r for r in summary['outputs']}
    assert set(records) == {'F_minus_W.npy','F_minus_B.npy','per_image_directions.csv','per_head_attention.csv','native_parity.csv'}
    for name, rec in records.items():
        assert Path(rec['path']).resolve() == source/name
        if not name.endswith('.npy'): check_record(rec)
    direction_rows = csvrows(source/'per_image_directions.csv')
    attention_rows = csvrows(source/'per_head_attention.csv')
    parity_rows = csvrows(source/'native_parity.csv')
    assert len(direction_rows) == 160 and len(attention_rows) == summary['attention_head_rows'] == 10240 and len(parity_rows) == 20
    assert [(int(r['step_index']),r['domain'],int(r['sample_id'])) for r in direction_rows] == expected_images
    assert [batch_identity(r) for r in parity_rows] == batches
    assert [(int(r['step_index']),r['domain'],r['branch'],int(r['decoder_block']),int(r['sample_id']),int(r['head_index'])) for r in attention_rows] == expected_attention
    for rows in [direction_rows,attention_rows,parity_rows]:
        for row in rows:
            assert float(row['t']) == times[int(row['step_index'])]
            if 'sample_id' in row: assert int(row['label']) == int(row['sample_id'])
    for row in direction_rows:
        assert int(row['source_row']) == request['source_rows'][int(row['sample_id'])]
        for name in ['full_norm','base_norm','weak_norm']: assert num(row,name) >= 0
    for row in parity_rows:
        for h in ['full','base']:
            assert row[h+'_native_dtype'] == row[h+'_helper_dtype'] == 'torch.bfloat16'
            assert flag(row[h+'_native_bitwise']) and num(row,h+'_native_max_abs_difference') == 0
        assert flag(row['first_decoder_keys_bitwise_same_full_weak']) and flag(row['first_decoder_values_bitwise_same_full_weak'])
    assert summary['native_Full_Base_bitwise_parity_all'] and summary['t1_teacher_rollout_state_duplicate']

    # Each new vector payload is read exactly once for both SHA and all reductions.
    streams = [open_vector(source/name) for name in ['F_minus_W.npy','F_minus_B.npy']]
    computed, max_error = [], defaultdict(float)
    first = None; t1_equal = None
    try:
        for start in range(0,160,8):
            arrays = []
            for stream, digest in streams:
                block = stream.read(8*D*4); assert len(block) == 8*D*4
                digest.update(block)
                x = np.frombuffer(block,dtype=np.float32).reshape(8,D).astype(np.float64)
                assert np.isfinite(x).all(); arrays.append(x)
            u, v = arrays
            if start == 0: first = (u.copy(),v.copy())
            if start == 8:
                t1_equal = np.array_equal(u,first[0]) and np.array_equal(v,first[1]); assert t1_equal
                first = None
            uw = np.sum(u*u,axis=1,dtype=np.float64); vb = np.sum(v*v,axis=1,dtype=np.float64)
            dot = np.sum(u*v,axis=1,dtype=np.float64)
            for local in range(8):
                i = start+local; row = direction_rows[i]
                nw, nb = math.sqrt(float(uw[local])), math.sqrt(float(vb[local]))
                coefficient = float(dot[local]/vb[local]) if nb>0 else None
                parallel = coefficient*v[local] if coefficient is not None else None
                pe = float(np.sum(parallel*parallel,dtype=np.float64)) if parallel is not None else None
                oe = float(np.sum((u[local]-parallel)**2,dtype=np.float64)) if parallel is not None else None
                calc = {'F_minus_W_norm':nw,'F_minus_B_norm':nb,'F_minus_W_rms':nw/math.sqrt(D),'F_minus_B_rms':nb/math.sqrt(D),
                        'direction_cosine':float(dot[local])/(nw*nb) if nw>0 and nb>0 else None,
                        'weak_gap_over_internal_gap':nw/nb if nb>0 else None,
                        'descriptive_projection_coefficient_on_internal_gap':coefficient,
                        'parallel_energy_to_internal_gap':pe,'orthogonal_energy_to_internal_gap':oe,
                        'parallel_energy_over_internal_gap_energy':pe/float(vb[local]) if nb>0 else None,
                        'orthogonal_energy_over_internal_gap_energy':oe/float(vb[local]) if nb>0 else None}
                for key,value in calc.items(): max_error[key] = max(max_error[key],match(value,num(row,key),key))
                assert flag(row['weak_gap_zero']) == (nw==0) and flag(row['internal_gap_zero']) == (nb==0)
                if nb>0:
                    match(pe+oe,float(uw[local]),'Pythagoras')
                    match(pe,float(dot[local]**2/vb[local]),'closed_form_parallel_energy')
                computed.append({'step_index':int(row['step_index']),'domain':row['domain'],'sample_id':int(row['sample_id']),**calc})
        for name,(stream,digest) in zip(['F_minus_W.npy','F_minus_B.npy'],streams):
            assert stream.read(1) == b''
            assert digest.hexdigest() == records[name]['sha256'] and (source/name).stat().st_size == records[name]['bytes']
    finally:
        for stream,_ in streams: stream.close()

    attention_metrics = ['query_row_max_abs_difference','original_query_row_max_abs_difference',
        'actual_sdpa_output_row_max_abs_difference','explicit_attention_probability_row_max_abs_difference',
        'explicit_key_nonuniform_KL_p_to_uniform_mean_query',
        'implemented_row_to_effective_input_reverseKL_barycenter_L1',
        'implemented_row_KL_to_effective_input_reverseKL_barycenter']
    attn = []; weak_flags = Counter()
    for row in attention_rows:
        rec = {k:num(row,k) for k in attention_metrics}
        assert rec['query_row_max_abs_difference'] >= 0 and rec['original_query_row_max_abs_difference'] >= 0
        assert rec['actual_sdpa_output_row_max_abs_difference'] >= 0
        assert 0 <= rec['explicit_attention_probability_row_max_abs_difference'] <= 1
        assert -1e-12 <= rec['explicit_key_nonuniform_KL_p_to_uniform_mean_query'] <= math.log(256)+1e-12
        assert flag(row['query_rows_bitwise_equal']) == (rec['query_row_max_abs_difference']==0)
        assert flag(row['actual_sdpa_output_rows_bitwise_equal']) == (rec['actual_sdpa_output_row_max_abs_difference']==0)
        if row['branch']=='weak':
            assert 0 <= rec['implemented_row_to_effective_input_reverseKL_barycenter_L1'] <= 2+1e-12
            assert rec['implemented_row_KL_to_effective_input_reverseKL_barycenter'] >= -1e-12
            for key in ['query_rows_bitwise_equal','effective_bf16_query_rows_bitwise_equal','actual_sdpa_output_rows_bitwise_equal']:
                weak_flags[key+'_true'] += int(flag(row[key]))
            weak_flags['explicit_probability_rows_exact'] += int(rec['explicit_attention_probability_row_max_abs_difference']==0)
        else:
            assert rec['implemented_row_to_effective_input_reverseKL_barycenter_L1'] is None
            assert rec['implemented_row_KL_to_effective_input_reverseKL_barycenter'] is None
        attn.append({'step_index':int(row['step_index']),'domain':row['domain'],'branch':row['branch'],'decoder_block':int(row['decoder_block']),**rec})

    batch_costs = read(source/'batch_costs.json')
    assert len(batch_costs)==20 and [batch_identity(x) for x in batch_costs]==batches
    expected = expected_events(); observed_counts = Counter(); stages = defaultdict(list)
    expected_stage_names = {'state_transfer','shared_encoder','full_preparation','full_decoder_and_readout','base_readout',
        'weak_decoder_and_readout','native_reference_forward','forward_count_verification','native_parity_and_trace_transfer',
        'direction_and_attention_statistics_cpu','output_write'}
    for batch in batch_costs:
        counts = batch['forward_counts']; events = counts['observed_module_events']
        assert events == expected and counts['all_module_events_match']
        rebuilt = event_counts(events)
        assert rebuilt == counts['observed'] == counts['expected']; observed_counts.update(rebuilt)
        assert set(batch['stages']) == expected_stage_names
        for name, cost in batch['stages'].items():
            wall = cost['wall_seconds']; assert math.isfinite(wall) and wall >= 0
            stages[name].append(wall)
            span = cost['cuda_event_span_seconds']
            if name in ['forward_count_verification','direction_and_attention_statistics_cpu','output_write']: assert span is None
            else: assert span is not None and math.isfinite(span) and span >= 0
        for branch in ['full','weak']:
            assert len(batch['trace_native_dtypes'][branch])==2
            for types in batch['trace_native_dtypes'][branch]:
                assert types['v']==types['sdpa_output']=='torch.bfloat16'
                assert all(v in ('torch.bfloat16','torch.float32') for v in types.values())
    assert dict(observed_counts)==summary['observed_forward_counts']==summary['expected_forward_counts']==request['expected_forward_counts']
    assert summary['all_batch_forward_module_events_match']
    stage_sums = {k:math.fsum(v) for k,v in stages.items()}
    assert set(stage_sums)==set(summary['stage_wall_seconds'])
    for key,value in stage_sums.items(): match(value,summary['stage_wall_seconds'][key],key)
    model_load = summary['loading']['model_load']['wall_seconds']
    unattributed = summary['wall_seconds_before_summary']-math.fsum(stage_sums.values())-model_load
    assert unattributed >= -1e-8
    prep = read(source/'prepare_cost.json')
    assert prep['request_sha256']==args.request_sha256 and prep['model_calls']==0 and prep['gpu_seconds']==0
    failed_dir = RESTART/'decoder_query_mean_v1'
    failed = read(failed_dir/'preflight_failure.json'); failed_prep = read(failed_dir/'prepare_cost.json')
    failed_request = read(failed_dir/'request.json')
    assert sha(failed_dir/'request.json')==failed['request_sha256']==failed_prep['request_sha256']
    assert failed['gpu_model_calls']==0 and failed['failed_attempt_wall_seconds'] is None
    assert not (failed_dir/'run_started.json').exists() and not (failed_dir/'summary.json').exists()
    old_source = Path(failed_request['sources']['experiments/audit_raev2_decoder_query_mean.py']['path']).read_text()
    new_source = Path(request['sources']['experiments/audit_raev2_decoder_query_mean.py']['path']).read_text()
    assert old_source.count("sha256_file(record['path'])")==2
    assert old_source.replace("sha256_file(record['path'])", "sha256_file(Path(record['path']))")==new_source
    failed_evidence = {'preflight_failure':failed,'prepare_cost':failed_prep,
        'metadata_sha256':{n:sha(failed_dir/n) for n in ['request.json','prepare_cost.json','preflight_failure.json']},
        'v1_to_v2_only_two_preflight_Path_wrappers_verified':True,
        'total_failed_attempt_cost_closed':False,
        'boundary':'Failed v1 preparation is known cost; failed execution wall remains unknown, never zero. It failed before run_started/CUDA and did not produce model outputs.'}
    assert summary['checkpoint_step']==100080 and summary['peak_reserved_bytes']>=summary['peak_allocated_bytes']>=0
    assert all(summary[k] for k in ['no_sampling','no_fid','no_decoder_images','no_new_noise'])
    assert all(sha(source/name)==digest for name,digest in initial_hashes.items())
    keys = list(max_error)
    per_batch = [{"step_index":k,"t":times[k],"domain":domain,
                  **aggregate([x for x in computed if batch_identity(x)==(k,domain)],keys)} for k,domain in batches]
    attention_groups = []
    for k,domain in batches:
        for branch in ['full','weak']:
            for block in range(2):
                subset=[a for a in attn if batch_identity(a)==(k,domain) and a['branch']==branch and a['decoder_block']==block]
                assert len(subset)==128
                attention_groups.append({'step_index':k,'domain':domain,'branch':branch,'decoder_block':block,**aggregate(subset,attention_metrics)})
    result = {'complete':True,'created_utc':datetime.now(timezone.utc).isoformat(),'request_sha256':args.request_sha256,
        'review_script_sha256':sha(__file__),'producer_runner_sha256':RUNNER_SHA,'input_metadata_sha256':initial_hashes,
        'all_new_output_sha256_verified':True,'outputs':summary['outputs'],
        'historical_snapshot_and_large_checkpoint_payload_rehashed':False,
        'all_160_row_ids_times_labels_source_rows_and_order_exact':True,'all_10240_attention_row_identities_exact':True,
        'native_parity_B8_records':20,'all_40_current_head_bitwise_parity_records_true':True,
        't1_direction_vectors_duplicate_exact':t1_equal,'unique_ids':8,'repeated_record_rows':160,
        'direction_vectors_streamed_payload_bytes':160*D*4*2,
        'vector_reduction_absolute_tolerance':ATOL,'vector_reduction_relative_tolerance':RTOL,
        'max_abs_difference_by_metric':dict(max_error),'direction_all_160':aggregate(computed,keys),
        'direction_by_domain_and_step':per_batch,'attention_weak_total_rows':5120,'attention_weak_flags':dict(weak_flags),
        'attention_all_rows_by_branch_and_block':[{'branch':branch,'decoder_block':block,
            **aggregate([a for a in attn if a['branch']==branch and a['decoder_block']==block],attention_metrics)}
            for branch in ['full','weak'] for block in range(2)],
        'attention_by_domain_step_branch_block':attention_groups,
        'attention_probability_or_KL_rebuilt_from_QK':False,'individual_F_B_W_norms_recomputed':False,
        'attention_scope':'All stored scalar rows checked for identity, finite values, equality-flag consistency, KL/L1 bounds and complete aggregation; Q/K/SDPA trace tensors were not retained, so probability/KL values cannot be independently reconstructed.',
        'observed_forward_counts_rebuilt_from_module_events':dict(observed_counts),
        'cost':{'prepare':prep,'model_load_wall_seconds':model_load,'stage_wall_seconds_rebuilt':stage_sums,
                'prior_v1_failed_attempt':failed_evidence,
                'run_wall_seconds_before_summary':summary['wall_seconds_before_summary'],
                'run_cpu_seconds_before_summary':summary['cpu_seconds_before_summary'],
                'unattributed_in_run_wall_seconds':unattributed,
                'external_process_wall_seconds':None,'peak_allocated_bytes':summary['peak_allocated_bytes'],
                'peak_reserved_bytes':summary['peak_reserved_bytes'],
                'boundary':'Stage breakdown is included in recorded run wall; do not add it twice. CUDA events are stream spans. Imports/final summary/exit and external process timing are not recovered here; no sampler or cost-matched quality claim.'},
        'limits':['All 160 rows retained; no response filtering or gain fitting.','Only 8 related IDs across 10 times and 2 domains; t=1 domains duplicate. No independent-trial CI.',
                  'Query-mean equal rows do not imply uniform keys or spatially constant final Full output.',
                  'Historical DC/AC direction is descriptive context, not a predictor of FID.',
                  'This audit establishes recorded structural response/identities, not weak-target error compatibility or generation benefit.'],
        'new_model_forwards':0,'new_vjps':0,'new_images':0,'fid_performed':False,
        'review_wall_seconds_before_final_write':time.perf_counter()-started,
        'review_cpu_seconds_before_final_write':time.process_time()-cpu}
    (out/'review.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'complete':True,'review_sha256':sha(out/'review.json'),'all_rows':160,
                      'attention_rows':10240,'weak_row_flags':dict(weak_flags),'max_metric_abs_error':max(max_error.values()),
                      'review_wall_seconds':result['review_wall_seconds_before_final_write']},indent=2))

if __name__=='__main__': main()
