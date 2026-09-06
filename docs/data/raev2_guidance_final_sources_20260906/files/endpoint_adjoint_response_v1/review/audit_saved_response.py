#!/usr/bin/env python3
"""Independent CPU-only audit of completed, frozen output; no model imports/calls.
Writes exclusively beside this script. SHA checks include actual external assets.
Time starts after importing time, before all other imports; final result serialization
and process exit are outside the embedded cost. This does not replay any dynamics.
"""
import time
START_WALL, START_CPU = time.perf_counter(), time.process_time()
from pathlib import Path
import collections
import csv
import datetime
import hashlib
import json
import resource
import sys
import numpy as np

REVIEW = Path(__file__).resolve().parent
P = REVIEW.parent
IDS = [0, 142, 285, 428, 570, 713, 856, 999]
checks, evidence, cache = [], [], {}
def check(name, passed, **details):
    checks.append({'name': name, 'passed': bool(passed), **details})
def close(name, actual, expected, rtol=2e-12, atol=1e-15):
    actual, expected = np.asarray(actual), np.asarray(expected)
    error = float(np.max(np.abs(actual-expected)))
    check(name, actual.shape == expected.shape and np.allclose(actual, expected, rtol=rtol, atol=atol), max_absolute_error=error, rtol=rtol, atol=atol)
def raw_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def sha(path):
    path = Path(path).resolve()
    if str(path) not in cache:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for data in iter(lambda: f.read(8*1024*1024), b''):
                h.update(data)
        cache[str(path)] = {'path':str(path), 'sha256': h.hexdigest(), 'bytes': path.stat().st_size}
    return cache[str(path)]
def load(path):
    return json.loads(Path(path).read_text())
def dump(name, value):
    with (REVIEW/name).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
def referenced_records(value, owner):
    if isinstance(value, dict):
        if {'path','sha256','bytes'} <= set(value):
            actual = sha(value['path'])
            check('record_identity', actual['sha256']==value['sha256'] and actual['bytes']==value['bytes'], owner=owner, expected=value, actual=actual)
        for item in value.values(): referenced_records(item, owner)
    elif isinstance(value,list):
        for item in value: referenced_records(item,owner)

docs = {}
# Pin all existing run evidence once; ignore interpreter bytecode and this new review.
for file in sorted(P.rglob('*')):
    if file.is_file() and REVIEW not in file.parents and '__pycache__' not in file.parts:
        sha(file)
        if file.suffix=='.json': docs[str(file.relative_to(P))]=load(file)
for name,value in docs.items(): referenced_records(value,name)
req=docs['request.json']; frozen=docs['frozen_control.json']; response=docs['response_summary.json']; execution=docs['execution_summary.json']
request_hash=sha(P/'request.json')['sha256']
check('fixed_ids_seed_grid', req['global_ids_and_labels']==IDS and req['seed']==202609111 and req['num_steps']==100 and req['batch_size']==1)
grid=np.asarray(req['time_grid'],dtype=np.float64); h=grid[:-1]-grid[1:]
check('positive_full_grid', grid.shape==(101,) and grid[0]==1 and grid[-1]==0 and np.all(h>0))
with np.load(P/'fixed_prototypes.npz',allow_pickle=False) as d:
    directions=d['directions']; center=d['center']; check('prototype_ids',np.array_equal(d['ids'],IDS))
with np.load(req['identities']['prototype_values']['path'],allow_pickle=False) as d:
    close('directions_from_fixed_A_divide_2048',directions,d['seed20260801_prototype_vectors'][IDS].astype(np.float64)/2048,rtol=0,atol=0)
    close('center_from_fixed_A',center,d['seed20260801_source_global_mean'].astype(np.float64),rtol=0,atol=0)
proto_summary=load(req['identities']['prototype_summary']['path'])
fold=next(v for v in proto_summary['results'] if v['name']=='train20260801_eval20260802')
close('delta_historical_frozen_scale',req['delta'],-fold['contrasts']['ig_minus_source']['class_equal_weighted_mean'],rtol=0,atol=0)
workers=[(name,d) for name,d in docs.items() if name.startswith('workers/') and name.endswith('/summary.json')]
check('eight_complete_workers',len(workers)==8 and all(d['complete'] for _,d in workers))
check('no_recorded_failures',not any('failure' in name for name in docs))
full_noise=None; full_rng=None; worker_counts=collections.Counter(); worker_costs=[]
for name,w in workers:
    with np.load(w['noise_artifact']['path'],allow_pickle=False) as d:
        noise=d['noise']; rng=d['rng_state']; ni=d['ids']
        check(name+'/noise_shape',noise.shape==(8,1024,16,16) and noise.dtype==np.float32 and np.array_equal(ni,IDS))
        if full_noise is None: full_noise=noise.copy(); full_rng=rng.copy()
        check(name+'/same_full_noise_and_rng',np.array_equal(noise,full_noise) and np.array_equal(rng,full_rng) and raw_sha(noise)==w['global_noise_sha256']==frozen['global_noise_sha256'])
    worker_counts.update(w['calls'])
    phase=w['phase']; expected_ids=[int(x) for x in name.split('/')[1].split('_')[1:]]
    check(name+'/ids_calls',expected_ids==w['ids'] and dict(sum((collections.Counter(d['calls']) for d in w['images']),collections.Counter()))==w['calls'])
    check(name+'/no_training_fid',w['fid_calls']==w['training_updates']==0)
    if phase=='replay': check(name+'/frozen_lambda_link',w['frozen_control']['sha256']==sha(P/'frozen_control.json')['sha256'])
    worker_costs.append({'worker':name, 'timing':w['timing'], 'process_cost':w['process_cost'], 'peak_allocated_bytes':w['peak_gpu_allocated_bytes_including_model_load'], 'peak_reserved_bytes':w['peak_gpu_reserved_bytes_including_model_load']})
rows=[]; norms_all=[]; timestep_rows=[]; image_counts=collections.Counter(); max_psi_error=0.
for index,image_id in enumerate(IDS):
    c=docs[f'collect/id{image_id:04d}/summary.json']; r=docs[f'replay/id{image_id:04d}/summary.json']
    row={'global_id':image_id}
    check(f'id{image_id}/identity',c['complete'] and r['complete'] and c['global_id']==r['global_id']==image_id and c['request_sha256']==r['request_sha256']==request_hash and c['noise_sha256']==r['noise_sha256']==raw_sha(full_noise[index]) and r['lambda']==frozen['lambda'])
    states=np.load(c['states']['path'],mmap_mode='r',allow_pickle=False)
    adjoints=np.load(c['adjoints']['path'],mmap_mode='r',allow_pickle=False)
    check(f'id{image_id}/state_adjoint_shape',states.shape==(101,1024,16,16) and adjoints.shape==(100,1024,16,16) and states.dtype==adjoints.dtype==np.float32)
    check(f'id{image_id}/all_101_collected_state_hashes',[raw_sha(v) for v in states]==c['state_sha256_by_step'] and np.array_equal(states[0],full_noise[index]))
    sq=[]; energy=0.; prediction=0.
    for k,a32 in enumerate(adjoints):
        a=np.asarray(a32,dtype=np.float64); u=(np.asarray(a32)*np.float32(frozen['lambda'])).astype(np.float64)
        squared=float(np.sum(a*a,dtype=np.float64)); sq.append(squared)
        ek=float(h[k]*np.sum(u*u,dtype=np.float64)); pk=float(h[k]*np.sum(a*u,dtype=np.float64)); energy+=ek; prediction+=pk
        timestep_rows.append({'global_id':image_id,'k':k,'time_current':float(grid[k]),'time_successor':float(grid[k+1]),'h':float(h[k]),'adjoint_norm':squared**.5,'adjoint_squared_norm':squared,'integrated_gram':float(h[k]*squared),'implemented_energy':ek,'implemented_prediction':pk})
    close(f'id{image_id}/all_100_adjoint_squared_norms',sq,c['post_state_squared_norms'])
    gram=float(np.dot(h,sq)); close(f'id{image_id}/integrated_gram',gram,c['integrated_gram'])
    norms_all.append(sq)
    row.update(integrated_gram=gram,implemented_control_energy=energy,implemented_linear_response=prediction,predicted_response=frozen['lambda']*gram)
    arms={'fp32_baseline':c['fp32_baseline'],**r['results']}
    for name,a in arms.items():
        files=a['artifacts']; features=np.load(files['features']['path'],allow_pickle=False); pixels=np.load(files['pixels']['path'],allow_pickle=False); latent=np.load(files['latent']['path'],allow_pickle=False)
        native=name.startswith('native')
        check(f'id{image_id}/{name}/finite_complete_shapes',features.shape==(1,2048) and features.dtype==np.float32 and pixels.shape==(1,3,256,256) and pixels.dtype==(np.uint8 if native else np.float32) and latent.shape==(1,1024,16,16) and latent.dtype==np.float32 and np.isfinite(features).all() and np.isfinite(pixels).all() and np.isfinite(latent).all())
        psi=float(np.sum((features.astype(np.float64)-center)*directions[index],dtype=np.float64)); row[name+'_psi']=psi
        max_psi_error=max(max_psi_error,abs(psi-a['psi'])); close(f'id{image_id}/{name}/psi',psi,a['psi'])
        hashes=c['state_sha256_by_step'] if name=='fp32_baseline' else a['state_sha256_by_step']
        check(f'id{image_id}/{name}/state_hash_endpoints',len(hashes)==101 and hashes[0]==raw_sha(full_noise[index]) and hashes[-1]==a['state_sha256']==raw_sha(latent))
        if name=='fp32_baseline':check(f'id{image_id}/saved_baseline_endpoint',np.array_equal(latent[0],states[-1]))
        if name.endswith('controlled'):
            close(f'id{image_id}/{name}/implemented_energy',energy,a['implemented_control_energy'])
            close(f'id{image_id}/{name}/implemented_prediction',prediction,a['linear_response_from_stored_fp32_control'])
    row['fp32_actual_response']=row['fp32_controlled_psi']-row['fp32_baseline_psi']
    row['native_actual_response']=row['native_controlled_psi']-row['native_baseline_psi']
    for precision in ['fp32','native']:
        row[precision+'_response_over_fixed_delta']=row[precision+'_actual_response']/req['delta']
        close(f'id{image_id}/{precision}/response',row[precision+'_actual_response'],r[precision+'_actual_response'])
    row['fp32_minus_prediction']=row['fp32_actual_response']-row['predicted_response']
    rows.append(row)
    for phase in ['collect','replay']:
        e=docs[f'{phase}/id{image_id:04d}/execution.json']; image_counts.update(e['calls']); ec=e['calls']
        check(f'id{image_id}/{phase}/expected_calls',ec['stage2_forward_calls']==(199 if phase=='collect' else 300) and ec.get('stage2_input_vjp_calls',0)==(99 if phase=='collect' else 0))
check('counts_all_levels',image_counts==worker_counts==collections.Counter(response['successful_image_calls']))
means={k:float(np.mean([row[k] for row in rows])) for k in rows[0] if k!='global_id'}
for key,value in response['eight_image_means'].items(): close('means/'+key,means[key],value)
close('frozen_G_mean',means['integrated_gram'],frozen['cohort_gram'])
close('unique_lambda_delta_over_G',frozen['lambda'],req['delta']/means['integrated_gram'])
close('ideal_minimum_energy',req['delta']*frozen['lambda'],frozen['minimum_surrogate_energy'])
with np.load(P/'collect/id0000/feature_coordinate_check.npz',allow_pickle=False) as d:
    coord_error=float(np.max(np.abs(d['expected']-d['actual'])))
    coordinate=docs['collect/id0000/summary.json']['coordinate_check']
    check('saved_coordinate_pair',d['pixels'].dtype==np.uint8 and d['expected'].shape==d['actual'].shape==(1,2048) and np.allclose(d['expected'],d['actual'],rtol=coordinate['rtol'],atol=coordinate['atol']))
    close('saved_coordinate_max_error',coord_error,coordinate['maximum_absolute_error'],rtol=0,atol=0)
allnorms=np.asarray(norms_all); contributions=allnorms*h; total=float(contributions.sum())
image_shares=contributions.sum(axis=1)/total; time_shares=contributions.sum(axis=0)/total
for row,share in zip(rows,image_shares):row['gram_share_percent']=float(100*share)
time_bins=[{'steps_inclusive':[k,k+9],'percent':float(100*time_shares[k:k+10].sum())} for k in range(0,100,10)]
for filename,data in [('per_image.csv',rows),('per_image_per_step.csv',timestep_rows)]:
    with (REVIEW/filename).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0])); writer.writeheader();writer.writerows(data)
processes=execution['processes']; expected_tags=[f'collect_rank{i}' for i in range(4)]+['finalize']+[f'replay_rank{i}' for i in range(4)]+['summarize']
check('one_complete_frozen_execution',execution['complete'] and [x['tag'] for x in processes]==expected_tags and all(x['exit_code']==0 for x in processes))
for process in processes:
    check(process['tag']+'/outer_record',process==docs[process['tag']+'.process.json'])
check('driver_identity',docs['driver_identity.json']['driver_sha256']==sha(P/'driver.py')['sha256'] and docs['driver_identity.json']['request_sha256']==request_hash and docs['driver_identity.json']['script_sha256']==req['sources']['experiments/audit_raev2_endpoint_adjoint_response.py']['archive']['sha256'])
source=req['sources']['experiments/audit_raev2_endpoint_adjoint_response.py']['archive']['path']
lines=Path(source).read_text().splitlines()
spans=[(11,29,'timer start before numpy/torch imports'),(170,202,'prototype and prepare timing'),(227,290,'CUDA span, IG Euler and successor suffix index'),(317,383,'observable and terminal psi.sum input VJP'),(423,470,'collect storage and suffix closure'),(547,595,'fixed full replay, postEuler control, all three arms'),(598,673,'full noise, B1, worker counts and timer boundary')]
dump('source_evidence.json',{'source':sha(source),'driver':sha(P/'driver.py'),'excerpts':[{'start_line':a,'end_line':b,'purpose':why,'text':'\n'.join(f'{i+1}: {lines[i]}' for i in range(a-1,b))} for a,b,why in spans], 'static_conclusions':{'suffix':'Save a at successor z[k+1] before VJP; k=99..0; VJP T_k only if k>0, hence T99..T1 = 99, excludes T0. terminal psi.sum, coordinate sum norms, cohort mean Gram.', 'control':'One lambda from all eight collects; same stored FP32 a used in each precision, added h*u after Euler; no ridge, control clipping, lambda/time scan or output selection.', 'native':'BF16 head arithmetic and decoder, clamp/mul255 uint8; FP32 continuous derivatives are not native Jacobians.', 'forward':'Full and Base share one stage2 call; B1 makes call and sample counts identical.', 'time':'Initial time import before all remaining runner imports; process snapshots exclude final write and exit. Driver main timer excludes its earlier imports and CPU prepare, final write and driver exit. CUDA events measure elapsed stream span including idle, not kernel busy time.'}})
phase_costs={}
for phase in ['collect','replay']:
    selected=[v for v in processes if v['tag'].startswith(phase+'_')]
    selected_workers=[w for n,w in workers if w['phase']==phase]
    phase_costs[phase]={'worker_outer_wall_sum_seconds':sum(v['outer_wall_seconds'] for v in selected),'worker_outer_wall_min_seconds':min(v['outer_wall_seconds'] for v in selected),'worker_outer_wall_max_seconds':max(v['outer_wall_seconds'] for v in selected),'child_user_cpu_sum_seconds':sum(v['child_user_cpu_seconds'] for v in selected),'child_system_cpu_sum_seconds':sum(v['child_system_cpu_seconds'] for v in selected),'source_weight_verification_wall_sum_seconds':sum(w['timing']['source_and_weight_verification']['wall_seconds'] for w in selected_workers),'model_load_wall_sum_seconds':sum(t['wall_seconds'] for w in selected_workers for t in w['timing']['model_loads'].values()),'phase_inner_wall_sum_seconds':sum(w['timing']['phase']['wall_seconds'] for w in selected_workers),'phase_cuda_event_span_sum_seconds':sum(w['timing']['phase']['cuda_stream_elapsed_seconds'] for w in selected_workers)}
dump('timing_evidence.json',{'prepare':req['prepare_cost'],'driver':execution,'workers':worker_costs,'phase_sums':phase_costs,'max_gpu_allocated_bytes':max(w['peak_allocated_bytes'] for w in worker_costs),'max_gpu_reserved_bytes':max(w['peak_reserved_bytes'] for w in worker_costs),'cost_boundary':'Parallel worker wall sums are resource occupation sums, not elapsed latency. Nested components must not be summed again into outer times. CUDA spans include idle, not kernel busy time. No fair production baseline benchmark or complete trained-method cost comparison exists.'})
dump('sha256_evidence.json',{'files':[cache[k] for k in sorted(cache)],'unique_file_count':len(cache),'total_unique_file_bytes':sum(v['bytes'] for v in cache.values()),'verification':'Actual SHA256 reread during this CPU audit, including external model weights, configs, prototype reference and all run files except interpreter bytecode; duplicate absolute paths resolved once.'})
result={'complete':True,'all_checks_passed':all(x['passed'] for x in checks),'checks_count':len(checks),'failed_checks':[v for v in checks if not v['passed']],'max_psi_absolute_error':max_psi_error,'global_noise_raw_sha256':raw_sha(full_noise),'per_image':rows,'means':means,'time_gram_share_per_step_percent':(100*time_shares).tolist(),'time_gram_share_ten_step_bins':time_bins,'calls':dict(image_counts),'coordinate_max_absolute_error':coord_error,'phase_costs':phase_costs,'preservation':{'cohort_images':8,'all_four_arms_per_image':True,'stored_endpoints':32,'stored_native_uint8_endpoints':16,'stored_continuous_fp32_endpoints':16,'baseline_full_state_hashes_recomputed':808,'all_arms_initial_and_terminal_hashes_recomputed':True,'suffix_adjoint_norms_recomputed':800,'replay_internal_states':'Only 101 hash entries per arm retained; intermediate replay tensor/dynamics not reexecuted in CPU review.'},'limits':['All psi recomputed from saved features; no decoder/Inception inference rerun, no pixels-to-features independent recomputation.','Suffix indexing/source/call counts reviewed; gradients not numerically reexecuted or checked by new epsilon/parameter sweeps.','Full saved noise and RNG states match bytewise; CUDA RNG draw not repeated.','FP32 is a differentiable surrogate, not native BF16/uint8 Jacobian.','Eight-image finite response, no FID/population quality conclusion, no fair-cost improvement claim.'],'decision':'Mechanism negative at fixed perturbation: FP32 only 23.5503% of prediction; native response -15.1659%. Stop deployment/distillation admission of this implementation; no lambda scan, clipping, time windows, sample reweighting or training.', 'cpu_review_cost':{'wall_seconds_since_initial_time_import':time.perf_counter()-START_WALL,'cpu_seconds_since_initial_time_import':time.process_time()-START_CPU,'host_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'scope':'NumPy imports, actual file SHA reads, all audit arithmetic and preceding evidence/CSV writes; excludes final results/checks/manifest serialization, print and interpreter exit.','gpu_calls':0,'model_calls':0,'tests_rerun':0},'python':sys.version,'numpy':np.__version__,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
dump('checks.json',checks)
dump('results.json',result)
dump('manifest.json',{'review_files':[sha(file) for file in sorted(REVIEW.iterdir()) if file.is_file()],'excludes':'This manifest itself; final document hash can be added in a separate document_identity.json without modifying manifest.','audit_script':sha(Path(__file__))})
print(json.dumps({k:result[k] for k in ['all_checks_passed','checks_count','failed_checks','means','phase_costs','cpu_review_cost']},indent=2))
raise SystemExit(0 if result['all_checks_passed'] else 1)
