#!/usr/bin/env python3
"""Independent CPU reconstruction; imports no producer or model code."""
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, math, os, time, zipfile
import numpy as np
import torch

SOURCE=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/query_mean_error_compatibility_v1')
OUT=Path(__file__).resolve().parent
ROOT=Path('/home/zhoushunyu/eqvae')
REQUEST_SHA='1f18f4374530e9c235d3c4a62bfe9349b729d20467d36194077797f229524ead'
SUMMARY_SHA='b3e49faf465b5b0a66a3ff2670505063e2ccf936219b822e47dce7d3fe754273'
STEPS=[0,47,67,77,84,89,92,95,97,99]
SHAPE=(1024,16,16)
D=262144
checks=0
compared=0
max_difference=0.

def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def put(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()
def ident(path):
    p=Path(path).resolve();return {'path':str(p),'sha256':digest(p),'bytes':p.stat().st_size}
def require(condition,reason):
    global checks
    if not condition:raise AssertionError(reason)
    checks+=1
def close(a,b,reason,atol=1e-11,rtol=5e-10):
    global compared,max_difference
    aa,bb=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    require(aa.shape==bb.shape and np.isfinite(aa).all() and np.isfinite(bb).all(),reason+' finite/shape')
    difference=np.abs(aa-bb)
    require(bool(np.all(difference<=atol+rtol*np.abs(bb))),reason+' value')
    compared+=aa.size;max_difference=max(max_difference,float(difference.max(initial=0)))
def table(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def write_table(path,rows):
    with Path(path).open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def dotmean(a,b):
    # Independent reduction: 256-coordinate einsum chunks then Python fsum,
    # instead of the producer's flat NumPy mean of products.
    a=np.asarray(a,dtype=np.float64).reshape(-1,256)
    b=np.asarray(b,dtype=np.float64).reshape(-1,256)
    return math.fsum(np.einsum('ij,ij->i',a,b))/a.size
def boolean(value):return value is True or value=='True'
def registered(value,result):
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value:
            path=str(Path(value['path']).resolve())
            if path in result:require(result[path]['sha256']==value['sha256'],'consistent duplicate artifact')
            result[path]=value
        for child in value.values():registered(child,result)
    elif isinstance(value,list):
        for child in value:registered(child,result)

def main():
    start,cpu=time.perf_counter(),time.process_time()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','GPU must be hidden')
    require(not torch.cuda.is_initialized(),'CUDA not initialized')
    torch.set_num_threads(4)
    require(not (OUT/'review.json').exists() and not (OUT/'request.json').exists(),'refuse overwrite')
    require(digest(SOURCE/'request.json')==REQUEST_SHA,'source request SHA')
    require(digest(SOURCE/'summary.json')==SUMMARY_SHA,'source summary SHA')
    req,summary=read(SOURCE/'request.json'),read(SOURCE/'summary.json')
    require(summary['complete'] and summary['rows']==80 and summary['unique_ids']==8,'completed fixed cohort')
    put(OUT/'request.json',{'review':'independent_cpu_teacher_query_mean_error_reconstruction_v1','created_utc':now(),
        'source_request':ident(SOURCE/'request.json'),'source_summary':ident(SOURCE/'summary.json'),'review_source':ident(__file__),
        'scope':'all80 saved heads/targets and rows; all registered producer sources/outputs SHA; no producer imports or GPU',
        'arithmetic':'independent FP64 vector subtraction, 256-coordinate einsum and math.fsum; all80/per-time means and ratios',
        'no_gain_selection':True,'no_fid':True})
    artifacts={};registered(req,artifacts);registered(summary,artifacts)
    verified=[]
    for path,record in artifacts.items():
        value=ident(path)
        require(value['sha256']==record['sha256'],'registered SHA '+path)
        if 'bytes' in record:require(value['bytes']==record['bytes'],'registered bytes '+path)
        verified.append(value)
    for relative,record in req['sources'].items():
        require(digest(ROOT/relative)==record['sha256'],'live producer source matches archive '+relative)
    require(digest(SOURCE/'frozen_protocol.md')==req['protocol_document']['sha256'],'frozen protocol copy')
    require(req['sample_ids']==req['labels']==list(range(8)),'fixed8 IDs/classes')
    require([s['step_index'] for s in req['snapshots']]==STEPS,'all10 fixed steps')
    require(req['coordinate_normalizer']==D and req['domain']=='teacher','normalization/domain')
    require(req['source_teacher_vector_rows']==[16*k+i for k in range(10) for i in range(8)],'v2 teacher offsets')
    # Independent join by the complete identity tuple, not array position or ID alone.
    with np.load(req['clean_bank']['path'],allow_pickle=False) as z:
        bank_id,bank_label,bank_source=[z[k] for k in ('ids','labels','rows')]
    tuples=list(zip(bank_id.tolist(),bank_label.tolist(),bank_source.tolist()))
    require(len(tuples)==1000 and len(set(tuples))==1000,'bank identities unique')
    locations=[]
    for identity in zip(req['sample_ids'],req['labels'],req['source_rows']):
        found=[i for i,v in enumerate(tuples) if v==identity]
        require(len(found)==1,'unique three-way clean join');locations.append(found[0])
    require(locations==req['bank_array_rows'],'source row mapping')
    clean=np.load(req['paired_clean']['path'],allow_pickle=False)
    require(clean.dtype==np.float16 and clean.shape==(8,*SHAPE),'saved clean dtype/shape')
    with zipfile.ZipFile(req['clean_bank']['path']) as z,z.open('latents.npy') as f:
        version=np.lib.format.read_magic(f);shape,fortran,dtype=np.lib.format._read_array_header(f,version)
        require(shape==(1000,*SHAPE) and not fortran and dtype==np.float16,'original latent header')
        first=f.tell()
        for local,index in enumerate(locations):
            f.seek(first+index*D*2);origin=np.frombuffer(f.read(D*2),dtype=np.float16).reshape(SHAPE)
            require(np.array_equal(origin.view(np.uint16),clean[local].view(np.uint16)),'paired X exact source bytes')
    meta=read(req['paired_clean_identity']['path'])
    for key in ('sample_ids','labels','source_rows','bank_array_rows'):require(meta[key]==req[key],'paired identity '+key)
    normal_req=read(req['state_source_request']['path'])
    original_hashes={str(Path(p).resolve()):h for p,h in normal_req['bank_metadata']['source_sha256'].items()}
    require(original_hashes[str(Path(req['clean_bank']['path']).resolve())]==req['clean_bank']['sha256'],'historical actual clean bank SHA')
    heads={name:np.load(SOURCE/(name+'.npy'),mmap_mode='r',allow_pickle=False) for name in ('F','B','W')}
    require(all(v.shape==(80,*SHAPE) and v.dtype==np.float32 for v in heads.values()),'all saved head shape/dtype')
    old_w=np.load(req['source_outputs']['F_minus_W.npy']['path'],mmap_mode='r',allow_pickle=False)
    old_b=np.load(req['source_outputs']['F_minus_B.npy']['path'],mmap_mode='r',allow_pickle=False)
    norm_rows=table(req['source_outputs']['per_image_directions.csv']['path'])
    produced=table(SOURCE/'per_image_risk.csv');parity=table(SOURCE/'v2_parity.csv')
    costs=read(SOURCE/'batch_costs.json');bridges=read(SOURCE/'bridge_identity_prepare.json')['rows']
    require(len(produced)==80 and len(parity)==len(costs)==len(bridges)==10,'all records retained')
    epsilon=torch.load(req['snapshots'][0]['path'],map_location='cpu',weights_only=True,mmap=True)['teacher']['state'].numpy().copy()
    require(req['snapshots'][0]['t']==1 and epsilon.dtype==np.float32,'original t1 epsilon')
    clean32=clean.astype(np.float32)
    expected_hooks={'shared_encoder':{'s_embedder':1,**{f'encoder_block_{i:02d}':1 for i in range(28)}},
        'full_decoder_and_readout':{**{f'decoder_{i}_{p}':1 for i in (28,29) for p in ('q','mlp')},'full_readout':1},
        'base_readout':{'base_readout':1},
        'weak_decoder_and_readout':{**{f'decoder_{i}_{p}':1 for i in (28,29) for p in ('q','mlp')},'full_readout':1}}
    expected_counts={'shared_encoder_passes':1,'native_model_forward_calls':0,'encoder_block_calls':28,
        'explicit_full_decoder_block_calls':2,'weak_decoder_block_calls':2,'native_reference_decoder_block_calls':0,
        'full_final_readout_calls_including_weak_and_reference':2,'base_readout_calls_including_reference':1}
    require(req['expected_module_events_per_batch']==expected_hooks,'expected hook protocol')
    rebuilt=[];bridge_results=[]
    for k,snap in enumerate(req['snapshots']):
        state=torch.load(snap['path'],map_location='cpu',weights_only=True,mmap=True)
        require(state['sample_ids'].tolist()==req['sample_ids'] and state['labels'].tolist()==req['labels'],'snapshot identity')
        require(state['step_index']==snap['step_index'] and state['t']==snap['t'],'snapshot time')
        z=state['teacher']['state'].numpy();t=snap['t']
        require(z.dtype==np.float32 and z.shape==(8,*SHAPE),'original teacher dtype/shape')
        expected=np.add(np.multiply(clean32,np.float32(1-t),dtype=np.float32),np.multiply(epsilon,np.float32(t),dtype=np.float32),dtype=np.float32)
        diff=z.astype(np.float64)-expected.astype(np.float64)
        bound=8*np.finfo(np.float32).eps*(abs(1-t)*clean.astype(np.float64).__abs__()+abs(t)*np.abs(epsilon.astype(np.float64))+np.finfo(np.float32).tiny)
        bridge={'bitwise_equal_to_cpu_original_order':np.array_equal(z.view(np.uint32),expected.view(np.uint32)),
            'max_abs_difference':float(np.max(np.abs(diff))),'rms_difference':math.sqrt(dotmean(diff,diff)),
            'max_elementwise_allowed_bound':float(np.max(bound)),'max_abs_difference_over_bound':float(np.max(np.abs(diff)/bound)),
            'elements_outside_bound':int(np.count_nonzero(np.abs(diff)>bound))}
        require(bridge['elements_outside_bound']==0,'teacher bridge roundoff bound')
        for recorded in (bridges[k],costs[k]['bridge_identity']):
            for key,value in bridge.items():
                if isinstance(value,bool):require(recorded[key]==value,'bridge exact flag')
                else:close(value,recorded[key],'bridge '+key)
        bridge_results.append({'step_index':snap['step_index'],'t':t,**bridge})
        require(costs[k]['observed_module_events']==expected_hooks,'actual per-phase every-block hooks')
        require(costs[k]['observed_forward_counts']==expected_counts,'actual derived forward counts')
        require(int(parity[k]['step_index'])==snap['step_index'] and parity[k]['domain']=='teacher','parity identity')
        require(boolean(parity[k]['F_minus_W_fp32_bitwise']) and boolean(parity[k]['F_minus_B_fp32_bitwise']),'producer bitwise flags')
        for i in range(8):
            j=8*k+i;vj=16*k+i;row=produced[j];refrow=norm_rows[vj]
            expected_identity={'output_row':j,'sample_id':i,'label':i,'source_row':req['source_rows'][i],
                'paired_X_row':i,'v2_vector_row':vj,'step_index':snap['step_index']}
            for key,value in expected_identity.items():require(int(row[key])==value,'output identity '+key)
            require(row['domain']=='teacher' and float(row['t'])==t,'output domain/time')
            require(refrow['domain']=='teacher' and int(refrow['sample_id'])==i and int(refrow['source_row'])==req['source_rows'][i],'v2 source row identity')
            f,b,w=(np.asarray(heads[key][j],dtype=np.float64) for key in ('F','B','W'));x=clean[i].astype(np.float64)
            require(all(np.isfinite(a).all() for a in (f,b,w,x)),'finite saved heads/X')
            dw32=np.subtract(heads['F'][j],heads['W'][j],dtype=np.float32)
            db32=np.subtract(heads['F'][j],heads['B'][j],dtype=np.float32)
            require(np.array_equal(dw32.view(np.uint32),old_w[vj].view(np.uint32)),'F-W exact v2 bits')
            require(np.array_equal(db32.view(np.uint32),old_b[vj].view(np.uint32)),'F-B exact v2 bits')
            for key,value in [('full',f),('base',b),('weak',w)]:close(math.sqrt(D*dotmean(value,value)),float(refrow[key+'_norm']),'v2 head norm',atol=1e-10,rtol=1e-12)
            r=x-f
            out={**expected_identity,'domain':'teacher','t':t,'R_F':dotmean(r,r),'R_W':dotmean(x-w,x-w),'R_B':dotmean(x-b,x-b)}
            for suffix,other,delta32 in [('W',w,dw32),('B',b,db32)]:
                delta=f-other;c=dotmean(r,delta);q=dotmean(delta,delta)
                rd=out['R_'+suffix]-out['R_F'];residual=rd-2*c-q
                rounding=delta32.astype(np.float64)-delta
                out.update({f'C_{suffix}':c,f'Q_{suffix}':q,f'Q_{suffix}_zero':q==0,
                    f'R_{suffix}_minus_R_F':rd,f'identity_{suffix}_residual':residual,
                    f'identity_{suffix}_allowed_bound':1e-12+1e-11*(abs(rd)+abs(2*c)+q),
                    f'oracle_C_over_Q_{suffix}_descriptive':c/q if q else None,
                    f'delta_{suffix}_fp32_minus_fp64_max_abs':float(np.max(np.abs(rounding))),
                    f'delta_{suffix}_fp32_minus_fp64_rms':math.sqrt(dotmean(rounding,rounding))})
                require(q>=0 and (q>0 or c==0),'zero-Q rule')
                require(abs(residual)<=out[f'identity_{suffix}_allowed_bound'],'independent risk identity')
                require(np.sign(c)==np.sign(float(row[f'C_{suffix}'])),'empirical C sign unchanged')
            for key,value in out.items():
                if key in expected_identity or key in ('t','domain'):continue
                if value is None:require(row[key]=='','undefined oracle retained')
                elif isinstance(value,bool):require(boolean(row[key])==value,'zero flags')
                else:close(value,float(row[key]),'all80 scalar '+key)
            rebuilt.append(out)
    # Group independently by time identity, not producer contiguous slice selection.
    primitive=('R_F','R_W','R_B','C_W','Q_W','C_B','Q_B','R_W_minus_R_F','R_B_minus_R_F')
    def aggregate(group):
        record={'rows':len(group)}
        for key in primitive:
            values=[r[key] for r in group];record.update({key+'_mean':math.fsum(values)/len(values),key+'_min':min(values),key+'_max':max(values)})
        for suffix in ('W','B'):
            q=record['Q_'+suffix+'_mean'];record['oracle_mean_C_over_mean_Q_'+suffix+'_descriptive']=record['C_'+suffix+'_mean']/q if q else None
            record['zero_Q_'+suffix+'_rows']=sum(r['Q_'+suffix]==0 for r in group)
            record['positive_C_'+suffix+'_rows']=sum(r['C_'+suffix]>0 for r in group)
            record['negative_C_'+suffix+'_rows']=sum(r['C_'+suffix]<0 for r in group)
            record['identity_'+suffix+'_max_abs_residual']=max(abs(r['identity_'+suffix+'_residual']) for r in group)
        return record
    per_time=[];published_time=table(SOURCE/'per_time_risk.csv')
    for k,step in enumerate(STEPS):
        group=[row for row in rebuilt if row['step_index']==step]
        require(len(group)==8,'all8 at each time');agg=aggregate(group)
        for reference in (summary['per_time'][k],published_time[k]):
            require(int(reference['step_index'])==step,'per-time step identity')
            for key,value in agg.items():
                if value is None:require(reference[key] in (None,''),'per-time null')
                else:close(value,float(reference[key]),'per-time '+key)
        per_time.append({'step_index':step,'t':req['snapshots'][k]['t'],**agg})
    all80=aggregate(rebuilt)
    for key,value in all80.items():
        if value is None:require(summary['all80_equal_time_descriptive'][key] is None,'all80 null')
        else:close(value,summary['all80_equal_time_descriptive'][key],'all80 '+key)
    oracle_algebra={}
    for suffix in ('W','B'):
        c,q=all80[f'C_{suffix}_mean'],all80[f'Q_{suffix}_mean']
        decrease=c*c/q if q else 0.
        oracle_algebra[suffix]={'mean_C':c,'mean_Q':q,'unconstrained_descriptive_C_over_Q':c/q if q else None,
            'unconstrained_empirical_risk_decrease_C_squared_over_Q':decrease,
            'unconstrained_decrease_relative_to_R_F':decrease/all80['R_F_mean'],
            'nonnegative_coefficient_empirical_risk_decrease':decrease if c>0 else 0.,
            'computed_by_closed_form_only':True,'coefficient_deployed':False}
    total_counts={k:10*v for k,v in expected_counts.items()}
    for value in (req['expected_forward_counts'],summary['expected_forward_counts'],summary['observed_forward_counts']):require(value==total_counts,'observed/expected aggregate counts')
    stage_sums={key:math.fsum(b['stages'][key]['wall_seconds'] for b in costs) for key in costs[0]['stages']}
    for key,value in stage_sums.items():close(value,summary['batch_stage_wall_seconds'][key],'stage cost '+key)
    stages=[s for b in costs for s in b['stages'].values()]+list(summary['setup_and_final_stages'].values())
    for stage in stages:
        require(math.isfinite(stage['wall_seconds']) and stage['wall_seconds']>=0,'stage wall valid')
        if stage['cuda_event_span_seconds'] is not None:require(math.isfinite(stage['cuda_event_span_seconds']) and stage['cuda_event_span_seconds']>=0,'CUDA event reported nonnegative')
    measured_stages=math.fsum(s['wall_seconds'] for s in stages)
    require(measured_stages<=summary['wall_seconds_before_summary']+1e-6,'disjoint stages fit complete run wall')
    started=read(SOURCE/'run_started.json');require(started['run_started_utc']==summary['run_started_utc'],'run start UTC')
    require(started['request']['sha256']==REQUEST_SHA,'run request identity')
    prepare=read(SOURCE/'prepare_cost.json');require(prepare['request_sha256']==REQUEST_SHA and prepare['gpu_calls']==0,'prepare identity/GPU boundary')
    for flag in ('no_sampling','no_new_noise','no_rollout_witness','no_decoder_images','no_fid'):require(summary[flag] is True,'scope '+flag)
    require(summary['oracle_coefficients_deployed'] is False and not torch.cuda.is_initialized(),'no oracle deployment/review CUDA')
    write_table(OUT/'reconstructed_per_image.csv',rebuilt)
    put(OUT/'reconstructed_per_time.json',per_time)
    put(OUT/'source_and_output_hash_checks.json',verified)
    result={'complete':True,'finished_utc':now(),'source_request_sha256':REQUEST_SHA,'source_summary_sha256':SUMMARY_SHA,
        'review_source':ident(__file__),'review_request':ident(OUT/'request.json'),'checks':checks,'numeric_values_compared':compared,
        'max_absolute_scalar_difference':max_difference,'registered_unique_artifacts_hashed':len(verified),
        'registered_artifact_bytes_hashed':sum(v['bytes'] for v in verified),'all80_reconstructed':True,'v2_gap_parity_rows':80,
        'all_source_X_identity_verified':True,'bridge_reconstruction':bridge_results,'observed_counts_checked':total_counts,
        'per_time':per_time,'all80_equal_time_descriptive':all80,'shared_scalar_empirical_oracle_algebra':oracle_algebra,
        'reported_cost_checked':{'prepare_wall_seconds':prepare['wall_seconds'],'run_wall_seconds_before_summary':summary['wall_seconds_before_summary'],
            'disjoint_stage_wall_seconds':measured_stages,'unallocated_run_wall_seconds':summary['wall_seconds_before_summary']-measured_stages,
            'batch_stage_wall_seconds':stage_sums,'scope':'arithmetic/identity of recorded costs, not a new timing benchmark'},
        'outputs':[ident(OUT/name) for name in ('reconstructed_per_image.csv','reconstructed_per_time.json','source_and_output_hash_checks.json')],
        'wall_seconds_before_review_write':time.perf_counter()-start,'cpu_seconds_before_review_write':time.process_time()-cpu,
        'model_calls':0,'gpu_calls':0,'fid_calls':0,'producer_imported':False,
        'interpretation':'Empirical C signs only on8 reused IDs across10 fixed times; no population sign/CI, collinearity proof, gain choice or FID inference.'}
    put(OUT/'review.json',result)
    print(json.dumps({'complete':True,'review':ident(OUT/'review.json'),'checks':checks,'numeric_values_compared':compared,
        'max_absolute_scalar_difference':max_difference,'wall_seconds':result['wall_seconds_before_review_write']},indent=2),flush=True)

if __name__=='__main__':main()
