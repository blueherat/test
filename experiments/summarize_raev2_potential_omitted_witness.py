#!/usr/bin/env python3
"""Aggregate the two frozen omitted-space witnesses with image-time pairing."""
from __future__ import annotations
import time
START, CPU_START = time.perf_counter(), time.process_time()
import argparse
import hashlib
import json
from pathlib import Path
import resource
import numpy as np

R = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906')
FIELDS = ('residual_dot_g1','residual_dot_g2','g1_energy','g1_dot_g2','g2_energy','residual_energy_fp64')


def record(path):
    path=Path(path).resolve()
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1<<20),b''): digest.update(chunk)
    return {'path':str(path),'sha256':digest.hexdigest(),'size_bytes':path.stat().st_size}


def verify(rec):
    current=record(rec['path'])
    if current['sha256']!=rec['sha256'] or current['size_bytes']!=rec['size_bytes']:
        raise ValueError(f'artifact changed: {rec["path"]}')


def write(path,payload):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def describe(values):
    return {'mean':float(values.mean()),'descriptive_class_sem':float(values.std(ddof=1)/np.sqrt(len(values))),
            'negative_fraction':float((values<0).mean()),'samples':len(values)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=R/'potential_omitted_witness_v1')
    parser.add_argument('--output-dir',type=Path,default=R/'potential_omitted_witness_summary_v1')
    args=parser.parse_args()
    out=args.output_dir.resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    requests,summaries=[],[]
    steps={}
    inputs=[]
    common=('protocol','mode','prepare_summary','prepare_request','runner','time_grid','time_probability',
            'time_weight_normalizer','validation_metadata','baseline','potential','batch_size','noise_seed','precision')
    for shard in range(4):
        path=args.input_dir/f'shard{shard}'
        request=json.loads((path/'request.json').read_text())
        summary=json.loads((path/'summary.json').read_text())
        if not summary['complete'] or not summary['old_residual_mse_bitwise_parity']:
            raise ValueError('incomplete or failed source')
        verify(summary['request'])
        if request['shard_index']!=shard or request['num_shards']!=4 or summary['time_indices']!=list(range(shard,100,4)):
            raise ValueError('unexpected time sharding')
        if requests and any(request[key]!=requests[0][key] for key in common):
            raise ValueError('inconsistent witness shard protocol')
        requests.append(request); summaries.append(summary)
        for k,rec in zip(summary['time_indices'],summary['step_artifacts'],strict=True):
            verify(rec)
            if k in steps: raise ValueError('duplicate query time')
            steps[k]=rec
        inputs.append({'request':record(path/'request.json'),'summary':record(path/'summary.json')})
    if sorted(steps)!=list(range(100)): raise ValueError('need all 100 times')
    reference=requests[0]
    for name in ('prepare_summary','prepare_request','validation_metadata'): verify(reference[name])
    with np.load(reference['validation_metadata']['path'],allow_pickle=False) as data:
        ids=data['ids'].copy(); labels=data['labels'].copy()
    grid=np.array(reference['time_grid'],dtype=np.float64)
    p=np.array(reference['time_probability'],dtype=np.float64)
    weights=(grid[:-1]-grid[1:])/grid[:-1]**2
    np.testing.assert_allclose(weights,p*reference['time_weight_normalizer'],rtol=1e-12,atol=1e-12)
    data=np.empty((100,1000,6),dtype=np.float64)
    old_c=np.empty((100,1000,3),dtype=np.float64)
    request={'protocol':'raev2_two_omitted_witness_summary_v1','inputs':inputs,
        'step_artifacts':[steps[k] for k in range(100)],'runner':record(Path(__file__)),
        'statistics':'integrate within each original image across all 100 times, then summarize across fixed images/classes',
        'time_units':'g is clean gradient, test velocity g/t, weights h/t^2',
        'plugin':'b^T H^-1 b uses same-bank estimated b/H; descriptive optimistic plug-in, not a certified population lower bound or deployable coefficient',
        'new_images':0,'new_fid':False,'old_step_artifacts':[]}
    write(out/'request.json',request)
    (out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    by_time=[]
    for k in range(100):
        with np.load(steps[k]['path'],allow_pickle=False) as value:
            if int(value['step'])!=k or float(value['time'])!=grid[k] or not np.array_equal(value['ids'],ids) or not np.array_equal(value['labels'],labels):
                raise ValueError('witness time/sample identity mismatch')
            for j,name in enumerate(FIELDS): data[k,:,j]=value[name]
            r2=value['residual_mse_fp32_parity'].copy()
        old_path=Path(reference['old_validation_root'])/f'shard{k%3}'/f'step{k:03d}.npz'
        request['old_step_artifacts'].append(record(old_path))
        with np.load(old_path,allow_pickle=False) as old:
            if not np.array_equal(old['ids'],ids) or not np.array_equal(old['labels'],labels) or not np.array_equal(old['residual_mse'],r2):
                raise ValueError('old R2 parity mismatch during aggregation')
            for j,key in enumerate(('residual_mse','correction_energy','residual_dot_correction')):
                old_c[k,:,j]=old[key]
        by_time.append({'step':k,'time':grid[k],'weight':weights[k],
                        **{name:describe(data[k,:,j]) for j,name in enumerate(FIELDS)}})
    if not np.isfinite(data).all() or not np.isfinite(old_c).all(): raise ValueError('nonfinite statistics')
    # An image, including all correlated times, is the statistical unit.
    integrated=np.einsum('k,knj->nj',weights,data)
    old_integrated=np.einsum('k,knj->nj',weights,old_c)
    b_by_image=integrated[:,:2]
    gram_by_image=np.empty((1000,2,2),dtype=np.float64)
    gram_by_image[:,0,0]=integrated[:,2]
    gram_by_image[:,0,1]=gram_by_image[:,1,0]=integrated[:,3]
    gram_by_image[:,1,1]=integrated[:,4]
    b=b_by_image.mean(0); gram=gram_by_image.mean(0)
    eigen=np.linalg.eigvalsh(gram)
    if eigen[0]<=0: raise ValueError('test Gram not positive definite; no implicit ridge/truncation')
    alpha=np.linalg.solve(gram,b)
    plugin=float(b@alpha)
    old_c2=old_integrated[:,1]; old_rc=old_integrated[:,2]
    own_witness=old_rc-old_c2
    own_gain=2*old_rc-old_c2
    own_additional=float(own_witness.mean()**2/old_c2.mean())
    baseline=float(old_integrated[:,0].mean())
    # Fixed formula, no bootstrap, per-time scale fitting, or deployment.
    np.savez(out/'paired_by_image.npz',ids=ids,labels=labels,integrated_b=b_by_image,
        integrated_gram=gram_by_image,old_residual_energy=old_integrated[:,0],old_c2=old_c2,old_rc=old_rc)
    write(out/'by_time.json',by_time)
    write(out/'request.json',request)
    summary={'protocol':request['protocol'],'complete':True,'samples':1000,'times':100,
        'witness_g1':describe(b_by_image[:,0]),'witness_g2':describe(b_by_image[:,1]),
        'covariance_of_image_integrals':np.cov(b_by_image,rowvar=False,ddof=1).tolist(),
        'gram':gram.tolist(),'gram_eigenvalues':eigen.tolist(),
        'gram_correlation':float(gram[0,1]/np.sqrt(gram[0,0]*gram[1,1])),
        'same_bank_critic_plugin':{'energy':plugin,'coefficients_for_diagnostic_only':alpha.tolist(),
            'fraction_of_old_residual_energy':plugin/baseline,
            'fraction_of_old_achieved_gain':plugin/float(own_gain.mean()),
            'boundary':'Optimistically estimated finite test-space energy; not a population certificate, fitted guidance or FID prediction.'},
        'existing_potential_competitor':{'own_witness':describe(own_witness),'achieved_gain':describe(own_gain),
            'same_bank_global_scale_plugin':float(old_rc.mean()/old_c2.mean()),
            'same_bank_additional_energy_plugin':own_additional,
            'additional_fraction_of_achieved_gain':own_additional/float(own_gain.mean())},
        'old_baseline_velocity_coupling_energy_per_dimension':baseline,
        'precision_orthogonality':[item['orthogonality'] for item in summaries],
        'old_R2_bitwise_parity_all_100000':True,
        'cost':{'stage2_forward_calls':sum(s['stage2_forward_calls'] for s in summaries),
            'stage2_sample_forwards':sum(s['stage2_sample_forwards'] for s in summaries),
            'potential_input_gradient_calls':sum(s['potential_input_gradient_calls'] for s in summaries),
            'potential_input_gradient_sample_forwards':sum(s['potential_input_gradient_sample_forwards'] for s in summaries),
            'sum_worker_phase_wall_seconds':sum(s['phase_wall_seconds'] for s in summaries),
            'sum_worker_script_wall_seconds':sum(s['cost']['wall_seconds_from_script_start'] for s in summaries),
            'max_worker_gpu_allocated_bytes':max(s['peak_gpu_allocated_bytes'] for s in summaries),
            'aggregation_wall_seconds':time.perf_counter()-START,'aggregation_cpu_seconds':time.process_time()-CPU_START,
            'aggregation_max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'worker_wall_boundary':'sum of parallel worker windows, not phase critical path or pure GPU kernel time'},
        'request':record(out/'request.json'),'paired_by_image':record(out/'paired_by_image.npz'),
        'new_fid':False,'goal_achieved':False}
    write(out/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
