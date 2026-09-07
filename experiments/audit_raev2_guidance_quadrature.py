#!/usr/bin/env python3
"""Verify completed quadrature images, paired inputs, identities and FID."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.audit_raev2_prefix_ratio64k_quality import independent_fid
from experiments.audit_raev2_screen_fid_20260907 import load_moments, resolve
from experiments.run_raev2_guidance_quadrature import sha


def read(path):
    return json.loads(Path(path).read_text())


def audit_pixels(folder, mode, count, summary, shard_count):
    sample = folder/mode/'samples.npz'
    assert sha(sample) == summary['sample_sha256']
    with np.load(sample) as data:
        merged = data['arr_0']
    assert merged.shape == (count,256,256,3) and merged.dtype == np.uint8
    ids, records = [], []
    for rank in range(shard_count):
        sub = folder/f'shard{rank}'/mode
        part = read(sub/'summary.json')
        assert part['complete'] and sha(sub/'samples.npz') == part['sample_sha256']
        with np.load(sub/'samples.npz') as data:
            assert np.array_equal(merged[data['ids']],data['arr_0'])
            ids.extend(data['ids'].tolist())
        records.extend(part['initial_noise'])
    assert sorted(ids) == list(range(count))
    records.sort(key=lambda row: row['batch'])
    assert [row['batch'] for row in records] == list(range(count//8))
    assert hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest() == summary['paired_noise_labels_sha256']
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--control-folder',type=Path,default=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_20260907/weak_confirm5k'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    began=time.perf_counter()
    run,control=a.folder.resolve(),a.control_folder.resolve()
    execution=read(run/'execution.json')
    assert execution['complete'] and execution['sampling_complete']
    for source,digest in execution['sources'].items():
        frozen=run/'frozen_source'/Path(source).relative_to(ROOT)
        assert sha(frozen)==digest
    metrics=read(run/'metrics.json')
    baseline=next(row for row in read(control/'metrics.json') if row['branch']=='official')
    bs=read(control/'official/summary.json')
    count=execution['args']['samples']
    assert count==5000 and execution['args']['seed']==read(control/'shard0/request.json')['seed']
    baseline_records=audit_pixels(control,'official',count,bs,4)
    assert sha(control/'official/samples.npz')==baseline['sample_sha256']
    ref_path=Path(resolve('imagenet_256_fid_stats'))
    assert sha(ref_path)=='925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac'
    mean_ref,cov_ref=load_moments(str(ref_path))
    eig,vec=np.linalg.eigh(cov_ref)
    assert eig.min()>-1e-9
    root_ref=(vec*np.sqrt(np.maximum(eig,0)))@vec.T
    def verify_metric(folder,row):
        assert row['evaluator_commit']=='19dfb4c2705333eb8b97e454fb354d47d1fe135b'
        assert row['fid_reference']=='imagenet_256_fid_stats'
        path=folder/'official_feature_cache'/f"{row['branch']}-{row['sample_sha256'][:16]}-inception.features.pt"
        features=torch.load(path,map_location='cpu',weights_only=True).double().numpy()
        assert features.shape==(count,2048) and np.isfinite(features).all()
        result=independent_fid(features,mean_ref,cov_ref,root_ref)
        assert abs(result['fid']-row['fid'])<1e-5
        return {'independent_fid':result,'feature_sha256':sha(path)}
    baseline_verified=verify_metric(control,baseline)
    base_cost=bs['trajectory_seconds_sum']+bs['decode_seconds_sum']
    rows=[]
    for metric in metrics:
        mode=metric['branch']
        summary=read(run/mode/'summary.json')
        assert summary['complete'] and summary['samples']==count and summary['sample_model_calls']==count*100
        assert summary['sample_sha256']==metric['sample_sha256']
        assert audit_pixels(run,mode,count,summary,len(execution['args']['gpus']))==baseline_records
        for rank in range(len(execution['args']['gpus'])):
            new,old=read(run/f'shard{rank}/request.json'),read(control/'shard0/request.json')
            for key in ('checkpoint_sha256','config_sha256','decoder_sha256','stats_sha256','state_key','time_grid','batch','torch_version'):
                assert new[key]==old[key],key
            assert new['steps']==100 and new['precision']=='native BF16 heads/mix, FP32 history and Euler, TF32 on'
        cost=summary['trajectory_seconds_sum']+summary['decode_seconds_sum']
        result={**metric,**verify_metric(run,metric),'mode':mode,'samples':count,'seed':summary['seed'],
            'relative_improvement_percent':100*(1-metric['fid']/baseline['fid']),
            'discovery_point_meets_3_percent':metric['fid']<=.97*baseline['fid'],
            'inference_seconds':cost,'cost_ratio':cost/base_cost,'sample_model_calls':summary['sample_model_calls']}
        rows.append(result)
        print(mode,result['fid'],result['relative_improvement_percent'],flush=True)
    result={'complete':True,'goal_achieved':False,
        'reason_goal_not_yet_certified':'Discovery bank; independent confirmation and mechanism assessment remain.',
        'baseline':{**baseline,**baseline_verified,'inference_seconds':base_cost},'rows':rows,
        'paired_all_batch_noise_labels_verified':True,'all_merged_pixels_and_shard_hashes_verified':True,
        'frozen_sources_verified':True,'reference_sha256':sha(ref_path),'execution_sha256':sha(run/'execution.json'),
        'audit_source_sha256':sha(Path(__file__).resolve()),'elapsed_seconds':time.perf_counter()-began}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
