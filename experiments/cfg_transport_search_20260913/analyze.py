"""Source-checked result tables and independent endpoint classification.

This reader deliberately whitelists metric fields: the ADM JSON's `samples`
is a path, whereas the sampling summary's `samples` is a count.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from experiments.guidance_pasted_20260912 import common as c

ROOT=c.EXPS/'cfg_transport_search_20260913'
WEIGHTS=Path('/home/zhoushunyu/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')
MAPPING=c.EXPS.parent/'imagenet_sit_flow/imagenet100_cmc/manifest.json'


def collect(phase, require_complete=False):
    root=ROOT/phase; request=c.read(root/'request.json'); rows=[]
    for config in request['configs']:
        folder=root/config['arm']
        if not (folder/'fid.json').exists():continue
        summary=c.read(folder/'summary.json'); metrics=c.read(folder/'fid.json')
        assert summary['complete'] and metrics['sample_count']==summary['samples']==request['samples']
        assert Path(metrics['samples'])==folder/'samples.npz'
        assert Path(metrics['reference'])==Path(request['reference'])
        assert c.sha(folder/'samples.npz')==summary['samples_sha256']
        assert summary['request_sha256']==c.sha(root/'request.json')
        row=dict(config,primary_samples=summary['samples'])
        for key in ['fid','sfid','inception_score']:row[key]=metrics[key]
        for key in ['seconds','full_calls_per_output','prefix_calls_per_output','saturation_fraction','latent_rms']:
            row[key]=summary[key]
        if (folder/'independent_classifier.json').exists():
            info=c.read(folder/'independent_classifier.json')
            assert info['samples_sha256']==summary['samples_sha256']
            row.update({key:info[key] for key in ['target_top1','target_top5','target_probability']})
        rows.append(row)
    if require_complete:assert len(rows)==len(request['configs']),(len(rows),len(request['configs']))
    fields=['arm','kind','alpha','eta','rho','apg_alpha','ctrl_alpha','steps','primary_samples','fid','sfid','inception_score',
            'target_top1','target_top5','target_probability','seconds','full_calls_per_output',
            'prefix_calls_per_output','saturation_fraction','latent_rms']
    with (root/'verified_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    winners={}
    for row in rows:
        family=f"{row['kind']}_{row.get('steps',64)}"
        if family not in winners or row['fid']<winners[family]['fid']: winners[family]=row
    c.atomic(root/'verified_results.json',dict(rows=rows,winners=winners,completed=len(rows),
             planned=len(request['configs']),sample_count_key='primary_samples',
             stage_interpretation='parameter search, not independent confirmation'))
    print(json.dumps(dict(phase=phase,completed=len(rows),winners=winners)),flush=True)
    return rows


def classify(phase,device):
    import torch
    from torchvision.models import resnet18,ResNet18_Weights
    torch.set_num_threads(2)
    root=ROOT/phase;request=c.read(root/'request.json')
    labels=np.load(root/'inputs.npz')['labels']
    classes=c.read(MAPPING)['classes']
    mapping=np.array([next(r['original_imagenet_label'] for r in classes if r['label']==i) for i in range(100)])
    targets=mapping[labels]
    model=resnet18(weights=None).eval().requires_grad_(False)
    model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True),strict=True)
    model=model.to(device); transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
    with torch.inference_mode():
        for config in request['configs']:
            folder=root/config['arm']
            if not (folder/'summary.json').exists() or (folder/'independent_classifier.json').exists():continue
            summary=c.read(folder/'summary.json')
            assert c.sha(folder/'samples.npz')==summary['samples_sha256']
            pixels=np.load(folder/'samples.npz')['arr_0']; p=[];begin=time.perf_counter()
            assert len(pixels)==len(labels)==request['samples']
            for start in range(0,len(pixels),32):
                batch=torch.from_numpy(pixels[start:start+32].copy()).permute(0,3,1,2)
                p.append(model(transform(batch).to(device)).softmax(-1).cpu().numpy())
            probs=np.concatenate(p);order=np.argsort(-probs,axis=1)
            pt=probs[np.arange(len(labels)),targets]
            top1=order[:,0]==targets;top5=(order[:,:5]==targets[:,None]).any(1)
            np.savez(folder/'independent_classifier.npz',labels=labels,target_probability=pt,top1=top1,top5=top5)
            c.atomic(folder/'independent_classifier.json',dict(primary_samples=len(labels),target_top1=float(top1.mean()),
                target_top5=float(top5.mean()),target_probability=float(pt.mean()),seconds=time.perf_counter()-begin,
                samples_sha256=summary['samples_sha256'],weights_sha256=c.sha(WEIGHTS),mapping_sha256=c.sha(MAPPING),
                interpretation='ResNet18 independent clean endpoint class proxy, not ground-truth perceptual quality'))
            print('classified',config['arm'],float(top1.mean()),flush=True)


def plot(phase):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=collect(phase,require_complete=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    groups={}
    for row in rows:groups.setdefault((row['kind'],row.get('steps',64)),[]).append(row)
    for (kind,steps),items in groups.items():
        items=sorted(items,key=lambda x:x['alpha']);label=f'{kind.upper()} / {steps} steps'
        axes[0].plot([x['alpha'] for x in items],[x['fid'] for x in items],'-o',label=label,markersize=4)
        axes[1].plot([x['fid'] for x in items],[x['inception_score'] for x in items],'-o',markersize=4)
        axes[2].scatter([x['full_calls_per_output'] for x in items],[x['fid'] for x in items],label=label,s=24)
    axes[0].set(xlabel='Extra guidance alpha',ylabel='ADM FID (lower is better)',title='Paired parameter search')
    axes[1].set(xlabel='FID',ylabel='Inception Score',title='Quality / score trade-off')
    axes[2].set(xlabel='Single-branch evaluations / image',ylabel='FID',title='Reported inference budget')
    axes[0].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle(f"SiT-S/2 · {rows[0]['primary_samples']} paired images per setting · search only",fontsize=11)
    fig.tight_layout();fig.savefig(ROOT/phase/'baseline_curves.png',dpi=160);fig.savefig(ROOT/phase/'baseline_curves.pdf')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['collect','classify','plot'])
    parser.add_argument('--phase',required=True);parser.add_argument('--device',default='cuda')
    parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
    if args.action=='collect':collect(args.phase,args.require_complete)
    elif args.action=='classify':classify(args.phase,args.device)
    else:plot(args.phase)


if __name__=='__main__':main()
