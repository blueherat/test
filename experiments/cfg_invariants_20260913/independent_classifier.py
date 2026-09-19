"""Check clean-confidence feedback with a classifier that did not guide it.

Uses only cached ResNet18 weights and existing paired generated images.
This is an additional semantic proxy, not a perceptual quality ground truth.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights

from experiments.lifting_scale_sweep_20260909 import atomic, sha

ROOT=Path('/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913')
WEIGHTS=Path('/home/zhoushunyu/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')
MAPPING=Path('/home/zhoushunyu/data/eqvae/imagenet_sit_flow/imagenet100_cmc/manifest.json')
ARMS=('cfg_base','cfg_half','clean_evidence_proxy','evidence_time_mean')


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait-seconds',type=int,default=900)
    args=parser.parse_args()
    torch.set_num_threads(4)
    out=ROOT/'independent_classifier'
    out.mkdir(parents=True,exist_ok=True)
    model=resnet18(weights=None).eval().requires_grad_(False)
    model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True),strict=True)
    transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
    classes=json.loads(MAPPING.read_text())['classes']
    mapping=np.array([next(r['original_imagenet_label'] for r in classes if r['label']==i) for i in range(100)])
    started=time.time()
    completed={}
    while len(completed)<len(ARMS):
        for arm in ARMS:
            if arm in completed:continue
            folder=ROOT/'images'/arm
            if not (folder/'summary.json').exists():continue
            summary=json.loads((folder/'summary.json').read_text())
            if not summary.get('complete'):continue
            images=np.load(folder/'samples.npz')['arr_0']
            labels=np.load(folder/'diagnostics.npz')['labels']
            assert len(images)==len(labels)==400
            targets=mapping[labels]
            begin=time.time()
            probabilities=[]
            for a in range(0,len(images),32):
                batch=torch.from_numpy(images[a:a+32].copy()).permute(0,3,1,2)
                probabilities.append(model(transform(batch)).softmax(-1).numpy())
            probabilities=np.concatenate(probabilities)
            ranks=np.argsort(-probabilities,axis=1)
            target=probabilities[np.arange(len(labels)),targets]
            top1=ranks[:,0]==targets
            top5=(ranks[:,:5]==targets[:,None]).any(1)
            np.savez(out/f'{arm}.npz',target_probability=target,top1=top1,top5=top5,labels=labels)
            completed[arm]=dict(n=len(labels),target_probability=float(target.mean()),
                top1=float(top1.mean()),top5=float(top5.mean()),seconds=time.time()-begin,
                input_sha256=sha(folder/'samples.npz'))
            print(json.dumps(dict(arm=arm,**completed[arm])),flush=True)
            atomic(out/'progress.json',dict(complete=len(completed)==len(ARMS),arms=completed))
        if len(completed)<len(ARMS):
            if time.time()-started>args.wait_seconds:
                raise TimeoutError('Required generated samples are not all ready; completed evaluations are preserved.')
            time.sleep(15)
    comparisons={}
    rng=np.random.default_rng(2026091333)
    indices=rng.integers(0,400,(2000,400))
    for control in ('cfg_base','evidence_time_mean'):
        a=dict(np.load(out/'clean_evidence_proxy.npz'))
        b=dict(np.load(out/f'{control}.npz'))
        np.testing.assert_array_equal(a['labels'],b['labels'])
        rows={}
        for metric in ('target_probability','top1','top5'):
            difference=a[metric].astype(float)-b[metric].astype(float)
            boot=difference[indices].mean(1)
            rows[metric]=dict(mean=float(difference.mean()),bootstrap_95=np.quantile(boot,[.025,.975]).tolist())
        comparisons[control]=rows
    atomic(out/'summary.json',dict(complete=True,arms=completed,comparisons=comparisons,
        classifier='ResNet18 ImageNet1K, not used for guidance; semantic proxy only',
        cpu_only=True,weights_sha256=sha(WEIGHTS),mapping_sha256=sha(MAPPING),
        script_sha256=sha(Path(__file__)),elapsed_seconds=time.time()-started))
    print(json.dumps(comparisons),flush=True)


if __name__=='__main__':
    main()
