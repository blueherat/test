"""Fixed twenty-bank exploratory evaluation, with freshly extracted real features."""
from __future__ import annotations
import csv
import json
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.raev2_training_core import file_sha256 as sha
from experiments.run_official_sit_native_sde_20260909 import array_sha,state_sha,write_json
from experiments.compute_raev2_predicted_clean_precision_recall import manifold_radii,precision_recall

RAW=Path('/home/zhoushunyu/data/eqvae/experiments')
DATA=RAW/'official_sit_xl_multimetrics_fixed_20260909'
OUT=ROOT/'docs/data/guidance_goal_20260909'
REAL=Path('/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz')
EVAL=Path('/home/zhoushunyu/data/eqvae/external_sources/nanogen-evals/fd_evaluator')
WEIGHTS=Path('/home/zhoushunyu/.cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth')


def offdiag_mean(k):
    return (k.sum()-k.diagonal().sum())/(len(k)*(len(k)-1))


@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert json.loads((RAW/'official_sit_pfr_upper_20260909/status.json').read_text())['phase']=='complete'
    DATA.mkdir(parents=True,exist_ok=False)
    with (OUT/'official_sit_pfr_extended.csv').open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==18 and json.loads((OUT/'official_sit_pfr_extended_audit.json').read_text())['passed']
    request=json.loads((RAW/'official_sit_pfr_symmetric_20260909/request.json').read_text())
    directories={arm:Path(r['directory']) for arm,r in request['references'].items()}
    directories.update({arm:RAW/'official_sit_pfr_symmetric_20260909'/arm for arm in request['arms']})
    upper=json.loads((RAW/'official_sit_pfr_upper_20260909/request.json').read_text())
    directories.update({arm:RAW/'official_sit_pfr_upper_20260909'/arm for arm in upper['arms']})
    for row in rows: row['directory']=str(directories[row['arm']]);row['vae']='mse';row['sampler']='euler'
    with (OUT/'official_sit_native_sde.csv').open() as f:
        for row in csv.DictReader(f):
            row.update(directory=str(RAW/'official_sit_native_sde_20260909'/row['vae']),method='native',scale=1.4,sampler='sde')
            rows.append(row)
    assert len(rows)==20
    inputs=[]
    for row in rows:
        directory=Path(row['directory']);feature,=(directory/'features').glob('*.features.pt')
        assert sha(directory/'samples.npz')==row['pixel_sha256']
        inputs.append(dict(arm=row['arm'],pixels=str(directory/'samples.npz'),pixel_sha256=row['pixel_sha256'],
                           features=str(feature),feature_sha256=sha(feature)))
    files=[Path(__file__),ROOT/'docs/OFFICIAL_SIT_XL_MULTIMETRIC_PROTOCOL_20260909_ZH.md',
        ROOT/'docs/OFFICIAL_SIT_XL_MULTIMETRIC_CUDNN_AMENDMENT_20260909_ZH.md',
        ROOT/'experiments/compute_raev2_predicted_clean_precision_recall.py',
        EVAL/'fd_evaluator/extractors.py',EVAL/'fd_evaluator/catalogue.yaml',
        OUT/'official_sit_pfr_extended.csv',OUT/'official_sit_pfr_extended_audit.json',
        OUT/'official_sit_native_sde.csv',OUT/'official_sit_native_sde_audit.json',WEIGHTS,
        Path('/home/zhoushunyu/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth')]
    req=dict(inputs=inputs,real_file=str(REAL),real_file_sha256=sha(REAL),
             supersedes_failed_run=str(RAW/'official_sit_xl_multimetrics_20260909'),
             failed_request_sha256=sha(RAW/'official_sit_xl_multimetrics_20260909/request.json'),
             sources={str(p):sha(p) for p in files},neighborhood=3,generated_samples=1000,reference_samples=10000,
             sampling_bank_role='reused_1k_discovery',matmul_tf32=False,inception_cudnn_tf32=True,
             classifier_cudnn_tf32=False,feature_dtype='float32',distance_kernel_dtype='float64',torch_version=torch.__version__)
    write_json(DATA/'request.json',req)
    sys.path.insert(0,str(EVAL))
    from fd_evaluator.extractors import InceptionExtractor,ExtractorSpec
    spec=yaml.safe_load((EVAL/'fd_evaluator/catalogue.yaml').read_text())['extractors']['inception']
    extractor=InceptionExtractor(ExtractorSpec(name='inception',**spec)).cuda().eval()
    with np.load(REAL) as f: real_pixels=f['arr_0']
    assert real_pixels.shape==(10000,256,256,3) and real_pixels.dtype==np.uint8
    def extract(pixels):
        x=torch.from_numpy(np.ascontiguousarray(pixels)).permute(0,3,1,2).contiguous().cuda()
        # Original nanogen subprocesses keep the torch default: convolution
        # TF32 on, matmul TF32 off.  Match those cached feature calculations.
        with torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=False,allow_tf32=True):
            return extractor(x).cpu()
    write_json(DATA/'status.json',dict(phase='real_features',images=0))
    real_parts=[];began=time.perf_counter()
    for start in range(0,len(real_pixels),32):
        real_parts.append(extract(real_pixels[start:start+32]))
        if (start+32)%1024==0:
            print(json.dumps(dict(phase='real_features',images=start+32,seconds=time.perf_counter()-began)),flush=True)
    real_cpu=torch.cat(real_parts)
    assert real_cpu.shape==(10000,2048) and torch.isfinite(real_cpu).all()
    torch.save(real_cpu,DATA/'real_features.pt')
    real_provenance=dict(pixel_array_sha256=array_sha(real_pixels),feature_sha256=sha(DATA/'real_features.pt'),
                         extractor_state_sha256=state_sha(extractor),seconds=time.perf_counter()-began)
    write_json(DATA/'real_features.json',real_provenance)
    del real_pixels,real_parts
    reference=real_cpu.cuda().double();radii=manifold_radii(reference,neighborhood=3,batch_size=250)
    torch.save(radii.cpu(),DATA/'real_radii.pt')
    real_kernel=(reference@reference.T/2048+1).pow(3)
    real_term=offdiag_mean(real_kernel);del real_kernel
    from torchvision.models import convnext_tiny,ConvNeXt_Tiny_Weights
    classifier=convnext_tiny(weights=None).cuda().eval().requires_grad_(False)
    classifier.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True),strict=True)
    classifier_sha=state_sha(classifier)
    transform=ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
    outputs=[];artifacts={}
    labels=torch.arange(1000)
    for row,entry in zip(rows,inputs):
        arm=row['arm'];write_json(DATA/'status.json',dict(phase='generated_metrics',arm=arm))
        assert sha(Path(entry['features']))==entry['feature_sha256']
        feats=torch.load(entry['features'],map_location='cpu',weights_only=True)
        assert feats.shape==(1000,2048) and torch.isfinite(feats).all()
        with np.load(entry['pixels']) as f: pixels=f['arr_0']
        assert pixels.shape==(1000,256,256,3) and pixels.dtype==np.uint8
        feature_error=(extract(pixels[:32])-feats[:32]).abs().max().item()
        assert feature_error<2e-4,(arm,feature_error)
        gen=feats.cuda().double()
        p,r,g_radii=precision_recall(gen,reference,radii,neighborhood=3,batch_size=250)
        kxx=(gen@gen.T/2048+1).pow(3)
        kxy=(gen@reference.T/2048+1).pow(3)
        kid=float(offdiag_mean(kxx)+real_term-2*kxy.mean())
        del gen,kxx,kxy
        logits=[]
        for start in range(0,1000,32):
            x=torch.from_numpy(pixels[start:start+32]).permute(0,3,1,2).cuda().float()/255
            logits.append(classifier(transform(x)).float().cpu())
        scores=torch.cat(logits);assert scores.shape==(1000,1000) and torch.isfinite(scores).all()
        probability=scores.softmax(1)[torch.arange(1000),labels]
        top1=scores.argmax(1).eq(labels);top5=scores.topk(5,dim=1).indices.eq(labels[:,None]).any(1)
        path=DATA/(arm+'.pt');torch.save(dict(logits=scores,labels=labels,target_probability=probability,
            top1=top1,top5=top5,generated_radii=g_radii.cpu()),path)
        artifacts[arm]=sha(path)
        result=dict(arm=arm,method=row['method'],scale=float(row['scale']),sampler=row['sampler'],vae=row['vae'],
            fid=float(row['fid']),inception_score=float(row['inception_score']),precision=p,recall=r,kid_u_statistic=kid,
            target_top1=float(top1.double().mean()),target_top5=float(top5.double().mean()),
            target_probability=float(probability.double().mean()),feature_reextract_max_error=feature_error)
        outputs.append(result);write_json(DATA/'results.json',outputs)
        print(json.dumps(result),flush=True)
    for p,d in req['sources'].items(): assert sha(Path(p))==d,p
    assert sha(REAL)==req['real_file_sha256']
    audit=dict(passed=True,research_goal_achieved=False,request_sha256=sha(DATA/'request.json'),
        real_provenance=real_provenance,classifier_state_sha256=classifier_sha,artifacts=artifacts,
        generated_features_reextract_max_error=max(r['feature_reextract_max_error'] for r in outputs),
        limitation='Twenty reused 1K banks, sparse unequal-size PR neighborhoods and fixed-class KID; no independent confirmation or semantic-quality equivalence.')
    csv_path=OUT/'official_sit_xl_multimetrics.csv';audit_path=OUT/'official_sit_xl_multimetrics_audit.json'
    assert not csv_path.exists() and not audit_path.exists()
    with csv_path.open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(outputs[0]));w.writeheader();w.writerows(outputs)
    write_json(audit_path,audit);write_json(DATA/'status.json',dict(phase='complete',research_goal_achieved=False))
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__': main()
