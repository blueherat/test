"""SiT counterpart of the frozen PFR conditional-erasure intervention."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torchdiffeq import odeint
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.run_imagenet100_sit_internal_early_two_segment_gamma_sweep import load_repo_modules,runtime_paths
from experiments.imagenet100_sit_multiscale_models import evaluate_internal_head_only
from experiments.internal_guidance_path_extrapolation import project_to_forward_ray
from experiments.train_imagenet100_sit_flow import sha256_file


@torch.inference_mode()
def run(a):
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    data=Path('/home/zhoushunyu/data/eqvae/imagenet_sit_flow')
    paths=runtime_paths(ROOT,data,Path('/data/shared/envs/adm-fid/bin/python'));m=load_repo_modules(ROOT)
    sit,metadata=m['load_official_sit_module'](m['DEFAULT_OFFICIAL_SIT_REPO'],verify_source=True)
    strong,semantics,_=m['load_sit_field_model'](checkpoint_path=paths['strong'],weights='ema',sit_module=sit,
        source_metadata=metadata,device=torch.device(a.device))
    assert semantics.prediction_target=='velocity'
    head=m['load_internal_head_for_source'](checkpoint_path=paths['depth4'],name='depth4_v',head_weights='ema',
        model=strong,sit_module=sit,source_checkpoint_path=paths['strong'],source_metadata=metadata,device=torch.device(a.device))
    from diffusers.models import AutoencoderKL
    vae=AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse',local_files_only=True).to(a.device).eval().requires_grad_(False)
    gen=torch.Generator(device='cpu').manual_seed(a.seed)
    yall=(torch.arange(a.samples)%100)[torch.randperm(a.samples,generator=gen)]
    noise=torch.randn(a.samples,4,32,32,generator=gen)
    classes=json.loads((data/'imagenet100_cmc/manifest.json').read_text())['classes']
    mapping=torch.tensor([c['original_imagenet_label'] for c in sorted(classes,key=lambda c:c['label'])])
    global_labels=mapping[yall]
    np.savez(a.output/'inputs.npz',labels=yall.numpy(),global_labels=global_labels.numpy(),noise=noise.numpy())
    cut=a.cut_time
    images={k:[] for k in ['conditional','unconditional']};middle=[];calls={'full':0,'prefix':0}
    started=time.perf_counter()
    for start in range(0,a.samples,8):
        z=noise[start:start+8].to(a.device);y=yall[start:start+8].to(a.device)
        def prefix(t,state):
            ts=t.expand(len(state));full,heads,_=m['evaluate_source_with_heads'](strong,state,ts,y,heads={'depth4_v':head})
            calls['full']+=1;weak=heads['depth4_v']
            gamma=.6 if float(t)<.25 else (.7 if float(t)<.5 else 0.)
            guided=full+gamma*(full-weak)
            if a.mode=='ig' or gamma==0:return guided
            h=min(1/32,max(0.,.5-float(t)))
            q=state+h*project_to_forward_ray((1+gamma)*(full-weak),guided).parallel
            future=evaluate_internal_head_only(strong,q,torch.full_like(ts,float(t)+h),y,spec=head)
            calls['prefix']+=1
            return guided+(1+gamma)*(weak-future)
        z=odeint(prefix,z,torch.tensor([0.,cut],device=a.device),method='dopri5',atol=1e-6,rtol=1e-3)[-1]
        middle.append(z.cpu())
        for kind in images:
            context=y if kind=='conditional' else torch.full_like(y,100)
            def suffix(t,state):
                full,_,_=m['evaluate_source_with_heads'](strong,state,t.expand(len(state)),context,heads={})
                calls['full']+=1;return full
            endpoint=odeint(suffix,z.clone(),torch.tensor([cut,1.],device=a.device),method='dopri5',atol=1e-6,rtol=1e-3)[-1]
            if not torch.isfinite(endpoint).all():raise FloatingPointError(kind)
            decoded=m['decode_latents_in_chunks'](vae,endpoint,scaling_factor=.18215,chunk_size=4)
            images[kind].append(m['official_pixel_quantization'](decoded))
        print(json.dumps({'done':min(start+8,a.samples),'mode':a.mode,'seconds':time.perf_counter()-started}),flush=True)
    torch.save(torch.cat(middle),a.output/'cut_latents.pt')
    del strong,head,vae;gc.collect();torch.cuda.empty_cache()
    from torchvision.models import convnext_tiny,ConvNeXt_Tiny_Weights
    weights_path=Path('/home/zhoushunyu/.cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth')
    classifier=convnext_tiny(weights=None).to(a.device).eval().requires_grad_(False)
    classifier.load_state_dict(torch.load(weights_path,map_location='cpu',weights_only=True),strict=True)
    transform=ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms();scores={};summary={}
    for kind,parts in images.items():
        pixels=np.concatenate(parts);np.savez(a.output/(kind+'.npz'),pixels)
        logits=[]
        for start in range(0,len(pixels),16):
            x=torch.from_numpy(pixels[start:start+16]).permute(0,3,1,2).to(a.device).float()/255
            logits.append(classifier(transform(x)).float().cpu())
        logit=torch.cat(logits);target=logit.softmax(1)[torch.arange(a.samples),global_labels]
        top1=logit.argmax(1).eq(global_labels);top5=logit.topk(5,1).indices.eq(global_labels[:,None]).any(1)
        scores[kind+'_target_prob']=target.numpy();scores[kind+'_top1']=top1.numpy()
        scores[kind+'_top5']=top5.numpy();scores[kind+'_logits']=logit.numpy()
        summary[kind]={'target_prob':float(target.mean()),'top1':float(top1.float().mean()),'top5':float(top5.float().mean())}
    np.savez(a.output/'scores.npz',labels=global_labels.numpy(),**scores)
    result={'complete':True,'mode':a.mode,'samples':a.samples,'seed':a.seed,'cut_time':cut,
        'strong_sha256':sha256_file(paths['strong']),'weak_sha256':sha256_file(paths['depth4']),
        'classifier_sha256':sha256_file(weights_path),'calls':calls,'seconds':time.perf_counter()-started,'results':summary}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['ig','pfr'],required=True);p.add_argument('--samples',type=int,default=128)
    p.add_argument('--seed',type=int,default=202609401)
    p.add_argument('--cut-time',type=float,default=1-.7975830435752869)
    p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
