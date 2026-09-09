"""Paired prefix intervention and conditional-erasure suffixes for PFR."""
import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.sample_raev2_pfr_retiming import (
    DEFAULT_CONFIG, DEFAULT_CHECKPOINT, load_config, instantiate_from_config,
    install_raev2_decoder_config_compat, shifted_time_grid, tensor_sha256,
    clean_to_velocity, evaluate_base_head_only, dataward_future_time, pfr_velocity,
    file_sha256)


@torch.inference_mode()
def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    install_raev2_decoder_config_compat();config=load_config(DEFAULT_CONFIG)
    decoder=instantiate_from_config(config.stage_1).to(a.device).eval().requires_grad_(False)
    del decoder.encoder;torch.cuda.empty_cache()
    model=instantiate_from_config(config.stage_2).to(a.device).eval().requires_grad_(False)
    ck=torch.load(DEFAULT_CHECKPOINT,map_location='cpu',weights_only=False,mmap=True)
    model.load_state_dict(ck['ema'],strict=True);step=int(ck['step']);del ck;gc.collect()
    shift=math.sqrt(config.misc.time_dist_shift_dim/config.misc.time_dist_shift_base)
    grid=shifted_time_grid(100,shift,torch.device(a.device))
    cut=int(torch.nonzero(grid<=a.cut_time)[0]);floor=float(config.transport.t_eps)
    gen=torch.Generator(device='cpu').manual_seed(a.seed)
    if a.imagenet100:
        manifest=Path('/home/zhoushunyu/data/eqvae/imagenet_sit_flow/imagenet100_cmc/manifest.json')
        classes=sorted(json.loads(manifest.read_text())['classes'],key=lambda c:c['label'])
        mapping=torch.tensor([c['original_imagenet_label'] for c in classes])
        labels=mapping[(torch.arange(a.samples)%100)[torch.randperm(a.samples,generator=gen)]]
    else:
        labels=torch.randperm(1000,generator=gen)[:a.samples]
    noise=torch.randn(a.samples,*config.misc.latent_size,generator=gen)
    np.savez(a.output/'inputs.npz',labels=labels.numpy(),noise=noise.numpy())
    images={k:[] for k in ['conditional','unconditional']};middle=[];counter={'full':0,'prefix':0}
    started=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for start in range(0,a.samples,4):
            z=noise[start:start+4].to(a.device);y=labels[start:start+4].to(a.device)
            for i in range(cut):
                t=grid[i].expand(len(z));kw={'context':y,'attn_mask':None}
                full,base=model(z,t,**kw);counter['full']+=1
                vf=clean_to_velocity(full,z,t,denominator_floor=floor)
                vb=clean_to_velocity(base,z,t,denominator_floor=floor)
                vq=vb
                if a.mode=='pfr':
                    tau=dataward_future_time(float(grid[i]),1/32,coordinate='raw_time',minimum_time=.1)
                    tq=torch.full_like(t,tau)
                    q=evaluate_base_head_only(model,z,tq,**kw);counter['prefix']+=1
                    vq=clean_to_velocity(q,z,tq,denominator_floor=floor)
                v=pfr_velocity(vf,vb,vq,guidance_scale=1.78,
                               revision_scale=.05 if a.mode=='pfr' else 0.)
                z=z-(grid[i]-grid[i+1])*v
            middle.append(z.float().cpu())
            for kind in images:
                state=z.clone()
                context=y if kind=='conditional' else torch.full_like(y,1000)
                for i in range(cut,len(grid)-1):
                    t=grid[i].expand(len(state))
                    full,_=model(state,t,context=context,attn_mask=None);counter['full']+=1
                    v=clean_to_velocity(full,state,t,denominator_floor=floor)
                    state=state-(grid[i]-grid[i+1])*v
                if not torch.isfinite(state).all():raise FloatingPointError(kind)
                im=decoder.decode(state).clamp(0,1)
                images[kind].append(im.mul(255).to(torch.uint8).permute(0,2,3,1).cpu().numpy())
            print(json.dumps({'done':min(start+4,a.samples),'mode':a.mode,'seconds':time.perf_counter()-started}),flush=True)
    middle=torch.cat(middle);torch.save(middle,a.output/'cut_latents.pt')
    del decoder,model;gc.collect();torch.cuda.empty_cache()
    from torchvision.models import convnext_tiny,ConvNeXt_Tiny_Weights
    weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    classifier=convnext_tiny(weights=None).to(a.device).eval().requires_grad_(False)
    weights_path=Path('/home/zhoushunyu/.cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth')
    classifier.load_state_dict(torch.load(weights_path,map_location='cpu',weights_only=True),strict=True)
    transform=weights.transforms();scores={};summary={}
    for kind,parts in images.items():
        pixels=np.concatenate(parts);np.savez(a.output/(kind+'.npz'),pixels)
        logits=[]
        for start in range(0,len(pixels),16):
            x=torch.from_numpy(pixels[start:start+16]).permute(0,3,1,2).to(a.device).float()/255
            logits.append(classifier(transform(x)).float().cpu())
        logit=torch.cat(logits);prob=logit.softmax(1)
        target=prob[torch.arange(len(labels)),labels]
        top1=logit.argmax(1).eq(labels);top5=logit.topk(5,dim=1).indices.eq(labels[:,None]).any(1)
        scores[kind+'_target_prob']=target.numpy();scores[kind+'_top1']=top1.numpy()
        scores[kind+'_top5']=top5.numpy();scores[kind+'_logits']=logit.numpy()
        summary[kind]={'target_prob':float(target.mean()),'top1':float(top1.float().mean()),'top5':float(top5.float().mean())}
    np.savez(a.output/'scores.npz',labels=labels.numpy(),**scores)
    result={'complete':True,'mode':a.mode,'samples':a.samples,'seed':a.seed,
            'imagenet100':a.imagenet100,'requested_cut_time':a.cut_time,
            'cut_index':cut,'cut_time':float(grid[cut]),'checkpoint_step':step,
            'checkpoint_sha256':file_sha256(DEFAULT_CHECKPOINT),'classifier_sha256':file_sha256(weights_path),
            'noise_sha256':tensor_sha256(noise),'labels_sha256':tensor_sha256(labels),
            'cut_latents_sha256':tensor_sha256(middle),'calls':counter,
            'seconds':time.perf_counter()-started,'results':summary}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['ig','pfr'],required=True)
    p.add_argument('--samples',type=int,default=128)
    p.add_argument('--seed',type=int,default=202609401)
    p.add_argument('--cut-time',type=float,default=.8)
    p.add_argument('--imagenet100',action='store_true')
    p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
