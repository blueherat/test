"""Paired unguided real-image sampling with native, scalar and PCA gates."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.cache_sit_subspace_gate import load_official_sit_module,DEFAULT_OFFICIAL_SIT_REPO,create_dual_output_sit,PROTOCOL,dual_output_velocities,sha256_file
from experiments.train_sit_subspace_gate import SiTGate
from experiments.sample_imagenet100_sit_flow import integrate_velocity
from experiments.sample_imagenet100_sit_dual_fid import conditional_dual_velocity
from experiments.sample_imagenet100_sit_fid import official_pixel_quantization,decode_latents_in_chunks


@torch.inference_mode()
def run(a):
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert ck['protocol']==PROTOCOL
    module,meta=load_official_sit_module(DEFAULT_OFFICIAL_SIT_REPO);assert ck['official_sit']==meta
    cfg=ck['config'];model=create_dual_output_sit(module,model_name=cfg['model_name'],cfg_dropout=cfg['cfg_dropout'])
    model.load_state_dict(ck['ema'],strict=True);model.to(a.device).eval().requires_grad_(False)
    del ck;gc.collect()
    gate_ck=torch.load(a.gates,map_location='cpu',weights_only=False)
    assert gate_ck['cache_manifest']['checkpoint_sha256']==sha256_file(a.checkpoint)
    p=gate_ck['basis']['basis'].to(a.device)
    gate=None
    if a.mode in ('scalar','pca'):
        gate=SiTGate().to(a.device).eval();gate.load_state_dict(gate_ck['models'][a.mode],strict=True)
    from diffusers.models import AutoencoderKL
    vae=AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse',local_files_only=True).to(a.device).eval().requires_grad_(False)
    gen=torch.Generator(device='cpu').manual_seed(a.seed)
    noise=torch.randn(a.samples,4,32,32,generator=gen)
    labels=(torch.arange(a.samples)%100)[torch.randperm(a.samples,generator=gen)]
    np.savez(a.output/'inputs.npz',noise=noise.numpy(),labels=labels.numpy())
    pixels=np.lib.format.open_memmap(a.output/'pixels.npy',mode='w+',dtype='uint8',shape=(a.samples,256,256,3))
    trace=[];started=time.perf_counter()
    for start in range(0,a.samples,32):
        initial=noise[start:start+32].to(a.device);y=labels[start:start+32].to(a.device)
        counter={'nfe':0}
        if a.mode in ('native','x','epsilon'):
            velocity,counter=conditional_dual_velocity(model,y,mode='dynamic' if a.mode=='native' else a.mode,
                gate_activation=cfg['gate_activation'],denominator_floor=cfg['denominator_floor'],autocast_dtype=torch.bfloat16)
        else:
            def velocity(time_value,state):
                counter['nfe']+=1;times=time_value.expand(len(state))
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model(state,times,y)
                f=dual_output_velocities(out,state=state,time_value=times,
                    gate_activation=cfg['gate_activation'],denominator_floor=cfg['denominator_floor'])
                delta=(f['x']-f['epsilon']).flatten(1);g=gate(state,times,y,a.mode)
                if a.mode=='scalar':v=f['epsilon'].flatten(1)+g[:,:1]*delta
                else:
                    parallel=(delta@p)@p.T
                    v=f['epsilon'].flatten(1)+g[:,:1]*parallel+g[:,1:]*(delta-parallel)
                v=v.reshape_as(state);floor=cfg['denominator_floor']
                v=torch.where((times<=floor)[:,None,None,None],f['x'],v)
                return torch.where((times>=1-floor)[:,None,None,None],f['epsilon'],v)
        tic=time.perf_counter()
        latent=integrate_velocity(initial,velocity,num_output_points=2,atol=1e-6,rtol=1e-3)
        if not torch.isfinite(latent).all():raise FloatingPointError('sampler')
        decoded=decode_latents_in_chunks(vae,latent,scaling_factor=.18215,chunk_size=4)
        images=official_pixel_quantization(decoded)
        pixels[start:start+len(images)]=images
        row={'start':start,'count':len(images),'nfe':counter['nfe'],'seconds':time.perf_counter()-tic}
        trace.append(row)
        if start%256==0:print(json.dumps(row),flush=True)
    pixels.flush();np.savez(a.output/'samples.npz',np.asarray(pixels))
    result={'complete':True,'mode':a.mode,'samples':a.samples,'seed':a.seed,'batch_size':32,
        'sampler':'original dopri5 atol1e-6 rtol1e-3, native pure-head endpoint conventions',
        'checkpoint_sha256':sha256_file(a.checkpoint),'gate_checkpoint_sha256':sha256_file(a.gates),
        'inputs_sha256':sha256_file(a.output/'inputs.npz'),'samples_sha256':sha256_file(a.output/'samples.npz'),
        'seconds':time.perf_counter()-started,'trace':trace}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=Path('/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/sit-s-2_dual-output_seed0/checkpoints/step_00450000.pt'))
    p.add_argument('--gates',type=Path,default=Path('/home/zhoushunyu/data/eqvae/experiments/sit_subspace_gate_20260908/gates/checkpoint.pt'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['native','scalar','pca','x','epsilon'],required=True)
    p.add_argument('--samples',type=int,default=5000);p.add_argument('--seed',type=int,default=202609321)
    p.add_argument('--device',required=True);run(p.parse_args())
