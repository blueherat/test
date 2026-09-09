"""Finite latent/pixel extrapolation against fixed original-image targets."""
import hashlib,inspect,json,time
from pathlib import Path
import numpy as np
import torch
from experiments import sample_raev2_pfr_retiming as native
from experiments.train_raev2_observable_potential import load_banks
from experiments.raev2_training_core import DeterministicImageNetPacked

OUT=Path('/home/zhoushunyu/data/eqvae/experiments/ig_decoded_target_risk_20260908')
def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
@torch.inference_mode()
def main():
    OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    bankdir=OUT.parent/'raev2_guidance_restart_20260906/potential_clean_bank_fp32_v1'
    banks,record=load_banks(bankdir);latents,meta=banks['validation']
    labels=np.arange(32)*999//31;indices=[int(np.flatnonzero(meta['labels']==c)[0]) for c in labels]
    packed=Path('/data/shared/imagenet-1k/random_access_v1')
    dataset=DeterministicImageNetPacked(packed,split='train',image_size=256,horizontal_flip=False,index_map_path=None)
    targets=[];target_record=[]
    for ix,c in zip(indices,labels):
        x,y,row=dataset[int(meta['rows'][ix])];assert y==c and row==meta['rows'][ix]
        targets.append(x)
        target_record.append(dict(index=ix,label=int(c),row=int(row),pixel_sha256=sha(x.numpy()),latent_sha256=sha(latents[ix])))
    np.save(OUT/'original_images.npy',torch.stack(targets).numpy());del dataset
    cfg=native.load_config(native.DEFAULT_CONFIG);native.install_raev2_decoder_config_compat()
    ckhash=native.file_sha256(native.DEFAULT_CHECKPOINT)
    assert ckhash=='723c56d7fa77ace9613909f7e38cb2386b898608218dc9b52649bb373d513c9a'
    decoder=native.instantiate_from_config(cfg.stage_1).cuda().eval().requires_grad_(False);del decoder.encoder
    model=native.instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
    ck=torch.load(native.DEFAULT_CHECKPOINT,map_location='cpu',mmap=True,weights_only=False);model.load_state_dict(ck['ema'],strict=True);del ck
    request=dict(targets=target_record,seed=202609441,times=[1.,.9,.75,.55],bank=record,checkpoint_sha256=ckhash,
                 sources={str(p):native.file_sha256(p) for p in [Path(__file__),Path(native.__file__),native.DEFAULT_CONFIG,Path(inspect.getfile(type(model)))]},
                 scope='previously explored 32 auxiliary-validation rows; not pretraining holdout or unused confirmation; original RGB targets')
    (OUT/'request.json').write_text(json.dumps(request,indent=2))
    gen=torch.Generator(device='cuda').manual_seed(202609441);rows=[];files={};started=time.perf_counter()
    for pos,(ix,c) in enumerate(zip(indices,labels)):
        clean=torch.from_numpy(np.array(latents[ix],copy=True)).float().unsqueeze(0).cuda()
        eps=torch.randn(clean.shape,generator=gen,device='cuda');target=targets[pos].numpy()
        for ev,tv in enumerate(request['times']):
            t=torch.tensor([tv],device='cuda');z=(1-t[:,None,None,None])*clean+t[:,None,None,None]*eps
            f,b=model(z,t,context=torch.tensor([int(c)],device='cuda'),attn_mask=None)
            g=b+1.78*(f-b)
            rgb=decoder.decode(torch.cat([b,f,g],dim=0)).cpu().numpy().astype(float)
            linear=rgb[1]+.78*(rgb[1]-rgb[0]);pred=np.concatenate([rgb,linear[None]],axis=0)
            err=(pred-target[None]).astype(np.float32)
            clipped=(np.clip(pred,0,1)-target[None]).astype(np.float32)
            p=OUT/f'id{ix:04d}_event{ev}.npz';np.savez_compressed(p,raw_errors=err,clipped_errors=clipped)
            files[p.name]=native.file_sha256(p)
            rows.append(dict(index=ix,label=int(c),event=ev,time=tv,names=['Base','Full','latent_IG','pixel_linear'],
                             raw_mse=np.square(err.astype(float)).mean(axis=(1,2,3)).tolist(),
                             clipped_mse=np.square(clipped.astype(float)).mean(axis=(1,2,3)).tolist()))
        if (pos+1)%8==0:print(json.dumps(dict(done=pos+1,seconds=time.perf_counter()-started)),flush=True)
    result=dict(complete=True,rows=rows,files=files,full_calls=128,decoder_calls=128,decoder_sample_calls=384,
                seconds=time.perf_counter()-started,request_sha256=native.file_sha256(OUT/'request.json'))
    (OUT/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(complete=True,seconds=result['seconds'])))
if __name__=='__main__':main()
