#!/usr/bin/env python3
"""Generic v800 checkpoint-reference sampler with arbitrary stage-local weak refs."""

from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from torchvision.utils import save_image

try:
    from experiments.imagenet100_sit_multiscale_models import evaluate_sit_field, load_sit_field_model
    from experiments.sample_imagenet100_sit_fid import configure_cuda_allocator, decode_latents_in_chunks, official_pixel_quantization
    from experiments.sample_imagenet100_sit_flow import integrate_velocity
    from experiments.train_imagenet100_sit_flow import DEFAULT_OFFICIAL_SIT_REPO, LATENT_SHAPE, NUM_CLASSES, SD_VAE_SCALING_FACTOR, atomic_json_dump, load_official_sit_module
except ModuleNotFoundError:
    from imagenet100_sit_multiscale_models import evaluate_sit_field, load_sit_field_model
    from sample_imagenet100_sit_fid import configure_cuda_allocator, decode_latents_in_chunks, official_pixel_quantization
    from sample_imagenet100_sit_flow import integrate_velocity
    from train_imagenet100_sit_flow import DEFAULT_OFFICIAL_SIT_REPO, LATENT_SHAPE, NUM_CLASSES, SD_VAE_SCALING_FACTOR, atomic_json_dump, load_official_sit_module

DATA = Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow")
DEFAULT_STRONG = DATA / "runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
STRONG_NAMES = {"strong","none","baseline","unguided"}

def parse_name_path(v: str):
    if "=" not in v: raise argparse.ArgumentTypeError("use NAME=PATH")
    n,p=v.split("=",1); n=n.strip(); p=p.strip()
    if not n or not p: raise argparse.ArgumentTypeError("use NAME=PATH")
    if n.lower() in STRONG_NAMES: raise argparse.ArgumentTypeError(f"{n!r} reserved")
    return n, Path(p)

def load_condition(path: Path):
    x=json.loads(path.read_text())
    if not isinstance(x,dict): raise ValueError("condition must be object")
    raw=x.get("stages")
    if not isinstance(raw,list) or not raw: raise ValueError("stages must be non-empty list")
    stages=[]
    for i,s in enumerate(raw):
        if not isinstance(s,dict): raise ValueError(f"stage {i} must be object")
        ref=str(s.get("reference","strong")).strip() or "strong"
        gamma=float(s.get("gamma",0.0))
        if not math.isfinite(gamma) or gamma<0: raise ValueError(f"bad gamma at stage {i}")
        if ref.lower() in STRONG_NAMES: ref,gamma="strong",0.0
        stages.append({"reference":ref,"gamma":gamma})
    boundaries=[float(v) for v in x.get("boundaries",[])]
    if len(boundaries)!=len(stages)-1: raise ValueError("len(boundaries) must be len(stages)-1")
    if boundaries and (boundaries[0]<=0 or boundaries[-1]>=1 or any(b<=a for a,b in zip(boundaries,boundaries[1:]))):
        raise ValueError("boundaries must be increasing in (0,1)")
    mode=str(x.get("mode","hard"))
    if mode not in {"hard","smooth"}: raise ValueError("mode must be hard or smooth")
    width=float(x.get("transition_width",0.08))
    if width<=0 or not math.isfinite(width): raise ValueError("transition_width must be positive")
    if mode=="smooth" and boundaries:
        edges=[0.0,*boundaries,1.0]
        if width>=min(b-a for a,b in zip(edges,edges[1:])): raise ValueError("transition too wide")
    return {**x,"format":"eqvae_checkpoint_reference_condition_v2","name":str(x.get("name",path.stem)),
            "stages":stages,"boundaries":boundaries,"mode":mode,"transition_width":width,
            "formula":"S + sum_i w_i(t)*gamma_i*(S-W_i)"}

def needed_refs(c):
    return {s["reference"] for s in c["stages"] if s["reference"]!="strong" and float(s["gamma"])!=0.0}

def smoothstep(x):
    x=min(1.0,max(0.0,float(x))); return x*x*(3-2*x)

def weights_for_time(t, n, boundaries, mode, width):
    t=float(t.detach().float().item())
    if n==1: return [1.0]
    if mode=="smooth":
        h=width/2
        for i,b in enumerate(boundaries):
            if b-h<=t<=b+h:
                a=smoothstep((t-(b-h))/width); w=[0.0]*n; w[i]=1-a; w[i+1]=a; return w
    i=0
    while i<len(boundaries) and t>=boundaries[i]: i+=1
    w=[0.0]*n; w[i]=1.0; return w

class Field:
    def __init__(self,c,strong,strong_sem,refs,ref_sem,labels):
        self.c=c; self.strong=strong; self.strong_sem=strong_sem; self.refs=refs; self.ref_sem=ref_sem; self.labels=labels
        self.nfe=0; self.strong_forwards=0; self.reference_forwards={k:0 for k in refs}; self.stage_weight_sums=[0.0]*len(c["stages"])
    def __call__(self,t,z):
        self.nfe+=1; times=t.expand(len(z))
        S=evaluate_sit_field(self.strong,self.strong_sem,z,times,self.labels); self.strong_forwards+=1
        ws=weights_for_time(t,len(self.c["stages"]),self.c["boundaries"],self.c["mode"],self.c["transition_width"])
        result=S; cache={}
        for i,(stage,w) in enumerate(zip(self.c["stages"],ws)):
            self.stage_weight_sums[i]+=float(w)
            ref=stage["reference"]; gamma=float(stage["gamma"])
            if w==0 or ref=="strong" or gamma==0: continue
            if ref not in cache:
                cache[ref]=evaluate_sit_field(self.refs[ref],self.ref_sem[ref],z,times,self.labels); self.reference_forwards[ref]+=1
            result=result+float(w)*gamma*(S-cache[ref])
        return result

def validate(strong_sem,strong_meta,ref_sem,ref_meta):
    if strong_sem.prediction_target!="velocity": raise ValueError("strong must be velocity")
    for name,sem in ref_sem.items():
        if sem.prediction_target!="velocity": raise ValueError(f"{name} must be velocity")
        for key in ("model_name","data_manifest_sha256"):
            if ref_meta[name].get(key)!=strong_meta.get(key): raise ValueError(f"{name} mismatch on {key}")

@torch.inference_mode()
def main(a):
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    c=load_condition(a.condition_json.expanduser().resolve()); supplied=dict(a.reference_checkpoint); need=needed_refs(c)
    miss=need-set(supplied)
    if miss: raise ValueError(f"missing refs: {sorted(miss)}")
    device=torch.device(a.device); torch.cuda.set_device(device)
    alloc=configure_cuda_allocator(device,limit_gib=a.cuda_allocator_limit_gib)
    torch.backends.cuda.matmul.allow_tf32=bool(a.allow_tf32); torch.backends.cudnn.allow_tf32=bool(a.allow_tf32)
    if hasattr(torch,"set_float32_matmul_precision"): torch.set_float32_matmul_precision("high" if a.allow_tf32 else "highest")
    od=a.output_dir.expanduser().resolve(); od.mkdir(parents=True,exist_ok=True)
    sit,source=load_official_sit_module(a.official_sit_repo.expanduser().resolve(),verify_source=a.verify_sit_source)
    strong,strong_sem,strong_meta=load_sit_field_model(checkpoint_path=a.strong_checkpoint.expanduser().resolve(),weights="ema",sit_module=sit,source_metadata=source,device=device)
    refs={}; ref_sem={}; ref_meta={}
    for n in sorted(need):
        m,s,meta=load_sit_field_model(checkpoint_path=supplied[n].expanduser().resolve(),weights="ema",sit_module=sit,source_metadata=source,device=device)
        refs[n]=m; ref_sem[n]=s; ref_meta[n]=meta
    validate(strong_sem,strong_meta,ref_sem,ref_meta)
    from diffusers.models import AutoencoderKL
    vae=AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse",local_files_only=True); vae.to(device).eval().requires_grad_(False)
    torch.manual_seed(a.seed); torch.cuda.manual_seed(a.seed)
    images=np.empty((a.num_samples,256,256,3),dtype=np.uint8); labels_arr=np.empty(a.num_samples,dtype=np.int16)
    nd=hashlib.sha256(); ld=hashlib.sha256(); preview=None; cursor=0; total_nfe=0; sf=0
    rf={k:0 for k in refs}; sw=[0.0]*len(c["stages"]); started=time.perf_counter()
    while cursor<a.num_samples:
        b=min(a.batch_size,a.num_samples-cursor); noise=torch.randn(b,*LATENT_SHAPE,device=device); labels=torch.randint(0,NUM_CLASSES,(b,),device=device)
        f=Field(c,strong,strong_sem,refs,ref_sem,labels)
        endpoint=integrate_velocity(noise,f,num_output_points=a.num_output_points,atol=a.atol,rtol=a.rtol)
        if not torch.isfinite(endpoint).all(): raise FloatingPointError("non-finite endpoint")
        dec=decode_latents_in_chunks(vae,endpoint,scaling_factor=SD_VAE_SCALING_FACTOR,chunk_size=a.vae_decode_batch_size)
        stop=cursor+b; images[cursor:stop]=official_pixel_quantization(dec); labels_arr[cursor:stop]=labels.cpu().numpy().astype(np.int16,copy=False)
        nd.update(noise.detach().cpu().contiguous().numpy().tobytes()); ld.update(labels.detach().cpu().contiguous().numpy().tobytes())
        if preview is None: preview=dec[:min(16,len(dec))].detach().cpu()
        total_nfe+=f.nfe; sf+=f.strong_forwards
        for k,v in f.reference_forwards.items(): rf[k]+=int(v)
        for i,v in enumerate(f.stage_weight_sums): sw[i]+=float(v)
        cursor=stop
        if cursor==b or cursor==a.num_samples or cursor%a.log_every==0:
            print(json.dumps({"condition":c["name"],"generated":cursor,"total":a.num_samples,"elapsed_seconds":time.perf_counter()-started,"last_batch_nfe":f.nfe}),flush=True)
    sp=od/f"samples_n{a.num_samples}.npz"; lp=od/f"labels_n{a.num_samples}.npy"; np.savez(sp,arr_0=images); np.save(lp,labels_arr,allow_pickle=False)
    assert preview is not None; save_image(preview,od/"preview.png",nrow=4,normalize=True,value_range=(-1,1))
    manifest={"format":"eqvae_imagenet100_sit_checkpoint_reference_schedule_samples_v2","condition":c,
      "formula":"S + sum_i w_i(t)*gamma_i*(S-W_i)",
      "sampling":{"num_samples":int(a.num_samples),"batch_size":int(a.batch_size),"seed":int(a.seed),"num_output_points":int(a.num_output_points),"integrator":"dopri5","atol":float(a.atol),"rtol":float(a.rtol),"precision":"fp32","allow_tf32":bool(a.allow_tf32)},
      "strong":strong_meta,"references":ref_meta,"noise_sha256":nd.hexdigest(),"label_sha256":ld.hexdigest(),
      "label_histogram":np.bincount(labels_arr.astype(np.int64),minlength=NUM_CLASSES).tolist(),
      "total_nfe":int(total_nfe),"strong_forwards":int(sf),"reference_forwards":rf,"stage_weight_sums":sw,
      "samples":str(sp),"labels":str(lp),"elapsed_seconds":time.perf_counter()-started,**alloc,
      "max_memory_allocated_bytes":int(torch.cuda.max_memory_allocated(device)),"max_memory_reserved_bytes":int(torch.cuda.max_memory_reserved(device))}
    atomic_json_dump(manifest,od/"sampling_manifest.json")
    print(json.dumps({"event":"complete","condition":c["name"],"samples":str(sp),"noise_sha256":manifest["noise_sha256"],"label_sha256":manifest["label_sha256"]},indent=2),flush=True)

def parser():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--condition-json",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--strong-checkpoint",type=Path,default=DEFAULT_STRONG); p.add_argument("--reference-checkpoint",action="append",type=parse_name_path,default=[],metavar="NAME=PATH")
    p.add_argument("--official-sit-repo",type=Path,default=DEFAULT_OFFICIAL_SIT_REPO); p.add_argument("--num-samples",type=int,default=1000); p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--vae-decode-batch-size",type=int,default=2); p.add_argument("--seed",type=int,default=0); p.add_argument("--num-output-points",type=int,default=250)
    p.add_argument("--atol",type=float,default=1e-6); p.add_argument("--rtol",type=float,default=1e-3); p.add_argument("--cuda-allocator-limit-gib",type=float,default=4.0)
    p.add_argument("--device",default="cuda:0"); p.add_argument("--log-every",type=int,default=512)
    p.add_argument("--allow-tf32",action=argparse.BooleanOptionalAction,default=True); p.add_argument("--verify-sit-source",action=argparse.BooleanOptionalAction,default=True)
    return p

if __name__=="__main__": main(parser().parse_args())
