#!/usr/bin/env python3
"""Checkpoint-reference geometry atlas for the v800 SiT guidance study.

Two complementary state families are measured:
1) unguided v800 rollout states: gap RMS, pairwise gap cosine, and projection
   residuals relative to v270;
2) held-out teacher interpolation states: alignment of each gap S-W_r with the
   strong-model supervised residual v*-S, including the locally optimal
   nonnegative scalar gamma in velocity MSE.

The atlas is diagnostic only; it never uses FID samples for fitting.
"""

from __future__ import annotations
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torchdiffeq import odeint

try:
    from experiments.imagenet100_sit_multiscale_models import evaluate_sit_field, load_sit_field_model
    from experiments.train_imagenet100_sit_flow import (
        DEFAULT_CACHE_DIR, DEFAULT_OFFICIAL_SIT_REPO, LATENT_SHAPE, NUM_CLASSES,
        NpyMomentsDataset, atomic_json_dump, load_official_sit_module,
        sample_sdvae_posterior, sha256_file,
    )
except ModuleNotFoundError:
    from imagenet100_sit_multiscale_models import evaluate_sit_field, load_sit_field_model
    from train_imagenet100_sit_flow import (
        DEFAULT_CACHE_DIR, DEFAULT_OFFICIAL_SIT_REPO, LATENT_SHAPE, NUM_CLASSES,
        NpyMomentsDataset, atomic_json_dump, load_official_sit_module,
        sample_sdvae_posterior, sha256_file,
    )

DATA=Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow")
DEFAULT_STRONG=DATA/"runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
DEFAULT_OUT=DATA/"checkpoint_reference_long_study_v1/atlas"
DEFAULT_TIMES=(0.05,0.15,0.25,0.35,0.50,0.65,0.75,0.85,0.95)

def parse_name_path(v):
    if "=" not in v: raise argparse.ArgumentTypeError("use NAME=PATH")
    n,p=v.split("=",1); n=n.strip(); p=p.strip()
    if not n or not p: raise argparse.ArgumentTypeError("use NAME=PATH")
    return n,Path(p)

def flat_dot(a,b): return (a.float()*b.float()).flatten(1).sum(1)
def flat_sq(a): return a.float().square().flatten(1).sum(1)
def rms(a): return a.float().square().flatten(1).mean(1).sqrt()
def cosine(a,b):
    return flat_dot(a,b)/(flat_sq(a).sqrt()*flat_sq(b).sqrt()).clamp_min(torch.finfo(torch.float32).tiny)

def region(t):
    if t<1/3: return "early"
    if t<2/3: return "mid"
    return "late"

def mean_std(values):
    x=np.asarray(values,dtype=np.float64)
    return float(x.mean()), float(x.std(ddof=1) if len(x)>1 else 0.0)

def save_rows(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: return
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

@torch.inference_mode()
def main(a):
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    refs_arg=dict(a.reference_checkpoint)
    if "v270" not in refs_arg: raise ValueError("atlas requires v270 as amplitude/projection anchor")
    device=torch.device(a.device); torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    if hasattr(torch,"set_float32_matmul_precision"): torch.set_float32_matmul_precision("high")
    out=a.output_dir.expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    sit,source=load_official_sit_module(a.official_sit_repo.expanduser().resolve(),verify_source=a.verify_sit_source)
    strong,strong_sem,strong_meta=load_sit_field_model(checkpoint_path=a.strong_checkpoint.expanduser().resolve(),weights="ema",sit_module=sit,source_metadata=source,device=device)
    refs={}; sem={}; meta={}
    for name,path in sorted(refs_arg.items()):
        m,s,mt=load_sit_field_model(checkpoint_path=path.expanduser().resolve(),weights="ema",sit_module=sit,source_metadata=source,device=device)
        if s.prediction_target!="velocity": raise ValueError(f"{name} is not velocity")
        if mt["model_name"]!=strong_meta["model_name"] or mt["data_manifest_sha256"]!=strong_meta["data_manifest_sha256"]: raise ValueError(f"{name} incompatible")
        refs[name]=m; sem[name]=s; meta[name]=mt
    names=sorted(refs,key=lambda n:int(meta[n]["checkpoint_step"]))
    times=tuple(float(t) for t in a.times)
    if any(not 0<t<1 for t in times) or any(b<=aa for aa,b in zip(times,times[1:])): raise ValueError("times must increase in (0,1)")

    rollout_vals=defaultdict(list); pair_vals=defaultdict(list); teacher_vals=defaultdict(list)
    global_ms=defaultdict(list); region_ms=defaultdict(list)

    # --- unguided strong rollout atlas ---
    torch.manual_seed(a.rollout_seed); torch.cuda.manual_seed(a.rollout_seed)
    cursor=0
    ode_times=torch.tensor((0.0,*times),device=device,dtype=torch.float32)
    while cursor<a.rollout_samples:
        b=min(a.batch_size,a.rollout_samples-cursor)
        noise=torch.randn(b,*LATENT_SHAPE,device=device)
        labels=torch.randint(0,NUM_CLASSES,(b,),device=device)
        def strong_field(t,z):
            return evaluate_sit_field(strong,strong_sem,z,t.expand(len(z)),labels)
        traj=odeint(strong_field,noise.float(),ode_times,method="dopri5",atol=a.atol,rtol=a.rtol)
        for ti,t in enumerate(times,1):
            z=traj[ti]; tv=torch.full((b,),t,device=device)
            S=evaluate_sit_field(strong,strong_sem,z,tv,labels)
            gaps={}
            for name in names:
                W=evaluate_sit_field(refs[name],sem[name],z,tv,labels)
                g=S-W; gaps[name]=g
                gr=rms(g); sr=rms(S)
                key=(name,t)
                rollout_vals[(key,"gap_rms")].extend(gr.cpu().tolist())
                rollout_vals[(key,"gap_over_strong")].extend((gr/sr.clamp_min(1e-12)).cpu().tolist())
                global_ms[name].extend(g.float().square().flatten(1).mean(1).cpu().tolist())
                region_ms[(name,region(t))].extend(g.float().square().flatten(1).mean(1).cpu().tolist())
            anchor=gaps["v270"]
            for name in names:
                g=gaps[name]
                c=cosine(g,anchor)
                coef=flat_dot(g,anchor)/flat_sq(anchor).clamp_min(1e-12)
                resid=g-coef[:,None,None,None]*anchor
                frac=rms(resid)/rms(g).clamp_min(1e-12)
                rollout_vals[((name,t),"cos_v270")].extend(c.cpu().tolist())
                rollout_vals[((name,t),"proj_resid_frac_v270")].extend(frac.cpu().tolist())
                rollout_vals[((name,t),"proj_coef_on_v270")].extend(coef.cpu().tolist())
            for i,left in enumerate(names):
                for right in names[i+1:]:
                    pair_vals[(left,right,t)].extend(cosine(gaps[left],gaps[right]).cpu().tolist())
        cursor+=b
        print(json.dumps({"phase":"rollout","done":cursor,"total":a.rollout_samples}),flush=True)

    # --- held-out teacher-state residual alignment ---
    cache=a.cache_dir.expanduser().resolve()
    ds=NpyMomentsDataset(cache,"validation")
    rng=np.random.default_rng(a.teacher_seed)
    indices=np.sort(rng.choice(len(ds),size=a.teacher_samples,replace=False))
    gen=torch.Generator(device=device).manual_seed(a.teacher_seed+991)
    cursor=0
    while cursor<a.teacher_samples:
        batch_idx=indices[cursor:cursor+a.batch_size]
        moments=np.stack([np.asarray(ds[int(i)][0],dtype=np.float32) for i in batch_idx])
        labels_np=np.asarray([int(ds[int(i)][1]) for i in batch_idx],dtype=np.int64)
        moments_t=torch.from_numpy(moments).to(device)
        labels=torch.from_numpy(labels_np).to(device)
        posterior_noise=torch.randn((len(batch_idx),*LATENT_SHAPE),generator=gen,device=device)
        data=sample_sdvae_posterior(moments_t,posterior_noise)
        eps=torch.randn(data.shape,generator=gen,device=device)
        target=data-eps
        for t in times:
            z=(1-t)*eps+t*data; tv=torch.full((len(z),),t,device=device)
            S=evaluate_sit_field(strong,strong_sem,z,tv,labels)
            err=target-S
            err_sq=flat_sq(err)
            for name in names:
                W=evaluate_sit_field(refs[name],sem[name],z,tv,labels); g=S-W
                dot=flat_dot(g,err); gsq=flat_sq(g).clamp_min(1e-12)
                gamma_ls=dot/gsq
                cos=dot/(gsq.sqrt()*err_sq.sqrt()).clamp_min(1e-12)
                gamma_pos=gamma_ls.clamp_min(0.0)
                after=flat_sq(err-gamma_pos[:,None,None,None]*g)
                reduction=(err_sq-after)/err_sq.clamp_min(1e-12)
                teacher_vals[((name,t),"cos_residual")].extend(cos.cpu().tolist())
                teacher_vals[((name,t),"gamma_ls")].extend(gamma_ls.cpu().tolist())
                teacher_vals[((name,t),"positive_reduction_fraction")].extend(reduction.cpu().tolist())
        cursor+=len(batch_idx)
        print(json.dumps({"phase":"teacher","done":cursor,"total":a.teacher_samples}),flush=True)

    rollout_rows=[]
    for name in names:
        for t in times:
            row={"reference":name,"step":int(meta[name]["checkpoint_step"]),"time":t,"region":region(t)}
            for metric in ("gap_rms","gap_over_strong","cos_v270","proj_resid_frac_v270","proj_coef_on_v270"):
                m,s=mean_std(rollout_vals[((name,t),metric)]); row[metric+"_mean"]=m; row[metric+"_std"]=s
            rollout_rows.append(row)
    pair_rows=[]
    for (left,right,t),vals in pair_vals.items():
        m,s=mean_std(vals); pair_rows.append({"left":left,"right":right,"time":t,"region":region(t),"cosine_mean":m,"cosine_std":s})
    teacher_rows=[]
    for name in names:
        for t in times:
            row={"reference":name,"step":int(meta[name]["checkpoint_step"]),"time":t,"region":region(t)}
            for metric in ("cos_residual","gamma_ls","positive_reduction_fraction"):
                vals=teacher_vals[((name,t),metric)]; m,s=mean_std(vals)
                row[metric+"_mean"]=m; row[metric+"_std"]=s
                if metric=="gamma_ls": row["gamma_ls_positive_fraction"]=float(np.mean(np.asarray(vals)>0))
            teacher_rows.append(row)
    save_rows(out/"rollout_gap_atlas.csv",rollout_rows); save_rows(out/"pairwise_gap_cosine.csv",pair_rows); save_rows(out/"teacher_residual_alignment.csv",teacher_rows)

    global_rms={name:math.sqrt(float(np.mean(global_ms[name]))) for name in names}
    region_rms={r:{name:math.sqrt(float(np.mean(region_ms[(name,r)]))) for name in names} for r in ("early","mid","late")}
    anchor_global=global_rms["v270"]
    gamma_match={name:3.5*anchor_global/global_rms[name] for name in names}
    gamma_match_region={r:{name:3.5*region_rms[r]["v270"]/region_rms[r][name] for name in names} for r in ("early","mid","late")}

    teacher_stage={}
    for r in ("early","mid","late"):
        candidates=[]
        for name in names:
            rows=[x for x in teacher_rows if x["reference"]==name and x["region"]==r]
            score=float(np.mean([x["positive_reduction_fraction_mean"] for x in rows]))
            gamma_vals=[]
            for t in times:
                if region(t)!=r: continue
                gamma_vals += [max(0.0,float(v)) for v in teacher_vals[((name,t),"gamma_ls")]]
            candidates.append((score,name,float(np.mean(gamma_vals))))
        candidates.sort(reverse=True)
        teacher_stage[r]={"reference":candidates[0][1],"score":candidates[0][0],"gamma_ls_positive_mean":candidates[0][2],
                          "ranking":[{"reference":n,"score":s,"gamma_ls_positive_mean":g} for s,n,g in candidates]}

    summary={"format":"eqvae_checkpoint_reference_atlas_v1","strong":strong_meta,"references":meta,"ordered_references":names,
      "times":list(times),"rollout_samples":a.rollout_samples,"teacher_samples":a.teacher_samples,
      "global_gap_rms":global_rms,"region_gap_rms":region_rms,
      "gamma_rms_match_to_v270_gamma3p5":gamma_match,"gamma_rms_match_by_region":gamma_match_region,
      "teacher_stage_recommendation":teacher_stage,
      "teacher_indices_sha256":__import__("hashlib").sha256(indices.astype(np.int64).tobytes()).hexdigest(),
      "files":{"rollout":str(out/"rollout_gap_atlas.csv"),"pairwise":str(out/"pairwise_gap_cosine.csv"),"teacher":str(out/"teacher_residual_alignment.csv")}}
    atomic_json_dump(summary,out/"summary.json")
    print(json.dumps({"event":"complete","summary":str(out/"summary.json"),"gamma_rms_match":gamma_match,"teacher_stage":teacher_stage},indent=2),flush=True)

def parser():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--strong-checkpoint",type=Path,default=DEFAULT_STRONG)
    p.add_argument("--reference-checkpoint",action="append",type=parse_name_path,default=[],required=True); p.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    p.add_argument("--cache-dir",type=Path,default=DEFAULT_CACHE_DIR); p.add_argument("--official-sit-repo",type=Path,default=DEFAULT_OFFICIAL_SIT_REPO)
    p.add_argument("--times",nargs="+",type=float,default=list(DEFAULT_TIMES)); p.add_argument("--rollout-samples",type=int,default=128); p.add_argument("--teacher-samples",type=int,default=128)
    p.add_argument("--batch-size",type=int,default=8); p.add_argument("--rollout-seed",type=int,default=20260818); p.add_argument("--teacher-seed",type=int,default=20260819)
    p.add_argument("--atol",type=float,default=1e-6); p.add_argument("--rtol",type=float,default=1e-3); p.add_argument("--device",default="cuda:0")
    p.add_argument("--verify-sit-source",action=argparse.BooleanOptionalAction,default=True); return p

if __name__=="__main__": main(parser().parse_args())
