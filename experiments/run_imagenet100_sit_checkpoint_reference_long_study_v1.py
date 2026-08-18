#!/usr/bin/env python3
"""Long v800 checkpoint-reference study on GPUs 0 and 2.

Phases
------
0. Geometry atlas on v800 rollout + held-out teacher states.
1. Static reference-maturity screen over 180/200/220/240/270/300/400/500/600K.
   Gamma centers are RMS-matched to the known v270 gamma=3.5 operating point;
   a 3-point local screen is expanded outward up to two rounds if the best
   point remains on a tested boundary.
2. Exact 2^3 causal factorial for early-v270 / mid-v400 / late-v500, plus
   replacement controls and reverse order.
3. Repeat the key schedules with the best newly discovered early reference.
4. Small principled schedule study: delayed boundaries, smooth switches,
   global/stage gain perturbations, v600 late controls, an amplitude-only
   v270 control matched to the forward forcing RMS, and a teacher-atlas
   identity schedule.
5. Aggregate paired FID, Shapley attribution, geometry, and all failures.

Every FID condition reseeds after model/VAE loading, so all successful
conditions use identical initial noise and class labels despite loading
different numbers of weak checkpoints.
"""

from __future__ import annotations
import argparse, csv, hashlib, json, math, os, sys, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch

try:
    from experiments.run_imagenet100_sit_fid_curve import DEFAULT_ADM_PYTHON, fid_environment, parse_gpu_indices, run_logged
    from experiments.train_imagenet100_sit_flow import atomic_json_dump, sha256_file
except ModuleNotFoundError:
    from run_imagenet100_sit_fid_curve import DEFAULT_ADM_PYTHON, fid_environment, parse_gpu_indices, run_logged
    from train_imagenet100_sit_flow import atomic_json_dump, sha256_file

ROOT=Path(__file__).resolve().parents[1]
SAMPLER=ROOT/"experiments/sample_imagenet100_sit_checkpoint_reference_schedule_v2.py"
ATLAS=ROOT/"experiments/analyze_imagenet100_sit_checkpoint_reference_atlas_v1.py"
FID_SCRIPT=ROOT/"experiments/compute_adm_fid.py"
DATA=Path("/home/zhoushunyu/data/eqvae/imagenet_sit_flow")
CKPT_DIR=DATA/"runs/sit-s-2_seed0/checkpoints"
DEFAULT_STRONG=CKPT_DIR/"step_00800000.pt"
DEFAULT_REFERENCE=DATA/"adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz"
DEFAULT_OUT=DATA/"checkpoint_reference_long_study_v1"
DEFAULT_STEPS=(180000,200000,220000,240000,270000,300000,400000,500000,600000)

def read_json(p):
    x=json.loads(Path(p).read_text())
    if not isinstance(x,dict): raise ValueError(f"expected object: {p}")
    return x

def safe_name(x):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(x))

def gtag(g):
    s=f"{float(g):.3f}".rstrip("0").rstrip(".")
    return s.replace(".","p").replace("-","m")

def step_name(step): return f"v{int(step)//1000}"
def ckpt_path(directory,step): return directory/f"step_{int(step):08d}.pt"

def checkpoint_meta(path):
    c=torch.load(path,map_location="cpu",weights_only=False,mmap=True); cfg=c["config"]
    m={"path":str(path.resolve()),"sha256":sha256_file(path),"step":int(c["step"]),"protocol":str(c.get("protocol")),
       "model_name":str(cfg["model_name"]),"prediction_target":str(cfg.get("prediction_target","velocity")),
       "seed":int(cfg.get("seed",-1)),"global_batch_size":int(cfg.get("global_batch_size",-1)),
       "data_manifest_sha256":c.get("data_manifest_sha256"),"official_sit":c.get("official_sit")}
    del c; return m

def validate_family(strong,refs):
    if strong["prediction_target"]!="velocity": raise ValueError("strong is not native velocity")
    for n,m in refs.items():
        if m["prediction_target"]!="velocity": raise ValueError(f"{n} not native velocity")
        for k in ("protocol","model_name","seed","global_batch_size","data_manifest_sha256","official_sit"):
            if m[k]!=strong[k]: raise ValueError(f"{n} differs from strong on {k}")

def stage(ref,gamma):
    if ref in {None,"strong","none"} or float(gamma)==0: return {"reference":"strong","gamma":0.0}
    return {"reference":str(ref),"gamma":float(gamma)}

def condition(name,stages,boundaries=None,mode="hard",width=0.08,group="misc",note=""):
    n=len(stages)
    if boundaries is None:
        boundaries=[] if n==1 else [i/n for i in range(1,n)]
    return {"format":"eqvae_checkpoint_reference_condition_v2","name":name,"group":group,"note":note,
            "stages":stages,"boundaries":[float(x) for x in boundaries],"mode":mode,
            "transition_width":float(width),"formula":"S + sum_i w_i(t)*gamma_i*(S-W_i)"}

def active_refs(c):
    return sorted({s["reference"] for s in c["stages"] if s["reference"]!="strong" and float(s["gamma"])!=0})

def condition_fingerprint(c,strong,refs,a):
    active=active_refs(c)
    payload={"condition":c,"strong":strong["sha256"],"refs":{n:refs[n]["sha256"] for n in active},
             "num_samples":a.num_samples,"batch_size":a.batch_size,"seed":a.seed,"num_output_points":a.num_output_points,
             "atol":a.atol,"rtol":a.rtol}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def valid_result(path,fp,n):
    if not path.is_file(): return False
    try:
        x=read_json(path)
        return x.get("experiment_fingerprint")==fp and int(x["sampling_manifest"]["sampling"]["num_samples"])==n and all(isinstance(x["metrics"].get(k),(int,float)) for k in ("fid","sfid","inception_score"))
    except Exception: return False

def valid_samples(od,c,a):
    p=od/"sampling_manifest.json"; s=od/f"samples_n{a.num_samples}.npz"
    if not p.is_file() or not s.is_file(): return False
    try:
        m=read_json(p)
        return m.get("condition")==c and int(m["sampling"]["num_samples"])==a.num_samples and int(m["sampling"]["seed"])==a.seed and bool(m.get("noise_sha256")) and bool(m.get("label_sha256"))
    except Exception: return False

def run_condition(gpu,c,strong,refs,a,phase):
    name=str(c["name"]); od=a.output_root/phase/safe_name(name); od.mkdir(parents=True,exist_ok=True)
    cp=od/"condition.json"; atomic_json_dump(c,cp)
    fp=condition_fingerprint(c,strong,refs,a); rp=od/"condition_result.json"
    if valid_result(rp,fp,a.num_samples):
        print(f"[reuse] {phase}/{name}",flush=True); return read_json(rp)
    env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(gpu); env.setdefault("OMP_NUM_THREADS","1"); env.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
    sp=od/f"samples_n{a.num_samples}.npz"; mp=od/"sampling_manifest.json"
    if not valid_samples(od,c,a):
        cmd=[sys.executable,str(SAMPLER),"--condition-json",str(cp),"--output-dir",str(od),"--strong-checkpoint",str(a.strong_checkpoint),
             "--num-samples",str(a.num_samples),"--batch-size",str(a.batch_size),"--vae-decode-batch-size",str(a.vae_decode_batch_size),
             "--seed",str(a.seed),"--num-output-points",str(a.num_output_points),"--atol",repr(float(a.atol)),"--rtol",repr(float(a.rtol)),
             "--cuda-allocator-limit-gib",repr(float(a.cuda_allocator_limit_gib)),"--device","cuda:0"]
        for n in active_refs(c): cmd += ["--reference-checkpoint",f"{n}={refs[n]['path']}"]
        run_logged(tuple(cmd),od/"sampling.log",env=env,monitored_gpu_indices=[gpu],memory_ceiling_mib=a.gpu_memory_ceiling_mib,
                   memory_poll_interval=a.memory_poll_interval,resource_audit_path=od/"sampling_resource_audit.json")
        if not valid_samples(od,c,a): raise RuntimeError(f"invalid sampler output: {name}")
    fpath=od/"adm_metrics.json"
    fcmd=(str(a.adm_python),str(FID_SCRIPT),"--reference",str(a.reference),"--samples",str(sp),"--batch-size",str(a.fid_batch_size),
          "--gpu-memory-fraction",str(a.fid_gpu_memory_fraction),"--output",str(fpath))
    run_logged(fcmd,od/"evaluation.log",env=fid_environment(env,cuda_visible_devices=str(gpu)),monitored_gpu_indices=[gpu],
               memory_ceiling_mib=a.gpu_memory_ceiling_mib,memory_poll_interval=a.memory_poll_interval,resource_audit_path=od/"fid_resource_audit.json")
    m=read_json(mp); metrics=read_json(fpath)
    x={"format":"eqvae_checkpoint_reference_long_condition_v1","experiment_fingerprint":fp,"phase":phase,"condition":c,"sampling_manifest":m,"metrics":metrics,"gpu":gpu}
    atomic_json_dump(x,rp)
    if not a.keep_samples: sp.unlink(missing_ok=True)
    print(f"[complete] {phase}/{name}: FID={float(metrics['fid']):.4f}",flush=True); return x

def run_lane(gpu,jobs,strong,refs,a,phase):
    results=[]; failures=[]
    for c in jobs:
        try: results.append(run_condition(gpu,c,strong,refs,a,phase))
        except Exception as e:
            fail={"phase":phase,"condition":c["name"],"gpu":gpu,"error":repr(e),"traceback":traceback.format_exc()}
            failures.append(fail); fd=a.output_root/"failures"; fd.mkdir(parents=True,exist_ok=True)
            atomic_json_dump(fail,fd/f"{safe_name(phase)}__{safe_name(c['name'])}.json")
            print(f"[FAILED] {phase}/{c['name']}: {e}",flush=True)
            if a.fail_fast: raise
    return results,failures

def run_conditions(conditions,strong,refs,a,phase):
    if not conditions: return [],[]
    lanes={g:[] for g in a.gpu_indices}
    for i,c in enumerate(conditions): lanes[a.gpu_indices[i%len(a.gpu_indices)]].append(c)
    results=[]; failures=[]
    with ThreadPoolExecutor(max_workers=len(a.gpu_indices)) as pool:
        futs=[pool.submit(run_lane,g,lanes[g],strong,refs,a,phase) for g in a.gpu_indices if lanes[g]]
        for f in futs:
            r,x=f.result(); results+=r; failures+=x
    return results,failures

def result_row(x):
    c=x["condition"]; m=x["sampling_manifest"]; q=x["metrics"]
    return {"phase":x["phase"],"condition":c["name"],"group":c.get("group",""),"note":c.get("note",""),
            "stages":"|".join(f"{s['reference']}:{float(s['gamma']):g}" for s in c["stages"]),
            "boundaries":"|".join(f"{float(v):.5g}" for v in c["boundaries"]),"mode":c["mode"],"transition_width":c["transition_width"],
            "fid":float(q["fid"]),"sfid":float(q["sfid"]),"inception_score":float(q["inception_score"]),
            "total_nfe":int(m["total_nfe"]),"noise_sha256":m["noise_sha256"],"label_sha256":m["label_sha256"]}

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: return
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def rounded_gamma(x):
    return round(max(0.25,min(8.0,float(x)))*20)/20

def initial_grid(name,atlas):
    center=float(atlas["gamma_rms_match_to_v270_gamma3p5"][name])
    if name=="v270": return [3.0,3.5,4.0]
    if name=="v400": center=max(center,4.0)
    if name=="v500": center=max(center,3.0)
    center=max(0.5,min(7.0,center))
    vals={rounded_gamma(center*m) for m in (0.8,1.0,1.2)}
    if name=="v400": vals.add(4.0)
    if name=="v500": vals.add(3.0)
    return sorted(vals)

def static_condition(ref,gamma):
    return condition(f"static_{ref}_g{gtag(gamma)}",[stage(ref,gamma)],group="static_maturity",note="full-time static checkpoint reference")

def summarize_static(results,refs):
    rows=[]
    for x in results:
        r=result_row(x); s=x["condition"]["stages"][0]; r["reference"]=s["reference"]; r["gamma"]=float(s["gamma"]); r["step"]=int(refs[s["reference"]]["step"]); rows.append(r)
    best={}
    for n in sorted({r["reference"] for r in rows},key=lambda n:refs[n]["step"]):
        cand=[r for r in rows if r["reference"]==n]; best[n]=min(cand,key=lambda r:r["fid"])
    return rows,best

def adaptive_static(atlas,strong,refs,a):
    tested={}; all_results=[]; failures=[]
    initial=[]
    for n in refs:
        for g in initial_grid(n,atlas):
            tested.setdefault(n,set()).add(float(g)); initial.append(static_condition(n,g))
    r,f=run_conditions(initial,strong,refs,a,"01_static_initial"); all_results+=r; failures+=f
    for round_i in (1,2):
        rows,best=summarize_static(all_results,refs); extra=[]
        for n,b in best.items():
            successful=sorted(r["gamma"] for r in rows if r["reference"]==n)
            if len(successful)<2: continue
            g=float(b["gamma"]); lo,hi=min(successful),max(successful)
            new=None
            if abs(g-lo)<1e-9 and lo>0.25: new=rounded_gamma(lo/1.25)
            elif abs(g-hi)<1e-9 and hi<8.0: new=rounded_gamma(hi*1.20)
            if new is not None and new not in tested[n] and 0.25<=new<=8.0:
                tested[n].add(new); extra.append(static_condition(n,new))
        if not extra: break
        r,f=run_conditions(extra,strong,refs,a,f"01_static_expand{round_i}"); all_results+=r; failures+=f
    rows,best=summarize_static(all_results,refs)
    return all_results,failures,rows,best

def factorial_conditions(g):
    names=("v270","v400","v500"); conditions=[]
    for mask in range(8):
        stages=[]
        bits=[]
        for i,n in enumerate(names):
            on=bool(mask&(1<<(2-i))); bits.append("1" if on else "0"); stages.append(stage(n,g[n]) if on else stage("strong",0))
        conditions.append(condition("factorial_"+"".join(bits),stages,group="factorial",note="2^3 early-v270 / mid-v400 / late-v500 causal factorial"))
    conditions += [
      condition("replace_mid_v270_v400_v270",[stage("v270",g["v270"]),stage("v400",g["v400"]),stage("v270",g["v270"])],group="replacement",note="replace only middle third of static v270"),
      condition("replace_late_v270_v270_v500",[stage("v270",g["v270"]),stage("v270",g["v270"]),stage("v500",g["v500"])],group="replacement",note="replace only late third of static v270"),
      condition("reverse_v500_v400_v270",[stage("v500",g["v500"]),stage("v400",g["v400"]),stage("v270",g["v270"])],group="ordering",note="reverse of forward maturity order"),
    ]
    return conditions

def shapley_from_factorial(rows):
    by={}
    for r in rows:
        if r["condition"].startswith("factorial_"): by[r["condition"].split("_",1)[1]]=float(r["fid"])
    if len(by)!=8: return {"available":False,"missing":sorted(set(f"{i:03b}" for i in range(8))-set(by))}
    base=by["000"]; benefit={k:base-v for k,v in by.items()}
    players=[0,1,2]; labels=["early_v270","mid_v400","late_v500"]; phi={}
    import itertools, math as _m
    for p,label in zip(players,labels):
        val=0.0; others=[q for q in players if q!=p]
        for r in range(3):
            for subset in itertools.combinations(others,r):
                bits=[0,0,0]
                for q in subset: bits[q]=1
                k0="".join(map(str,bits)); bits[p]=1; k1="".join(map(str,bits))
                w=_m.factorial(r)*_m.factorial(2-r)/_m.factorial(3)
                val+=w*(benefit[k1]-benefit[k0])
        phi[label]=val
    interaction={
      "early_mid":benefit["110"]-benefit["100"]-benefit["010"],
      "early_late":benefit["101"]-benefit["100"]-benefit["001"],
      "mid_late":benefit["011"]-benefit["010"]-benefit["001"],
      "triple_mobius":benefit["111"]-benefit["110"]-benefit["101"]-benefit["011"]+benefit["100"]+benefit["010"]+benefit["001"],
    }
    return {"available":True,"baseline_fid":base,"full_fid":by["111"],"full_benefit":benefit["111"],"shapley_fid_benefit":phi,"interactions":interaction,"fid_by_mask":by}

def best_ref_conditions(best_ref,g,available):
    if best_ref=="v270": return []
    gb=g[best_ref]
    return [
      condition(f"beststart_{best_ref}_v400_v500",[stage(best_ref,gb),stage("v400",g["v400"]),stage("v500",g["v500"])],group="best_start",note="replace v270 by best static maturity reference"),
      condition(f"beststart_{best_ref}_v400_{best_ref}",[stage(best_ref,gb),stage("v400",g["v400"]),stage(best_ref,gb)],group="best_start",note="middle replacement relative to best static"),
      condition(f"beststart_{best_ref}_{best_ref}_v500",[stage(best_ref,gb),stage(best_ref,gb),stage("v500",g["v500"])],group="best_start",note="late replacement relative to best static"),
      condition(f"beststart_{best_ref}_only_early",[stage(best_ref,gb),stage("strong",0),stage("strong",0)],group="best_start",note="early-only best reference"),
    ]

def tuning_conditions(best_ref,g,atlas,available):
    a0=g[best_ref]; seq=[best_ref,"v400","v500"]; gs=[a0,g["v400"],g["v500"]]
    c=[]
    def seqcond(name,bounds=None,mode="hard",width=0.08,scale=1.0,later_scale=1.0,note=""):
        stages=[stage(seq[0],gs[0]*scale),stage(seq[1],gs[1]*scale*later_scale),stage(seq[2],gs[2]*scale*later_scale)]
        return condition(name,stages,boundaries=bounds,mode=mode,width=width,group="schedule_tuning",note=note)
    c += [
      seqcond(f"{best_ref}_v400_v500_switch_025_055",[0.25,0.55],note="earlier-than-equal switching control"),
      seqcond(f"{best_ref}_v400_v500_switch_045_072",[0.45,0.72],note="keep early reference longer"),
      seqcond(f"{best_ref}_v400_v500_switch_055_078",[0.55,0.78],note="keep early reference substantially longer"),
      seqcond(f"{best_ref}_v400_v500_switch_065_085",[0.65,0.85],note="late switching; close to static early reference"),
      seqcond(f"{best_ref}_v400_v500_smooth_w008",mode="smooth",width=0.08,note="smooth cross-fade around equal-third boundaries"),
      seqcond(f"{best_ref}_v400_v500_smooth_w016",mode="smooth",width=0.16,note="wider smooth cross-fade"),
      seqcond(f"{best_ref}_v400_v500_alpha075",scale=0.75,note="global schedule-gain sensitivity"),
      seqcond(f"{best_ref}_v400_v500_alpha125",scale=1.25,note="global schedule-gain sensitivity"),
      seqcond(f"{best_ref}_v400_v500_alpha150",scale=1.50,note="global schedule-gain sensitivity"),
      seqcond(f"{best_ref}_v400_v500_later125",later_scale=1.25,note="stage-local compensation: boost only mid/late"),
      seqcond(f"{best_ref}_v400_v500_later150",later_scale=1.50,note="stage-local compensation: boost only mid/late"),
    ]
    if "v600" in available:
        c += [
          condition(f"{best_ref}_v400_v600",[stage(best_ref,g[best_ref]),stage("v400",g["v400"]),stage("v600",g["v600"])],group="schedule_tuning",note="test more mature late reference"),
          condition(f"{best_ref}_v500_v600",[stage(best_ref,g[best_ref]),stage("v500",g["v500"]),stage("v600",g["v600"])],group="schedule_tuning",note="monotone maturity schedule with v600 late"),
        ]
    # Mean-RMS amplitude control: use only v270 identity, but match forward forcing RMS in each region.
    rr=atlas["region_gap_rms"]
    gam_mid=g["v400"]*float(rr["mid"]["v400"])/float(rr["mid"]["v270"])
    gam_late=g["v500"]*float(rr["late"]["v500"])/float(rr["late"]["v270"])
    c.append(condition("v270_amplitude_only_match_forward",[stage("v270",g["v270"]),stage("v270",gam_mid),stage("v270",gam_late)],
                       group="amplitude_control",note="v270-only direction with region-RMS forcing matched to v270/v400/v500 forward"))
    # Teacher-state diagnostic chooses identity without looking at FID.
    tr=atlas["teacher_stage_recommendation"]; ids=[tr[r]["reference"] for r in ("early","mid","late")]
    if all(n in g for n in ids):
        c.append(condition("teacher_atlas_identity_schedule",[stage(n,g[n]) for n in ids],group="teacher_predictor",
                           note="stage identities chosen only by held-out teacher residual projection; static-calibrated gamma"))
    return c

def run_atlas(strong,refs,a):
    od=a.output_root/"00_atlas"; od.mkdir(parents=True,exist_ok=True)
    request={"strong":strong["sha256"],"refs":{n:refs[n]["sha256"] for n in refs},"rollout_samples":a.atlas_rollout_samples,"teacher_samples":a.atlas_teacher_samples,"times":a.atlas_times}
    fp=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
    if (od/"summary.json").is_file() and (od/"request.sha256").is_file() and (od/"request.sha256").read_text().strip()==fp:
        print("[reuse] atlas",flush=True); return read_json(od/"summary.json")
    env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(a.gpu_indices[0]); env.setdefault("OMP_NUM_THREADS","1")
    cmd=[sys.executable,str(ATLAS),"--strong-checkpoint",str(a.strong_checkpoint),"--output-dir",str(od),"--rollout-samples",str(a.atlas_rollout_samples),
         "--teacher-samples",str(a.atlas_teacher_samples),"--batch-size",str(a.atlas_batch_size),"--device","cuda:0","--times",*map(str,a.atlas_times)]
    for n in refs: cmd += ["--reference-checkpoint",f"{n}={refs[n]['path']}"]
    run_logged(tuple(cmd),od/"atlas.log",env=env,monitored_gpu_indices=[a.gpu_indices[0]],memory_ceiling_mib=a.gpu_memory_ceiling_mib,
               memory_poll_interval=a.memory_poll_interval,resource_audit_path=od/"resource_audit.json")
    if not (od/"summary.json").is_file(): raise RuntimeError("atlas did not produce summary.json")
    (od/"request.sha256").write_text(fp+"\n"); return read_json(od/"summary.json")

def make_readme(summary,path):
    s=summary; lines=["# Checkpoint Reference Long Study v1","","## Headline",""]
    lines += [f"- best static reference: **{s['best_static']['reference']}**, gamma={s['best_static']['gamma']:.3g}, FID-1K={s['best_static']['fid']:.4f}"]
    if s["shapley"].get("available"):
        lines += [f"- factorial forward FID: {s['shapley']['full_fid']:.4f}",
                  f"- factorial baseline FID: {s['shapley']['baseline_fid']:.4f}",
                  "- Shapley FID-benefit attribution: "+", ".join(f"{k}={v:+.3f}" for k,v in s["shapley"]["shapley_fid_benefit"].items())]
    if s.get("best_dynamic"):
        b=s["best_dynamic"]; lines += [f"- best dynamic/tuning condition: **{b['condition']}**, FID-1K={b['fid']:.4f}, delta vs best static={b['fid']-s['best_static']['fid']:+.4f}"]
    lines += ["","## Files","",f"- all paired conditions: `{s['files']['all_conditions_csv']}`",f"- static maturity: `{s['files']['static_csv']}`",f"- atlas: `{s['files']['atlas_summary']}`","",
              "Interpret FID-1K differences below roughly 0.5 as screening-scale unless later confirmed with larger samples/seeds."]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--gpus",default="0,2"); p.add_argument("--checkpoint-dir",type=Path,default=CKPT_DIR)
    p.add_argument("--candidate-steps",nargs="+",type=int,default=list(DEFAULT_STEPS)); p.add_argument("--strong-checkpoint",type=Path,default=DEFAULT_STRONG)
    p.add_argument("--reference",type=Path,default=DEFAULT_REFERENCE); p.add_argument("--adm-python",type=Path,default=DEFAULT_ADM_PYTHON); p.add_argument("--output-root",type=Path,default=DEFAULT_OUT)
    p.add_argument("--num-samples",type=int,default=1000); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--vae-decode-batch-size",type=int,default=2); p.add_argument("--seed",type=int,default=0)
    p.add_argument("--num-output-points",type=int,default=250); p.add_argument("--atol",type=float,default=1e-6); p.add_argument("--rtol",type=float,default=1e-3)
    p.add_argument("--cuda-allocator-limit-gib",type=float,default=4.0); p.add_argument("--fid-batch-size",type=int,default=8); p.add_argument("--fid-gpu-memory-fraction",type=float,default=0.25)
    p.add_argument("--gpu-memory-ceiling-mib",type=int,default=15*1024); p.add_argument("--memory-poll-interval",type=float,default=0.25)
    p.add_argument("--atlas-rollout-samples",type=int,default=128); p.add_argument("--atlas-teacher-samples",type=int,default=128); p.add_argument("--atlas-batch-size",type=int,default=8)
    p.add_argument("--atlas-times",nargs="+",type=float,default=[0.05,0.15,0.25,0.35,0.50,0.65,0.75,0.85,0.95])
    p.add_argument("--keep-samples",action="store_true"); p.add_argument("--fail-fast",action="store_true"); p.add_argument("--dry-run",action="store_true")
    a=p.parse_args(); a.gpu_indices=parse_gpu_indices(a.gpus)
    if a.gpu_indices!=[0,2]: print(f"[note] requested GPUs are {a.gpu_indices}; project default is [0,2]",flush=True)
    if len(a.gpu_indices)!=2: raise ValueError("long study expects exactly two GPUs")
    a.checkpoint_dir=a.checkpoint_dir.expanduser().resolve(); a.strong_checkpoint=a.strong_checkpoint.expanduser().resolve(); a.reference=a.reference.expanduser().resolve()
    a.adm_python=a.adm_python.expanduser().absolute(); a.output_root=a.output_root.expanduser().resolve(); a.output_root.mkdir(parents=True,exist_ok=True)
    for f in (SAMPLER,ATLAS,FID_SCRIPT,a.strong_checkpoint,a.reference,a.adm_python):
        if not f.is_file(): raise FileNotFoundError(f)
    strong=checkpoint_meta(a.strong_checkpoint)
    refs={}
    missing=[]
    for step in a.candidate_steps:
        path=ckpt_path(a.checkpoint_dir,step); name=step_name(step)
        if not path.is_file(): missing.append({"reference":name,"step":step,"path":str(path)}); continue
        refs[name]=checkpoint_meta(path)
    for required in ("v270","v400","v500"):
        if required not in refs: raise FileNotFoundError(f"required {required} checkpoint is missing")
    validate_family(strong,refs)
    atomic_json_dump({"strong":strong,"references":refs,"missing_optional":missing,"gpus":a.gpu_indices,"candidate_steps":a.candidate_steps},a.output_root/"inventory.json")
    print("available refs:",", ".join(refs),flush=True); print("GPUs:",a.gpu_indices,flush=True)
    if "v700" in refs: print("[warning] v700 is present but was not a default candidate because the repository flags its standalone FID as anomalous",flush=True)
    if a.dry_run:
        print("dry-run: atlas -> adaptive static maturity -> factorial/replacement -> best-start -> schedule tuning",flush=True); return

    atlas=run_atlas(strong,refs,a)
    static_results,failures,static_rows,static_best=adaptive_static(atlas,strong,refs,a)
    summary_dir=a.output_root/"summary"; summary_dir.mkdir(parents=True,exist_ok=True)
    write_csv(summary_dir/"static_maturity_all.csv",sorted(static_rows,key=lambda r:(r["step"],r["gamma"])))
    best_rows=[{k:v for k,v in row.items()} for _,row in sorted(static_best.items(),key=lambda kv:refs[kv[0]]["step"])]
    write_csv(summary_dir/"static_maturity_best.csv",best_rows)
    best_ref=min(static_best,key=lambda n:static_best[n]["fid"]); best_static=static_best[best_ref]
    g={n:float(row["gamma"]) for n,row in static_best.items()}
    print(f"[static winner] {best_ref} gamma={g[best_ref]:g} FID={best_static['fid']:.4f}",flush=True)

    core=factorial_conditions(g); r,f=run_conditions(core,strong,refs,a,"02_causal_factorial"); failures+=f
    causal_rows=[result_row(x) for x in r]; write_csv(summary_dir/"causal_factorial_and_replacements.csv",causal_rows)
    shapley=shapley_from_factorial(causal_rows)

    bc=best_ref_conditions(best_ref,g,set(refs)); r3,f3=run_conditions(bc,strong,refs,a,"03_best_start"); failures+=f3
    beststart_rows=[result_row(x) for x in r3]; write_csv(summary_dir/"best_start_schedules.csv",beststart_rows)

    tc=tuning_conditions(best_ref,g,atlas,set(refs)); r4,f4=run_conditions(tc,strong,refs,a,"04_schedule_tuning"); failures+=f4
    tuning_rows=[result_row(x) for x in r4]; write_csv(summary_dir/"schedule_tuning.csv",tuning_rows)

    all_rows=static_rows+causal_rows+beststart_rows+tuning_rows
    write_csv(summary_dir/"all_conditions.csv",all_rows)
    noises={r["noise_sha256"] for r in all_rows}; labels={r["label_sha256"] for r in all_rows}
    if len(noises)!=1 or len(labels)!=1: raise RuntimeError(f"pairing failure: noise={len(noises)}, labels={len(labels)}")
    dynamic=[r for r in causal_rows+beststart_rows+tuning_rows if r["condition"]!="factorial_000"]
    best_dynamic=min(dynamic,key=lambda r:r["fid"]) if dynamic else None
    replacement={}
    by={r["condition"]:r for r in causal_rows}
    if "replace_mid_v270_v400_v270" in by:
        replacement["mid_replacement_minus_static_v270"]=by["replace_mid_v270_v400_v270"]["fid"]-static_best["v270"]["fid"]
    if "replace_late_v270_v270_v500" in by:
        replacement["late_replacement_minus_static_v270"]=by["replace_late_v270_v270_v500"]["fid"]-static_best["v270"]["fid"]
    final={"format":"eqvae_checkpoint_reference_long_study_v1","pairing_verified":True,"noise_sha256":next(iter(noises)),"label_sha256":next(iter(labels)),
      "strong":strong,"references":refs,"missing_optional":missing,"gammas_selected":g,
      "best_static":{"reference":best_ref,"gamma":g[best_ref],"fid":float(best_static["fid"]),"row":best_static},
      "best_static_by_reference":static_best,"shapley":shapley,"replacement_deltas":replacement,"best_dynamic":best_dynamic,
      "atlas_headline":{"gamma_rms_match":atlas["gamma_rms_match_to_v270_gamma3p5"],"teacher_stage_recommendation":atlas["teacher_stage_recommendation"]},
      "failures":failures,"files":{"all_conditions_csv":str(summary_dir/"all_conditions.csv"),"static_csv":str(summary_dir/"static_maturity_best.csv"),
                                   "atlas_summary":str(a.output_root/"00_atlas/summary.json"),"causal_csv":str(summary_dir/"causal_factorial_and_replacements.csv"),
                                   "tuning_csv":str(summary_dir/"schedule_tuning.csv")}}
    atomic_json_dump(final,summary_dir/"final_summary.json"); make_readme(final,summary_dir/"README.md")
    print("\n=== LONG STUDY COMPLETE ===",flush=True)
    print(f"best static: {best_ref} gamma={g[best_ref]:g} FID={best_static['fid']:.4f}",flush=True)
    if shapley.get("available"): print("Shapley FID benefit:",json.dumps(shapley["shapley_fid_benefit"],indent=2),flush=True)
    if best_dynamic: print(f"best dynamic: {best_dynamic['condition']} FID={best_dynamic['fid']:.4f} delta_vs_static={best_dynamic['fid']-best_static['fid']:+.4f}",flush=True)
    print(f"failures: {len(failures)}",flush=True); print("summary:",summary_dir/"final_summary.json",flush=True)

if __name__=="__main__": main()
