#!/usr/bin/env python3
"""One documented exploratory range expansion, preserving the original run."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from experiments.audit_ag_ig_smoothing_20260911 import (
    REPO, DATA, OUT, RAW, DELTAS, TIMES, SEED, digest, write_json, write_csv,
    mean_square, score_from_velocity,
)
from experiments.imagenet100_sit_multiscale_models import load_sit_field_model, evaluate_sit_field
from experiments.train_imagenet100_sit_flow import load_official_sit_module, DEFAULT_OFFICIAL_SIT_REPO

EXTRA=[.001,.003,.005,.02,.05,.075,.15,.2,.35,.75,1.5,3.,6.,8.,12.,16.,32.,64.]


def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    original=json.loads((RAW/'manifest.json').read_text())
    original_code=REPO/'experiments/audit_ag_ig_smoothing_20260911.py'
    assert digest(original_code)==original['script_sha256']
    stage=RAW/'range_check_20260912';stage.mkdir(exist_ok=True)
    if (stage/'manifest.json').exists():raise FileExistsError('Never overwrite an existing range check')
    shutil.copyfile(original_code,RAW/'original_audit_script.py')
    write_json(stage/'manifest.json',dict(completed=False,extra_deltas=EXTRA,script_sha256=digest(__file__),
               amendment_sha256=digest(REPO/'docs/AG_IG_SMOOTHING_DIAGNOSTIC_AMENDMENT_20260912_ZH.md')))
    start=time.monotonic()
    module,source=load_official_sit_module(DEFAULT_OFFICIAL_SIT_REPO,verify_source=True)
    model,semantics,metadata=load_sit_field_model(checkpoint_path=Path(original['strong']['checkpoint']),weights='ema',
                                                sit_module=module,source_metadata=source,device=torch.device('cpu'))
    labels=torch.from_numpy(np.load(RAW/'inputs.npz')['labels'])
    candidates=sorted(set(DELTAS+EXTRA))
    rng=np.random.default_rng(SEED+1);boot=rng.integers(0,16,(2000,16))
    curves=[];summaries=[];per_image=[]
    with torch.inference_mode():
        for ti,t in enumerate(TIMES):
            obs=np.load(RAW/f'time_{ti}.npz');q=float(obs['q']);s=obs['strong']
            y=torch.from_numpy(obs['z'])/t
            added={}
            for delta in EXTRA:
                shifted_t=1/(1+np.sqrt(q*(1+delta)));z=y*shifted_t
                v=evaluate_sit_field(model,semantics,z,torch.full((32,),shifted_t),labels)
                added[f'shift_{delta:g}']=score_from_velocity(z,shifted_t,v).numpy()
            np.savez_compressed(stage/f'time_{ti}.npz',**added)
            for name in [*original['heads'],'external_v500']:
                w=obs[f'weak_{name}'];gap2=mean_square(w.astype(np.float64)-s)
                errors=[]
                for delta in candidates:
                    field=added[f'shift_{delta:g}'] if delta in EXTRA else obs[f'shift_{delta:g}']
                    e=mean_square(w.astype(np.float64)-field);errors.append(e)
                    curves.append(dict(weak=name,t=t,q=q,delta=delta,tau=q*delta,
                                       fit_ratio=float(e[:16].sum()/gap2[:16].sum()),test_ratio=float(e[16:].sum()/gap2[16:].sum())))
                for mode in ['nonnegative','signed']:
                    allowed=[j for j,d in enumerate(candidates) if mode=='signed' or d>=0]
                    j=min(allowed,key=lambda k:errors[k][:16].sum());e=errors[j]
                    ratios=e[16:][boot].sum(1)/gap2[16:][boot].sum(1)
                    ci=np.quantile(ratios,[.025,.975]);d=candidates[j]
                    summaries.append(dict(weak=name,t=t,q=q,fit_type=f'finite_{mode}',delta=d,tau=d*q,
                                          fit_ratio=float(e[:16].sum()/gap2[:16].sum()),test_ratio=float(e[16:].sum()/gap2[16:].sum()),
                                          test_ratio_ci_low=ci[0],test_ratio_ci_high=ci[1],gap_rms=float(np.sqrt(gap2[16:].mean()))))
                    for i in range(32):
                        per_image.append(dict(weak=name,t=t,id=original['ids'][i],split=original['split'][i],fit_type=f'finite_{mode}',
                                              delta=d,tau=d*q,squared_residual=float(e[i]),squared_gap=float(gap2[i])))
            print(f'Range check t={t:.2f} complete',flush=True)
    manifest=json.loads((stage/'manifest.json').read_text());manifest.update(completed=True,wall_seconds=time.monotonic()-start,
               model_calls=len(EXTRA)*len(TIMES),strong=metadata,cuda_initialized=torch.cuda.is_initialized(),
               observations={p.name:digest(p) for p in stage.glob('time_*.npz')},original_manifest_sha256=digest(RAW/'manifest.json'))
    write_json(stage/'manifest.json',manifest);write_json(OUT/'refined_manifest.json',manifest)
    write_csv(OUT/'refined_fit_curves.csv',curves);write_csv(OUT/'refined_summary.csv',summaries);write_csv(OUT/'refined_per_image.csv',per_image)
    print(json.dumps([r for r in summaries if r['fit_type']=='finite_nonnegative'],indent=2),flush=True)


if __name__=='__main__':main()
