"""Real SiT numerical contracts, representation controls, and five projections.

These are numerical/implementation diagnostics, not image-quality evidence.
No fresh-noise img2img operation is described as identity or refeeding here.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from experiments.guidance_pasted_20260912 import common


def rows_csv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rms(x):
    return x.double().flatten(1).square().mean(1).sqrt()


def pair(rt, z, t):
    labels = rt.labels
    vc = rt.field(z, t, 'full')
    try:
        rt.labels = torch.full_like(labels, 100)
        vu = rt.field(z, t, 'full')
    finally:
        rt.labels = labels
    return vc, vu


def field(rt, z, t, w):
    vc, vu = pair(rt, z, t)
    return vu+w*(vc-vu)


def integrate(rt, start, grid, w, save=False):
    z = start.clone()
    states = {0: z.clone()} if save else {}
    steps = len(grid)-1
    for i, (t, u) in enumerate(zip(grid[:-1], grid[1:])):
        h = u-t
        k1 = field(rt, z, t, w)
        k2 = field(rt, z+h*k1, u, w)
        z = z+.5*h*(k1+k2)
        if not torch.isfinite(z).all():
            raise FloatingPointError(f'Nonfinite trajectory at {float(u)}')
        if save and (i+1) % (steps//4) == 0:
            states[i+1] = z.clone()
    return z, states


def gallery(path, columns, labels):
    width, height, header = 160, 180, 40
    canvas = Image.new('RGB', (width*len(columns), header+height*len(labels)), 'white')
    draw = ImageDraw.Draw(canvas)
    for j, (title, pixels) in enumerate(columns):
        draw.text((width*j+4, 6), title, fill='black')
        for i, p in enumerate(pixels):
            y = header+i*height
            canvas.paste(Image.fromarray(p).resize((width, width)), (width*j, y))
            draw.text((width*j+4, y+width+2), f'image {i}, class {int(labels[i])}', fill='black')
    canvas.save(path)


def representation_checks(gaps):
    """The same frozen gap sequence, expressed in two prediction units.

    Its canonical history advances once per listed snapshot. This isolates
    the representation law rather than proposing a new physical sampler.
    """
    k, lam = .3, .05
    history = None
    naive = {'epsilon': None, 'clean': None}
    fixed_history = {'epsilon': None, 'clean': None}
    previous_scale = {'epsilon': None, 'clean': None}
    records, project_rows = [], []
    for t, gap in gaps:
        if not 0 < t < 1:
            continue
        gap = gap.double()
        prior = gap if history is None else history
        canonical = gap-k*torch.sign(gap-prior+lam*prior)
        for representation, scale in [('epsilon', -t), ('clean', 1-t)]:
            encoded = scale*gap
            # Consistent transport of stored units and control amplitude.
            old = fixed_history[representation]
            transported = encoded if old is None else old*(scale/previous_scale[representation])
            corrected = encoded-abs(scale)*k*torch.sign(encoded-transported+lam*transported)
            fixed_history[representation] = corrected
            previous_scale[representation] = scale
            # Deliberately naive reuse of numeric K and untransported history.
            old_naive = encoded if naive[representation] is None else naive[representation]
            bad = encoded-k*torch.sign(encoded-old_naive+lam*old_naive)
            naive[representation] = bad
            records.append(dict(t=t, representation=representation,
                fixed_error_max=float((corrected/scale-canonical).abs().max()),
                naive_error_rms=float(rms(bad/scale-canonical).mean()),
                frozen_sequence_not_sample_quality=True))
        # Project a controller proposal to one fixed line segment, then apply
        # the identical operator five times, without re-estimating its set.
        denominator=gap.flatten(1).square().sum(1).clamp_min(1e-24)
        def projection(proposal):
            coefficient=(proposal*gap).flatten(1).sum(1)/denominator
            return coefficient.clamp(0., 1.)[:,None,None,None]*gap
        projected = canonical
        first = None
        for iteration in range(1, 6):
            before = projected
            projected = projection(projected)
            if first is None:
                first = projected.clone()
            project_rows.append(dict(t=t, iteration=iteration,
                change_previous_rms=float(rms(projected-before).mean()),
                change_first_max=float((projected-first).abs().max()),
                frozen_set=True))
        history = canonical
    return records, project_rows


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=Path('/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/contracts'))
    p.add_argument('--samples',type=int,default=8)
    p.add_argument('--steps',type=int,default=64)
    p.add_argument('--seed',type=int,default=2026091329)
    args=p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    begin=time.time()
    common.atomic(args.out/'status.json',dict(status='loading',started=begin))
    rt=common.runtime('sit_small')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    rng=torch.Generator(device='cuda').manual_seed(args.seed)
    noise=torch.randn((args.samples,4,32,32),generator=rng,device='cuda')
    labels=torch.arange(args.samples,device='cuda')*7 % 100
    rt.labels=labels
    grid=torch.linspace(0,1,args.steps+1,device='cuda')
    fine=torch.linspace(0,1,args.steps*2+1,device='cuda')
    w=2.25
    request=dict(samples=args.samples,seed=args.seed,steps=args.steps,guidance=w,
        guidance_schedule='constant for this numerical contract',
        checkpoints=[0,.25,.5,.75,1.],independent_suffix_recomputation=True,
        quality_experiment=False,noising_or_refeeding=False,
        source_hashes=rt.sources,script_sha256=common.sha(Path(__file__)))
    common.atomic(args.out/'request.json',request)
    before=rt.counts.copy()
    endpoint,states=integrate(rt,noise,grid,w,True)
    fine_endpoint,_=integrate(rt,noise,fine,w)
    rows=[]
    columns={name:[] for name in ['clean_predictions','recomputed_endpoints','changed_future']}
    gaps=[]
    for step,z in states.items():
        t=grid[step]
        vc,vu=pair(rt,z,t)
        gaps.append((float(t),vc-vu))
        predicted=z+(1-t)*vc
        predicted_guided=z+(1-t)*(vu+w*(vc-vu))
        same,_=integrate(rt,z,grid[step:],w)
        refined,_=integrate(rt,z,fine[2*step:],w)
        changed,_=integrate(rt,z,grid[step:],1.)
        same_error=float((same-endpoint).abs().max())
        if same_error > 2e-5:
            raise AssertionError(f'Same discrete suffix contract failed: {same_error}')
        pixels={name:rt.decode(value) for name,value in
                [('clean',predicted),('endpoint',same),('changed',changed)]}
        np.savez(args.out/f't{step:03d}.npz',t=float(t),latents=z.cpu().numpy(),
            labels=labels.cpu().numpy(),clean_prediction=predicted.cpu().numpy(),
            guided_clean_prediction=predicted_guided.cpu().numpy(),
            recomputed_endpoint=same.cpu().numpy(),refined_suffix=refined.cpu().numpy(),
            changed_future_endpoint=changed.cpu().numpy(),**pixels)
        for i in range(args.samples):
            Image.fromarray(pixels['clean'][i]).save(args.out/f'image{i:02d}_t{step:03d}_clean.png')
            Image.fromarray(pixels['endpoint'][i]).save(args.out/f'image{i:02d}_t{step:03d}_endpoint.png')
        columns['clean_predictions'].append((f't={float(t):.2f} clean',pixels['clean']))
        columns['recomputed_endpoints'].append((f't={float(t):.2f} endpoint',pixels['endpoint']))
        columns['changed_future'].append((f't={float(t):.2f} future w=1',pixels['changed']))
        row=dict(t=float(t),same_discrete_suffix_max_error=same_error,
            refined_suffix_rms_to_discrete_endpoint=float(rms(refined-endpoint).mean()),
            posterior_clean_rms_to_endpoint=float(rms(predicted-endpoint).mean()),
            guided_clean_rms_to_endpoint=float(rms(predicted_guided-endpoint).mean()),
            changed_future_rms_to_endpoint=float(rms(changed-endpoint).mean()))
        rows.append(row)
        common.atomic(args.out/'status.json',dict(status='running',finished_times=len(rows),total_times=5))
        print(json.dumps(row),flush=True)
    representation,projections=representation_checks(gaps)
    rows_csv(args.out/'trajectory.csv',rows)
    rows_csv(args.out/'representation.csv',representation)
    rows_csv(args.out/'five_projections.csv',projections)
    for name,values in columns.items():
        gallery(args.out/f'{name}.png',values,labels.cpu().numpy())
    fixed_error=max(r['fixed_error_max'] for r in representation)
    projection_error=max(r['change_first_max'] for r in projections)
    assert fixed_error < 1e-10
    assert projection_error < 1e-10
    summary=dict(complete=True,seconds=time.time()-begin,quality_experiment=False,
        same_discrete_suffix_max_error=max(r['same_discrete_suffix_max_error'] for r in rows),
        coarse_fine_full_endpoint_rms=float(rms(fine_endpoint-endpoint).mean()),
        fixed_representation_max_error=fixed_error,
        naive_representation_max_rms=max(r['naive_error_rms'] for r in representation),
        five_frozen_projection_max_error=projection_error,
        model_calls={k:rt.counts[k]-before[k] for k in before},
        explanation='Same-solver identity is algebraic; refined suffix measures numerical error. Prediction change and changed-future results do not establish image quality.')
    common.atomic(args.out/'summary.json',summary)
    common.atomic(args.out/'status.json',dict(status='complete',finished_times=5,total_times=5))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
