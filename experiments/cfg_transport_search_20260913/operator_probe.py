"""Paired real-SiT finite-time CFG transport diagnostics (no fresh noise).

All flows use the physical clock noise=0, image=1. An inverse uses a
descending grid, re-evaluating the velocity at the current state and time.
These 8-state operator measurements do not measure image quality.

CUDA_VISIBLE_DEVICES=0 python -m experiments.cfg_transport_search_20260913.operator_probe
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.guidance_pasted_20260912 import common as c


DEFAULT_SOURCE = c.EXPS / 'cfg_invariants_20260913/contracts'
DEFAULT_OUT = c.EXPS / 'cfg_transport_search_20260913/operator_probe'
DEFAULT_PAIRS = ((3., 1.), (3., 0.), (2., 0.), (2.5, 1.))


def rms(x):
    return x.double().flatten(1).square().mean(1).sqrt()


def dot(x, y):
    return (x.double()*y.double()).flatten(1).mean(1)


def branches(w):
    return 1 if w in (0., 1.) else 2


def pair(rt, z, t):
    labels = rt.labels
    conditional = rt.field(z, t, 'full')
    try:
        rt.labels = torch.full_like(labels, 100)
        unconditional = rt.field(z, t, 'full')
    finally:
        rt.labels = labels
    return conditional, unconditional


def velocity(rt, z, t, w):
    """Total CFG coefficient: 0=null, 1=conditional. Single branches are real costs."""
    if w == 1.:
        return rt.field(z, t, 'full')
    if w == 0.:
        labels = rt.labels
        try:
            rt.labels = torch.full_like(labels, 100)
            return rt.field(z, t, 'full')
        finally:
            rt.labels = labels
    vc, vu = pair(rt, z, t)
    return vu+w*(vc-vu)


def flow(rt, z, t0, t1, w, steps, solver='heun'):
    """Integrate the same field on either orientation; never negate twice."""
    assert steps > 0 and solver in ('heun', 'euler')
    grid = torch.linspace(t0, t1, steps+1, device=z.device, dtype=z.dtype)
    result = z.clone()
    for t, u in zip(grid[:-1], grid[1:]):
        dt = u-t
        first = velocity(rt, result, t, w)
        if solver == 'heun':
            second = velocity(rt, result+dt*first, u, w)
            result = result+.5*dt*(first+second)
        else:
            result = result+dt*first
    if not torch.isfinite(result).all():
        raise FloatingPointError(f'Nonfinite flow {t0}->{t1}, w={w}, steps={steps}')
    return result


def measured_flow(rt, z, t0, t1, w, steps, solver='heun'):
    torch.cuda.synchronize()
    begin = time.perf_counter()
    before = rt.counts.copy()
    result = flow(rt, z, t0, t1, w, steps, solver)
    torch.cuda.synchronize()
    record = dict(t_start=t0, t_end=t1, w=w, steps=steps, solver=solver,
                  seconds=time.perf_counter()-begin,
                  full_branch_nfe=rt.counts['full']-before['full'],
                  prefix_branch_nfe=rt.counts['prefix']-before['prefix'],
                  signed_step=(t1-t0)/steps,
                  state_and_time_reevaluated=True)
    expected = (2 if solver == 'heun' else 1)*steps*branches(w)
    assert record['full_branch_nfe'] == expected
    assert record['prefix_branch_nfe'] == 0
    return result, record


def geometry(delta, gap, scale=1.):
    gg = dot(gap, gap).clamp_min(1e-30)
    coefficient = dot(delta, gap)/gg
    perpendicular = delta.double()-coefficient[:, None, None, None]*gap.double()
    dr = rms(delta)
    gr = rms(gap)
    return dict(rms=dr,
                parallel_coefficient=coefficient,
                parallel_gain=coefficient/scale,
                parallel_signed_rms=coefficient*gr,
                orthogonal_rms=rms(perpendicular),
                orthogonal_fraction=rms(perpendicular)/dr.clamp_min(1e-30),
                cosine=dot(delta, gap)/(dr*gr).clamp_min(1e-30))


def columns(prefix, values):
    return {prefix+'_'+key: value for key, value in values.items()}


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_case(rt, x, gap, t, h, a, b, n):
    """R=F_b^{-1}F_a, cancel=F_b R, triple=F_a R; all F span [t,t+h]."""
    w = 2*a-b
    states, costs = {}, {}

    def leg(name, start, left, right, coeff, steps=n):
        states[name], costs[name] = measured_flow(rt, start, left, right, coeff, steps)
        return states[name]

    high = leg('high', x, t, t+h, a)
    same = leg('same_field_return', high, t+h, t, a)
    relative = leg('relative', high, t+h, t, b)
    cancel = leg('cancel', relative, t, t+h, b)
    triple = leg('triple', relative, t, t+h, a)
    ordinary = leg('ordinary', x, t, t+h, w)
    triple_nfe = sum(costs[key]['full_branch_nfe'] for key in ('high', 'relative', 'triple'))
    budget_steps = triple_nfe//(2*branches(w))
    assert budget_steps*(2*branches(w)) == triple_nfe
    budget = leg('ordinary_budget', x, t, t+h, w, budget_steps)
    delta = relative-x
    metrics = dict(
        state_rms=rms(x), gap_rms=rms(gap),
        same_return_rms=rms(same-x),
        same_return_over_transport=rms(same-x)/rms(delta).clamp_min(1e-30),
        cancel_rms=rms(cancel-high),
        cancel_over_high_displacement=rms(cancel-high)/rms(high-x).clamp_min(1e-30),
        linear_transport_error_rms=rms(delta-h*(a-b)*gap),
        linear_transport_relative_error=rms(delta-h*(a-b)*gap)/rms(delta).clamp_min(1e-30),
        triple_ordinary_difference_rms=rms(triple-ordinary),
        triple_ordinary_budget_difference_rms=rms(triple-budget),
        ordinary_budget_refinement_rms=rms(ordinary-budget),
        triple_ordinary_over_ordinary_displacement=rms(triple-ordinary)/rms(ordinary-x).clamp_min(1e-30),
        **columns('transport', geometry(delta, gap, h*(a-b))),
        **columns('triple_residual', geometry(triple-ordinary, gap)),
        **columns('triple_displacement', geometry(triple-x, gap)),
        **columns('ordinary_displacement', geometry(ordinary-x, gap)),
    )
    base = dict(t=t, h=h, a=a, b=b, effective_w=w, substeps=n,
                relative_nfe=sum(costs[key]['full_branch_nfe'] for key in ('high', 'relative')),
                cancellation_nfe=sum(costs[key]['full_branch_nfe'] for key in ('high', 'relative', 'cancel')),
                triple_nfe=triple_nfe, ordinary_nfe=costs['ordinary']['full_branch_nfe'],
                ordinary_budget_nfe=costs['ordinary_budget']['full_branch_nfe'],
                ordinary_budget_steps=budget_steps,
                triple_seconds=sum(costs[key]['seconds'] for key in ('high', 'relative', 'triple')),
                ordinary_seconds=costs['ordinary']['seconds'],
                ordinary_budget_seconds=costs['ordinary_budget']['seconds'],
                all_diagnostic_nfe=sum(v['full_branch_nfe'] for v in costs.values()),
                all_diagnostic_seconds=sum(v['seconds'] for v in costs.values()))
    assert base['triple_nfe'] == base['ordinary_budget_nfe']
    cpu = {key: value.detach().cpu().numpy() for key, value in metrics.items()}
    samples = [dict(base, sample=i, label=int(rt.labels[i]), **{k: float(v[i]) for k, v in cpu.items()})
               for i in range(len(x))]
    aggregate = dict(base, **{k: float(v.mean()) for k, v in cpu.items()})
    return states, costs, samples, aggregate


def refinement_rows(low, high, x, gap, base):
    transport_scale = rms(high['relative']-x).clamp_min(1e-30)
    triple_difference = rms(high['triple']-high['ordinary']).clamp_min(1e-30)
    values = {}
    for key in high:
        values[key+'_refinement_rms'] = rms(low[key]-high[key])
    values['transport_refinement_fraction'] = values['relative_refinement_rms']/transport_scale
    values['triple_refinement_over_structural_difference'] = values['triple_refinement_rms']/triple_difference
    # The final reference leg is compared to its own high target, with a
    # separate check against the finer forward target to avoid target mixing.
    values['coarse_cancel_to_fine_high_rms'] = rms(low['cancel']-high['high'])
    values['fine_cancel_to_fine_high_rms'] = rms(high['cancel']-high['high'])
    values['same_return_refinement_reduction'] = rms(low['same_field_return']-x)/rms(high['same_field_return']-x).clamp_min(1e-30)
    cpu = {key: value.cpu().numpy() for key, value in values.items()}
    return [dict(base, sample=i, **{k: float(v[i]) for k, v in cpu.items()}) for i in range(len(x))], dict(base, **{k: float(v.mean()) for k, v in cpu.items()})


def report(out, summary, rows, refinement, matched):
    fine = [row for row in rows if row['substeps'] == max(summary['substeps'])]
    lines = ['# Real-SiT CFG transport operator probe', '',
             'Eight paired states at each physical time; noise=0, image=1. No new noise, no pixel refeeding, no image-quality claim.', '',
             'The saved trajectories were generated at constant total CFG w=2.25. This probe uses constant fields over each local interval, including after t=.75.', '',
             'F_a maps t to t+h. R=F_b^-1 F_a returns to t; F_b R is a cancellation control; F_a R is the three-leg candidate. A descending grid alone implements reversal.', '',
             'Total CFG uses v_u+w(v_c-v_u), so w=0 is null and w=1 is conditional. Those endpoints use one model branch; other coefficients use two. Triple and ordinary_budget costs are exactly matched in branch NFE.', '',
             'All figures below are operator distances, not classifier/FID scores. RMS is per latent coordinate, then averaged over the 8 paired examples.', '',
             '|t|h|a,b|W=2a-b|R parallel gain|R orthogonal fraction|same-field RMS|cancel RMS|triple vs W RMS|triple vs budget W RMS|',
             '|---|---|---|---|---|---|---|---|---|---|']
    for row in fine:
        lines.append('|'+ '|'.join([f"{row['t']:.2f}", f"{row['h']:.5f}", f"{row['a']:g},{row['b']:g}", f"{row['effective_w']:g}",
                  *[f"{row[k]:.6g}" for k in ('transport_parallel_gain', 'transport_orthogonal_fraction', 'same_return_rms', 'cancel_rms', 'triple_ordinary_difference_rms', 'triple_ordinary_budget_difference_rms')]])+'|')
    lines.extend(['', 'Parallel gain is <R(x)-x,g> / [h(a-b)||g||²]; the corresponding signed coefficient and signed parallel RMS are retained in CSV. Orthogonal fraction measures departure from the current gap line, not superiority to a future CFG trajectory.', '',
                  'The 8/16-step comparisons are in refinement.csv. Both grids are finite-precision approximations; tiny error ratios near the float32 floor should not be treated as convergence orders.', '',
                  'The same effective W=4 null and conditional reference comparison is in matched_w4.csv, with the identical source state and ordinary-CFG reference. Equality of first-order W does not guarantee equality of the finite displacement projection.', '',
                  f"Total diagnostic model branch calls: {summary['full_branch_calls']}; measured integration seconds: {summary['integration_seconds']:.3f}; elapsed seconds including loading and output: {summary['elapsed_seconds']:.3f}.", '',
                  'Each branch call processes all 8 examples. Per-output logical NFE is reported separately from the number of output paths and from wall time. Diagnostics include independently computed checks and must not be confused with the deployment cost of the three-leg candidate.', '',
                  'GPU0 contained unrelated small workloads at launch; observed timing can include contention. No other process was modified.', ''])
    (out/'report.md').write_text('\n'.join(lines))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--times', type=float, nargs='+', default=[.25, .5, .75])
    parser.add_argument('--widths', type=float, nargs='+', default=[1/32, 1/16, 1/8])
    parser.add_argument('--substeps', type=int, nargs=2, default=[8, 16])
    args = parser.parse_args()
    assert 0 < args.substeps[0] < args.substeps[1]
    assert all(0 < t < t+h <= 1 for t in args.times for h in args.widths)
    if args.out.exists():
        raise FileExistsError(f'Use a fresh output directory: {args.out}')
    args.out.mkdir(parents=True)
    begin = time.perf_counter()
    c.atomic(args.out/'status.json', dict(status='loading', complete=False))
    files = {t: args.source/f't{round(t*64):03d}.npz' for t in args.times}
    source_request = c.read(args.source/'request.json')
    for path, digest in source_request['source_hashes'].items():
        assert c.sha(path) == digest, f'Source changed since contracts: {path}'
    rt = c.runtime('sit_small')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    request = dict(model='sit_small', samples=8, times=args.times, widths=args.widths,
                   substeps=args.substeps, pairs=[dict(a=a, b=b, effective_w=2*a-b) for a,b in DEFAULT_PAIRS],
                   physical_clock='noise=0,image=1', local_guidance_schedule='constant on the full interval',
                   inverse='descending physical-time grid; state and original time reevaluated every query',
                   source_request=source_request, source_request_sha256=c.sha(args.source/'request.json'),
                   input_files={str(p): c.sha(p) for p in files.values()},
                   original_noise_sha256=c.sha(args.source/'t000.npz'),
                   source_hashes={**rt.sources, str(Path(c.__file__).resolve()): c.sha(c.__file__), str(Path(__file__).resolve()): c.sha(__file__)},
                   asset_hashes={str(p): c.sha(p) for p in c.asset_paths('sit_small')},
                   dtype='float32 model and states; float64 metric accumulation', tf32=False,
                   visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   cuda_device=torch.cuda.get_device_name(), no_fresh_noise=True,
                   quality_experiment=False, independent_inverse=True,
                   nfe='number of full-network branch calls per example, batched over 8 examples',
                   budget_match='ordinary CFG Heun steps chosen to match three-leg full-branch NFE exactly')
    c.atomic(args.out/'request.json', request)
    request_hash = c.sha(args.out/'request.json')
    rows, sample_rows, refinement, refinement_samples, matched = [], [], [], [], []
    all_costs = []
    before = rt.counts.copy()
    finished = 0
    total = len(args.times)*len(args.widths)*len(DEFAULT_PAIRS)
    labels_expected = None
    for t in args.times:
        with np.load(files[t]) as saved:
            assert float(saved['t']) == t
            x = c.cuda(saved['latents'])
            labels = c.cuda(saved['labels'])
        assert x.shape == (8, 4, 32, 32)
        if labels_expected is None:
            labels_expected = labels.clone()
        assert torch.equal(labels, labels_expected)
        rt.labels = labels
        vc, vu = pair(rt, x, torch.tensor(t, device=x.device, dtype=x.dtype))
        gap = vc-vu
        for h in args.widths:
            w4 = {}
            for a, b in DEFAULT_PAIRS:
                case = f't{t:.5f}_h{h:.5f}_a{a:g}_b{b:g}'
                outputs = {}
                for n in args.substeps:
                    states, costs, per_sample, aggregate = run_case(rt, x, gap, t, h, a, b, n)
                    outputs[n] = states
                    rows.append(aggregate)
                    sample_rows.extend(per_sample)
                    all_costs.append(dict(case=case, substeps=n, legs=costs))
                    np.savez_compressed(args.out/f'{case}_n{n}.npz',
                        request_sha256=request_hash, source_latents=x.cpu().numpy(),
                        labels=labels.cpu().numpy(), gap=gap.cpu().numpy(),
                        **{key: value.cpu().numpy() for key, value in states.items()})
                base = dict(t=t, h=h, a=a, b=b, effective_w=2*a-b,
                            coarse_substeps=args.substeps[0], fine_substeps=args.substeps[1])
                refined_samples, refined = refinement_rows(outputs[args.substeps[0]], outputs[args.substeps[1]], x, gap, base)
                refinement_samples.extend(refined_samples)
                refinement.append(refined)
                if 2*a-b == 4:
                    w4[b] = outputs[args.substeps[-1]]
                finished += 1
                c.atomic(args.out/'status.json', dict(status='running', complete=False, finished_cases=finished, total_cases=total, latest_case=case))
                print(json.dumps(dict(case=case, finished=finished, total=total,
                    transport_orth=rows[-1]['transport_orthogonal_fraction'],
                    transport_gain=rows[-1]['transport_parallel_gain'],
                    same_return_rms=rows[-1]['same_return_rms'],
                    cancel_rms=rows[-1]['cancel_rms'],
                    triple_vs_cfg=rows[-1]['triple_ordinary_difference_rms'],
                    numerical_fraction=refined['triple_refinement_over_structural_difference'])), flush=True)
                write_csv(args.out/'operators.csv', rows)
                write_csv(args.out/'refinement.csv', refinement)
            difference = w4[1.]['triple']-w4[0.]['triple']
            mm = geometry(difference, gap)
            mm['difference_over_null_vs_ordinary'] = rms(difference)/rms(w4[0.]['triple']-w4[0.]['ordinary']).clamp_min(1e-30)
            mm['ordinary_reference_difference_rms'] = rms(w4[1.]['ordinary']-w4[0.]['ordinary'])
            matched.append(dict(t=t, h=h, effective_w=4., substeps=args.substeps[-1],
                                **{key: float(value.mean()) for key, value in mm.items()}))
    for path, digest in request['source_hashes'].items():
        assert c.sha(path) == digest, f'Source changed while running: {path}'
    write_csv(args.out/'operators_per_sample.csv', sample_rows)
    write_csv(args.out/'refinement_per_sample.csv', refinement_samples)
    write_csv(args.out/'matched_w4.csv', matched)
    c.atomic(args.out/'leg_costs.json', all_costs)
    summary = dict(complete=True, quality_experiment=False, finished_cases=finished,
                   examples_per_case=8, times=args.times, widths=args.widths,
                   substeps=args.substeps, request_sha256=request_hash,
                   full_branch_calls=rt.counts['full']-before['full'],
                   prefix_branch_calls=rt.counts['prefix']-before['prefix'],
                   integration_seconds=sum(row['all_diagnostic_seconds'] for row in rows),
                   elapsed_seconds=time.perf_counter()-begin,
                   source_manifest_verified_at_completion=True,
                   max_same_return_rms=max(row['same_return_rms'] for row in rows),
                   max_fine_cancel_rms=max(row['cancel_rms'] for row in rows if row['substeps']==args.substeps[-1]))
    assert summary['full_branch_calls'] == sum(row['all_diagnostic_nfe'] for row in rows)+2*len(args.times)
    c.atomic(args.out/'summary.json', summary)
    report(args.out, summary, rows, refinement, matched)
    c.atomic(args.out/'status.json', dict(status='complete', complete=True, finished_cases=finished, total_cases=total))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
