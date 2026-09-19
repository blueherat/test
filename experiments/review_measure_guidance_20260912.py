"""Read-only independent fit audit and paired exploratory bootstrap."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import csv
from experiments.lifting_scale_sweep_20260909 import sha

ROOT = Path('/home/zhoushunyu/data/eqvae/experiments/sit_measure_guidance_20260912')
OUT = Path('/home/zhoushunyu/eqvae/docs/data/sit_measure_guidance_20260912')


def main():
    raw = np.load(ROOT/'mechanism_raw.npz')
    source = json.loads((ROOT/'mechanism_check.json').read_text())
    assert sha(ROOT/'mechanism_raw.npz') == source['raw_sha256']
    assert len(set(raw['source_indices'])) == 128
    recorded = list(csv.DictReader((ROOT/'mechanism_results.csv').open()))
    rng = np.random.default_rng(2026120930)
    draws = rng.integers(64, size=(2000, 64))
    rows, maximum = [], 0.
    conditions = []
    for tv in (.15, .5, .85):
        values = {key:raw[f't{tv}_{key}'].astype(float).reshape(128, -1)
                  for key in ['strong', 'state', 'native', 'depth4_v', 'depth6_v', 'depth10_v',
                              'clt', 'heat', 'plain_mix', 'factor']}
        for head in ('depth4_v', 'depth6_v', 'depth10_v'):
            target = values['strong']-values[head]
            energies = np.square(target[64:]).sum(1)
            errors = {}
            for method in ('nuisance', 'clt', 'heat', 'plain_mix', 'factor'):
                basis = [values['strong'], values['state']]
                if method != 'nuisance':
                    basis += [values['native']-values[method]]
                gram = np.array([[np.sum(a[:64]*b[:64]) for b in basis] for a in basis])
                rhs = np.array([np.sum(a[:64]*target[:64]) for a in basis])
                condition = float(np.linalg.cond(gram))
                assert condition < 1e12
                conditions.append(condition)
                coefficient = np.linalg.solve(gram, rhs)
                prediction = sum(w*a[64:] for w, a in zip(coefficient, basis))
                errors[method] = np.square(target[64:]-prediction).sum(1)
                explained = 1-errors[method].sum()/energies.sum()
                if method != 'nuisance':
                    old = next(r for r in recorded if float(r['time']) == tv and
                               r['head'] == head and r['operator'] == method)
                    maximum = max(maximum, abs(explained-float(old['combined_explained'])))
            for alternative in ('nuisance', 'heat', 'plain_mix', 'factor'):
                per_image = errors[alternative]-errors['clt']
                distribution = per_image[draws].sum(1)/energies[draws].sum(1)
                low, high = np.quantile(distribution, [.025, .975])
                rows.append(dict(time=tv, head=head, comparison='clt_minus_'+alternative,
                    extra_explained=float(per_image.sum()/energies.sum()),
                    paired_bootstrap_low=float(low), paired_bootstrap_high=float(high),
                    images=64, bootstrap_draws=2000, fixed_fit=True, exploratory=True))
    assert maximum < 1e-8, maximum
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'paired_bootstrap.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # The historical metadata label counted all parameters, including fixed pos_embed.
    import torch
    model = torch.load(ROOT/'training/native/model.pt', map_location='cpu', weights_only=True)['ema']
    total = sum(p.numel() for p in model.values())
    fixed = model['pos_embed'].numel()
    metadata = json.loads((ROOT/'training/native/complete.json').read_text())
    assert metadata['trainable_parameters'] == total
    result = dict(passed=True, raw_sha256=source['raw_sha256'], independent_solver='normal equations',
        original_solver='least squares', maximum_explained_difference=maximum,
        maximum_gram_condition=max(conditions), confidence_intervals='paired 64-image percentile bootstrap, fixed fits',
        no_multiple_testing_correction=True, source=sha(__file__), no_gpu_used=True,
        metadata_clarification=dict(recorded_parameter_field_counts_all_parameters=total,
            fixed_position_parameters=fixed, actual_parameters_with_requires_grad=total-fixed),
        inverse_preflight_scope='two original-bank conditional trajectories, repeated on four GPUs; rho=1 FID controls still required')
    (OUT/'independent_mechanism_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    for row in rows:
        if row['head'] == 'depth4_v' and row['comparison'] in ('clt_minus_heat', 'clt_minus_plain_mix'):
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
