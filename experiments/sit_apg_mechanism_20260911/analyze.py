"""Reproducible analysis of the held-out mechanism study; never reads sweep FID."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from experiments.sit_apg_mechanism_20260911 import study
from experiments.lifting_scale_sweep_20260909 import WORK, atomic, read, sha

OUT = WORK/'docs/data/apg_mechanism_extension_20260911'
KEY = ['sample_id', 'amount', 'time']


def interval(frame, column):
    """Resample the 32 independent seed/class pairs, keeping repeated states together."""
    values = frame.groupby('sample_id')[column].mean().to_numpy()
    rng = np.random.default_rng(20260911)
    means = values[rng.integers(0, len(values), (4000, len(values)))].mean(1)
    lo, hi = np.quantile(means, [.025, .975])
    return dict(mean=float(values.mean()), ci_low=float(lo), ci_high=float(hi),
                seeds=len(values), observations=len(frame))


def main():
    request = study.verify()
    completed = read(study.STUDY/'completed.json')
    assert completed['passed'] and completed['request_sha256'] == sha(study.STUDY/'request.json')
    curls, effects, hashes = [], [], {}
    for rank, result in enumerate(completed['ranks']):
        root = study.STUDY/f'rank{rank}'
        for name, digest in result['output_hashes'].items():
            assert sha(root/name) == digest
            hashes[str(root/name)] = digest
        for record in result['files']:
            assert sha(root/record['file']) == record['sha256']
        curls.append(pd.read_json(root/'curl_rows.json'))
        effects.append(pd.read_json(root/'effect_rows.json'))
    c, e = pd.concat(curls, ignore_index=True), pd.concat(effects, ignore_index=True)
    assert len(c) == 17760 and len(e) == 8160 and c.sample_id.nunique() == 32
    OUT.mkdir(parents=True, exist_ok=True)
    c.to_csv(OUT/'curl_observations.csv', index=False)
    e.to_csv(OUT/'effect_observations.csv', index=False)
    fd = c[c.field == 'gap_fd_scale_check']
    cc = c[c.probe >= 0].copy()
    field_rows = []
    for (amount, field), group in cc.groupby(['amount', 'field']):
        by_seed = group.groupby('sample_id')[['antisymmetric_frobenius_sq', 'jacobian_frobenius_sq']].mean()
        values = by_seed.to_numpy()
        rng = np.random.default_rng(20260911)
        resampled = values[rng.integers(0, len(values), (4000, len(values)))].mean(1)
        boot = resampled[:, 0]/resampled[:, 1]
        lo, hi = np.quantile(boot, [.025, .975])
        field_rows.append(dict(amount=amount, field=field, skew_sq=values[:, 0].mean(),
            jacobian_sq=values[:, 1].mean(), ratio_sq=values[:, 0].mean()/values[:, 1].mean(),
            ci_low=lo, ci_high=hi))
    pd.DataFrame(field_rows).to_csv(OUT/'field_summary.csv', index=False)
    component_rows = []
    components = e[(e.kind == 'component') & (e.grid == 16)]
    for (amount, variant), group in components.groupby(['amount', 'variant']):
        for metric in ('q_gain', 'energy_change', 'moments_change'):
            component_rows.append(dict(amount=amount, variant=variant, metric=metric,
                **interval(group, metric), positive_fraction=float((group[metric] > 0).mean())))
    pd.DataFrame(component_rows).to_csv(OUT/'component_summary.csv', index=False)
    grid = e[(e.kind == 'component') & (e.variant != 'zero')].pivot(
        index=KEY+['variant'], columns='grid', values='q_gain').reset_index()
    grid['same_sign'] = (grid[8] > 0) == (grid[16] > 0)
    grid['absolute_disagreement'] = (grid[8]-grid[16]).abs()
    grid_rows = []
    for variant, group in grid.groupby('variant'):
        grid_rows.append(dict(variant=variant, sign_agreement=group.same_sign.mean(),
            pearson=group[8].corr(group[16]), spearman=spearmanr(group[8], group[16]).statistic,
            mean_absolute_disagreement=group.absolute_disagreement.mean(),
            fine_positive_fraction=(group[16] > 0).mean()))
    pd.DataFrame(grid_rows).to_csv(OUT/'future_grid_agreement.csv', index=False)
    integration = e[e.kind == 'integration_future'].pivot(
        index=KEY+['variant'], columns='grid', values='q').reset_index()
    integration['q_refinement_gain'] = integration[4]-integration[1]
    integration_rows = [dict(amount=amount, variant=variant, **interval(group, 'q_refinement_gain'),
        mean_absolute_gain=group.q_refinement_gain.abs().mean())
        for (amount, variant), group in integration.groupby(['amount', 'variant'])]
    pd.DataFrame(integration_rows).to_csv(OUT/'refinement_summary.csv', index=False)
    local = e[(e.kind == 'integration') & (e.grid == 1)].copy()
    field_state = cc.groupby(KEY+['field'])[['antisymmetric_frobenius_sq', 'jacobian_frobenius_sq']].mean().reset_index()
    correlations = []
    for variant in ('cfg', 'projected'):
        joined = local[local.variant == variant].merge(field_state[field_state.field == variant], on=KEY)
        for metric in ('antisymmetric_frobenius_sq', 'jacobian_frobenius_sq'):
            raw = spearmanr(joined[metric], joined.local_error_rms).statistic
            # Remove the known time/strength strata by ranking within each stratum.
            ranks = joined.groupby(['amount', 'time'])[[metric, 'local_error_rms']].rank(pct=True)
            correlations.append(dict(variant=variant, metric=metric, spearman=raw,
                within_time_amount_rank_correlation=ranks.iloc[:, 0].corr(ranks.iloc[:, 1])))
    pd.DataFrame(correlations).to_csv(OUT/'local_error_correlations.csv', index=False)
    reliable = fd[fd.derivative_relative_disagreement <= .1][KEY]
    reliable_fields = cc.merge(reliable, on=KEY).groupby('field')[
        ['antisymmetric_frobenius_sq', 'jacobian_frobenius_sq']].mean()
    reliable_fields['ratio_sq'] = reliable_fields.iloc[:, 0]/reliable_fields.iloc[:, 1]
    reliable_fields.to_csv(OUT/'fd_reliable_subset.csv')
    parallel = components[components.variant == 'parallel']
    gain = e[e.kind == 'integration_future'].pivot(index=KEY+['grid'], columns='variant', values='q')
    null = e[e.kind == 'passive_grid_defect'].set_index(KEY).q
    release = gain.reset_index().query('grid == 4').set_index(KEY)
    release['conditional_value'] = release.cfg-null
    release['projected_value'] = release.projected-null
    release.reset_index().to_csv(OUT/'matched_release_value.csv', index=False)
    summary = dict(passed=True, request_sha256=sha(study.STUDY/'request.json'),
        inputs=hashes, independent_seed_class_pairs=32, state_observations=480,
        random_bilinear_probes_per_state=4, archived_state_batches=120,
        bootstrap='4000 resamples of independent seed/class pairs; repeated states kept together; descriptive, unadjusted intervals',
        estimator='ratio of aggregated estimates of ||J-J^T||_F^2 and ||J||_F^2; finite-probe ratio is biased; neither spectral norm nor probability',
        fd_median=float(fd.derivative_relative_disagreement.median()),
        fd_p90=float(fd.derivative_relative_disagreement.quantile(.9)),
        fd_p99=float(fd.derivative_relative_disagreement.quantile(.99)),
        fd_above_10pct=int((fd.derivative_relative_disagreement > .1).sum()),
        clean_gap_norm_median=float(components[components.variant == 'zero'].clean_gap_norm.median()),
        published_clean_apg_radius_center=5., radius_grid=[2.5, 5., 10.],
        parallel_q_gain=interval(parallel, 'q_gain'),
        orthogonal_q_gain=interval(components[components.variant == 'orthogonal'], 'q_gain'),
        parallel_positive_fraction=float((parallel.q_gain > 0).mean()),
        all_full_calls=sum(r['full_calls'] for r in completed['ranks']),
        all_decoder_images=sum(r['decoder_images'] for r in completed['ranks']),
        all_classifier_images=sum(r['classifier_images'] for r in completed['ranks']),
        max_worker_wall_seconds=max(r['wall_seconds'] for r in completed['ranks']),
        no_fid=True, no_sweep_results_read=True, no_claim_of_image_quality_improvement=True,
        source_hashes={str(Path(__file__).resolve()):sha(__file__)})
    atomic(OUT/'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
