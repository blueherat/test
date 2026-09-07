"""Export the completed paired-5K discovery comparison; no quality selection."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    assert summary['complete'] and summary['completed_additional_idea_rounds'] == 3
    rows = summary['paired5k_rows'][1:]
    labels = {
        'guidance_2m': 'Guidance quadrature', 'full_2m': 'Full quadrature',
        'exponential_2m': 'Complete second-order quadrature',
        'directional_variance': 'Global covariance (prior)',
        'spatial_full': 'Spatial covariance: full', 'spatial_diagonal': 'Spatial covariance: diagonal',
        'causal_reference': 'Round 2: finite Base response',
        'full_read_after_write': 'Round 3: Full reads written latent',
    }
    gains = np.array([r['improvement_percent'] for r in rows])
    costs = np.array([r['cost_ratio_to_official'] for r in rows])
    y = np.arange(len(rows))
    fig, (quality, cost) = plt.subplots(1, 2, figsize=(14, 7.2), sharey=True,
        gridspec_kw={'width_ratios': [2.6, 1]})
    colors = ['#278179' if x >= 0 else '#b25e4b' for x in gains]
    quality.barh(y, gains, color=colors, height=.52)
    quality.axvline(0, color='#737373', lw=.8)
    quality.axvline(3, color='#252525', lw=1.2, ls='--')
    quality.text(3, -.65, '3% target', ha='center', va='bottom', fontsize=10)
    left, right = min(-.65, float(gains.min())-.7), max(3.65, float(gains.max())+.7)
    quality.set_xlim(left, right)
    for yy, value in zip(y, gains):
        quality.text(value+(.04 if value >= 0 else -.04), yy, f'{value:+.3f}%',
                     va='center', ha='left' if value >= 0 else 'right', fontsize=9)
    quality.set_yticks(y, [f"{labels[r['method']]}\nFID {r['fid']:.5f}" for r in rows], fontsize=10)
    quality.invert_yaxis()
    quality.set_xlabel('Relative FID reduction (%)  |  larger is better')
    quality.set_title('Quality point estimates', fontsize=12, pad=21)
    cost.barh(y, costs, color='#748b9d', height=.52)
    cost.axvline(1, color='#252525', lw=1, ls='--')
    cost.set_xlim(0, max(2.25, float(costs.max())+.28))
    cost.set_xlabel('Inference / official time')
    cost.set_title('Measured inference cost', fontsize=12, pad=21)
    for yy, value in zip(y, costs):
        cost.text(value+.035, yy, f'{value:.3f}x', va='center', fontsize=9)
    for ax in (quality, cost):
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', alpha=.16)
        ax.set_axisbelow(True)
    fig.suptitle('RAEv2: final three rounds and recent paired controls', x=.58, y=.98, fontsize=15)
    fig.text(.03, .025,
        f"Same 5,000 noises/classes; official FID {summary['baseline_fid']:.5f}; seed 202609072 was repeatedly explored.\n"
        'No independent quality confirmation. Round 1 failed its mechanism gate and has no FID. '
        'Costs exclude preparation and model loading.', fontsize=9, color='#454545')
    fig.subplots_adjust(left=.29, right=.98, bottom=.14, top=.86, wspace=.18)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(out/f'paired5k.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'complete': True, 'source_summary': str(args.summary.resolve()),
        'source_summary_sha256': digest(args.summary), 'plot_source_sha256': digest(Path(__file__).resolve()),
        'outputs': {f'paired5k.{suffix}': digest(out/f'paired5k.{suffix}') for suffix in ('png', 'pdf')},
        'interpretation': 'Descriptive paired discovery point estimates; no confidence intervals or new trials.'}
    (out/'figure_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
