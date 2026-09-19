"""CPU analysis and standalone plot of the frozen copy-calibration toy."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.identity_copy_toy_20260913.run import TARGET, STRONG

ROOT = Path('docs/research/identity_copy_toy_20260913')
data = json.loads((ROOT/'results.json').read_text())
bank_data = json.loads((ROOT/'nonoracle_results.json').read_text())
arrays = np.load(ROOT/'records.npz')
prior = arrays['prior_quantiles']
target = arrays['target_quantiles']
reference_codes = STRONG.inverse(target)
prior_mismatch = float(np.mean((reference_codes-prior)**2))
best = {}
for family, choice in data['selections'].items():
    chosen = next(row for row in data['rows'] if row['family'] == family and row['alpha'] == choice['oracle_copy_selected_alpha'])
    baseline = next(row for row in data['rows'] if row['family'] == family and row['alpha'] == 0)
    best[family] = {'selected_alpha': chosen['alpha'],
                    'heldout_W2_squared': chosen['independent_generation_W2_squared'],
                    'reduction_fraction': 1-chosen['independent_generation_W2_squared']/baseline['independent_generation_W2_squared']}
analysis = {'biased_reference_aggregate_prior_W2_squared': prior_mismatch,
            'selected': best,
            'high_dimensional_rank_counterexample': {
                'true_target': 'N(0,I_2), true inverse identity',
                'G_A=90degree_rotation': {'one_pass_copy_mse': 4., 'generation_W2_squared': 0.},
                'G_B=.9I': {'one_pass_copy_mse': .02, 'generation_W2_squared': .02},
                'scope': 'Learned ODEs can include a rotation; not two distinct exact minimizers of the same canonical FM regression.'}}
(ROOT/'analysis.json').write_text(json.dumps(analysis, indent=2)+'\n')
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
colors = {'strong': '#52525b', 'weak': '#a1a1aa', 'cfg': '#2563eb', 'same_target_ag': '#059669'}
for family, title in [('cfg', 'CFG'), ('same_target_ag', 'AG')]:
    rows = [row for row in data['rows'] if row['family'] == family]
    alpha = [row['alpha'] for row in rows]
    bank_risks = np.array([[row['real_copy_mse'] for row in bank_data['rows'] if row['seed'] == seed and row['family'] == family]
                           for seed in bank_data['bank_seeds']])
    axes[0].plot(alpha, [row['independent_generation_W2_squared'] for row in rows], 'o-', color=colors[family], label=title+' generation W2 squared')
    axes[0].plot(alpha, [row['real_oracle_copy_mse'] for row in rows], ':', color=colors[family], label=title+' true-inverse copy risk')
    axes[0].plot(alpha, bank_risks.mean(0), '--', color=colors[family], label=title+' bank-inverse copy risk')
    axes[0].fill_between(alpha, bank_risks.min(0), bank_risks.max(0), color=colors[family], alpha=.12)
axes[0].annotate('Self inverse selects alpha = 0\nwhile generation W2 squared = 0.1086',
                 xy=(0, .10861126657100212), xytext=(.33, .91), textcoords='axes fraction', fontsize=8,
                 bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':.9},
                 arrowprops={'arrowstyle':'->', 'color':'#444444'})
axes[0].set(title='Choose guidance using real-image copy risk', xlabel='Extra guidance coefficient alpha', ylabel='Squared error', yscale='log')
axes[0].grid(alpha=.2); axes[0].legend(fontsize=7.5, loc='lower left')
for family, title in [('strong', 'Strong'), ('cfg', 'CFG'), ('same_target_ag', 'AG')]:
    rows = data['rounds'][family]
    bank_curves = np.array([[row['mse_to_initial'] for row in bank_data['rounds'][f'{seed}_{family}']]
                            for seed in bank_data['bank_seeds']])
    axes[1].plot(range(6), [row['mse_to_initial'] for row in rows], 'o-', color=colors[family], label=title+' / true inverse')
    axes[1].plot(range(6), bank_curves.mean(0), 's--', color=colors[family], label=title+' / bank inverse')
    axes[1].fill_between(range(6), bank_curves.min(0), bank_curves.max(0), color=colors[family], alpha=.12)
axes[1].set(title='Every copy round: error to the original', xlabel='Copy round', ylabel='MSE to original', xticks=range(6))
axes[1].set_yscale('symlog', linthresh=1e-4)
axes[1].set_ylim(0, 3.2)
axes[1].grid(alpha=.2); axes[1].legend(fontsize=8, loc='lower right')
fig.suptitle('Canonical 1D Gaussian-mixture FM: feasibility of copy-based calibration', fontsize=12)
fig.savefig(ROOT/'copy_calibration.png', dpi=180)
fig.savefig(ROOT/'copy_calibration.pdf')
plt.close(fig)
print(json.dumps(analysis, indent=2))
