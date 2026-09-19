"""Fixed-rule empirical KDE inverse; no analytic target CDF enters calibration."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from scipy.special import ndtri
from experiments.identity_copy_toy_20260913.run import (
    ROOT, ALPHAS, Mixture, TARGET, STRONG, WEAK, NULL, generate, metrics)

BANK_SIZE = 1024
BANK_SEEDS = [2026091351, 2026091352, 2026091353]


def make_empirical_inverse(bank):
    scale = min(float(np.std(bank, ddof=1)), float(np.subtract(*np.percentile(bank, [75, 25])))/1.34)
    bandwidth = .9*scale*len(bank)**(-.2)
    return Mixture.make(np.full(len(bank), 1/len(bank)), bank, np.full(len(bank), bandwidth)), bandwidth


def main():
    started = time.monotonic()
    original = json.loads((ROOT/'results.json').read_text())
    original_arrays = np.load(ROOT/'records.npz')
    real = original_arrays['real']
    fits, codes, assets = [], [], {}
    for seed in BANK_SEEDS:
        rng = np.random.default_rng(seed)
        # Simulator only: no target component label, CDF, mean or variance enters the KDE fit.
        components = rng.choice(len(TARGET.weights), size=BANK_SIZE, p=TARGET.weights)
        bank = TARGET.means[components] + TARGET.stds[components]*rng.normal(size=BANK_SIZE)
        fit, bandwidth = make_empirical_inverse(bank)
        encoded = fit.inverse(real)
        fits.append((seed, fit, bandwidth))
        codes.append(encoded)
        assets[f'bank_{seed}'] = bank
        assets[f'calibration_codes_{seed}'] = encoded
    encoded_batch = np.concatenate(codes)
    rows, selections, cost = [], {}, 0
    for family, reference in [('cfg', NULL), ('same_target_ag', WEAK)]:
        for alpha in ALPHAS:
            copied, nfev = generate(encoded_batch, reference, alpha)
            cost += nfev
            original_row = next(row for row in original['rows'] if row['family'] == family and row['alpha'] == alpha)
            assets[f'calibration_copy_{family}_{alpha:g}'] = copied
            for i, (seed, _, bandwidth) in enumerate(fits):
                row = {'seed': seed, 'bandwidth': bandwidth, 'family': family, 'alpha': alpha,
                       'real_copy_mse': float(np.mean((copied[i*len(real):(i+1)*len(real)]-real)**2)),
                       'independent_generation_W2_squared': original_row['independent_generation_W2_squared']}
                rows.append(row)
        for seed in BANK_SEEDS:
            choice = min([row for row in rows if row['seed'] == seed and row['family'] == family], key=lambda row: row['real_copy_mse'])
            selections[f'{seed}_{family}'] = choice
    # Oracle information is used only below, for independent audit and test quality.
    # It is never used to choose bandwidth or alpha above.
    probabilities = (np.arange(1024)+.5)/1024
    prior = ndtri(probabilities)
    original_x = TARGET.quantile(probabilities)
    assets['round_initial_true_quantiles'] = original_x
    prior_checks, rounds = [], {}
    for seed, encoder, bandwidth in fits:
        heldout_codes = encoder.inverse(original_x)
        prior_checks.append({'seed': seed, 'bandwidth': bandwidth,
                             'heldout_aggregate_prior_W2_squared': float(np.mean((heldout_codes-prior)**2)),
                             'heldout_aggregate_prior_mean': float(heldout_codes.mean()),
                             'heldout_aggregate_prior_std': float(heldout_codes.std())})
        for family, reference in [('strong', None), ('cfg', NULL), ('same_target_ag', WEAK)]:
            alpha = 0. if family == 'strong' else selections[f'{seed}_{family}']['alpha']
            current = original_x.copy()
            snapshots = [current.copy()]
            records = [dict(round=0, mse_to_previous=0., **metrics(current, original_x))]
            for k in range(1, 6):
                copied, nfev = generate(encoder.inverse(current), reference, alpha)
                cost += nfev
                records.append(dict(round=k, mse_to_previous=float(np.mean((copied-current)**2)),
                                    **metrics(copied, original_x)))
                current = copied
                snapshots.append(current.copy())
            key = f'{seed}_{family}'
            rounds[key] = records
            assets['rounds_'+key] = np.stack(snapshots)
    data = {'bank_size': BANK_SIZE, 'bank_seeds': BANK_SEEDS,
            'bandwidth_rule': '.9 * min(sample_std_ddof1, sample_IQR/1.34) * n^(-1/5); fixed before evaluation, no oracle or heldout quality tuning.',
            'inverse': 'E_B(x)=Phi^-1[(1/n) sum_i Phi((x-bank_i)/h)] with stable log tails.',
            'calibration_real_samples': len(real), 'calibration_seed_from_original': original['seed'],
            'selection_rule': 'One-pass squared reconstruction error on bank-independent real anchors; every bank uses the same anchors for paired comparison.',
            'generation_test_rule': 'Reuse frozen 2048-quantile generation quality from original run; never select alpha or bandwidth by this metric.',
            'round_evaluation_quantiles': len(original_x), 'rows': rows, 'selections': selections,
            'prior_checks': prior_checks, 'rounds': rounds,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'original_source_sha256': original['source_sha256'],
            'original_results_sha256': hashlib.sha256((ROOT/'results.json').read_bytes()).hexdigest(),
            'total_vector_rhs_evaluations': cost, 'seconds': time.monotonic()-started}
    (ROOT/'nonoracle_results.json').write_text(json.dumps(data, indent=2)+'\n')
    np.savez_compressed(ROOT/'nonoracle_records.npz', **assets)
    print(json.dumps({'complete': True, 'selections': selections, 'prior_checks': prior_checks,
                      'seconds': data['seconds']}, indent=2))


if __name__ == '__main__':
    main()
