"""CPU canonical 1D Gaussian-mixture FM with an externally fixed true inverse.

All guidance coefficients are chosen by real-data one-pass reconstruction only.
No GPU, neural model, FID, or generated data masquerading as real anchors.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import ndtr, ndtri, ndtri_exp, log_ndtr, logsumexp

ROOT = Path('docs/research/identity_copy_toy_20260913')
ALPHAS = [-.25, 0., .25, .5, .75, 1., 1.25, 1.5, 2., 2.5]


@dataclass
class Mixture:
    weights: np.ndarray
    means: np.ndarray
    stds: np.ndarray

    @classmethod
    def make(cls, weights, means, stds):
        return cls(*[np.asarray(value, dtype=np.float64) for value in (weights, means, stds)])

    def cdf(self, x):
        return (ndtr((np.asarray(x)[..., None] - self.means) / self.stds) * self.weights).sum(-1)

    def quantile(self, u):
        u = np.asarray(u)
        low = np.full(u.shape, np.min(self.means - 14 * self.stds))
        high = np.full(u.shape, np.max(self.means + 14 * self.stds))
        for _ in range(65):
            mid = (low + high) / 2
            left = self.cdf(mid) < u
            low = np.where(left, mid, low)
            high = np.where(left, high, mid)
        return (low + high) / 2

    def inverse(self, x):
        standardized = (np.asarray(x)[..., None]-self.means)/self.stds
        log_cdf = logsumexp(np.log(self.weights)+log_ndtr(standardized), axis=-1)
        log_survival = logsumexp(np.log(self.weights)+log_ndtr(-standardized), axis=-1)
        # Use log tails, never clip large-drift copies back to a finite prior band.
        return np.where(log_cdf < np.log(.5), ndtri_exp(log_cdf), -ndtri_exp(log_survival))

    def velocity(self, t, x):
        variance = t*t*self.stds**2 + (1-t)**2
        centered = np.asarray(x)[..., None] - t*self.means
        log_weights = np.log(self.weights) - .5*np.log(variance) - .5*centered**2/variance
        responsibility = np.exp(log_weights - logsumexp(log_weights, axis=-1, keepdims=True))
        velocity = self.means + (t*self.stds**2-(1-t))/variance*centered
        return np.sum(responsibility*velocity, axis=-1)

    def specification(self):
        return {name: getattr(self, name).tolist() for name in ('weights', 'means', 'stds')}


TARGET = Mixture.make([.5, .5], [-2., 2.], [.55, .55])
STRONG = Mixture.make([.5, .5], [-1.7, 1.7], [.70, .70])
WEAK = Mixture.make([.5, .5], [-1.4, 1.4], [.85, .85])
# The true null law mixes the requested class and a separate central class.
NULL = Mixture.make([.25, .25, .5], [-2., 2., 0.], [.55, .55, 1.5])


def generate(z, reference=None, alpha=0., model=STRONG, tolerance=2e-9):
    def field(t, x):
        vc = model.velocity(t, x)
        return vc if reference is None or alpha == 0 else vc + alpha*(vc-reference.velocity(t, x))
    solution = solve_ivp(field, (0., 1.), np.asarray(z).copy(), method='DOP853',
                         rtol=tolerance, atol=tolerance*.1)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[:, -1], solution.nfev


def metrics(copied, original):
    squared = (copied-original)**2
    return {'mse_to_initial': float(squared.mean()),
            'mean': float(copied.mean()), 'std': float(copied.std()),
            'p95_absolute_error_to_initial': float(np.quantile(np.sqrt(squared), .95))}


def main():
    started = time.monotonic()
    ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026091347)
    count = 256
    labels = rng.choice(2, size=count, p=TARGET.weights)
    real = TARGET.means[labels] + TARGET.stds[labels]*rng.normal(size=count)
    true_codes = TARGET.inverse(real)
    biased_codes = STRONG.inverse(real)
    # Independent deterministic quadrature, not used to choose alpha.
    probabilities = (np.arange(2048)+.5)/2048
    prior = ndtri(probabilities)
    target_samples = TARGET.quantile(probabilities)
    all_codes = np.concatenate((true_codes, biased_codes, prior))
    rows, endpoint_arrays, cost = [], {}, 0
    for family, reference in [('cfg', NULL), ('same_target_ag', WEAK)]:
        for alpha in ALPHAS:
            generated, nfev = generate(all_codes, reference, alpha)
            cost += nfev
            key = f'{family}_{alpha:g}'
            endpoint_arrays[key] = generated
            rows.append({'family': family, 'alpha': alpha,
                         'real_oracle_copy_mse': float(np.mean((generated[:count]-real)**2)),
                         'real_biased_reference_copy_mse': float(np.mean((generated[count:2*count]-real)**2)),
                         'independent_population_copy_mse': float(np.mean((generated[2*count:]-target_samples)**2)),
                         'independent_generation_W2_squared': float(np.mean((np.sort(generated[2*count:])-target_samples)**2)),
                         'generation_strictly_monotone': bool(np.all(np.diff(generated[2*count:]) > 0)),
                         'nfev': nfev})
            print(json.dumps(rows[-1]), flush=True)
    selections = {}
    for family in ('cfg', 'same_target_ag'):
        candidates = [row for row in rows if row['family'] == family]
        selections[family] = {'oracle_copy_selected_alpha': min(candidates, key=lambda row: row['real_oracle_copy_mse'])['alpha'],
                              'biased_reference_selected_alpha': min(candidates, key=lambda row: row['real_biased_reference_copy_mse'])['alpha']}
    rounds = {}
    round_arrays = {}
    selected_configs = [('strong', STRONG, None, 0.), ('weak', WEAK, None, 0.)]
    selected_configs.extend((family, STRONG, NULL if family == 'cfg' else WEAK,
                             selection['oracle_copy_selected_alpha'])
                            for family, selection in selections.items())
    for name, model, reference, alpha in selected_configs:
        current = target_samples.copy()
        records = [dict(round=0, mse_to_previous=0., **metrics(current, target_samples))]
        images = [current.copy()]
        for k in range(1, 6):
            codes = TARGET.inverse(current)
            copied, nfev = generate(codes, reference, alpha, model)
            cost += nfev
            records.append(dict(round=k, mse_to_previous=float(np.mean((copied-current)**2)),
                                **metrics(copied, target_samples)))
            current = copied
            images.append(current.copy())
        rounds[name] = records
        round_arrays[name] = np.stack(images)
    # Independent numerical check of the canonical FM and oracle identities.
    check_codes = prior[::16]
    checks = {}
    for name, model in [('target', TARGET), ('strong', STRONG), ('weak', WEAK), ('null', NULL)]:
        actual, nfev = generate(check_codes, model=model, tolerance=2e-11)
        cost += nfev
        checks[name+'_max_error_against_exact_quantile'] = float(np.max(np.abs(actual-model.quantile(ndtr(check_codes)))))
    checks['oracle_inverse_source_max_error'] = float(np.max(np.abs(TARGET.quantile(ndtr(true_codes))-real)))
    checks['population_risk_identity_max_error'] = max(abs(row['independent_population_copy_mse']-row['independent_generation_W2_squared']) for row in rows)
    assert max(value for name, value in checks.items() if 'quantile' in name) < 2e-6, checks
    assert checks['population_risk_identity_max_error'] < 1e-12, checks
    # Analytic reversal: five-round stability need not rank one-pass generators.
    iteration_counterexample = {
        'target': 'N(0,1), true inverse identity',
        'translation_G(z)=z+.3': {'one_pass_W2_squared': .3**2, 'five_round_copy_mse': (5*.3)**2},
        'contraction_G(z)=.5z': {'one_pass_W2_squared': (.5-1)**2, 'five_round_copy_mse': (.5**5-1)**2}}
    summary = {'seed': 2026091347, 'real_calibration_samples': count,
               'population_midpoint_quantiles': len(prior), 'alpha_grid': ALPHAS,
               'models': {name: model.specification() for name, model in [('target', TARGET), ('strong', STRONG), ('weak', WEAK), ('null', NULL)]},
               'selection_rule': 'Minimize one-pass paired reconstruction MSE on 256 independent draws from the true target; no generation metric used.',
               'selections': selections, 'rows': rows, 'rounds': rounds,
               'checks': checks, 'iteration_counterexample': iteration_counterexample,
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'total_vector_rhs_evaluations': cost, 'seconds': time.monotonic()-started}
    (ROOT/'results.json').write_text(json.dumps(summary, indent=2)+'\n')
    np.savez_compressed(ROOT/'records.npz', real=real, true_codes=true_codes,
                        biased_codes=biased_codes, prior_quantiles=prior, target_quantiles=target_samples,
                        **{'generated_'+key: value for key, value in endpoint_arrays.items()},
                        **{'rounds_'+key: value for key, value in round_arrays.items()})
    print(json.dumps({'complete': True, 'selections': selections, 'checks': checks,
                      'seconds': summary['seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
