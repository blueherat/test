"""Conditional Monte Carlo uncertainty for a frozen, independent 5K comparison.

Only pool_3 FID is primary here. The ADM reference mean/covariance stay fixed.
Within each class, resample paired noise IDs with replacement, keeping 50/class.
This is NOT FID-infinity, a correction for model-dependent finite-N bias, or a
bootstrap of the real reference. Coverage is not guaranteed at N=5000, D=2048.

Protocol: plan BEFORE inspecting confirmation outcomes; bind the resulting ADM
caches to the runner's ordered inputs; then benchmark/run on an allocated CPU or
GPU. Legacy caches lack row IDs: bind requires an explicit ordered-extraction
attestation. No metadata can detect a feature-only permutation made BEFORE that
binding. Later changes and any supplied row-ID misalignment are rejected.

The self-test uses small CPU matrices and the unchanged local ADM FID class.
No TensorFlow, Inception graph, model, or CUDA context is needed for self-test.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
LIMITATIONS = [
    "Fixed empirical ADM reference: no real-reference sampling uncertainty.",
    "Finite-N FID bias is generator dependent; paired equal N does not remove it.",
    "Class counts are conditioned on (100 classes, 50 each), not resampled.",
    "No uncertainty over model training, class taxonomy, or feature extractor.",
    "The centered interval describes bootstrap Monte Carlo variation of the finite-N comparison; its nominal coverage is not guaranteed.",
    "Percentile/basic alternatives expose bootstrap mean-shift sensitivity, not certified bias correction.",
    "A new frozen bank reduces selection reuse; 1K-selected outcomes are not themselves confirmation.",
    "Caches lack native row IDs; pre-binding within-class feature-only permutation cannot be detected retrospectively.",
    "Bonferroni adjusts the declared comparisons only, not undisclosed method/seed selection.",
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez(f, **arrays)
    tmp.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def row_ids(noise, labels):
    require(len(noise) == len(labels), 'Noise/label length mismatch')
    # These IDs mean the same stochastic input, not output-image identity.
    return np.asarray([
        hashlib.sha256(np.ascontiguousarray(z, dtype='<f4').tobytes()
                       + np.asarray(label, dtype='<i8').tobytes()).hexdigest()
        for z, label in zip(noise, labels)
    ], dtype='U64')


def validate_pair_ids(expected_ids, expected_labels, actual_ids, actual_labels):
    require(len(set(expected_ids.tolist())) == len(expected_ids), 'Duplicate source IDs')
    require(np.array_equal(expected_ids, actual_ids),
            'Paired row IDs differ in identity or order (including within-class permutations)')
    require(np.array_equal(expected_labels, actual_labels), 'Paired labels differ in order')


def inspect_bank(stage, formal=True):
    stage = Path(stage).resolve()
    request = read(stage/'request.json')
    require(sha(stage/'inputs.npz') == request['inputs_sha256'], 'Frozen inputs changed')
    with np.load(stage/'inputs.npz') as f:
        noise, labels = f['noise'], f['labels']
    require(array_sha(noise) == request['noise_sha256'], 'Noise digest mismatch')
    require(array_sha(labels) == request['labels_sha256'], 'Label digest mismatch')
    require(len(labels) == request['samples'], 'Request count mismatch')
    require(np.isfinite(noise).all(), 'Nonfinite noise')
    if formal:
        unique, counts = np.unique(labels, return_counts=True)
        require(len(labels) == 5000 and np.array_equal(unique, np.arange(100))
                and np.all(counts == 50), 'Formal comparison requires exactly 100 classes x 50 = 5K')
    return request, row_ids(noise, labels), labels


def plan(args):
    stage, out = Path(args.stage).resolve(), Path(args.output).resolve()
    require(not (out/'plan.json').exists(), 'Plan exists; use a fresh output directory')
    req, ids, labels = inspect_bank(stage)
    require(args.reps >= 200, 'At least 200 draws required; 500+ preferable for tail precision')
    arms = [args.candidate, *args.baseline]
    require(len(set(arms)) == len(arms), 'Candidate/baselines must be distinct arms')
    configs = {v['arm']: v for v in req['configs']}
    require(set(arms).issubset(configs), 'Every declared arm must be in the same frozen stage')
    # This avoids retrospectively selecting a favorable 5K outcome. The 1K
    # selection stages may of course already have results.
    require(not any((stage/a/'fid.json').exists() for a in arms),
            'Confirmation FID already exists; this CLI only preregisters future confirmation')
    selection = []
    for old in args.selection_stage:
        old = Path(old).resolve()
        old_req, old_ids, _ = inspect_bank(old, formal=False)
        require(old_req['seed'] != req['seed'], 'Selection and confirmation reuse the RNG seed')
        overlap = set(ids.tolist()).intersection(old_ids.tolist())
        require(not overlap, 'Selection and confirmation reuse stochastic source IDs')
        selection.append(dict(stage=str(old), request_sha256=sha(old/'request.json'),
                              seed=old_req['seed'], source_overlap=len(overlap)))
    ref = Path(req['reference']).resolve()
    if str(ref) in req.get('assets', {}):
        require(sha(ref) == req['assets'][str(ref)], 'Reference differs from sampling request')
    out.mkdir(parents=True, exist_ok=True)
    save_npz(out/'expected_ids.npz', row_ids=ids, labels=labels)
    value = dict(version=1, stage=str(stage), candidate=args.candidate, baselines=args.baseline,
                 configs={a: configs[a] for a in arms}, request_sha256=sha(stage/'request.json'),
                 reference=str(ref), reference_sha256=sha(ref), source_sha256=sha(HERE),
                 extractor_sha256=sha(REPO/'experiments/compute_adm_fid.py'),
                 evaluator_sha256=sha(REPO/'train_gen/evaluator.py'),
                 expected_ids_sha256=sha(out/'expected_ids.npz'),
                 selection_stages=selection, seed=args.seed, reps=args.reps,
                 confidence=.95, metric='ADM pool_3 FID, fixed reference, float64',
                 primary_interval='centered_bootstrap_monte_carlo',
                 alternatives=['percentile', 'basic'], adjustment='Bonferroni over declared baselines',
                 covariance_ddof=1, point_adm_tolerance=args.adm_tolerance,
                 created_unix=time.time(), limitations=LIMITATIONS)
    write(out/'plan.json', value)
    print(json.dumps(dict(plan=str(out/'plan.json'), arms=arms, samples=5000)), flush=True)


def load_plan(output):
    out = Path(output).resolve()
    p = read(out/'plan.json')
    stage = Path(p['stage'])
    require(sha(HERE) == p['source_sha256'], 'Uncertainty source changed since plan')
    require(sha(stage/'request.json') == p['request_sha256'], 'Sampling request changed')
    require(sha(p['reference']) == p['reference_sha256'], 'Reference changed')
    require(sha(out/'expected_ids.npz') == p['expected_ids_sha256'], 'Expected identities changed')
    require(sha(REPO/'experiments/compute_adm_fid.py') == p['extractor_sha256'], 'Extractor changed')
    require(sha(REPO/'train_gen/evaluator.py') == p['evaluator_sha256'], 'ADM evaluator changed')
    _, ids, labels = inspect_bank(stage)
    with np.load(out/'expected_ids.npz') as f:
        validate_pair_ids(ids, labels, f['row_ids'], f['labels'])
    return out, p, ids, labels


def bind(args):
    out, p, ids, labels = load_plan(args.output)
    require(args.attest_ordered_extraction,
            'ADM caches lack native IDs: --attest-ordered-extraction must explicitly attest trusted runner/extractor order')
    require(not (out/'bindings.json').exists(), 'Bindings already exist; do not overwrite')
    bindings = {}
    for arm in [p['candidate'], *p['baselines']]:
        directory = Path(p['stage'])/arm
        summary, fid = read(directory/'summary.json'), read(directory/'fid.json')
        require(summary['complete'] and summary['samples'] == len(ids), 'Incomplete sampling')
        require(summary['config'] == p['configs'][arm], 'Arm configuration changed')
        require(summary['request_sha256'] == p['request_sha256'], 'Arm from another request')
        require(sha(directory/'samples.npz') == summary['samples_sha256'], 'Images changed')
        require(Path(fid['samples']).resolve() == directory/'samples.npz', 'FID uses different samples')
        require(Path(fid['reference']).resolve() == Path(p['reference']), 'FID reference mismatch')
        require(fid['sample_count'] == len(ids), 'FID count mismatch')
        with np.load(directory/'endpoints.npz') as f:
            require(np.array_equal(f['labels'], labels), 'Endpoint label order mismatch')
        feature_path = directory/'inception_activations.npz'
        with np.load(feature_path) as f:
            x = f['pool_3']
            require(x.shape == (5000, 2048) and np.isfinite(x).all(), 'Invalid ADM pool features')
            # Future extractors may provide native IDs; enforce them when present.
            has_native_ids = 'row_ids' in f and 'labels' in f
            if has_native_ids:
                validate_pair_ids(ids, labels, f['row_ids'], f['labels'])
        sidecar = out/f'{arm}.identities.npz'
        save_npz(sidecar, row_ids=ids, labels=labels)
        bindings[arm] = dict(feature_path=str(feature_path), feature_sha256=sha(feature_path),
                             native_feature_ids=has_native_ids, identity_path=str(sidecar),
                             identity_sha256=sha(sidecar), summary_sha256=sha(directory/'summary.json'),
                             samples_sha256=summary['samples_sha256'], fid_sha256=sha(directory/'fid.json'),
                             adm_fid=fid['fid'], full_calls_per_output=summary['full_calls_per_output'],
                             prefix_calls_per_output=summary['prefix_calls_per_output'],
                             sample_seconds=summary['seconds'])
    write(out/'bindings.json', dict(plan_sha256=sha(out/'plan.json'), arms=bindings,
                                   attestation='Trusted current runner collects ascending source rows; ADM reads samples in order. No retrospective proof for feature-only permutations.',
                                   bound_unix=time.time()))
    print(json.dumps(dict(bound=list(bindings), native_ids=[v['native_feature_ids'] for v in bindings.values()])), flush=True)


def psd_eigenvalues(matrix, vectors=False):
    matrix = (matrix + matrix.T) * .5
    values, basis = torch.linalg.eigh(matrix) if vectors else (torch.linalg.eigvalsh(matrix), None)
    minimum, maximum = float(values.min()), float(values.abs().max())
    # Clip roundoff only; no ridge regularization that would change the metric.
    tolerance = 1e-10 * max(1., maximum)
    require(minimum >= -tolerance, f'Covariance is not PSD: min={minimum}, tolerance={tolerance}')
    return values.clamp_min(0.), basis, dict(min_eigenvalue=minimum, negative_tolerance=tolerance)


class FixedReferenceFID:
    """FP64 covariance FID with precomputed X @ sqrt(reference covariance)."""
    def __init__(self, mu, covariance, device='cpu'):
        self.device = torch.device(device)
        self.mu = torch.as_tensor(np.array(mu, dtype=np.float64), device=self.device)
        covariance = torch.as_tensor(np.array(covariance, dtype=np.float64), device=self.device)
        require(covariance.shape == (len(mu), len(mu)), 'Reference dimensions mismatch')
        require(bool(torch.isfinite(covariance).all() and torch.isfinite(self.mu).all()), 'Nonfinite reference')
        values, vectors, self.reference_diagnostics = psd_eigenvalues(covariance, vectors=True)
        self.sqrt_reference = (vectors * values.sqrt()[None, :]) @ vectors.T
        self.trace_reference = torch.trace(covariance)

    def prepare_features(self, x):
        x = torch.as_tensor(np.array(x, dtype=np.float64), device=self.device)
        require(x.ndim == 2 and x.shape[1] == len(self.mu), 'Feature dimensions mismatch')
        require(bool(torch.isfinite(x).all()), 'Nonfinite features')
        return x, x @ self.sqrt_reference, x.square().sum(1)

    def evaluate(self, prepared, counts=None):
        x, y, row_squares = prepared
        w = torch.ones(len(x), dtype=torch.float64, device=self.device) if counts is None else torch.as_tensor(counts, dtype=torch.float64, device=self.device)
        require(w.shape == (len(x),) and bool(torch.all(w >= 0)), 'Invalid resampling counts')
        n = w.sum()
        require(float(n) > 1, 'At least two observations required')
        mux, muy = (w @ x)/n, (w @ y)/n
        trace = ((w @ row_squares) - n * mux.square().sum())/(n-1)
        transformed_covariance = ((y.T * w[None, :]) @ y - n * torch.outer(muy, muy))/(n-1)
        eigenvalues, _, diagnostics = psd_eigenvalues(transformed_covariance)
        mean_term = (mux-self.mu).square().sum()
        covariance_term = trace + self.trace_reference - 2*eigenvalues.sqrt().sum()
        fid = mean_term + covariance_term
        require(bool(torch.isfinite(fid)), 'Nonfinite FID')
        return dict(fid=float(fid), mean_component=float(mean_term),
                    covariance_component=float(covariance_term), **diagnostics)


def stratified_counts(labels, rng):
    counts = np.zeros(len(labels), dtype=np.int64)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        np.add.at(counts, rng.choice(indices, size=len(indices), replace=True), 1)
    return counts


def interval_summary(point, draws, confidence=.95):
    draws = np.asarray(draws, dtype=np.float64)
    alpha = 1-confidence
    low, high = np.quantile(draws, [alpha/2, 1-alpha/2])
    mean = float(draws.mean())
    return dict(nominal_confidence=confidence, observed_delta=float(point),
                bootstrap_mean=mean, bootstrap_mean_shift=mean-float(point),
                bootstrap_std=float(draws.std(ddof=1)),
                centered_bootstrap_monte_carlo=[float(point-(high-mean)), float(point-(low-mean))],
                percentile=[float(low), float(high)], basic=[float(2*point-high), float(2*point-low)],
                expected_draws_in_each_tail=len(draws)*alpha/2)


def load_bound(args):
    out, p, ids, labels = load_plan(args.output)
    bound = read(out/'bindings.json')
    require(bound['plan_sha256'] == sha(out/'plan.json'), 'Plan changed after binding')
    features = {}
    for arm, b in bound['arms'].items():
        directory = Path(p['stage'])/arm
        for file, digest in [(b['feature_path'], b['feature_sha256']),
                             (b['identity_path'], b['identity_sha256']),
                             (directory/'summary.json', b['summary_sha256']),
                             (directory/'fid.json', b['fid_sha256'])]:
            require(sha(file) == digest, f'Bound file changed: {file}')
        with np.load(b['identity_path']) as f:
            validate_pair_ids(ids, labels, f['row_ids'], f['labels'])
        with np.load(b['feature_path']) as f:
            features[arm] = f['pool_3']
            if b['native_feature_ids']:
                validate_pair_ids(ids, labels, f['row_ids'], f['labels'])
    return out, p, bound, labels, features


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


@torch.inference_mode()
def run(args, benchmark=False):
    out, p, bound, labels, features = load_bound(args)
    require(benchmark or not (out/'uncertainty.json').exists(), 'Completed output exists; do not overwrite')
    torch.set_num_threads(args.threads)
    begin = time.perf_counter()
    with np.load(p['reference']) as f:
        metric = FixedReferenceFID(f['mu'], f['sigma'], args.device)
    prepared = {arm: metric.prepare_features(x) for arm, x in features.items()}
    point = {arm: metric.evaluate(x) for arm, x in prepared.items()}
    for arm in point:
        difference = point[arm]['fid'] - bound['arms'][arm]['adm_fid']
        point[arm]['difference_from_original_adm'] = difference
        require(abs(difference) <= p['point_adm_tolerance'],
                f'FP64 and original ADM point FID differ for {arm}: {difference}; investigate before bootstrap')
    synchronize(args.device)
    setup_seconds = time.perf_counter()-begin
    rng = np.random.default_rng(p['seed'])
    reps = 2 if benchmark else p['reps']
    values = np.empty((reps, len(prepared)), dtype=np.float64)
    count_hash = hashlib.sha256()
    arms = list(prepared)
    begin = time.perf_counter()
    for i in range(reps):
        counts = stratified_counts(labels, rng)
        count_hash.update(counts.tobytes())
        values[i] = [metric.evaluate(prepared[a], counts)['fid'] for a in arms]
        if (i+1) % 10 == 0 or i == 0 or i+1 == reps:
            synchronize(args.device)
            elapsed = time.perf_counter()-begin
            progress = dict(completed=i+1, planned=reps, bootstrap_seconds=elapsed,
                            estimated_remaining_seconds=elapsed/(i+1)*(reps-i-1))
            write(out/('benchmark_progress.json' if benchmark else 'progress.json'), progress)
            print(json.dumps(progress), flush=True)
    synchronize(args.device)
    seconds = time.perf_counter()-begin
    base = dict(device=args.device, dtype='float64', samples=len(labels), feature_dimension=2048,
                reps=reps, setup_seconds=setup_seconds, bootstrap_seconds=seconds,
                seconds_per_paired_draw=seconds/reps, point=point,
                reference_diagnostics=metric.reference_diagnostics,
                counts_sha256=count_hash.hexdigest(), plan_sha256=sha(out/'plan.json'),
                bindings_sha256=sha(out/'bindings.json'), source_sha256=sha(HERE))
    if benchmark:
        base.update(estimated_full_seconds=setup_seconds+seconds/reps*p['reps'],
                    interpretation='Timing only; 2 bootstrap draws cannot estimate uncertainty.')
        write(out/'benchmark.json', base)
        return
    candidate_idx = arms.index(p['candidate'])
    comparisons = {}
    for baseline in p['baselines']:
        delta = values[:, candidate_idx] - values[:, arms.index(baseline)]
        point_delta = point[p['candidate']]['fid'] - point[baseline]['fid']
        comparisons[baseline] = dict(
            point_delta=point_delta,
            unadjusted=interval_summary(point_delta, delta, p['confidence']),
            bonferroni=interval_summary(point_delta, delta, 1-(1-p['confidence'])/len(p['baselines'])))
    save_npz(out/'bootstrap_values.npz', fids=values, arms=np.asarray(arms), labels=labels)
    base.update(comparisons=comparisons, limitations=LIMITATIONS,
                delta_sign='candidate minus baseline; negative favors candidate',
                primary_interval=p['primary_interval'],
                arm_costs={a: {k: b[k] for k in ('full_calls_per_output','prefix_calls_per_output','sample_seconds')}
                           for a,b in bound['arms'].items()},
                finished_unix=time.time())
    write(out/'uncertainty.json', base)


def local_adm_class():
    """Execute ONLY the unchanged class AST, avoiding TensorFlow imports."""
    from scipy import linalg
    source = REPO/'train_gen/evaluator.py'
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FIDStatistics')
    namespace = dict(np=np, linalg=linalg, warnings=warnings)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['FIDStatistics']


def self_test(args):
    torch.set_num_threads(2)
    require(not torch.cuda.is_initialized(), 'Self-test must start without a CUDA context')
    rng = np.random.default_rng(73)
    labels = np.repeat(np.arange(4), 12)
    x, reference = rng.normal(size=(48, 8)), rng.normal(size=(96, 8))
    mu, covariance = reference.mean(0), np.cov(reference, rowvar=False)
    metric = FixedReferenceFID(mu, covariance, 'cpu')
    prepared = metric.prepare_features(x)
    adm = local_adm_class()
    old_fid = adm(x.mean(0), np.cov(x, rowvar=False)).frechet_distance(adm(mu, covariance))
    new_fid = metric.evaluate(prepared)['fid']
    require(abs(new_fid-old_fid) < 1e-10, 'FP64 SPD FID differs from unchanged ADM')
    # Verify actual weighted resampling, not just a duplicate implementation.
    errors, same_deltas = [], []
    for _ in range(32):
        counts = stratified_counts(labels, rng)
        require(all(counts[labels == c].sum() == 12 for c in range(4)), 'Class counts changed')
        sampled = x[np.repeat(np.arange(len(x)), counts)]
        expected = adm(sampled.mean(0), np.cov(sampled, rowvar=False)).frechet_distance(adm(mu, covariance))
        actual = metric.evaluate(prepared, counts)['fid']
        errors.append(abs(actual-expected))
        same_deltas.append(actual-metric.evaluate(prepared, counts)['fid'])
    require(max(errors) < 1e-10 and np.all(np.asarray(same_deltas) == 0), 'Weighted or identical-arm failure')
    interval = interval_summary(0., same_deltas)
    require(interval['centered_bootstrap_monte_carlo'] == [0., 0.], 'Same-method interval must be zero')
    # Rank-deficient covariance is handled without changing the metric by ridge.
    singular = x[:5]
    actual = metric.evaluate(metric.prepare_features(singular))['fid']
    expected = adm(singular.mean(0), np.cov(singular, rowvar=False)).frechet_distance(adm(mu, covariance))
    singular_error = abs(actual-expected)
    require(singular_error < 1e-6, 'Singular FID differs excessively from ADM')
    ids = row_ids(rng.normal(size=(48, 4, 2, 2)).astype(np.float32), labels)
    validate_pair_ids(ids, labels, ids.copy(), labels.copy())
    permutation = np.arange(len(ids)); permutation[[0, 1]] = permutation[[1, 0]]
    caught = False
    try:
        validate_pair_ids(ids, labels, ids[permutation], labels[permutation])
    except ValueError:
        caught = True
    require(caught, 'Within-class paired-ID permutation not detected')
    require(array_sha(x) != array_sha(x[permutation]), 'Feature permutation must change bound digest')
    require(not torch.cuda.is_initialized(), 'Self-test unexpectedly initialized CUDA')
    result = dict(passed=True, device='cpu', samples=48, dimensions=8, bootstrap_reps=32,
                  same_method_delta_exact_zero=True, same_method_interval=interval,
                  paired_within_class_id_permutation_detected=caught,
                  feature_permutation_changes_digest=True,
                  fp64_vs_unchanged_adm_spd_absolute_error=abs(new_fid-old_fid),
                  weighted_resample_max_adm_absolute_error=max(errors),
                  singular_vs_unchanged_adm_absolute_error=singular_error,
                  class_counts_preserved=True, cuda_initialized=False,
                  source_sha256=sha(HERE), evaluator_sha256=sha(REPO/'train_gen/evaluator.py'),
                  limitations=['No full-dimensional 2048-feature or GPU benchmark run by self-test.',
                               'A silent pre-binding feature-only permutation is not detectable from legacy caches.'])
    if args.output:
        write(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--stage', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--baseline', action='append', required=True)
    p.add_argument('--selection-stage', action='append', required=True)
    p.add_argument('--reps', type=int, default=500)
    p.add_argument('--seed', type=int, default=202609135001)
    p.add_argument('--adm-tolerance', type=float, default=1e-3)
    p = sub.add_parser('bind')
    p.add_argument('--output', required=True)
    p.add_argument('--attest-ordered-extraction', action='store_true')
    for action in ('run', 'benchmark'):
        p = sub.add_parser(action)
        p.add_argument('--output', required=True)
        p.add_argument('--device', default='cpu')
        p.add_argument('--threads', type=int, default=4)
    p = sub.add_parser('self-test')
    p.add_argument('--output')
    args = parser.parse_args()
    if args.action == 'plan': plan(args)
    elif args.action == 'bind': bind(args)
    elif args.action == 'self-test': self_test(args)
    else: run(args, benchmark=args.action == 'benchmark')


if __name__ == '__main__':
    main()
