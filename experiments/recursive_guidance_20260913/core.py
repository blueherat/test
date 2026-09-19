"""Counted, frozen-model regeneration probes and guidance interventions.

SiT uses z=t*x+(1-t)*epsilon. All probe differences are clean-space quantities.
Probes are sampled only at accepted-step boundaries; their transform is held
fixed for eight steps, including both Heun evaluations. They are not ground truth.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path
import numpy as np
import torch

from experiments.guidance_pasted_20260912 import common as common
from experiments.context_reference_5k_20260912 import reference
from experiments.context_reference_5k_20260912 import core as reference_core
from experiments.guidance_distribution_20260912 import local_head
from experiments.small_sit_guidance_tuning_20260910 import adg_field
from experiments import lifting_scale_sweep_20260909 as infrastructure
from . import catalog

EPS = 1e-12
PROBE_INTERVAL = 8
PROBE_START = .125
DIFFERENCE_STEP = .1
DIAGNOSTICS = ('relative_direction_change', 'effective_extra_strength',
               'direction_cosine', 'probe_response_gain', 'probe_drift_to_gap_ratio')
IDEAS = {row['key']: row for row in catalog.IDEAS}
FAMILIES = catalog.FAMILIES
configurations = catalog.configurations


def install():
    """Compatibility entry; never monkeypatch another experiment's dispatch."""


def make_runtime():
    rt = common.runtime('sit_small')
    rt.recursive_heads = {'native': rt.head}
    rt.recursive_mlp_provenance = reference.install(rt)
    rt.recursive_heads['mlp'] = rt.head
    rt.head = rt.recursive_heads['native']
    return rt


def asset_paths():
    training = reference_core.training_root('sit_small')
    paths = list(infrastructure.asset_paths('sit_small'))
    paths += [training / name for name in ('head.pt', 'request.json', 'summary.json')]
    # Include the dependencies validated by the retained reference's provenance.
    request = infrastructure.read(training / 'request.json')
    for category in ('assets', 'heads'):
        paths.extend(Path(path) for path in request.get(category, {}))
    return sorted(set(path.resolve() for path in paths))


def source_paths():
    paths = list(infrastructure.source_paths('sit_small'))
    paths += [Path(__file__), Path(catalog.__file__), Path(common.__file__),
              Path(reference.__file__), Path(reference_core.__file__), Path(local_head.__file__)]
    # Local transitively imported code is part of the executable dependency set.
    root = infrastructure.WORK.resolve()
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        # Torch can register generated modules with a synthetic <...>.py name.
        # Real file dependencies still fail closed if they disappear.
        if name and not Path(name).name.startswith('<'):
            path = Path(name).resolve()
            if path.suffix == '.py' and path.is_relative_to(root) and path.is_file():
                paths.append(path)
    return sorted(set(path.resolve() for path in paths))


def inner(x, y):
    return (x*y).flatten(1).sum(1).reshape(-1, *([1]*(x.ndim-1)))


def norm(x):
    return inner(x, x).clamp_min(0).sqrt()


def unit(x):
    return x / norm(x).clamp_min(EPS)


def cosine(x, y):
    return (inner(x, y) / (norm(x)*norm(y)).clamp_min(EPS)).clamp(-1, 1)


def project(x, onto):
    return onto * (inner(x, onto) / inner(onto, onto).clamp_min(EPS))


def cap(x, radius):
    return x * (radius / norm(x).clamp_min(EPS)).clamp_max(1)


@contextlib.contextmanager
def selected_head(rt, name):
    before = rt.head
    rt.head = rt.recursive_heads[name]
    try:
        yield
    finally:
        rt.head = before


def branch(rt, z, t, null=False):
    before = rt.labels
    if null:
        rt.labels = torch.full_like(before, 100)
    try:
        return rt.field(z, z.new_tensor(float(t)), 'full')
    finally:
        rt.labels = before


def pair(rt, z, t, source):
    if source == 'ig':
        return rt.pair(z, z.new_tensor(float(t)))
    return branch(rt, z, t), branch(rt, z, t, null=True)


def amount_at(config, left, zero=False):
    if zero or left >= config['cutoff']:
        return 0.
    peak = config['strength']
    return peak*(6/7) if config['source'] == 'ig' and left < .25 else peak


class Regeneration:
    """Fixed-time, antithetic re-noise/denoise; never an ODE inverse."""

    def __init__(self, rt, tau, xi, source):
        self.rt, self.tau, self.xi, self.source = rt, float(tau), xi, source

    def single(self, x, sign, which='strong'):
        y = self.tau*x + (1-self.tau)*sign*self.xi
        if which == 'strong':
            value = branch(self.rt, y, self.tau)
        elif self.source == 'cfg':
            value = branch(self.rt, y, self.tau, null=True)
        else:
            # The shared traversal is deliberately counted as one full query.
            _, value = self.rt.pair(y, y.new_tensor(self.tau))
        return y + (1-self.tau)*value

    def mean(self, x, which='strong'):
        return .5*(self.single(x, 1, which)+self.single(x, -1, which))

    def paired(self, x):
        strong, weak, gaps = [], [], []
        for sign in (1, -1):
            y = self.tau*x + (1-self.tau)*sign*self.xi
            s, w = pair(self.rt, y, self.tau, self.source)
            strong.append(y+(1-self.tau)*s)
            weak.append(y+(1-self.tau)*w)
            gaps.append((1-self.tau)*(s-w))
        return .5*(strong[0]+strong[1]), .5*(weak[0]+weak[1]), gaps

    def derivative(self, x, direction, which='strong'):
        h = DIFFERENCE_STEP
        plus = self.mean(x+h*direction, which)
        minus = self.mean(x-h*direction, which)
        return (plus-minus)/(2*h), plus, minus


def probe(rt, z, t, strong, weak, config, xi):
    key = config['key']
    tau = max(float(t), config['theta'])
    assert 0 < tau < 1
    m = z+(1-t)*strong
    d = (1-t)*(strong-weak)
    re = Regeneration(rt, tau, xi, config['source'])
    gain = torch.ones_like(norm(d))
    result = dict(key=key, gain=gain, response_gain=torch.zeros_like(gain),
                  drift_ratio=torch.zeros_like(gain))
    coefficient = IDEAS[key]['coefficient'] if key in IDEAS else 0.
    result['coefficient'] = coefficient
    if key in ('cfg_cycle_gain', 'cfg_cycle_transport', 'cfg_cycle_curvature', 'ig_cycle_weakgain'):
        js, plus, minus = re.derivative(m, d)
        response = norm(js)/norm(d).clamp_min(EPS)
        result['response_gain'] = response
        if key == 'cfg_cycle_gain':
            result['gain'] = 1/(1+coefficient*(response-1).clamp_min(0))
        elif key == 'cfg_cycle_transport':
            # A zero response contains no usable direction; retain the current direction.
            result.update(axis=unit(js), valid=norm(js)>EPS)
        elif key == 'cfg_cycle_curvature':
            center = re.mean(m)
            curvature = norm(plus+minus-2*center)/(DIFFERENCE_STEP*norm(d)).clamp_min(EPS)
            result.update(gain=1/(1+coefficient*curvature), response_gain=curvature,
                          drift_ratio=norm(center-m)/norm(d).clamp_min(EPS))
        else:
            jw, _, _ = re.derivative(m, d, 'weak')
            extra = ((norm(jw)-norm(js))/norm(d).clamp_min(EPS)).clamp_min(0)
            result.update(gain=1/(1+coefficient*extra), response_gain=extra)
    elif key == 'cfg_cycle_antidrift':
        residual = re.mean(m)-m
        result.update(axis=unit(residual), drift_ratio=norm(residual)/norm(d).clamp_min(EPS))
    else:
        ms, mw, gaps = re.paired(m)
        rs, rw = ms-m, mw-m
        replay_gap = ms-mw
        result['drift_ratio'] = norm(rs)/norm(d).clamp_min(EPS)
        if key == 'cfg_cycle_common_drift':
            agreement = cosine(rs, rw).clamp_min(0)
            bias = .5*(rs+rw)
            result['bias_ratio'] = agreement*cap(bias, norm(d))/norm(d).clamp_min(EPS)
        elif key == 'ig_cycle_agreement':
            result['gain'] = cosine(d, replay_gap).clamp_min(0)**coefficient
        elif key == 'ig_cycle_extrapolate':
            result.update(axis=unit(replay_gap), valid=norm(replay_gap)>EPS)
        elif key == 'ig_cycle_antithetic':
            result['gain'] = cosine(gaps[0], gaps[1]).clamp_min(0)**coefficient
        elif key == 'ig_cycle_twohop':
            # Compose deterministic antithetic MEAN maps on separate branch
            # paths. This is Rbar_S(Rbar_S(m))-Rbar_W(Rbar_W(m)), not the
            # expectation of a stochastic two-step regeneration chain.
            d2 = re.mean(ms, 'strong')-re.mean(mw, 'weak')
            growth = norm(d2)/norm(replay_gap).clamp_min(EPS)
            result.update(gain=cosine(replay_gap, d2).clamp_min(0)/
                          (1+coefficient*(growth-1).clamp_min(0)), response_gain=growth)
        elif key != 'probe_only':
            raise KeyError(key)
    for name, value in result.items():
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise FloatingPointError(f'Nonfinite regeneration probe {key}/{name}')
    return result


def transform(d, state, config):
    if state is None or config['parameters'].get('correction_multiplier', 1.) == 0:
        return d
    key, coefficient = state['key'], state['coefficient']
    if key == 'cfg_cycle_transport':
        transported = torch.where(state['valid'], norm(d)*state['axis'], d)
        value = (1-coefficient)*d+coefficient*transported
    elif key == 'cfg_cycle_antidrift':
        axis = state['axis']
        value = d-coefficient*inner(d, axis).clamp_min(0)*axis
    elif key == 'cfg_cycle_common_drift':
        bias = norm(d)*state['bias_ratio']
        value = d-coefficient*(bias-project(bias, d))
    elif key == 'ig_cycle_extrapolate':
        transported = torch.where(state['valid'], norm(d)*state['axis'], d)
        value = cap(d+coefficient*(d-transported), 2*norm(d))
    else:
        value = d*state['gain']
    if config['parameters'].get('reverse', False):
        value = cap(2*d-value, 2*norm(d))
    if config['parameters'].get('norm_only', False):
        value = d*(norm(value)/norm(d).clamp_min(EPS))
    return value


def evaluate(rt, z, t, amount, config, state, memory, values=None):
    if amount == 0:
        strong = branch(rt, z, t)
        return strong, None, z.new_zeros((len(z), 5))
    strong, weak = pair(rt, z, t, config['source']) if values is None else values
    d = strong-weak
    key = config['key']
    if key == 'adg':
        value, _ = adg_field(z, strong, weak, z.new_tensor(t), amount)
        modified = (value-strong)/amount
        update = None
    elif key == 'apg':
        update = d+config['theta']*(torch.zeros_like(d) if memory is None else memory)
        bounded = cap(update, 2*norm(d))
        modified = bounded-project(bounded, z+(1-t)*strong)
        value = strong+amount*modified
    elif key == 'ctrl':
        prev = d if memory is None else memory
        sliding = d+(config['parameters'].get('decay', 5.)-1)*prev
        update = d-config['theta']*sliding.sign()
        # Existing repository CFG-Ctrl convention, including correction of the base gap.
        value = weak+(1+amount)*update
        modified = (value-strong)/amount
    else:
        modified = transform(d, state, config)
        value = strong+amount*modified
        update = None
    response = torch.zeros_like(norm(d)) if state is None else state['response_gain']
    drift = torch.zeros_like(norm(d)) if state is None else state['drift_ratio']
    diagnostic = torch.cat((norm(modified-d)/norm(d).clamp_min(EPS),
        amount*norm(modified)/norm(d).clamp_min(EPS), cosine(d, modified), response, drift), dim=1)
    return value, update, diagnostic.reshape(len(z), 5)


@torch.inference_mode()
def sample(rt, noise, labels, config, *, zero=False, start_step=0, probe_seed=None):
    assert config['solver'].startswith('heun')
    steps = int(config['solver'][4:])
    assert steps in (64, 96, 128)
    assert 0 <= start_step < steps
    begin = rt.counts.copy()
    old_labels = rt.labels
    z, state, memory = noise.clone(), None, None
    diag = torch.zeros((len(z), 5), device=z.device, dtype=torch.float64)
    history, diag_queries, probe_events, probe_calls = [], 0, 0, 0
    # For feedback rounds, callers supply a seed derived from the shared fresh
    # noise bank, NEVER from the method-dependent mixed input state.
    seed = (int(infrastructure.array_sha(noise.detach().cpu().numpy())[:15], 16)
            if probe_seed is None else int(probe_seed))
    generator = torch.Generator(device=z.device).manual_seed(seed)
    disabled = config['parameters'].get('correction_multiplier', 1.) == 0
    try:
        with selected_head(rt, config.get('reference', 'native')), rt.context():
            rt.labels = labels
            grid = torch.linspace(0, 1, steps+1, device=z.device, dtype=z.dtype)
            for step in range(start_step, steps):
                t_tensor, u_tensor = grid[step], grid[step+1]
                t, u, h = float(t_tensor), float(u_tensor), u_tensor-t_tensor
                amount = amount_at(config, t, zero)
                values = None
                if (amount and not disabled and config['key'] in (*IDEAS, 'probe_only')
                        and step % PROBE_INTERVAL == 0 and t >= PROBE_START):
                    values = pair(rt, z, t, config['source'])
                    # Same input batch and event => same probes for all settings and methods.
                    xi = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator)
                    before_probe = rt.counts['full']
                    state = probe(rt, z, t, *values, config, xi)
                    probe_calls += rt.counts['full']-before_probe
                    probe_events += 1
                first, update, first_diag = evaluate(rt, z, t, amount, config, state, memory, values)
                second, _, second_diag = evaluate(rt, z+h*first, u, amount, config, state, memory)
                z = z+(h/2)*(first+second)
                if amount:
                    diag.add_(first_diag.double()+second_diag.double())
                    diag_queries += 2
                    history.append(first_diag[:, 1].detach())
                if update is not None:
                    memory = update.detach()
                if not torch.isfinite(z).all() or z.abs().max() > 1e6:
                    raise FloatingPointError(f'{config["arm"]}: invalid accepted state at step {step}')
    finally:
        rt.labels = old_labels
    counts = {key: rt.counts[key]-begin[key] for key in begin}
    active_steps = sum(amount_at(config, float(t), zero) != 0 for t in grid[start_step:-1])
    expected_native = 2*(steps-start_step)+(2*active_steps if config['source'] == 'cfg' else 0)
    assert counts['prefix'] == 0, counts
    assert counts['full'] == expected_native+probe_calls, (config['arm'], counts, expected_native, probe_calls)
    effective = torch.cat(history) if history else z.new_zeros(1)
    quantiles = torch.quantile(effective.float(), z.new_tensor((0., .25, .5, .75, 1.))).cpu().numpy()
    return z, dict(full_calls=counts['full'], prefix_calls=0, auxiliary_full_calls=probe_calls,
        diagnostic_queries=diag_queries, diagnostics=(diag/max(diag_queries, 1)).cpu().numpy(),
        probe_events=probe_events, probe_seed=seed, effective_strength_quantiles=quantiles,
        generated_paths_per_output=1, steps=steps, start_step=start_step,
        accepted_steps=steps-start_step)


@torch.inference_mode()
def limiting_checks(rt, noise, labels):
    """Independent native integrator comparisons and state restoration checks."""
    before_head, before_labels = rt.head, rt.labels
    rows = []
    for source, head, strength in (('ig', 'native', .8), ('ig', 'mlp', .8), ('cfg', 'native', 1.25)):
        cfg = dict(arm='parity', key='native', source=source, strength=strength, theta=0.,
                   cutoff=.5 if source == 'ig' else .75, parameters={}, reference=head, solver='heun64')
        actual, stats = sample(rt, noise, labels, cfg)
        assert rt.head is before_head and rt.labels is before_labels
        with selected_head(rt, head), rt.context():
            rt.grid = torch.linspace(0, 1, 65, device=noise.device)
            def native_field(z, t, left, step, substage):
                a = amount_at(cfg, left)
                if not a:
                    return rt.field(z, t, 'full')
                if source == 'ig':
                    return rt.guided(z, t, a)
                strong, weak = pair(rt, z, t, source)
                return strong+a*(strong-weak)
            expected, _ = common.integrate(rt, noise, labels, native_field)
        rt.labels = before_labels
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        rows.append(dict(source=source, reference=head, full=stats['full_calls'], exact=True))
    rt.labels = labels
    with selected_head(rt, 'native'):
        native_s, _ = rt.pair(noise, noise.new_tensor(.25))
    with selected_head(rt, 'mlp'):
        mlp_s, _ = rt.pair(noise, noise.new_tensor(.25))
    rt.labels = before_labels
    torch.testing.assert_close(native_s, mlp_s, rtol=0, atol=0)
    # Disabling only the new correction must preserve positive native guidance,
    # unlike zero=True, which removes native guidance as well. Cache references
    # shared by several mechanisms; probes must not execute in this limit.
    native_cache, correction_rows = {}, []
    for key in IDEAS:
        cfg = [c for c in configurations() if c['key'] == key and c['role'] == 'candidate'][-1]
        disabled = dict(cfg, parameters={**cfg['parameters'], 'correction_multiplier': 0.})
        actual, stats = sample(rt, noise, labels, disabled)
        identity = tuple(cfg[name] for name in ('source', 'reference', 'strength', 'solver', 'cutoff'))
        if identity not in native_cache:
            native = dict(cfg, key='native', parameters={})
            native_cache[identity] = sample(rt, noise, labels, native)
        expected, native_stats = native_cache[identity]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert stats['auxiliary_full_calls'] == stats['probe_events'] == 0
        assert stats['full_calls'] == native_stats['full_calls']
        assert rt.head is before_head and rt.labels is before_labels
        correction_rows.append(dict(key=key, full=stats['full_calls'], exact=True,
                                    auxiliary_full_calls=0, probe_events=0))
    return dict(passed=True, native_parity=rows, head_install_preserves_strong_exact=True,
                labels_and_head_restored=True, extra_prefix_calls=0,
                zero_correction_preserves_native_guidance=correction_rows)


@torch.inference_mode()
def preflight_grid(rt, noise, labels, configs):
    rows = []
    before_head, before_labels = rt.head, rt.labels
    generator = torch.Generator(device=noise.device).manual_seed(2026131302)
    xi = torch.randn(noise.shape, device=noise.device, generator=generator)
    for cfg in configs:
        if cfg['role'] != 'candidate':
            continue
        with selected_head(rt, cfg['reference']):
            rt.labels = labels
            for t in (.125, cfg['cutoff']-.125):
                strong, weak = pair(rt, noise, t, cfg['source'])
                state = probe(rt, noise, t, strong, weak, cfg, xi)
                out = transform(strong-weak, state, cfg)
                assert torch.isfinite(out).all(), cfg['arm']
                disabled = dict(cfg, parameters={'correction_multiplier': 0.})
                assert torch.equal(transform(strong-weak, state, disabled), strong-weak)
                if cfg['key'] == 'cfg_cycle_common_drift':
                    d = strong-weak
                    relative = inner(out-d, d).abs()/inner(d, d).clamp_min(EPS)
                    assert relative.max() < 2e-5
        rows.append(cfg['arm'])
        if len(rows) % 20 == 0:
            print(f'grid preflight {len(rows)}/200', flush=True)
    rt.labels = before_labels
    assert rt.head is before_head
    assert len(rows) == 200
    return dict(passed=True, configurations=rows, states_per_configuration=2,
                zero_transform_exact=True, common_drift_orthogonality=True)


@torch.inference_mode()
def feedback_checks(rt, noise, labels):
    """Actual five-round suffix chains, plus independently integrated native suffixes."""
    configs = configurations()
    generator = torch.Generator(device=noise.device).manual_seed(2026131351)
    banks = [torch.randn(noise.shape, device=noise.device, generator=generator) for _ in range(5)]
    seed_values = [int(infrastructure.array_sha(x.cpu().numpy())[:15], 16) for x in banks]
    before_head, before_labels = rt.head, rt.labels
    parity = []
    for source, head in (('cfg', 'native'), ('ig', 'native'), ('ig', 'mlp')):
        cfg = next(c for c in configs if c['key'] == 'native' and c['source'] == source
                   and c['reference'] == head and c['solver'] == 'heun64' and c['strength'] > 0)
        start = .25*noise+.75*banks[0]
        actual, stats = sample(rt, start, labels, cfg, start_step=16, probe_seed=seed_values[0])
        with selected_head(rt, head), rt.context():
            rt.grid = torch.linspace(0, 1, 65, device=noise.device)[16:]
            def native_field(z, t, left, step, substage):
                a = amount_at(cfg, left)
                if not a:
                    return rt.field(z, t, 'full')
                if source == 'ig':
                    return rt.guided(z, t, a)
                strong, weak = pair(rt, z, t, source)
                return strong+a*(strong-weak)
            expected, _ = common.integrate(rt, start, labels, native_field)
        rt.labels = before_labels
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert stats['full_calls'] == (160 if source == 'cfg' else 96)
        parity.append(dict(source=source, reference=head, full=stats['full_calls'], exact=True))
    trajectories = []
    for idea in catalog.IDEAS:
        cfg = next(c for c in configs if c['idea_id'] == idea['id'])
        current, stats = sample(rt, noise, labels, cfg)
        counts = [stats['full_calls']]
        hashes = [infrastructure.array_sha(current.cpu().numpy())]
        for round_index, bank in enumerate(banks):
            mixed = .25*current+.75*bank
            current, stats = sample(rt, mixed, labels, cfg, start_step=16,
                                    probe_seed=seed_values[round_index])
            assert stats['accepted_steps'] == 48 and stats['probe_seed'] == seed_values[round_index]
            assert torch.isfinite(current).all()
            counts.append(stats['full_calls'])
            hashes.append(infrastructure.array_sha(current.cpu().numpy()))
        repeated, _ = sample(rt, mixed, labels, cfg, start_step=16, probe_seed=seed_values[-1])
        torch.testing.assert_close(repeated, current, rtol=0, atol=0)
        assert rt.head is before_head and rt.labels is before_labels
        trajectories.append(dict(key=idea['key'], rounds=list(range(6)), full_calls=counts,
            output_latent_sha256=hashes, repeated_last_round_exact=True))
        print(f'feedback preflight {idea["id"]}/10: {counts}', flush=True)
    rt.grid = torch.linspace(0, 1, 65, device=noise.device)
    return dict(passed=True, native_suffix_parity=parity, full_feedback_rounds=5,
                fixed_labels=True, shared_probe_seeds=seed_values, independent_round_noise=True,
                trajectories=trajectories, head_and_labels_restored=True)


def cpu_checks():
    """Nonlinear operator tests, native limits, and exact Gaussian counterexample."""
    configs = configurations()
    torch.manual_seed(1309)
    x, d, r = [torch.randn(3, 4, 4, 4, dtype=torch.float64) for _ in range(3)]
    # Finite differences must measure amplification, not displacement bias.
    h, a = DIFFERENCE_STEP, 1.7
    plus, minus = a*(x+h*d)+r, a*(x-h*d)+r
    torch.testing.assert_close((plus-minus)/(2*h), a*d, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(plus+minus-2*(a*x+r), torch.zeros_like(x), rtol=0, atol=2e-15)
    for key, row in IDEAS.items():
        cfg = next(c for c in configs if c['key'] == key and c['role'] == 'candidate')
        state = dict(key=key, coefficient=row['coefficient'], gain=torch.ones_like(norm(d))*.6,
                     axis=unit(r), valid=norm(r)>EPS, bias_ratio=.5*unit(r))
        y = transform(d, state, cfg)
        assert torch.isfinite(y).all()
        zero = dict(cfg, parameters={'correction_multiplier': 0.})
        assert torch.equal(transform(d, state, zero), d)
        if key == 'cfg_cycle_common_drift':
            assert inner(y-d, d).abs().max() < 1e-12
        if key == 'cfg_cycle_antidrift':
            assert (norm(y) <= norm(d)+1e-12).all()
        if key == 'ig_cycle_extrapolate':
            assert (norm(y) <= 2*norm(d)+1e-12).all()
    # Exact posterior-mean denoising may shrink a clean input; zero drift is not the target.
    variance, sigma2 = 2., .7
    shrink = variance/(variance+sigma2)
    assert 0 < shrink < 1
    # Posterior sampling (unlike its mean) preserves marginal variance.
    recovered_variance = shrink**2*(variance+sigma2)+variance*sigma2/(variance+sigma2)
    assert abs(recovered_variance-variance) < 1e-12
    return dict(passed=True, ideas=10, candidate_settings=200,
                controls=len(configs)-200, affine_derivative_exact=True,
                common_drift_orthogonality=True, gaussian_pointwise_identity_rejected=True,
                gaussian_posterior_stationarity=True)
