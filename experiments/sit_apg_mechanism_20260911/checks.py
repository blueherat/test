"""Independent audits of accepted decisions, source contracts, and limiting cases."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from experiments.sit_apg_mechanism_20260911 import core, catalog, study
from experiments.sit_control_50_20260910.checks import FakeRuntime
from experiments.lifting_scale_sweep_20260909 import atomic, sha


def audit_trace(stats, config):
    if 'extension_trace_kind' not in stats:
        return dict(events=0)
    kind = str(stats['extension_trace_kind'])
    get = lambda name: np.asarray(stats['extension_trace_'+name])
    steps = get('step')
    assert kind == config['key'] and steps.ndim == 1
    if kind in ('future_moment_apg', 'semantic_direction', 'moment_without_semantic'):
        accepted, feasible = get('accepted'), get('feasible')
        qb, qp, qr = get('q_before'), get('q_proposed'), get('q_raw')
        objective, eligible, index = get('objectives'), get('eligible'), get('selected')
        assert accepted.dtype == bool and feasible.dtype == bool
        assert (get('shift_norm') <= get('radius')*(1+2e-5)+1e-7).all()
        masked = np.where(eligible, objective, np.inf)
        expected = np.where(feasible, masked.argmin(-1), 0)
        np.testing.assert_array_equal(index, expected)
        if kind != 'moment_without_semantic':
            assert np.all(qp[accepted] > qb[accepted])
        if kind == 'future_moment_apg':
            np.testing.assert_allclose(get('semantic_floor'), qb+.5*np.maximum(qr-qb, 0), rtol=1e-6, atol=1e-8)
            assert np.all(qp[accepted] >= get('semantic_floor')[accepted])
        if kind != 'semantic_direction':
            assert np.all(get('moment_change')[accepted] <= get('raw_moment_change')[accepted]+1e-8)
        return dict(events=len(steps), accepted=int(accepted.sum()), finite_plane_optimum_verified=True,
                    actual_semantic_guard=kind != 'moment_without_semantic')
    if kind == 'future_eta':
        q, penalty, scores = get('q_all'), get('penalty_all'), get('scores')
        index, eligible = get('selected'), get('eligible')
        np.testing.assert_allclose(scores, q-config['theta']*penalty, rtol=1e-6, atol=1e-8)
        np.testing.assert_array_equal(eligible, q >= q[..., :1])
        np.testing.assert_array_equal(index, np.where(eligible, scores, -np.inf).argmax(-1))
        chosen = np.take_along_axis(q, index[..., None], -1)[..., 0]
        np.testing.assert_array_equal(chosen, get('q_selected'))
        assert np.all(chosen >= q[..., 0])
        return dict(events=len(steps), choices=int(index.size), finite_eta_optimum_verified=True)
    if kind in ('marginal_release', 'probability_null_release'):
        before, after, null = get('released_before'), get('released_after'), get('q_null')
        if kind == 'marginal_release':
            value = get('q_keep')-null
            np.testing.assert_array_equal(value, get('marginal_value'))
            expected = before | ((null >= .5) & (value <= config['theta']))
        else:
            expected = before | (null >= config['theta'])
        np.testing.assert_array_equal(after, expected)
        np.testing.assert_array_equal(before[1:], after[:-1])
        assert not before[0].any() and np.all(after | ~before)
        return dict(events=len(steps), released=int(after[-1].sum()), latch_and_value_verified=True)
    if kind in ('terminal_horizon', 'fixed_horizon_verified'):
        q, index, eligible = get('q_all'), get('selected'), get('eligible')
        threshold = config['theta'] if kind == 'terminal_horizon' else 0.
        expected = q > q[..., :1]+threshold
        expected[..., 0] = True
        np.testing.assert_array_equal(eligible, expected)
        np.testing.assert_array_equal(index, np.where(eligible, q, -np.inf).argmax(-1))
        assert np.all(get('q_selected') >= q[..., 0])
        assert np.all(np.abs(get('coefficients')) <= 1.)
        return dict(events=len(steps), choices=int(index.size), common_terminal_optimum_verified=True)
    if kind in ('curl_refine', 'curl_gain_shrink', 'embedded_refinement'):
        if kind == 'embedded_refinement':
            expected = get('severity') > config['theta']
        else:
            expected = np.where(get('fd_error') <= .1, get('severity') > config['theta'], get('fallback_error') > .003)
        np.testing.assert_array_equal(get('refined'), expected)
        if kind != 'curl_gain_shrink':
            assert int(stats['refined_sample_steps']) == int(expected.sum())*8
        return dict(events=len(steps), allocation_verified=True)
    raise AssertionError(kind)


def disabled_reference(config):
    if config['parameters'].get('inherited_exact') or config['key'] in ('clean_apg', 'projection_apg', 'uniform_refinement'):
        return None
    eta = 1. if config['parameters'].get('main') == 'cfg' else 0.
    return next(c for c in catalog.configurations() if c['key'] == 'projection_apg'
                and c['strength'] == config['strength'] and c['theta'] == eta)


@torch.inference_mode()
def cpu_checks():
    rows = catalog.configurations()
    rt = FakeRuntime()
    generator = torch.Generator().manual_seed(202609111)
    noise = torch.randn((2, 4, 8, 8), generator=generator, dtype=torch.float64)
    labels = torch.tensor([2, 9])
    rt.labels = labels
    # Parameterization equivalence is checked against a velocity-space recurrence
    # with the analytically required changing coefficient, not a duplicate clean buffer.
    parameterization_error = 0.
    for beta in (-.75, -.5, -.25):
        clean_memory = torch.zeros_like(noise)
        velocity_memory = torch.zeros_like(noise)
        previous_b = 1.
        for step in range(48):
            b = 1-step/64
            gap = torch.randn(noise.shape, generator=generator, dtype=noise.dtype)
            clean_memory = b*gap+beta*clean_memory
            velocity_memory = gap+beta*(previous_b/b)*velocity_memory
            parameterization_error = max(parameterization_error, float((clean_memory/b-velocity_memory).abs().max()))
            previous_b = b
    assert parameterization_error < 1e-12
    for t in (.125, .5, .7):
        plain = core.guided(rt, noise, t, labels, 2., eta=0.)[0]
        clean = core.guided(rt, noise, t, labels, 2., eta=0., radius=1e10)[0]
        torch.testing.assert_close(plain, clean, rtol=1e-12, atol=1e-12)
        cfg = core.guided(rt, noise, t, labels, 2., eta=1.)[0]
        pair = core.ops.bundle(rt, noise, t, labels, 2.)['cfg']
        torch.testing.assert_close(cfg, pair, rtol=0, atol=0)
    # Independently formed disk points stay inside the declared input budget.
    raw = torch.randn((100, 2), generator=generator, dtype=noise.dtype)
    raw = raw/raw.norm(dim=-1, keepdim=True)
    points = core.coefficient_candidates(raw, raw)
    assert points.shape == (100, 259, 2) and (points.norm(dim=-1) <= 1+1e-12).all()
    probes, trajectories = [], []
    for config in rows:
        if config['parameters'].get('inherited_exact'):
            continue
        # Every candidate parameter combination and every new control gets a
        # complete nonlinear toy trajectory, including all stateful events.
        latent, stats = core.sample(rt, noise, labels, config)
        assert torch.isfinite(latent).all() and rt.labels is labels
        trace = audit_trace(stats, config)
        probes.append(config['arm'])
        if config['strength'] == 2.75:
            trajectories.append(dict(arm=config['arm'], full_calls=stats['full_calls'], trace=trace))
    # One representative per family checks no hidden intervention survives disable.
    for family in dict.fromkeys(c['family'] for c in rows):
        config = next(c for c in reversed(rows) if c['family'] == family)
        reference = disabled_reference(config)
        if reference:
            actual, _ = core.sample(rt, noise, labels, config, disable_calibration=True)
            expected, _ = core.sample(rt, noise, labels, reference)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    result = dict(passed=True, no_gpu=True, no_fid=True, new_configurations=probes,
        trajectories=trajectories, clean_momentum_conversion_max_error=parameterization_error,
        clean_velocity_memory_equivalence=True, clean_beta0_unclipped_projection_exact=True,
        disabled_controllers_equal_references=True, finite_choices_and_guards_audited=True,
        source_hashes={str(Path(p).resolve()):sha(p) for p in (__file__, core.__file__, catalog.__file__)})
    atomic(study.ROOT/'cpu_checks.json', result)
    print(dict(passed=True, configurations=len(probes), representative_trajectories=len(trajectories)), flush=True)
    return result


if __name__ == '__main__':
    cpu_checks()
