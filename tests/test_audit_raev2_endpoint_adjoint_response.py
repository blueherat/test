from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import audit_raev2_endpoint_adjoint_response as audit


def test_suffix_adjoint_is_successor_gradient_uses_transpose_and_omits_a0():
    matrices = [torch.tensor([[3., 2.], [0., .5]]), torch.tensor([[1., 2.], [0., 1.]])]
    initial = torch.tensor([[.7, -.4]])
    states = [initial[0].numpy()]
    state = initial
    for matrix in matrices:
        state = state @ matrix.T
        states.append(state[0].numpy())
    calls, saved, counts = [], {}, Counter()

    def step(value, k):
        calls.append(k)
        return value @ matrices[k].T

    h = [.5, .5]
    norms, gram = audit.suffix_adjoint(np.asarray(states), step, torch.tensor([[1., 0.]]),
                                     h, lambda k, value: saved.update({k: value.clone()}), counts)
    assert calls == [1]
    assert counts['stage2_input_vjp_calls'] == 1
    assert torch.equal(saved[1], torch.tensor([[1., 0.]]))
    assert torch.equal(saved[0], torch.tensor([[1., 2.]]))
    np.testing.assert_array_equal(norms, [5., 1.])
    assert gram == 3.
    _, lam = audit.solve_cohort_gram([gram], expected_count=1, delta=1.)
    controlled = initial
    energy = 0.
    for k, matrix in enumerate(matrices):
        u = lam * saved[k]
        controlled = controlled @ matrix.T + h[k]*u
        energy += h[k]*float(u.double().square().sum())
    assert float(controlled[0, 0]-state[0, 0]) == pytest.approx(1., abs=3e-7)
    assert energy == pytest.approx(1./3., rel=2e-7)


def test_full_nonlinear_suffix_gradient_matches_direct_autograd():
    steps = [lambda x: 1.7*x+x.square()/20., lambda x: torch.sin(x)+x/3.,
             lambda x: x+x.square()/7.]
    state = torch.tensor([[.12, -.31]], dtype=torch.float32)
    states = [state[0].numpy()]
    for step in steps:
        state = step(state)
        states.append(state[0].numpy())
    terminal = state.detach().requires_grad_(True)
    psi = terminal.square().sum()+terminal[0, 0]
    terminal_gradient, = torch.autograd.grad(psi, terminal)
    saved = {}
    audit.suffix_adjoint(np.asarray(states), lambda x, k: steps[k](x), terminal_gradient,
                        [.2, .3, .5], lambda k, a: saved.update({k: a.clone()}), Counter())
    for k in range(3):
        successor = torch.tensor(states[k+1][None], requires_grad=True)
        value = successor
        for step in steps[k+1:]:
            value = step(value)
        direct, = torch.autograd.grad(value.square().sum()+value[0, 0], successor)
        torch.testing.assert_close(saved[k], direct)


def test_observable_sum_gradient_and_gram_use_distinct_averages():
    features = torch.tensor([[1., 4.], [2., 8.]], requires_grad=True)
    direction = torch.tensor([[.2, .7], [1.1, -.3]], dtype=torch.float64)
    center = torch.tensor([-.5, .6], dtype=torch.float64)
    gradients, = torch.autograd.grad(audit.observable(features, direction, center).sum(), features)
    torch.testing.assert_close(gradients, direction.float())
    gram, lam = audit.solve_cohort_gram([2., 6.], expected_count=2, delta=.25)
    assert gram == 4.
    assert lam == .0625
    assert np.mean([lam*2., lam*6.]) == .25
    for values in ([], [2.], [2., -1.], [np.inf, 1.], [0., 0.]):
        with pytest.raises((ValueError, FloatingPointError)):
            audit.solve_cohort_gram(values, expected_count=2)


def test_control_is_added_after_euler_without_clean_denominator_rescaling():
    state = torch.tensor([[[[.8, -.2]]]])
    clean = torch.tensor([[[[.3, .1]]]])
    u = torch.tensor([[[[.05, -.15]]]])
    current, following = .03, .02
    result = audit.euler_from_clean(state, clean, current, following)+(current-following)*u
    expected = state-.01*((state-clean)/.05)+.01*u
    torch.testing.assert_close(result, expected)
    assert not torch.allclose(result, audit.euler_from_clean(state, clean+u, current, following))


def test_native_head_and_pixel_arithmetic_keep_bf16_rounding():
    full = torch.tensor([1.03125, 1.03125], dtype=torch.bfloat16).reshape(2, 1, 1, 1)
    base = torch.tensor([.3125, .3125], dtype=torch.bfloat16).reshape_as(full)
    result = audit.official_heads(full, base, torch.tensor([.5, .074]))
    assert result.dtype == torch.bfloat16
    assert torch.equal(result[0], (base+1.78*(full-base))[0])
    assert torch.equal(result[1], full[1])
    decoded = torch.tensor([.50390625, -.2, 1.1], dtype=torch.bfloat16)
    assert torch.equal(audit.native_pixels(decoded), decoded.clamp(0, 1).mul(255).to(torch.uint8))
    with pytest.raises(ValueError, match='BF16'):
        audit.native_pixels(decoded.float())


def test_fixed_ids_and_grid():
    assert tuple(np.linspace(0, 999, 8, dtype=np.int64)) == audit.IDS
    assert audit.parse_ids('999,0') == (0, 999)
    for text in ('', '0,0', '1', '0,1000', 'cat'):
        with pytest.raises(Exception):
            audit.parse_ids(text)
    grid = audit.time_grid()
    assert len(grid) == 101 and grid[0] == 1. and grid[-1] == 0.
    assert all(a > b for a, b in zip(grid[:-1], grid[1:]))
    assert grid[-2] > audit.T_EPS


def test_frozen_control_rejects_partial_cohort_tampered_coefficient_and_artifacts(tmp_path):
    request_hash = 'frozen-request'
    records = []
    values = []
    for index, image_id in enumerate(audit.IDS):
        value = float(index+1)
        path = tmp_path / f'{image_id}.json'
        audit.write_json(path, {'global_id': image_id, 'request_sha256': request_hash,
                               'complete': True, 'integrated_gram': value})
        records.append(audit.record(path))
        values.append(value)
    gram, lam = audit.solve_cohort_gram(values)
    frozen = {'protocol': audit.PROTOCOL, 'phase': 'finalize', 'complete': True,
              'request_sha256': request_hash, 'global_ids': list(audit.IDS),
              'delta': audit.DELTA, 'cohort_gram': gram, 'lambda': lam,
              'collect_summaries': records}
    path = tmp_path / 'frozen_control.json'
    path.write_text(json.dumps(frozen))
    assert audit.load_frozen(tmp_path, request_hash)['lambda'] == lam
    for changes in ({'lambda': lam*2}, {'collect_summaries': records[:-1]}, {'delta': .01}):
        path.write_text(json.dumps({**frozen, **changes}))
        with pytest.raises(ValueError):
            audit.load_frozen(tmp_path, request_hash)
    path.write_text(json.dumps(frozen))
    Path(records[0]['path']).write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        audit.load_frozen(tmp_path, request_hash)


def test_durable_results_cannot_be_overwritten(tmp_path):
    path = tmp_path / 'request.json'
    audit.write_json(path, {'delta': audit.DELTA})
    with pytest.raises(FileExistsError):
        audit.write_json(path, {'delta': .1})
    assert json.loads(path.read_text())['delta'] == audit.DELTA


def test_finalize_requires_all_eight_and_recomputes_norms_from_saved_adjoint(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'STEPS', 2)
    monkeypatch.setattr(audit, 'LATENT_SHAPE', (2,))
    monkeypatch.setattr(audit, 'time_grid', lambda: [1., .5, 0.])
    monkeypatch.setattr(audit, 'load_request', lambda *args, **kwargs: {})
    request = tmp_path / 'request.json'
    request.write_text('{}')
    request_hash = audit.sha256(request)
    integrated = []
    for index, image_id in enumerate(audit.IDS):
        directory = tmp_path / 'collect' / f'id{image_id:04d}'
        directory.mkdir(parents=True)
        adjoint = np.asarray([[index+1., 0.], [0., 1.]], dtype=np.float32)
        np.save(directory / 'adjoint.npy', adjoint)
        np.save(directory / 'states.npy', np.zeros((3, 2), dtype=np.float32))
        for kind in ('latent', 'features', 'pixels'):
            np.save(directory / f'{kind}.npy', np.zeros(2, dtype=np.float32))
        norms = np.square(adjoint.astype(np.float64)).sum(axis=1)
        gram = float(.5*norms.sum())
        integrated.append(gram)
        summary = {'protocol': audit.PROTOCOL, 'phase': 'collect', 'complete': True,
                   'global_id': image_id, 'request_sha256': request_hash,
                   'global_noise_sha256': 'same-full-cohort', 'noise_sha256': f'noise-{image_id}',
                   'states': audit.record(directory / 'states.npy'),
                   'adjoints': audit.record(directory / 'adjoint.npy'),
                   'fp32_baseline': {'artifacts': {kind: audit.record(directory / f'{kind}.npy')
                                                 for kind in ('latent', 'features', 'pixels')}},
                   'post_state_squared_norms': norms.tolist(), 'integrated_gram': gram,
                   'coordinate_check': {'passed': True, 'artifact': audit.record(directory / 'pixels.npy')}
                   if index == 0 else None}
        audit.write_json(directory / 'summary.json', summary)
        if index < 7:
            with pytest.raises(FileNotFoundError):
                audit.finalize(tmp_path)
    audit.finalize(tmp_path)
    result = audit.load_frozen(tmp_path, request_hash)
    assert result['cohort_gram'] == np.mean(integrated)
    assert result['lambda'] == audit.DELTA / np.mean(integrated)
    assert result['predicted_mean_response'] == pytest.approx(audit.DELTA)


def test_fp32_ig_single_step_retains_input_graph_and_counts_one_shared_forward():
    class Model:
        def __call__(self, x, times, context, attn_mask):
            assert torch.is_grad_enabled()
            return 2*x, .5*x

    state = torch.tensor([[[[.3, -.8]]]], requires_grad=True)
    counts = Counter()
    output = audit.unit_step(Model(), state, torch.tensor([142]), .5, .4, native=False, counts=counts)
    gradient, = torch.autograd.grad(output.sum(), state)
    expected_clean_slope = .5+1.78*1.5
    assert torch.allclose(gradient, torch.full_like(state, 1.-.1*(1.-expected_clean_slope)/.5))
    assert counts == {'stage2_forward_calls': 1, 'stage2_sample_forwards': 1}
