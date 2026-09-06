from contextlib import nullcontext
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import sample_raev2_affine_reflection as sampler
from experiments.train_raev2_observable_potential import official_clean_from_heads


def geometry(dtype=torch.float64):
    mean = torch.tensor([[[.2, -.1]], [[-.3, .4]], [[.1, .5]]], dtype=torch.float64)
    variance = torch.tensor([[[1., 2.]], [[3., .5]], [[.2, 1.2]]], dtype=torch.float64)
    u, c = sampler.affine_geometry(mean, variance)
    return u.to(dtype), c.to(dtype), mean, variance


def test_geometry_is_the_encoder_constraint_in_normalized_coordinates():
    u, c, mean, variance = geometry()
    assert torch.allclose(u.square().sum(0), torch.ones(1, 2, dtype=torch.float64), atol=3e-16)
    raw = torch.arange(48, dtype=torch.float64).reshape(8, 3, 1, 2)/13
    raw -= raw.mean(1, keepdim=True)
    clean = (raw-mean)/(variance+1e-5).sqrt()
    assert sampler.normal_part(clean-c, u).abs().max() < 1e-15
    assert torch.allclose(sampler.normal_part(c[None], u)[0], c, atol=1e-16)
    with pytest.raises(ValueError):
        sampler.affine_geometry(mean, -variance)
    with pytest.raises(ValueError):
        sampler.affine_geometry(mean, torch.full_like(variance, float('nan')))
    with pytest.raises(ValueError):
        sampler.affine_geometry(mean.flatten(), variance.flatten())


def test_projection_orthogonality_pythagoras_and_idempotence():
    u, c, _, _ = geometry()
    x = torch.sin(torch.arange(48, dtype=torch.float64)).reshape(8, 3, 1, 2)
    clean = sampler.project_clean(x*.3+.2, u, c)
    projected = sampler.project_clean(x, u, c)
    normal = sampler.normal_part(x-c, u)
    assert torch.allclose(projected+normal, x, atol=2e-16)
    assert torch.allclose(sampler.project_clean(projected, u, c), projected, atol=3e-16)
    assert abs(float(((projected-clean)*normal).sum())) < 1e-14
    improvement = (x-clean).square().sum()-(projected-clean).square().sum()
    assert float(improvement) == pytest.approx(float(normal.square().sum()), abs=4e-15)


def test_reflection_is_involution_at_correct_moving_center_and_preserves_tangent():
    u, c, _, _ = geometry()
    z = torch.cos(torch.arange(48, dtype=torch.float64)).reshape(8, 3, 1, 2)
    for t in (1., .7, 0.):
        r = sampler.mirror_normal_noise(z, t, u, c)
        assert torch.allclose(sampler.mirror_normal_noise(r, t, u, c), z, atol=6e-16)
        d = r-z
        assert torch.allclose(d, sampler.normal_part(d, u), atol=5e-16)
        assert torch.allclose(sampler.normal_part(r-(1-t)*c, u),
                              -sampler.normal_part(z-(1-t)*c, u), atol=5e-16)
    with pytest.raises(ValueError):
        sampler.mirror_normal_noise(z, -1., u, c)


def test_C2_average_is_invariant_but_retains_even_normal_dependence():
    u = torch.tensor([[[1.]], [[0.]]], dtype=torch.float64)
    c = torch.zeros_like(u)
    def raw(z):
        # Deliberately nonconservative: tangent output depends on squared normal.
        return torch.cat((z[:, 1:2]+.3, z[:, :1].square()+z[:, 1:2]), dim=1)
    def guided(z, t):
        return sampler.project_clean((raw(z)+raw(sampler.mirror_normal_noise(z, t, u, c)))*.5, u, c)
    z = torch.tensor([2., 1.], dtype=torch.float64).reshape(1, 2, 1, 1)
    assert torch.equal(guided(z, .4), guided(sampler.mirror_normal_noise(z, .4, u, c), .4))
    z2 = z.clone(); z2[:, 0] = 3.
    assert not torch.equal(guided(z, .4), guided(z2, .4))
    assert torch.count_nonzero(sampler.normal_part(guided(z, .4)-c, u)) == 0


def test_exact_nonconservative_closed_loop_equivariance_and_normal_bridge():
    u, c, _, _ = geometry()
    noise = torch.sin(torch.arange(48, dtype=torch.float64)).reshape(8, 3, 1, 2)
    state = noise.clone()
    reflected_state = sampler.mirror_normal_noise(noise, 1., u, c)
    grid = sampler.shifted_time_grid(100, 8, torch.device('cpu')).tolist()
    assert min(grid[:-1]) > sampler.T_EPS
    def raw(z):
        return .15*z.roll(1, 1)+.01*z.square()+.04
    def guided(z, t):
        return sampler.project_clean((raw(z)+raw(sampler.mirror_normal_noise(z, t, u, c)))*.5, u, c)
    for t, s in zip(grid[:-1], grid[1:]):
        state = (s/t)*state+(1-s/t)*guided(state, t)
        reflected_state = (s/t)*reflected_state+(1-s/t)*guided(reflected_state, t)
        assert torch.allclose(reflected_state, sampler.mirror_normal_noise(state, s, u, c), atol=2e-14)
        expected = (1-s)*c+s*sampler.normal_part(noise, u)
        assert torch.allclose(sampler.normal_part(state, u), expected, atol=2e-14)
    assert sampler.normal_part(state-c, u).abs().max() < 1e-15


def test_preserved_official_floor_does_not_have_the_exact_normal_bridge():
    z = torch.tensor([[[[.03]]]], dtype=torch.float32)
    clean = torch.zeros_like(z)
    actual = sampler.euler_update(z, clean, None, .03, 0.)
    assert torch.equal(actual, z-.03*(z/.05))
    assert torch.count_nonzero(actual) == 1  # endpoint would be zero without floor


def test_native_BF16_mixing_window_then_FP32_Euler(monkeypatch):
    monkeypatch.setattr(torch, 'autocast', lambda *a, **k: nullcontext())
    class Model:
        def __call__(self, z, t, **kwargs):
            return torch.full_like(z, 1.03125, dtype=torch.bfloat16), torch.full_like(z, .3125, dtype=torch.bfloat16)
    z = torch.ones(8, 3, 1, 2)*.7
    labels = torch.arange(8)
    full, base = Model()(z, None)
    wrong = base.float()+1.78*(full.float()-base.float())
    assert not torch.equal((base+1.78*(full-base)).float(), wrong)
    for t, s in ((1., .99), (.1, .08), (.074, 0.), (.03, .01)):
        native = (base+1.78*(full-base)).float() if t >= .1 else full.float()
        expected = z-(t-s)*((z-native)/max(t, .05))
        actual = sampler.successor(Model(), z, labels, t, s, mode='official')
        direct = sampler.production.sampling_step(Model(), None, z, labels, t, s, use_potential=False)
        assert torch.equal(actual, expected) and torch.equal(actual, direct)


def test_reflection_uses_two_separate_native_B8_calls_all_times(monkeypatch):
    monkeypatch.setattr(torch, 'autocast', lambda *a, **k: nullcontext())
    u, c, _, _ = geometry(torch.float32)
    z = torch.sin(torch.arange(48, dtype=torch.float32)).reshape(8, 3, 1, 2)
    labels = torch.arange(8)
    for t in (1., .074):
        calls = []
        class Model:
            def __call__(self, value, times, **kwargs):
                calls.append((value.clone(), times.clone(), kwargs['context'].clone()))
                return (value*.2+.1).bfloat16(), (value.roll(1, 1)*.3-.04).bfloat16()
        actual = sampler.reflection_guided(Model(), z, labels, t, u, c)
        assert len(calls) == 2 and all(len(call[0]) == 8 for call in calls)
        assert torch.equal(calls[0][0], z)
        reflected = sampler.mirror_normal_noise(z, t, u, c)
        assert torch.equal(calls[1][0], reflected)
        assert all(torch.equal(call[2], labels) for call in calls)
        assert torch.equal(calls[0][1], calls[1][1])
        def predict(value):
            f, b = (value*.2+.1).bfloat16(), (value.roll(1, 1)*.3-.04).bfloat16()
            return official_clean_from_heads(f, b, calls[0][1]).float()
        expected = sampler.project_clean((predict(z)+predict(reflected))*.5, u, c)
        assert actual.dtype == torch.float32 and torch.equal(actual, expected)


def test_official_bypasses_all_geometry_and_100step_path_matches_production(monkeypatch):
    def clean(model, z, times, labels):
        return z*.125+labels.reshape(-1, 1, 1, 1).float()/31
    monkeypatch.setattr(sampler, 'clean_forward', clean)
    monkeypatch.setattr(sampler.production, 'clean_forward', clean)
    monkeypatch.setattr(sampler, 'reflection_guided', lambda *a, **k: pytest.fail('official must bypass geometry'))
    z = torch.linspace(-1, 1, 48).reshape(8, 3, 1, 2)
    labels = torch.arange(8)
    grid = sampler.shifted_time_grid(100, 8, torch.device('cpu')).tolist()
    a, _ = sampler.trajectory(None, z.clone(), labels, grid, mode='official')
    b, _ = sampler.trajectory(None, z.clone(), labels, grid, mode='official', reference=True)
    assert torch.equal(a, b)


def test_noise_pairing_and_one_full_CUDA_draw_without_using_CUDA(monkeypatch):
    a, ra = sampler.paired_noise(sampler.SEED, 'cpu', count=16, latent_shape=(3, 1, 2))
    b, rb = sampler.paired_noise(sampler.SEED, 'cpu', count=16, latent_shape=(3, 1, 2))
    assert torch.equal(a, b) and torch.equal(ra, rb)
    calls = []
    class Generator:
        def __init__(self, *, device): calls.append(('device', str(device)))
        def manual_seed(self, seed): calls.append(('seed', seed)); return self
        def get_state(self): return torch.arange(4, dtype=torch.uint8)
    def randn(shape, *, generator, device, dtype):
        calls.append(('draw', shape, str(device), dtype)); return torch.empty(0)
    monkeypatch.setattr(torch, 'Generator', Generator)
    monkeypatch.setattr(torch, 'randn', randn)
    sampler.paired_noise(sampler.SEED, 'cuda:0', count=1000)
    assert calls == [('device', 'cuda:0'), ('seed', sampler.SEED),
                     ('draw', (1000, 1024, 16, 16), 'cuda:0', torch.float32)]


def test_fixed_CLI_only_official_can_receive_cost_steps(tmp_path):
    common = ['--output-dir', str(tmp_path)]
    assert sampler.parse_args([*common, '--mode', 'parity']).num_samples == 16
    args = sampler.parse_args([*common, '--mode', 'official', '--parity-dir', 'parity', '--num-steps', '220'])
    assert args.num_samples == 1000 and args.num_steps == 220
    assert sampler.parse_args([*common, '--mode', 'reflection', '--parity-dir', 'parity']).num_steps == 100
    for extra in (['--mode', 'reflection', '--parity-dir', 'parity', '--num-steps', '200'],
                  ['--mode', 'official'], ['--mode', 'parity', '--seed', '1'],
                  ['--mode', 'parity', '--num-samples', '1000']):
        with pytest.raises(SystemExit): sampler.parse_args([*common, *extra])


@pytest.mark.parametrize('mode,factor', [('official', 1), ('reflection', 2)])
def test_actual_hook_counts_and_evaluator_compatible_archive(mode, factor, tmp_path, monkeypatch):
    monkeypatch.setattr(sampler, 'COUNT', 16)
    monkeypatch.setattr(sampler, 'LATENT_SHAPE', (3, 1, 2))
    monkeypatch.setattr(torch, 'autocast', lambda *a, **k: nullcontext())
    class Model(torch.nn.Module):
        def forward(self, z, t, **kwargs):
            return (z*.2+.1).bfloat16(), (z*.1+.05).bfloat16()
    class Decoder:
        def __init__(self): self.decoder = torch.nn.Identity()
    def decode(decoder, state):
        decoder.decoder(state)
        return np.full((len(state), 256, 256, 3), 127, dtype=np.uint8)
    monkeypatch.setattr(sampler, 'decode', decode)
    model, decoder = Model(), Decoder()
    audit = sampler.ForwardAudit(model, decoder, 'cpu')
    noise = torch.ones(16, 3, 1, 2)
    u, c, _, _ = geometry(torch.float32)
    result = sampler.sample(model, decoder, noise, torch.arange(16), [1., .5, 0.], u, c,
                            SimpleNamespace(mode=mode), tmp_path, torch.device('cpu'), audit)
    audit.close()
    assert result['stage2_forward_calls'] == 4*factor
    assert result['stage2_sample_forwards'] == 32*factor
    assert result['decoder_forward_calls'] == 2 and result['decoder_sample_forwards'] == 16
    assert result['forward_counts_observed'] == result['forward_counts_expected']
    assert result['reflection_calls'] == (4 if mode == 'reflection' else 0)
    assert result['average_projection_calls'] == result['reflection_calls']
    assert result['inference_trajectory_plus_decode_wall_seconds'] == result['trajectory_wall_seconds']+result['decode_and_uint8_wall_seconds']
    with np.load(tmp_path/'samples.npz', allow_pickle=False) as z:
        assert z['arr_0'].shape == (16, 256, 256, 3) and z['arr_0'].dtype == np.uint8
        assert np.array_equal(z['ids'], np.arange(16)) and np.array_equal(z['labels'], np.arange(16))
    assert len(json.loads((tmp_path/'batch_manifest.json').read_text())['batches']) == 2


def test_actual_hook_rejects_B16_and_native_pixel_rounding_is_preserved():
    model = torch.nn.Identity()
    decoder = SimpleNamespace(decoder=torch.nn.Identity())
    audit = sampler.ForwardAudit(model, decoder, 'cpu')
    with pytest.raises(ValueError, match='B16'):
        model(torch.zeros(16, 1))
    audit.close()
    values = torch.linspace(-.01, 1.01, 3*256*256).reshape(1, 3, 256, 256).bfloat16()
    actual = sampler.native_uint8(values)
    expected = values.clamp(0, 1).mul(255).permute(0, 2, 3, 1).to(torch.uint8).numpy()
    assert np.array_equal(actual, expected)
    with pytest.raises(ValueError, match='BF16'):
        sampler.native_uint8(values.float())


def test_parity_two_official_paths_and_fixed_four_forward_candidate_check(tmp_path, monkeypatch):
    monkeypatch.setattr(torch, 'autocast', lambda *a, **k: nullcontext())
    class Model(torch.nn.Module):
        def forward(self, z, t, **kwargs):
            return (z*.2+.1).bfloat16(), (z*.1+.05).bfloat16()
    decoder = SimpleNamespace(decoder=torch.nn.Identity())
    def decode(decoder, state):
        decoder.decoder(state)
        return np.full((len(state), 256, 256, 3), 127, dtype=np.uint8)
    monkeypatch.setattr(sampler, 'decode', decode)
    model = Model()
    audit = sampler.ForwardAudit(model, decoder, 'cpu')
    noise, _ = sampler.paired_noise(sampler.SEED, 'cpu', count=16, latent_shape=(3, 1, 2))
    u, c, _, _ = geometry()
    grid = sampler.shifted_time_grid(100, 8, torch.device('cpu')).tolist()
    result = sampler.run_parity(model, decoder, noise, torch.arange(16), grid, u, c,
                                tmp_path, torch.device('cpu'), audit)
    audit.close()
    assert result['complete'] and result['endpoint_bitwise'] and result['pixel_bitwise']
    assert result['off_branch_forward_counts_observed']['stage2_forward_calls'] == 400
    assert result['forward_counts_observed']['stage2_forward_calls'] == 404
    assert result['forward_counts_observed']['decoder_forward_calls'] == 4
    assert result['candidate_formula_check_forward_counts']['observed']['stage2_forward_calls'] == 4
    check = json.loads((tmp_path/'candidate_formula_check.json').read_text())
    assert check['all_finite'] and check['sample_ids'] == list(range(8))
    assert check['operator_counts'] == {'stage2_forward': 4, 'reflection': 2, 'average_projection': 2}


def test_frozen_source_verification_rejects_mutation_and_missing_coverage(tmp_path):
    source = tmp_path/'source.py'; source.write_text('frozen')
    record = sampler.artifact(source)
    plan = {'frozen_files': {str(source): record['sha256']}}
    sampler.verify_frozen_files(plan, {'config': record}, {})
    with pytest.raises(ValueError, match='absent'):
        sampler.verify_frozen_files({'frozen_files': {}}, {'config': record}, {})
    source.write_text('mutated')
    with pytest.raises(ValueError, match='changed'):
        sampler.verify_frozen_files(plan, {'config': record}, {})
