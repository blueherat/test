from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import sample_raev2_paired_bridge as sampler
from experiments.raev2_paired_bridge import PairedBridgeField, native_euler


class ToyModel(torch.nn.Module):
    def forward(self, state, times, **kwargs):
        label = kwargs['context'][:, None, None, None]
        return (state*.125+label/31).bfloat16(), (state*.0625-.125).bfloat16()


@pytest.fixture(autouse=True)
def no_cuda_autocast(monkeypatch):
    monkeypatch.setattr(torch, 'autocast', lambda *args, **kwargs: nullcontext())
    torch.set_num_threads(1)


def test_fixed_interface_and_only_official_allows_different_cost():
    common = ['--output-dir', '/tmp/unused']
    assert sampler.parse_args(['--mode', 'parity', *common]).num_samples == 16
    for mode in ('official', 'candidate', 'control'):
        args = sampler.parse_args(['--mode', mode, *common, '--parity-dir', '/tmp/parity'])
        assert args.num_samples == 1000 and args.seed == 202609151
    assert sampler.parse_args(['--mode', 'official', *common, '--parity-dir', '/tmp/parity',
                               '--num-steps', '201']).num_steps == 201
    for extra in (['--mode', 'candidate', '--parity-dir', '/tmp/parity', '--num-steps', '101'],
                  ['--mode', 'parity', '--num-samples', '8'],
                  ['--mode', 'parity', '--seed', '202609152'], ['--mode', 'official']):
        with pytest.raises(SystemExit):
            sampler.parse_args([*common, *extra])


def test_every_native_step_matches_production_and_training_floor_arithmetic():
    model = ToyModel()
    labels = torch.arange(8)
    state = torch.linspace(-1, 1, 48).reshape(8, 3, 1, 2)
    for steps in (100, 201):
        a, b = state.clone(), state.clone()
        grid = sampler.shifted_time_grid(steps, 8, torch.device('cpu')).tolist()
        for t, s in zip(grid[:-1], grid[1:]):
            times = torch.full((8,), t)
            g = sampler.clean_forward(model, a, times, labels)
            expected = native_euler(a, g, t, s)
            a = sampler.successor(model, a, labels, t, s, mode='official')
            b = sampler.production.sampling_step(model, None, b, labels, t, s, use_potential=False)
            assert torch.equal(a, b) and torch.equal(a, expected)


def test_zero_full_field_preserves_all_native_steps_for_candidate_and_control():
    field = PairedBridgeField(latent_channels=3, spatial_size=(1, 2), num_classes=8).eval()
    model = ToyModel()
    labels = torch.arange(8)
    original = torch.linspace(-2, 2, 48).reshape(8, 3, 1, 2)
    grid = sampler.shifted_time_grid(100, 8, torch.device('cpu')).tolist()
    states = {mode: original.clone() for mode in ('official', 'candidate', 'control')}
    for t, s in zip(grid[:-1], grid[1:]):
        for mode in states:
            states[mode] = sampler.successor(model, states[mode], labels, t, s, mode=mode,
                                            field=None if mode == 'official' else field)
        assert torch.equal(states['official'], states['candidate'])
        assert torch.equal(states['official'], states['control'])


def test_nonzero_field_uses_euler_successor_and_evolving_midpoint_without_window():
    labels = torch.arange(8)
    z = torch.linspace(-1, 1, 48).reshape(8, 3, 1, 2)
    model = ToyModel()
    for mode in ('candidate', 'control'):
        for t, s in ((1., .998), (.074, 0.), (.03, .01)):
            calls = []
            def field(state, times, following, tau, classes):
                calls.append((state.clone(), tau.clone(), classes.clone()))
                return state*.25+tau[:, None, None, None]
            native = sampler.successor(model, z, labels, t, s, mode='official')
            actual = sampler.successor(model, z, labels, t, s, mode=mode, field=field)
            beta = (torch.tensor(t)-torch.tensor(s))/torch.tensor(t).clamp_min(.05)
            mid = native+.5*beta*(native*.25)
            tau = .5 if mode == 'candidate' else 0.
            expected = native+beta*(mid*.25+tau)
            assert torch.equal(actual, expected)
            assert len(calls) == 2 and torch.equal(calls[0][0], native)
            assert torch.equal(calls[1][0], mid)
            assert torch.equal(calls[1][1], torch.full((8,), tau))
            assert all(torch.equal(row[2], labels) for row in calls)
    with pytest.raises(ValueError):
        sampler.successor(model, z, labels, 1., .9, mode='official', field=field)


def test_actual_module_counts_are_distinct_and_forbid_accidental_B16():
    model = ToyModel()
    field = PairedBridgeField(latent_channels=3, spatial_size=(1, 2), num_classes=8)
    audit = sampler.ForwardAudit({'stage2': model, 'candidate': field})
    before = audit.snapshot()
    z, labels = torch.ones(8, 3, 1, 2), torch.arange(8)
    sampler.successor(model, z, labels, 1., .9, mode='candidate', field=field)
    result = audit.check_delta(before, {'stage2': 1, 'candidate': 2})
    assert result['observed']['stage2_sample_forwards'] == 8
    assert result['observed']['candidate_sample_forwards'] == 16
    with pytest.raises(ValueError):
        model(z.repeat(2, 1, 1, 1), torch.ones(16), context=labels.repeat(2))
    audit.close()


def test_BF16_export_and_evaluator_archive_layout(tmp_path):
    decoded = torch.full((8, 3, 256, 256), .501953125, dtype=torch.bfloat16)
    images = sampler.native_uint8(decoded)
    assert images.dtype == np.uint8 and images.shape == (8, 256, 256, 3)
    expected = decoded.clamp(0, 1).mul(255).permute(0, 2, 3, 1).byte().numpy()
    assert np.array_equal(images, expected)
    path = tmp_path/'samples.npz'
    ids = np.arange(8, dtype=np.int64)
    sampler.save_npz(path, images, ids=ids, labels=ids)
    with np.load(path, allow_pickle=False) as payload:
        assert set(payload.files) == {'arr_0', 'ids', 'labels'}
        assert np.array_equal(payload['arr_0'], images)
        assert np.array_equal(payload['ids'], payload['labels'])


def test_source_archive_keeps_colliding_basenames_separate(tmp_path, monkeypatch):
    root = tmp_path/'repo'
    names = ('external/stage2/model_utils.py', 'external/utils/model_utils.py')
    for i, name in enumerate(names):
        file = root/name
        file.parent.mkdir(parents=True)
        file.write_text(str(i))
    monkeypatch.setattr(sampler, 'ROOT', root)
    monkeypatch.setattr(sampler, 'SOURCE_FILES', names)
    records = sampler.archive_sources(tmp_path/'output')
    assert set(records) == set(names)
    assert len({record['path'] for record in records.values()}) == 2
    assert len({record['sha256'] for record in records.values()}) == 2
    for item in records.values():
        sampler.verify_record(item)


def test_cpu_noise_draw_is_paired_by_full_cohort_not_loop_steps():
    first, state = sampler.paired_noise(sampler.SEED, 'cpu', count=16, latent_shape=(3, 1, 2))
    second, state2 = sampler.paired_noise(sampler.SEED, 'cpu', count=16, latent_shape=(3, 1, 2))
    assert torch.equal(first, second) and torch.equal(state, state2)
    assert first.dtype == torch.float32


def test_parity_rejects_failed_intermediate_step_even_if_endpoints_marked_equal(tmp_path):
    request = {'protocol': sampler.PROTOCOL, 'seed': sampler.SEED, 'num_steps': 100}
    sampler.atomic_json(tmp_path/'request.json', request)
    sampler.atomic_json(tmp_path/'summary.json', {
        'complete': True, 'command': 'parity', 'samples_per_loop': 16,
        'stepwise_bitwise': False, 'endpoint_bitwise': True, 'pixel_bitwise': True,
        'request': sampler.artifact(tmp_path/'request.json'),
    })
    with pytest.raises(ValueError, match='production/official/zero-bridge parity'):
        sampler.verify_parity(tmp_path, {}, {}, {})


def test_official_request_binds_training_without_deserializing_auxiliary_weights(tmp_path, monkeypatch):
    # Exercise the actual make_request -> verify_training route. The checkpoint
    # deliberately is not a Torch archive: official needs only its frozen bytes.
    root = tmp_path/'repo'
    root.mkdir()
    paths = {}
    for name in ('config', 'baseline', 'decoder', 'stats', 'source.py'):
        paths[name] = root/name
        paths[name].write_text(name)
    train = tmp_path/'fit/train'
    train.mkdir(parents=True)
    checkpoint = train/'final.pt'
    checkpoint.write_bytes(b'frozen auxiliary checkpoint bytes')
    monkeypatch.setattr(sampler, 'ROOT', root)
    monkeypatch.setattr(sampler, 'SOURCE_FILES', ('source.py',))
    monkeypatch.setattr(sampler, 'DEFAULT_BRIDGE_CHECKPOINT', checkpoint)
    for constant, path in (('BRIDGE_SHA256', checkpoint), ('CONFIG_SHA256', paths['config']),
                           ('BASELINE_SHA256', paths['baseline']), ('DECODER_SHA256', paths['decoder']),
                           ('STATS_SHA256', paths['stats'])):
        monkeypatch.setattr(sampler, constant, sampler.sha256_file(path))
    request = {'protocol': sampler.TRAINING_PROTOCOL, 'mode': 'train',
               'config': sampler.artifact(paths['config']), 'baseline_checkpoint': sampler.artifact(paths['baseline']),
               'baseline_checkpoint_step': 100080, 'sources': {'source.py': sampler.artifact(paths['source.py'])}}
    sampler.atomic_json(train/'request.json', request)
    sampler.atomic_json(train/'summary.json', {
        'complete': True, 'protocol': sampler.TRAINING_PROTOCOL, 'mode': 'train', 'updates': 2048,
        'checkpoint': sampler.artifact(checkpoint), 'request': sampler.artifact(train/'request.json')})
    sampler.atomic_json(train.parent/'supplemental_source_environment.json', {
        'sources': [{'path': 'source.py', 'sha256': sampler.sha256_file(paths['source.py'])}]})
    config = SimpleNamespace(
        misc=SimpleNamespace(latent_size=sampler.LATENT_SHAPE, num_classes=1000),
        transport=SimpleNamespace(prediction='x', t_eps=.05),
        guidance=SimpleNamespace(ig=SimpleNamespace(scale=1.78, t_min=.1, t_max=1), cfg=SimpleNamespace(scale=1)),
        stage_1=SimpleNamespace(params={'pretrained_decoder_path': str(paths['decoder']),
                                        'normalization_stat_path': str(paths['stats'])}))
    monkeypatch.setattr(sampler, 'load_config', lambda path: config)
    monkeypatch.setattr(sampler, 'verify_parity', lambda *args: {'already_checked': True})
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: pytest.fail('official must not deserialize auxiliary weights'))
    args = sampler.parse_args(['--mode', 'official', '--output-dir', str(tmp_path/'sample'),
                               '--parity-dir', str(tmp_path/'parity'), '--config', str(paths['config']),
                               '--checkpoint', str(paths['baseline']), '--bridge-checkpoint', str(checkpoint)])
    actual, _, weights = sampler.make_request(args, args.output_dir)
    assert weights is None
    assert actual['auxiliary_checkpoint_deserialized'] is False
    assert actual['training']['checkpoint'] == sampler.artifact(checkpoint)
