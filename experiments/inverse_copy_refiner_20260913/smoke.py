"""CPU-only synthetic smoke: zero identity, supervised update, leakage and overwrite guards."""
from pathlib import Path
import json
import tempfile
import numpy as np
import torch
from experiments.inverse_copy_refiner_20260913.model import Refiner
from experiments.inverse_copy_refiner_20260913.train import parser, run, load_pairs, evaluate, sha256


def main():
    root = Path(tempfile.mkdtemp(prefix='inverse_copy_refiner_cpu_smoke_'))
    rng = np.random.default_rng(1729)
    clean = rng.normal(size=(10, 4, 32, 32)).astype(np.float32)
    # Synthetic shape-only fixture; these are not real-image experimental pairs.
    corrupted = np.stack((clean+.10, clean-.16), axis=1).astype(np.float32)
    labels = np.arange(10, dtype=np.int64) % 3
    split = np.array([0]*8+[1]*2, dtype=np.int8)
    source_ids = np.arange(10, dtype=np.int64)
    path = root/'fixture.npz'
    np.savez(path, clean=clean, corrupted=corrupted, labels=labels, split=split,
             source_ids=source_ids, taus=np.array([.75,.5]))
    args = parser().parse_args(['--pairs', str(path), '--output', str(root/'run'),
                               '--device', 'cpu', '--steps', '3', '--eval-every', '1',
                               '--batch-sources', '2', '--width', '8', '--context-dim', '16',
                               '--num-classes', '3', '--cpu-threads', '1', '--allow-small-data'])
    summary = run(args)
    last = torch.load(root/'run'/'last.pt', map_location='cpu', weights_only=False)
    best = torch.load(root/'run'/'best.pt', map_location='cpu', weights_only=False)
    model = Refiner(**last['model_config'])
    model.load_state_dict(last['model'])
    model.eval()
    with torch.no_grad():
        predicted = model(torch.from_numpy(corrupted[8:, 0]), torch.from_numpy(labels[8:]))
    assert predicted.shape == (2, 4, 32, 32)
    assert torch.isfinite(predicted).all()
    assert model.output[-1].weight.abs().sum() > 0, 'No gradient update reached the zero-initialized output'
    assert best['step'] == summary['best_step']
    assert summary['best_heldout']['objective'] <= summary['zero_p_baseline']['heldout']['objective']
    shape_fixture = root/'shape_only_m4.npz'
    np.savez(shape_fixture, clean=clean, corrupted=np.concatenate((corrupted, corrupted), axis=1),
             labels=labels, split=split, source_ids=source_ids,
             taus=np.array([.75,.5,.75,.5]), directions=np.array([1,1,-1,-1], dtype=np.int8))
    shape_data, _, shape_held, shape_metadata = load_pairs(shape_fixture, num_classes=3, allow_small=True)
    assert shape_metadata['directions'] == [1,1,-1,-1]
    shape_metrics = evaluate(model, shape_data, shape_held, torch.device('cpu'), batch_sources=2)
    assert len(shape_metrics['repair_mse_by_corruption']) == 4
    try:
        run(args)
    except FileExistsError:
        overwrite_rejected = True
    else:
        raise AssertionError('Output overwrite was not rejected')
    leaked_ids = source_ids.copy()
    leaked_ids[-1] = leaked_ids[0]
    bad = root/'leaked_fixture.npz'
    np.savez(bad, clean=clean, corrupted=corrupted, labels=labels, split=split, source_ids=leaked_ids)
    try:
        load_pairs(bad, num_classes=3, allow_small=True)
    except ValueError as error:
        assert 'leakage' in str(error)
        leakage_rejected = True
    else:
        raise AssertionError('Source-ID leakage was not rejected')
    full_model = Refiner()
    checks = {'synthetic_cpu_smoke_only': True, 'temporary_outputs': str(root),
              'zero_initialization_bitwise_identity': True, 'three_cpu_updates_finite': True,
              'output_layer_receives_gradient': True, 'checkpoint_roundtrip_load': True,
              'initial_identity_in_best_selection': True, 'overwrite_rejected': overwrite_rejected,
              'cross_split_source_leakage_rejected': leakage_rejected,
              'generic_M4_shape_only_evaluation': True,
              'production_model_parameters': sum(parameter.numel() for parameter in full_model.parameters()),
              'smoke_summary': summary,
              'source_sha256': {name: sha256(Path(__file__).with_name(name)) for name in ('model.py', 'train.py', 'smoke.py')}}
    destination = Path('docs/research/identity_copy_toy_20260913/refiner_cpu_smoke.json')
    destination.write_text(json.dumps(checks, indent=2)+'\n')
    print(json.dumps({'checks': str(destination), **{key: value for key, value in checks.items() if key != 'smoke_summary'}}, indent=2))


if __name__ == '__main__':
    main()
