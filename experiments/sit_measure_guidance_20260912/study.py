"""Offline mechanism comparison and a full-covariance Gaussian control."""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from . import core, train
from .data import ROOT, WORK, METHODS
from experiments.sit_guidance_portfolio_20260910 import operators
from experiments.lifting_scale_sweep_20260909 import atomic, read, sha, array_sha


def assets():
    train.verify()
    output = ROOT/'gaussian_reference.pt'
    if output.exists():
        assert sha(output) == read(ROOT/'gaussian_reference.json')['sha256']
        return
    moments = np.load(ROOT/'train_moments.npy', mmap_mode='r')
    means = np.load(ROOT/'class_means.npy')
    factors, diagonal = [], []
    for c in range(100):
        scaled = moments[c, :, :4]*.18215
        factors.append((scaled-means[c]).reshape(256, -1))
        diagonal.append(np.sqrt(np.mean((moments[c, :, 4:]*.18215)**2, axis=0)))
    value = dict(means=torch.from_numpy(means), factors=torch.from_numpy(np.stack(factors)),
                 posterior_std=torch.from_numpy(np.stack(diagonal)))
    assert value['factors'].mean(1).abs().max() < 1e-6
    torch.save(value, output)
    atomic(ROOT/'gaussian_reference.json', dict(passed=True, sha256=sha(output),
        training_request_sha256=sha(ROOT/'training_request.json'),
        includes_full_empirical_covariance=True, includes_vae_posterior_variance=True,
        no_fid_used=True))


def regression(target, basis, train_count=64):
    columns = [np.asarray(b, dtype=np.float64).reshape(128, -1) for b in basis]
    y = np.asarray(target, dtype=np.float64).reshape(128, -1)
    x = np.stack([b[:train_count].reshape(-1) for b in columns], -1)
    coefficient = np.linalg.lstsq(x, y[:train_count].reshape(-1), rcond=1e-10)[0]
    prediction = sum(weight*b[train_count:] for weight, b in zip(coefficient, columns))
    energy = np.square(y[train_count:]).sum()
    return dict(explained=1-float(np.square(y[train_count:]-prediction).sum()/energy),
                coefficients=coefficient.tolist())


@torch.inference_mode()
def run():
    assets()
    assert read(ROOT/'training_status.json')['phase'] == 'complete'
    rt = operators.make_runtime()
    moments = np.load(ROOT/'validation_moments.npy', mmap_mode='r')
    labels = np.arange(128) % 100
    slots = np.arange(128)//100
    indices = np.load(ROOT/'validation_indices.npy')[labels, slots]
    assert len(set(indices)) == 128
    gen = torch.Generator(device='cuda').manual_seed(2026120920)
    block = torch.from_numpy(np.array(moments[labels, slots])).cuda()
    clean = (block[:, :4]+block[:, 4:]*torch.randn((128, 4, 32, 32),
             device='cuda', generator=gen))*.18215
    noise = torch.randn(clean.shape, device='cuda', generator=gen)
    ys = torch.from_numpy(labels).cuda()
    records, raw = [], {}
    begin = time.perf_counter()
    for tv in (.15, .5, .85):
        states = tv*clean+(1-tv)*noise
        values = {}
        for start in range(0, 128, 8):
            z, y = states[start:start+8], ys[start:start+8]
            time_vector = z.new_full((len(z),), tv)
            strong, heads, _ = rt.small.evaluate_source_with_heads(rt.model, z, time_vector, y,
                heads=rt.portfolio_heads, source_semantics=rt.semantics)
            for key, value in [('strong', strong), *heads.items()]:
                values.setdefault(key, []).append(value.cpu().numpy())
        values = {key:np.concatenate(items) for key, items in values.items()}
        for method in METHODS:
            model = core.weak_model(rt, method)
            values[method] = np.concatenate([model(states[i:i+8], states.new_full((8,), tv),
                ys[i:i+8]).cpu().numpy() for i in range(0, 128, 8)])
        state_array = states.cpu().numpy()
        for head in ('depth4_v', 'depth6_v', 'depth10_v'):
            gap = values['strong']-values[head]
            nuisance = [values['strong'], state_array]
            base = regression(gap, nuisance)
            for method in METHODS[1:]:
                signature = values['native']-values[method]
                fit = regression(gap, nuisance+[signature])
                plain = regression(gap, [signature])
                records.append(dict(time=tv, head=head, operator=method,
                    nuisance_explained=base['explained'], operator_alone_explained=plain['explained'],
                    combined_explained=fit['explained'],
                    incremental_explained=fit['explained']-base['explained'],
                    operator_coefficient=fit['coefficients'][-1],
                    diagnostic_only=True))
        for name, value in values.items():
            raw[f't{tv}_{name}'] = value
        raw[f't{tv}_state'] = state_array
    raw.update(labels=labels, source_indices=indices, clean=clean.cpu().numpy(), noise=noise.cpu().numpy())
    np.savez(ROOT/'mechanism_raw.npz', **raw)
    with (ROOT/'mechanism_results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    result = dict(passed=True, records=len(records), states=128*3,
        fit_images=64, heldout_images=64, all_source_images_distinct=True,
        raw_sha256=sha(ROOT/'mechanism_raw.npz'), results_sha256=sha(ROOT/'mechanism_results.csv'),
        training_request_sha256=sha(ROOT/'training_request.json'),
        complete_checkpoints={m:sha(ROOT/'training'/m/'model.pt') for m in METHODS},
        elapsed_seconds=time.perf_counter()-begin, no_fid_used=True,
        algorithm_uses_no_diagnostic_fit=True, input_noise_sha256=array_sha(noise.cpu().numpy()))
    atomic(ROOT/'mechanism_check.json', result)
    portable = WORK/'docs/data/sit_measure_guidance_20260912'
    portable.mkdir(parents=True, exist_ok=True)
    (portable/'mechanism_results.csv').write_bytes((ROOT/'mechanism_results.csv').read_bytes())
    atomic(portable/'mechanism_check.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets-only', action='store_true')
    args = parser.parse_args()
    if args.assets_only:
        assets()
    else:
        run()
