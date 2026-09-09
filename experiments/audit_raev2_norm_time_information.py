"""Noise-time information in latent norm, conditional on cached clean radii.

Exact noncentral-chi-square simulation of the norm of a Gaussian corruption.
No neural-model inference or evidence of model use of this statistic.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.train_raev2_observable_potential import load_banks


def main():
    started = time.perf_counter()
    root = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/potential_clean_bank_fp32_v1')
    banks, identity = load_banks(root)
    dimension = 1024 * 16 * 16
    radii = {}
    for name, (x, metadata) in banks.items():
        radii[name] = np.concatenate([
            np.square(np.asarray(x[i:i+32], dtype=np.float64)).reshape(-1, dimension).mean(1)
            for i in range(0, len(x), 32)])
    mean = float(radii['train'].mean())
    variance = float(radii['train'].var(ddof=1))
    rng = np.random.default_rng(202609435)

    def norm_moments(t):
        a = 1 - t
        return a*a*mean + t*t, a**4*variance + (4*a*a*t*t*mean + 2*t**4)/dimension

    def draw(t):
        # ||a X+t epsilon||²/t² conditional on X is noncentral chi-square.
        noncentrality = (1-t)**2 * radii['validation'] * dimension / t**2
        return t*t*rng.noncentral_chisquare(dimension, noncentrality)/dimension

    rows = []
    for t in [1., .95, .9, .75, .6]:
        r = t - 1/32
        current, future = draw(t), draw(r)
        mt, vt = norm_moments(t)
        mr, vr = norm_moments(r)
        threshold = (mt+mr)/2
        sign = np.sign(mt-mr)
        accuracy = .5*(np.mean(sign*(current-threshold)>0)+np.mean(sign*(future-threshold)<0))
        predicted = (mean+np.sqrt(np.maximum((1+mean)*current-mean, 0)))/(1+mean)
        rows.append(dict(noise_time=t, future_time=r,
                         separation_in_pooled_sd=abs(mt-mr)/np.sqrt((vt+vr)/2),
                         fixed_midpoint_balanced_accuracy=float(accuracy),
                         current_time_mae=float(np.abs(predicted-t).mean()),
                         current_time_rmse=float(np.sqrt(np.square(predicted-t).mean())),
                         empirical_current_norm_mean=float(current.mean()),
                         predicted_current_norm_mean=mt,
                         empirical_current_norm_variance=float(current.var(ddof=1)),
                         predicted_current_norm_variance=vt))
    out = ROOT/'experiments/results/terminal_defect_20260908/raev2_norm_time_information.json'
    result = dict(complete=True, seed=202609435, dimension=dimension, bank=identity,
                  train_count=len(radii['train']), validation_count=len(radii['validation']),
                  train_clean_norm_mean=mean, train_clean_norm_variance=variance,
                  validation_clean_norm_mean=float(radii['validation'].mean()),
                  validation_clean_norm_variance=float(radii['validation'].var(ddof=1)),
                  rows=rows, seconds=time.perf_counter()-started,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Norm-only information under real cached clean radii plus simulated exact Gaussian corruption. No SiT comparison, model-use or causal quality claim.')
    with out.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
