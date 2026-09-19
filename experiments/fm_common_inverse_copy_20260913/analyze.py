"""Read completed copy outputs and make a compact, reproducible comparison."""
from pathlib import Path
import hashlib
import json

import numpy as np

ROOT = Path('/home/zhoushunyu/data/eqvae/experiments/fm_common_inverse_copy_20260913_v2')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    request = json.loads((ROOT/'request.json').read_text())
    inputs = np.load(ROOT/'inputs.npz')
    n, rounds = request['samples'], request['rounds']
    expected_rows = rounds*len(request['arms'])
    rows, checks = [], {}
    for reference in request['references']:
        folder = ROOT/reference
        summary = json.loads((folder/'summary.json').read_text())
        assert summary['complete'] and summary['outputs'] == n*expected_rows
        raw = json.loads((folder/'results.json').read_text())
        assert len(raw) == expected_rows
        for arm in request['arms']:
            previous = None
            original = inputs['clean']
            for r in range(rounds+1):
                with np.load(folder/arm/f'round{r:02d}.npz') as d:
                    z, pixels = d['latents'], d['pixels']
                    assert z.shape == (n,4,32,32) and np.isfinite(z).all()
                    assert pixels.shape == (n,256,256,3) and pixels.dtype == np.uint8
                    np.testing.assert_array_equal(d['labels'], inputs['labels'])
                    if r == 0:
                        np.testing.assert_array_equal(z, original)
                    else:
                        matches = [a for a in raw if a['arm'] == arm and a['round'] == r]
                        assert len(matches) == 1
                        a = matches[0]
                        orig_mse = ((z.astype(np.float64)-original)**2).mean(axis=(1,2,3))
                        prev_mse = ((z.astype(np.float64)-previous)**2).mean(axis=(1,2,3))
                        np.testing.assert_allclose(orig_mse, a['latent_mse_original'], rtol=1e-10)
                        np.testing.assert_allclose(prev_mse, a['latent_mse_previous'], rtol=1e-10)
                        rows.append(dict(reference=reference, arm=arm, round=r,
                            latent_mse_original=float(orig_mse.mean()),
                            latent_mse_previous=float(prev_mse.mean()),
                            psnr_original=float(np.mean(a['psnr_original'])),
                            calls=a['full_calls_per_image']))
                    previous = z.copy()
        check = json.loads((folder/'solver_check.json').read_text())
        checks[reference] = {k: float(np.mean(v)) for k,v in check.items()}
    result = dict(rows=rows, first_four_solver_check=checks,
        request_sha256=hashlib.sha256((ROOT/'request.json').read_bytes()).hexdigest(),
        verified_outputs=n*expected_rows*len(request['references']), round00_verified_against_frozen_real_inputs=True,
        all_roundwise_latent_metrics_recomputed=True, quality_benefit_established=False)
    (ROOT/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    fig, axes = plt.subplots(1, len(request['references']), figsize=(5*len(request['references']),3.8), sharey=True, squeeze=False)
    for ax, ref in zip(axes[0], request['references']):
        for arm in request['arms']:
            selected = [r for r in rows if r['reference'] == ref and r['arm'] == arm]
            ax.plot([r['round'] for r in selected], [r['latent_mse_original'] for r in selected], 'o-', label=arm)
        ax.set(title=ref, xlabel='Recursive copy round', ylabel='Latent MSE to original', xticks=range(1,rounds+1))
        ax.set_yscale('log')
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle(f'{n} real-image latents; fixed reference inverse; no fresh noise')
    fig.tight_layout()
    fig.savefig(ROOT/'copy_drift.png', dpi=180)
    fig.savefig(ROOT/'copy_drift.pdf')
    for r in rows:
        if r['round'] in (1,rounds):
            print(json.dumps(r))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT)
    ROOT = parser.parse_args().out
    main()
