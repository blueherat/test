"""Recompute intervention Grams from saved Full-output differences on CPU."""
import json
from pathlib import Path
import numpy as np
from experiments.raev2_training_core import file_sha256

ROOT = Path(__file__).resolve().parents[1]
RUN = Path('/home/zhoushunyu/data/eqvae/experiments/fsg_class_token_persistence_20260908')


def main():
    result = json.loads((RUN / 'result.json').read_text())
    assert result['complete'] and len(result['rows']) == 32
    for source, expected in result['sources'].items():
        assert file_sha256(Path(source)) == expected
    summaries, hashes = [], {}
    for step in result['steps']:
        grams = []
        for row in result['rows']:
            if row['step'] != step:
                continue
            path = RUN / f"id{row['id']:04d}_step{step:03d}.npz"
            v = np.load(path)['vectors'].astype(np.float64)
            assert v.shape == (6, 262144) and np.isfinite(v).all()
            g = v @ v.T / v.shape[1]
            np.testing.assert_allclose(g, row['gram'], rtol=1e-10, atol=1e-12)
            grams.append(g)
            hashes[str(path)] = file_sha256(path)
        g = sum(grams)
        def compare(a, b):
            return dict(energy_ratio=float(g[a,a]/g[b,b]),
                        cosine=float(g[a,b]/np.sqrt(g[a,a]*g[b,b])),
                        target_projection=float(g[a,b]/g[b,b]),
                        relative_squared_error=float((g[a,a]+g[b,b]-2*g[a,b])/g[b,b]))
        summaries.append(dict(step=step, time=row['time'] if row['step']==step else
                              next(r['time'] for r in result['rows'] if r['step']==step),
                              fresh_class_to_gap=compare(1,0),
                              previous_raw_to_fresh_class=compare(2,1),
                              previous_contrast_to_fresh_class=compare(3,1),
                              previous_raw_to_gap=compare(2,0),
                              previous_contrast_to_gap=compare(3,0),
                              cached_clean_gap=compare(4,0), cached_velocity_gap=compare(5,0)))
    result.update(summary=summaries, vector_files_sha256=hashes,
                  audit='32 saved vector Grams independently recomputed on CPU; sources unchanged')
    out = ROOT / 'experiments/results/terminal_defect_20260908/raev2_class_token_persistence.json'
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
