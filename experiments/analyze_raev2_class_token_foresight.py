"""Independent CPU MSE identities from saved residual/correction vectors."""
import json
from pathlib import Path
import numpy as np
from experiments.raev2_training_core import file_sha256


def main():
    root = Path(__file__).resolve().parents[1]
    run = Path('/home/zhoushunyu/data/eqvae/experiments/fsg_class_token_foresight_20260908')
    result = json.loads((run/'result.json').read_text())
    assert result['complete'] and len(result['rows']) == 128
    for source, expected in result['sources'].items():
        assert file_sha256(Path(source)) == expected
    rows, hashes = [], {}
    for row in result['rows']:
        path = run/f"id{row['index']:04d}_event{row['event']}.npz"
        v = np.load(path)['vectors'].astype(np.float64)
        assert v.shape == (2,262144) and np.isfinite(v).all()
        g = v@v.T/v.shape[1]
        np.testing.assert_allclose(g,row['gram'],rtol=1e-10,atol=1e-12)
        direct = np.mean((v[0]+v[1])**2)
        np.testing.assert_allclose(direct,g[0,0]+2*g[0,1]+g[1,1],rtol=1e-12)
        hashes[str(path)] = file_sha256(path)
        rows.append(dict(event=row['event'], base=float(g[0,0]), corrected=float(direct),
                         correction=float(g[1,1]), linear=float(2*g[0,1])))
    summaries=[]
    for event in range(4):
        selected=[r for r in rows if r['event']==event]
        means={k:float(np.mean([r[k] for r in selected])) for k in ['base','corrected','correction','linear']}
        means.update(event=event, relative_change=means['corrected']/means['base']-1,
                     improved_fraction=float(np.mean([r['corrected']<r['base'] for r in selected])))
        summaries.append(means)
    result.update(summary=summaries,vector_files_sha256=hashes,
                  audit='CPU recomputed all 128 Grams and actual summed residual MSE')
    (root/'experiments/results/terminal_defect_20260908/raev2_class_token_foresight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__ == '__main__':
    main()
