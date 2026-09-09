"""Recompute token-subset risks against actual clean targets, CPU only."""
import json
from pathlib import Path
import numpy as np
from experiments.raev2_training_core import file_sha256


def main():
    root=Path(__file__).resolve().parents[1]
    run=Path('/home/zhoushunyu/data/eqvae/experiments/fsg_carrier_real_risk_20260908')
    r=json.loads((run/'result.json').read_text())
    old=json.loads((root/'experiments/results/terminal_defect_20260908/raev2_class_token_foresight.json').read_text())
    assert r['complete'] and len(r['rows'])==128
    for source,expected in r['sources'].items():
        assert file_sha256(Path(source))==expected
    matrices,hashes={},{}
    for row in r['rows']:
        path=run/f"id{row['index']:04d}_event{row['event']}.npz"
        v=np.load(path)['vectors'].astype(np.float64)
        assert v.shape==(5,262144) and np.isfinite(v).all()
        g=v@v.T/v.shape[1]
        np.testing.assert_allclose(g,row['gram'],rtol=1e-10,atol=1e-12)
        prev=next(p for p in old['rows'] if (p['index'],p['event'])==(row['index'],row['event']))
        for key in ['noise_sha256','clean_sha256','state_sha256']:
            assert prev[key]==row[key]
        oldpath=Path('/home/zhoushunyu/data/eqvae/experiments/fsg_class_token_foresight_20260908')/path.name
        np.testing.assert_array_equal(v[1],np.load(oldpath)['vectors'][0])
        matrices.setdefault(row['event'],[]).append(g)
        hashes[str(path)]=file_sha256(path)
    summary=[]
    for event,gg in matrices.items():
        g=np.mean(gg,axis=0)
        vals={name:float(g[j,j]) for j,name in enumerate(r['rows'][0]['names'])}
        vals.update(event=event, class_gain_fraction=(vals['null']-vals['class_only'])/(vals['null']-vals['conditional']),
                    image_gain_fraction=(vals['null']-vals['image_only'])/(vals['null']-vals['conditional']),
                    class_beats_full_fraction=float(np.mean([x[2,2]<x[1,1] for x in gg])),
                    image_beats_full_fraction=float(np.mean([x[3,3]<x[1,1] for x in gg])))
        summary.append(vals)
    r.update(summary=summary,vector_files_sha256=hashes,audit='all128 Grams CPU recomputed; inputs and native residual match preceding real-data probe bitwise')
    (root/'experiments/results/terminal_defect_20260908/raev2_carrier_real_risk.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
