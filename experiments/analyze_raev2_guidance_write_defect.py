import json
from pathlib import Path
import numpy as np
from experiments.raev2_training_core import file_sha256

def main():
    root=Path(__file__).resolve().parents[1]
    run=Path('/home/zhoushunyu/data/eqvae/experiments/guidance_write_defect_20260908')
    result=json.loads((run/'result.json').read_text());assert result['complete']
    for path,h in result['sources'].items():assert file_sha256(Path(path))==h
    old=json.loads((root/'experiments/results/terminal_defect_20260908/raev2_condition_handoff.json').read_text())
    assert result['noise_sha256']==old['noise_sha256']
    hashes={};summary=[]
    for kind in ['ig','cfg']:
        total=np.zeros((4,4))
        for start in range(0,64,4):
            path=run/f'{kind}_{start:03d}.npz';v=np.load(path)['vectors'].astype(float)
            assert v.shape==(4,4,262144) and np.isfinite(v).all()
            for j in range(4):
                g=v[j]@v[j].T/262144
                row=next(r for r in result['rows'] if r['kind']==kind and r['id']==start+j)
                np.testing.assert_allclose(g,row['gram'],rtol=1e-10,atol=1e-12)
                total+=g
            hashes[str(path)]=file_sha256(path)
        g=total/64
        summary.append(dict(kind=kind,raw_energy=float(g[0,0]),nuisance_energy=float(g[1,1]),
                            signal_energy=float(g[2,2]),nuisance_to_signal_energy=float(g[1,1]/g[2,2]),
                            signal_to_impulse_cosine=float(g[2,3]/np.sqrt(g[2,2]*g[3,3])),
                            raw_to_signal_cosine=float(g[0,2]/np.sqrt(g[0,0]*g[2,2]))))
    result.update(summary=summary,vector_files_sha256=hashes,checkpoint_sha256_audited_after_run=file_sha256(Path('/home/zhoushunyu/data/eqvae/models/RAEv2/stage2/imagenet/dinov3l-k7/checkpoint.pt')))
    (root/'experiments/results/terminal_defect_20260908/raev2_guidance_write_defect.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
