"""Recompute raw-RGB finite extrapolation Grams without clipping/quantization."""
import json
from pathlib import Path
import numpy as np
from experiments.raev2_training_core import file_sha256
def main():
    root=Path('/home/zhoushunyu/data/eqvae/experiments/ig_head_correspondence_20260908/commutator')
    data=json.loads((root/'results.json').read_text());assert data['complete']
    groups={}
    for row in data['rows']:
        p=root/f"id{row['id']:04d}_step{row['step']:03d}.npz";assert file_sha256(p)==data['files'][p.name]
        v=np.load(p)['raw_rgb'].reshape(3,-1).astype(float);a=v[2]-v[1];b=.78*(v[1]-v[0]);m=np.stack([a,b,a-b])
        gram=m@m.T/m.shape[1];np.testing.assert_allclose(gram,row['gram'],atol=1e-12,rtol=1e-12)
        groups.setdefault(row['step'],[]).append(gram)
    rows=[]
    for k,gs in groups.items():
        g=np.mean(gs,axis=0);rows.append(dict(step=k,actual_energy=g[0,0],linear_energy=g[1,1],
                                            residual_to_actual_energy=g[2,2]/g[0,0],
                                            actual_linear_cosine=g[0,1]/np.sqrt(g[0,0]*g[1,1])))
    result=dict(complete=True,rows=rows,seconds=data['seconds'],decoder_batch_calls=32,
                source_result_sha256=file_sha256(root/'results.json'),scope='32 fixed native readouts, finite gamma=.78; no terminal quality or global decoder claim')
    p=Path(__file__).resolve().parents[1]/'experiments/results/terminal_defect_20260908/raev2_decoder_guidance_commutator.json'
    p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
