"""Independent saved-vector audit; no quality claims from head-space geometry."""
import json
from pathlib import Path
import numpy as np
from experiments.raev2_training_core import file_sha256

def main():
    root=Path('/home/zhoushunyu/data/eqvae/experiments/ig_guidance_history_20260908')
    data=json.loads((root/'results.json').read_text());assert data['complete'] and data['calls']==832
    for path,digest in data['request']['sources'].items():assert file_sha256(Path(path))==digest
    groups={};max_error=0.
    for row in data['rows']:
        p=root/f"id{row['id']:04d}_step{row['step']:03d}.npz"
        assert file_sha256(p)==data['files'][p.name]
        raw=np.load(p)['vectors'];v=raw.reshape(5,-1).astype(float)
        g=v@v.T/v.shape[1]
        np.testing.assert_allclose(g,row['gram'],atol=1e-12,rtol=1e-12)
        err=float(np.max(np.abs(v[0]-v[1]-v[2]+v[3])))
        max_error=max(max_error,err);assert err<2e-6
        if row['step']==0:assert np.count_nonzero(raw[2:])==0
        groups.setdefault(row['step'],[]).append(g)
    result=[]
    def cosine(g,a,b):
        return float((a@g@b)/max(np.sqrt((a@g@a)*(b@g@b)),1e-30))
    for k,gs in groups.items():
        g=np.mean(gs,axis=0);e=np.eye(5);h=e[2]-e[3]
        result.append(dict(step=k,n=len(gs),gap_ig_energy=g[0,0],gap_full_energy=g[1,1],
                           weak_history_to_gap_energy=g[3,3]/g[0,0],
                           strong_history_to_gap_energy=g[2,2]/g[0,0],
                           head_history_interaction_to_gap_energy=float(h@g@h/g[0,0]),
                           history_head_cosine=cosine(g,e[2],e[3]),
                           gap_history_cosine=cosine(g,e[0],e[1]),
                           cross_state_gap_to_native_cosine=cosine(g,e[0]+e[3],e[0])))
    out=dict(complete=True,rows=result,max_identity_abs_error=max_error,calls=data['calls'],seconds=data['seconds'],
             source_result_sha256=file_sha256(root/'results.json'),scope='8 paired trajectories, descriptive; no images/FID; no causal quality conclusion')
    dest=Path(__file__).resolve().parents[1]/'experiments/results/terminal_defect_20260908/raev2_guidance_history.json'
    dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
