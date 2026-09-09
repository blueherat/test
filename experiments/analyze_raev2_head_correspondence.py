"""Exact assignment upper bound and held-channel correspondence check."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from experiments.raev2_training_core import file_sha256

def match(a,b):
    cost=np.square(a).sum(1)[:,None]+np.square(b).sum(1)[None,:]-2*a@b.T
    i,j=linear_sum_assignment(cost);np.testing.assert_array_equal(i,np.arange(len(a)))
    return j
def main():
    root=Path('/home/zhoushunyu/data/eqvae/experiments/ig_head_correspondence_20260908')
    data=json.loads((root/'results.json').read_text());assert data['complete'] and data['calls']==32
    for p,h in data['request']['sources'].items():assert file_sha256(Path(p))==h
    rows=[];xy=np.stack(np.unravel_index(np.arange(256),(16,16)),axis=1)
    for i in data['request']['ids']:
        for k in data['request']['steps']:
            p=root/f'id{i:04d}_step{k:03d}.npz';assert file_sha256(p)==data['files'][p.name]
            d=np.load(p);a=d['full'].reshape(1024,256).T.astype(float);b=d['base'].reshape(1024,256).T.astype(float)
            perm=match(a,b);fitperm=match(a[:,::2],b[:,::2])
            raw=float(np.square(a-b).mean());aligned=float(np.square(a-b[perm]).mean())
            test_raw=float(np.square(a[:,1::2]-b[:,1::2]).mean())
            test_new=float(np.square(a[:,1::2]-b[fitperm,1::2]).mean())
            assert aligned<=raw+1e-10
            if i==0 and k==0:
                np.testing.assert_array_equal(match(a,a),np.arange(256))
                rolled=np.roll(a,31,axis=0)
                np.testing.assert_array_equal(rolled[match(a,rolled)],a)
            rows.append(dict(id=i,step=k,raw=raw,aligned=aligned,removed_fraction=1-aligned/raw,
                             test_raw=test_raw,test_aligned=test_new,test_removed_fraction=1-test_new/test_raw,
                             moved_fraction=float(np.mean(perm!=np.arange(256))),
                             displacement=float(np.linalg.norm(xy-xy[perm],axis=1).mean())))
    grouped=[]
    for k in data['request']['steps']:
        rr=[r for r in rows if r['step']==k]
        grouped.append(dict(step=k,removed_fraction=1-sum(r['aligned'] for r in rr)/sum(r['raw'] for r in rr),
                            held_channel_removed_fraction=1-sum(r['test_aligned'] for r in rr)/sum(r['test_raw'] for r in rr),
                            moved_fraction=float(np.mean([r['moved_fraction'] for r in rr])),
                            displacement=float(np.mean([r['displacement'] for r in rr]))))
    result=dict(complete=True,rows=rows,grouped=grouped,calls=data['calls'],seconds=data['seconds'],
                source_result_sha256=file_sha256(root/'results.json'),controls='identity and known permutation exact',
                scope='permutation matching only, no semantic/quality claim; channel split is not independent data')
    out=Path(__file__).resolve().parents[1]/'experiments/results/terminal_defect_20260908/raev2_head_correspondence.json'
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(grouped,indent=2))
if __name__=='__main__':main()
