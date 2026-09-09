"""Paired condition-erasure contrasts with class-cluster standard errors."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def statistics(v, labels):
    v=np.asarray(v,dtype=np.float64);groups=np.unique(labels);mu=float(v.mean())
    influences=np.array([(v[labels==g]-mu).sum() for g in groups])
    se=float(np.sqrt(len(groups)/(len(groups)-1)*(influences**2).sum())/len(v))
    return {'mean':mu,'class_cluster_se':se,'samples':len(v),'classes':len(groups)}


def analyze(folder,family):
    a=np.load(folder/'ig/inputs.npz');b=np.load(folder/'pfr/inputs.npz')
    assert a.files==b.files
    assert all(np.array_equal(a[k],b[k]) for k in a.files)
    labels=a['global_labels'] if 'global_labels' in a else a['labels']
    scores={mode:np.load(folder/mode/'scores.npz') for mode in ['ig','pfr']}
    for mode in scores:
        meta=json.loads((folder/mode/'summary.json').read_text());assert meta['complete']
        assert np.array_equal(scores[mode]['labels'],labels)
    rows=[]
    for metric in ['target_prob','top1','top5']:
        v={(mode,suffix):scores[mode][suffix+'_'+metric].astype(float)
           for mode in scores for suffix in ['conditional','unconditional']}
        contrasts={mode+'_erasure':v[mode,'unconditional']-v[mode,'conditional'] for mode in scores}
        contrasts.update({suffix+'_pfr_minus_ig':v['pfr',suffix]-v['ig',suffix]
                          for suffix in ['conditional','unconditional']})
        contrasts['erasure_interaction']=contrasts['pfr_erasure']-contrasts['ig_erasure']
        for name,values in contrasts.items():
            rows.append({'family':family,'metric':metric,'contrast':name,**statistics(values,labels)})
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raev2-folder',type=Path,required=True);p.add_argument('--sit-folder',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=analyze(a.raev2_folder,'raev2')+analyze(a.sit_folder,'sit')
    with a.output.open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps([r for r in rows if r['metric']=='target_prob'],indent=2))
