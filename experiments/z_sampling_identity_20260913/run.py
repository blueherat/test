"""Independent Z-Sampling screen using the existing paired evaluation runner."""
from pathlib import Path
import csv
import json
from experiments.cfg_transport_search_20260913 import runner

ROOT=Path('/home/zhoushunyu/data/eqvae/experiments/z_sampling_identity_20260913')


def report(root):
    request=runner.c.read(root/'request.json');rows=[]
    for config in request['configs']:
        out=root/config['arm']
        if not (out/'summary.json').exists(): continue
        summary=runner.c.read(out/'summary.json')
        row={**config,**{key:summary[key] for key in ('samples','seconds',
            'full_calls_per_output','prefix_calls_per_output','saturation_fraction','latent_rms')}}
        if (out/'fid.json').exists():
            metrics=runner.c.read(out/'fid.json')
            assert metrics['sample_count']==summary['samples']
            row.update({key:metrics[key] for key in ('fid','sfid','inception_score')})
        rows.append(row)
    columns=['arm','kind','alpha','backward_w','forward_w','steps','samples','fid','sfid',
             'inception_score','seconds','full_calls_per_output','prefix_calls_per_output',
             'saturation_fraction','latent_rms']
    with (root/'results.tmp').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,columns,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    (root/'results.tmp').replace(root/'results.csv')
    runner.c.atomic(root/'status.json',dict(completed_sampling=len(rows),
        completed_fid=sum('fid' in r for r in rows),planned=len(request['configs']),
        samples_per_arm=request['samples'],timing='seconds includes sampler and VAE decode'))


if __name__=='__main__':
    check=json.loads((ROOT/'model_check/summary.json').read_text())
    assert check['passed']
    for path,digest in check['source_sha256'].items():
        assert runner.c.sha(path)==digest,('Model check source changed',path)
    runner.ROOT=ROOT
    runner.MODULE='experiments.z_sampling_identity_20260913.run'
    runner.report=report
    runner.main()
