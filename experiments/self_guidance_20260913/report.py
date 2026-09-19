"""Archive completed, paired SG transfer results and unselected image examples."""
from pathlib import Path
import argparse
import csv
import json
import shutil
import numpy as np
from PIL import Image, ImageDraw
from experiments.cfg_transport_search_20260913 import runner


def main(phase):
    root = runner.c.EXPS/'self_guidance_20260913'/phase
    request = runner.verify(root)
    completion = runner.c.read(root/'controller_complete.json')
    assert completion['complete'] and all(code == 0 for code in completion['exit_codes'])
    evidence = runner.c.WORK/'docs/data/self_guidance_20260913'/phase
    evidence.mkdir(parents=True, exist_ok=True)
    with np.load(root/'inputs.npz') as bank:
        labels = bank['labels'].copy()
        assert np.array_equal(np.bincount(labels, minlength=100), np.full(100, request['samples']//100))
    rows, receipts = [], []
    canvas = Image.new('RGB', (190+6*144, 30+len(request['configs'])*165), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f'{phase}: first 6 fixed sample IDs, identical noise and class across rows', fill='black')
    for index, config in enumerate(request['configs']):
        out = root/config['arm']
        summary = runner.c.read(out/'summary.json')
        metric = runner.c.read(out/'fid.json')
        assert summary['complete'] and metric['sample_count'] == request['samples'] == summary['samples']
        assert summary['request_sha256'] == runner.c.sha(root/'request.json')
        assert summary['samples_sha256'] == runner.c.sha(out/'samples.npz')
        assert np.isfinite([metric[k] for k in ('fid', 'sfid', 'inception_score')]).all()
        for record in summary['records']:
            assert runner.c.sha(record['path']) == record['sha256']
        with np.load(out/'endpoints.npz') as z:
            assert np.array_equal(z['labels'], labels) and np.isfinite(z['latents']).all()
        with np.load(out/'inception_activations.npz') as a:
            assert len(a['pool_3']) == request['samples'] and np.isfinite(a['pool_3']).all()
            assert len(a['spatial']) == request['samples'] and np.isfinite(a['spatial']).all()
        row = {**config, **{k:metric[k] for k in ('fid', 'sfid', 'inception_score')},
               **{k:summary[k] for k in ('samples', 'seconds', 'full_calls_per_output')}}
        rows.append(row)
        top = 30+index*165
        draw.text((6, top+20), config['arm'], fill='black')
        draw.text((6, top+40), f"FID {row['fid']:.3f} | NFE {row['full_calls_per_output']}", fill='black')
        with np.load(out/'batch000000.npz') as batch:
            for j, pixels in enumerate(batch['pixels'][:6]):
                x = 190+j*144
                canvas.paste(Image.fromarray(pixels).resize((140,140)), (x,top))
                draw.text((x,top+142), f'ID {j}, class {labels[j]}', fill='black')
        receipts.append(dict(arm=config['arm'], metrics_sha256=runner.c.sha(out/'fid.json'),
                             summary_sha256=runner.c.sha(out/'summary.json'),
                             activations_sha256=runner.c.sha(out/'inception_activations.npz')))
    baseline = {row['arm']:row for row in rows}
    for row in rows:
        name = 'cfg_e64' if row['alpha'] else 'strong_e64'
        if name in baseline:
            base = baseline[name]
            row['baseline'] = name
            row['delta_fid'] = row['fid']-base['fid']
            row['relative_fid_percent'] = 100*row['delta_fid']/base['fid']
    canvas.save(evidence/'comparison.png')
    runner.c.atomic(evidence/'results.json', dict(rows=rows, root=str(root),
        request_sha256=runner.c.sha(root/'request.json'), receipts=receipts,
        audit='source/asset/input/sample/batch hashes, balanced labels, finite endpoints/features, 1000 outputs per arm'))
    shutil.copy2(root/'request.json', evidence/'request.json')
    columns = ['arm','kind','alpha','omega','solver','steps','samples','fid','sfid','inception_score',
               'baseline','delta_fid','relative_fid_percent','full_calls_per_output','seconds']
    with (evidence/'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, columns, extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    table = ['|配置|FID ↓|相对同源Euler64 ΔFID|sFID ↓|IS ↑|前向/图|采样+解码GPU秒|',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        delta = f"{r['delta_fid']:+.3f}" if 'delta_fid' in r else '—'
        table.append(f"|`{r['arm']}`|{r['fid']:.3f}|{delta}|{r['sfid']:.3f}|{r['inception_score']:.3f}|{r['full_calls_per_output']}|{r['seconds']:.1f}|")
    (evidence/'table.md').write_text('\n'.join(table)+'\n')
    print('\n'.join(table))
    print('Evidence:', evidence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--phase', default='screen_1k')
    main(parser.parse_args().phase)
