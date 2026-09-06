"""Collect every completed study; verify sample identities and paired inputs."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_20260907')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def main():
    baseline=json.loads((DATA/'ancestral_screen1k/metrics.json').read_text())[0]
    control=json.loads((DATA/'ancestral_screen1k/official/summary.json').read_text())
    rows=[]
    for folder in sorted(DATA.glob('*screen1k')):
        execution=folder/'execution.json'
        if not execution.exists() or not json.loads(execution.read_text())['complete']: continue
        for metric in json.loads((folder/'metrics.json').read_text()):
            summary=json.loads((folder/metric['branch']/'summary.json').read_text())
            if sha(Path(metric['sample_path']))!=metric['sample_sha256'] or summary['sample_sha256']!=metric['sample_sha256']:
                raise ValueError('sample identity mismatch')
            paired=summary['paired_noise_labels_sha256']==control['paired_noise_labels_sha256']
            if not paired: raise ValueError('unpaired screen')
            row={**metric,**summary,'study':folder.name,'execution_sha256':sha(execution),
                'relative_fid_improvement_percent':100*(1-metric['fid']/baseline['fid']),
                'inference_gpu_seconds':summary['trajectory_seconds_sum']+summary['decode_seconds_sum']}
            rows.append(row)
    out=ROOT/'experiments/results/raev2_guidance_20260907'
    out.mkdir(parents=True,exist_ok=True)
    (out/'screen_ledger.json').write_text(json.dumps({'protocol':str(ROOT/'docs/RAEV2_GUIDANCE_GOAL_20260907_ZH.md'),
        'baseline_fid':baseline['fid'],'target_fid':.97*baseline['fid'],
        'independent_seed_confirmed':False,'goal_achieved':False,'rows':rows},indent=2)+'\n')
    print('| 方法 | 1K FID | 相对改善 | GPU 秒 |\n|---|---:|---:|---:|')
    for r in rows:
        print(f"| {r['mode']} | {r['fid']:.6f} | {r['relative_fid_improvement_percent']:+.4f}% | {r['inference_gpu_seconds']:.1f} |")


if __name__=='__main__': main()
