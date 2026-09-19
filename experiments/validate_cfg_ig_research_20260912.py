"""Independent arithmetic and provenance checks of the research diagnostics."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'docs/data/cfg_ig_research_20260912'
RAW = Path('/home/zhoushunyu/data/eqvae/imagenet_sit_flow/ag_ig_smoothing_cpu_20260911')
CFG_RAW = RAW.parent / 'cfg_bayes_compatibility_cpu_20260912'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def main():
    hashes_checked = 0
    manifests = [json.loads((OUT/n).read_text()) for n in ['ig_nuisance_manifest.json', 'cfg_manifest.json']]
    for m in manifests:
        assert m['completed']
        assert digest(REPO/'experiments/audit_cfg_ig_research_20260912.py') == m['script_sha256']
    for path, expected in manifests[0]['sources'].items():
        assert digest(path) == expected
        hashes_checked += 1
    for name, expected in manifests[1]['observations'].items():
        assert digest(CFG_RAW/name) == expected
        hashes_checked += 1
    assert not manifests[1]['cuda_initialized']
    rows = read_csv(OUT/'ig_nuisance_summary.csv')
    detail = read_csv(OUT/'ig_nuisance_per_image.csv')
    assert len(rows) == 75 and len(detail) == 2400
    max_group_error = 0.
    max_normal_error = 0.
    for r in rows:
        part = [d for d in detail if (d['weak'], d['t'], d['model']) == (r['weak'], r['t'], r['model']) and d['split']=='test']
        assert len(part)==16
        ratio = sum(float(d['squared_residual']) for d in part)/sum(float(d['squared_gap']) for d in part)
        max_group_error = max(max_group_error, abs(ratio-float(r['test_residual_ratio'])))
        ti = [.15,.3,.5,.7,.9].index(float(r['t']))
        a = np.load(RAW/f'time_{ti}.npz')
        strong = a['strong'].astype(np.float64)
        y = a['z'].astype(np.float64)/float(r['t'])
        gap = a['weak_'+r['weak']].astype(np.float64)-strong
        basis = {'score_scale':[strong], 'radial':[y], 'score_scale_radial':[strong,y]}[r['model']]
        A = np.stack([b[:16].reshape(-1) for b in basis], axis=1)
        coeff = np.linalg.solve(A.T@A, A.T@gap[:16].reshape(-1))
        predicted = sum(c*b[16:] for c,b in zip(coeff,basis))
        ratio = np.sum((gap[16:]-predicted)**2)/np.sum(gap[16:]**2)
        max_normal_error = max(max_normal_error, abs(ratio-float(r['test_residual_ratio'])))
    assert max_group_error < 1e-12 and max_normal_error < 1e-10
    cfg_rows = read_csv(OUT/'cfg_bayes_geometry.csv')
    weights = read_csv(OUT/'cfg_hull_weights.csv')
    assert len(cfg_rows)==12 and len(weights)==1200
    max_affine_error=max_convex_error=max_state_difference=max_strong_rms=0.
    pooled=[]
    for r in cfg_rows:
        t=float(r['t']); pos=int(r['position']); label=int(r['label'])
        ti=[.15,.5,.9].index(t)
        raw=np.load(CFG_RAW/f'time_{ti}_position_{pos}.npz')
        v=raw['fields'].astype(np.float64).reshape(101,-1)
        differences=(v[1:100]-v[0]).T
        coeff=np.linalg.lstsq(differences, v[100]-v[0], rcond=1e-6)[0]
        residual=v[100]-v[0]-differences@coeff
        denom=np.sum((v[label]-v[100])**2)
        ratio=np.sum(residual**2)/denom
        max_affine_error=max(max_affine_error,abs(ratio-float(r['affine_gap_energy_ratio_1e-05'])))
        w=np.array([float(d['weight']) for d in weights if d['id']==r['id'] and d['t']==r['t']])
        assert len(w)==100 and w.min()>-1e-10 and abs(w.sum()-1)<1e-10
        ratio=np.sum((w@v[:100]-v[100])**2)/denom
        max_convex_error=max(max_convex_error,abs(ratio-float(r['hull_gap_energy_ratio'])))
        assert r['hull_solver_success']=='True' and float(r['hull_fw_gap'])<1e-6
        assert int(r['affine_rank_1e-06'])==int(r['affine_rank_1e-05'])==int(r['affine_rank_0.0001'])==99
        old=np.load(RAW/f'time_{[.15,.3,.5,.7,.9].index(t)}.npz')
        max_state_difference=max(max_state_difference,float(np.max(np.abs(old['z'][pos]-raw['z'][0]))))
        reconstructed=((1-t)*old['strong'][pos].astype(np.float64)/t+old['z'][pos])/t
        max_strong_rms=max(max_strong_rms,float(np.sqrt(np.mean((reconstructed.reshape(-1)-v[label])**2))))
    assert max_affine_error<1e-10 and max_convex_error<1e-10
    assert max_state_difference<1e-6 and max_strong_rms<2e-5
    for t in [.15,.5,.9]:
        group=[r for r in cfg_rows if float(r['t'])==t]
        den=np.array([float(r['target_gap_rms'])**2 for r in group])
        ae=np.array([float(r['affine_gap_energy_ratio_1e-05']) for r in group])
        ce=np.array([float(r['hull_gap_energy_ratio']) for r in group])
        pooled.append(dict(t=t, states=4, affine_energy_ratio=float(ae@den/den.sum()),
            convex_energy_ratio=float(ce@den/den.sum()), affine_min=float(ae.min()), affine_max=float(ae.max())))
    with (OUT/'cfg_pooled_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(pooled[0]));writer.writeheader();writer.writerows(pooled)
    source_manifest=json.loads((REPO/'readings/cfg_ig_research_20260912/manifest.json').read_text())
    for source in source_manifest:
        assert digest(REPO/'readings/cfg_ig_research_20260912'/(source['key']+'.html'))==source['sha256']
        hashes_checked+=1
    result=dict(passed=True, source_hashes_checked=hashes_checked, nuisance_summary_rows=len(rows),
        nuisance_per_image_rows=len(detail), cfg_state_rows=len(cfg_rows), cfg_weight_rows=len(weights),
        max_group_ratio_error=max_group_error,max_independent_normal_equation_error=max_normal_error,
        max_independent_affine_error=max_affine_error,max_convex_reconstruction_error=max_convex_error,
        max_old_input_state_difference=max_state_difference,max_old_strong_field_rms_difference=max_strong_rms,
        max_hull_fw_gap=max(float(r['hull_fw_gap']) for r in cfg_rows),
        cuda_initialized=manifests[1]['cuda_initialized'])
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
