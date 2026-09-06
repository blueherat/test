"""Exact CPU target-identifiability audit; no RAE models or generated images."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import sympy as sp


ROOT = Path(__file__).resolve().parents[1]
OUT = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/temporal_score_pde_v1')
PROTOCOL = ROOT / 'docs/RAEV2_TEMPORAL_SCORE_PDE_PROTOCOL_20260906_ZH.md'


def identity(path: Path) -> dict:
    data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}


def write_json(path: Path, value: dict) -> None:
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def main() -> None:
    wall, cpu = time.perf_counter(), time.process_time()
    OUT.mkdir(parents=True, exist_ok=False)
    request = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'round': 3,
        'kind': 'analytic_target_identifiability_no_quality_experiment',
        'protocol': identity(PROTOCOL), 'script': identity(Path(__file__).resolve()),
        'python': sys.version, 'sympy': sp.__version__, 'platform': platform.platform(),
        'cases': [{'name': 'reference', 'mu': 0, 'variance': 1},
                  {'name': 'wrong_mean', 'mu': 1, 'variance': 1},
                  {'name': 'wrong_variance', 'mu': 0, 'variance': 4}],
        'times': ['1/100', '1/4', '1/2', '3/4', '99/100'],
        'positions': [-2, -1, 0, 1, 2],
        'case_selection': 'analytic derivation before execution; not blinded',
        'gpu_model_decoder_training_and_real_fid_calls': 0,
    }
    write_json(OUT / 'request.json', request)
    t = sp.symbols('t', positive=True)
    x, mu, z1 = sp.symbols('x mu z1', real=True)
    v = sp.symbols('v', positive=True)
    a, V, m = 1-t, (1-t)**2*v+t**2, (1-t)*mu
    score = -(x-m)/V
    f, diffusion_half = -x/a, t/a

    def residual(s):
        potential = diffusion_half*(sp.diff(s, x)+s*s)-f*s-sp.diff(f, x)
        return sp.factor(sp.diff(s, t)-sp.diff(potential, x))

    score_residual = residual(score)
    assert score_residual == 0
    drift = sp.factor(f-diffusion_half*score)
    direct_drift = (t-a*v)*x/V-t*mu/V
    assert sp.simplify(drift-direct_drift) == 0
    exact_path = m+sp.sqrt(V)*z1
    path_residual = sp.simplify(sp.diff(exact_path,t)-drift.subs(x,exact_path))
    assert path_residual == 0
    endpoint_score = sp.simplify(score.subs(t,1))
    assert endpoint_score == -x
    e = t*(1-t)*(1-2*t)
    assert sp.integrate(e,(t,0,1)) == 0
    s_reference = score.subs({mu:0,v:1})
    s_cancellation = s_reference-a*e*x/t
    cancellation_residual = residual(s_cancellation)
    assert cancellation_residual != 0
    assert sp.simplify(s_cancellation.subs(t,1)) == -x
    recovered_delta_drift = sp.simplify((f-diffusion_half*s_cancellation)-drift.subs({mu:0,v:1}))
    assert sp.simplify(recovered_delta_drift-e*x) == 0
    # The linear ODE's centered variance multiplier is exp(2 integral b dt).
    cancellation_endpoint_multiplier = sp.exp(2*sp.integrate(e,(t,1,0)))
    assert cancellation_endpoint_multiplier == 1
    rows, cases = [], []
    for case in request['cases']:
        sub = {mu:case['mu'],v:case['variance']}
        scase = score.subs(sub)
        rcase = residual(scase)
        assert rcase == 0
        fid = case['mu']**2+(sp.sqrt(case['variance'])-1)**2
        cases.append({**case,'score':str(scase),'symbolic_pde_residual':str(rcase),
                      'noise_endpoint_score':str(scase.subs(t,1)),
                      'clean_endpoint_map':str(exact_path.subs(sub).subs(t,0)),
                      'identity_feature_gaussian_fid_exact':str(fid),
                      'identity_feature_gaussian_fid_float':float(fid)})
        for ts in request['times']:
            for xs in request['positions']:
                # Substitute before differentiating only the Gaussian parameters;
                # the full x,t derivatives remain exact in rcase.
                value = rcase.subs({t:sp.Rational(ts),x:xs})
                assert value == 0
                rows.append({'case':case['name'],'t':ts,'x':xs,
                             'score_exact':str(scase.subs({t:sp.Rational(ts),x:xs})),
                             'pde_residual_exact':str(value)})
    with (OUT/'grid.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    witness = sp.simplify(cancellation_residual.subs({t:sp.Rational(1,2),x:1}))
    assert witness != 0
    summary = {
        'completed_utc':datetime.now(timezone.utc).isoformat(),
        'request':identity(OUT/'request.json'),'grid':identity(OUT/'grid.csv'),
        'general_gaussian_score_pde_residual':str(score_residual),
        'general_analytic_path_ode_residual':str(path_residual),
        'all_cases_share_exact_gaussian_noise_boundary':True,
        'grid_points':len(rows),'cases':cases,
        'cancellation':{'added_drift_coefficient':str(e),
                        'integral_0_to_1':str(sp.integrate(e,(t,0,1))),
                        'score_residual':str(cancellation_residual),
                        'score_residual_at_t_half_x_one':str(witness),
                        'endpoint_variance_multiplier':str(cancellation_endpoint_multiplier),
                        'identity_feature_gaussian_fid_exact':'0'},
        'decision':'Reject target identification from zero score PDE residual and t=1 Gaussian boundary alone; no new training or RAE sampling.',
        'scope':'Exact scalar Gaussian counterexample, not ImageNet FID, not a refutation of data-anchored PDE regularization.',
        'gpu_model_decoder_training_and_real_fid_calls':0,
        'script_wall_seconds':time.perf_counter()-wall,
        'script_cpu_seconds':time.process_time()-cpu,
    }
    write_json(OUT/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
