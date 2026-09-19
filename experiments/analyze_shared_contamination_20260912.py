"""Check the user's mixture derivation and re-read existing geometry; no GPU runs.

The Gaussian examples check algebra, not the empirical premise or image quality.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments import report_replica_reference_20260912 as workbook_writer

WORK = Path(__file__).resolve().parents[1]
OUT = WORK / 'docs/data/shared_contamination_20260912'


def grad(y, x):
    return torch.autograd.grad(y.sum(), x, create_graph=True, retain_graph=True)[0]


def div(j, x):
    return sum(grad(j[:, k], x)[:, k] for k in range(x.shape[1]))


def normal(x, mean):
    return torch.exp(-.5 * (x - mean).square().sum(1)) / (2 * np.pi)


def rotate(v):
    return torch.stack((-v[:, 1], v[:, 0]), 1)


def check_algebra():
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(2026091207)
    x = torch.tensor(rng.uniform(-1.5, 1.5, (256, 2)), requires_grad=True)
    t = torch.full((len(x), 1), .37, requires_grad=True)
    zero = torch.zeros_like(t)
    pg = normal(x, torch.cat((t, zero), 1))
    pe = normal(x, torch.cat((zero, t), 1))
    a, b = .2, .6
    kappa = a / b
    ps = (1-a)*pg+a*pe
    pw = (1-b)*pg+b*pe
    sg, se, ss, sw = [grad(p.log(), x) for p in (pg, pe, ps, pw)]
    ratio = ps/pw
    gamma = kappa/(ratio-kappa)
    srec = ss+gamma[:, None]*(ss-sw)
    eta_s, eta_w = a*pe/ps, b*pe/pw
    errors = {
        'density_recovery_max_abs': ((ps-kappa*pw)/(1-kappa)-pg).abs().max().item(),
        'score_recovery_max_abs': (srec-sg).abs().max().item(),
        'responsibility_gap_max_abs': (ss-sw-(eta_w-eta_s)[:, None]*(sg-se)).abs().max().item(),
    }
    # Actual probability currents may have divergence-free additions. They need
    # not equal the canonical score-derived fields of their own marginals.
    vg = torch.tensor([1., 0.]).expand_as(x)
    ve = torch.tensor([0., 1.]).expand_as(x)
    vs = ((1-a)*pg[:, None]*vg+a*pe[:, None]*ve)/ps[:, None] + .7*rotate(ss)
    vw = ((1-b)*pg[:, None]*vg+b*pe[:, None]*ve)/pw[:, None] - .3*rotate(sw)
    vrec = vs+gamma[:, None]*(vs-vw)
    for name, p, v in [('strong', ps, vs), ('weak', pw, vw), ('recovered', pg, vrec)]:
        errors[name+'_continuity_max_abs'] = (grad(p, t)[:, 0]+div(p[:, None]*v, x)).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(vg-vs, vs-vw, dim=1)
    ratio_input_coefficient = ((ss-sw)*(vs-vw)).sum(1)
    # A variable mixture weight adds a source term to the continuity equation.
    kt = .15+.1*t[:, 0]
    qt = (ps-kt*pw)/(1-kt)
    vt = (ps[:, None]*vs-kt[:, None]*pw[:, None]*vw)/(ps-kt*pw)[:, None]
    residual = grad(qt, t)[:, 0]+div(qt[:, None]*vt, x)
    expected = .1*(ps-pw)/(1-kt).square()
    errors['time_weight_source_identity_max_abs'] = (residual-expected).abs().max().item()
    # Spatially varying kappa also invalidates the naive score formula.
    kx = .1+.05*torch.sigmoid(x[:, 0])
    qx = (ps-kx*pw)/(1-kx)
    naive = ss+(kx/(ratio-kx))[:, None]*(ss-sw)
    extra = ((ratio-1)/((1-kx)*(ratio-kx)))[:, None]*grad(kx, x)
    errors['space_weight_score_identity_max_abs'] = (grad(qx.log(), x)-naive-extra).abs().max().item()
    # Shared-error misspecification: small density perturbations must be bounded
    # relatively, together with their derivatives, to control score error.
    pew = normal(x, torch.cat((zero+.02, t), 1))
    pwm = (1-b)*pg+b*pew
    qnom = (ps-kappa*pwm)/(1-kappa)
    eps = a/(1-kappa)*(pew-pe)/pg
    assert qnom.min()>0
    errors['mismatch_density_identity_max_abs'] = (qnom-pg*(1-eps)).abs().max().item()
    errors['mismatch_score_identity_max_abs'] = (sg-grad(qnom.log(), x)-grad(eps, x)/(1-eps)[:, None]).abs().max().item()
    assert max(errors.values())<1e-11, errors
    return dict(passed=True,checks=errors,points=len(x),dtype='float64',device='cpu',
        constant_weight_flow_example=dict(common_initial_density='N(0,I)',a=a,b=b,
            noncanonical_velocity_error_cosine_mean=cos.mean().item(),
            noncanonical_velocity_error_cosine_min=cos.min().item(),
            noncanonical_velocity_error_cosine_max=cos.max().item(),
            mean_orthogonal_fraction=(1-cos.square()).mean().item(),
            negative_logratio_input_coefficient_fraction=(ratio_input_coefficient<0).double().mean().item(),
            meaning='Exact density recovery without matching the canonical target velocity.'),
        time_varying_weight_source_max_abs=expected.abs().max().item(),
        limitations='Constructed identity checks only; no model-density identification, sampler convergence or FID claim.')


def historical_geometry():
    source = WORK/'docs/data/imagenet100_sit_depth_difference_mechanism_v1/depth_difference_geometry_per_sample.csv'
    with source.open() as f:
        rows = list(csv.DictReader(f))
    groups = []
    for context in sorted({r['context'] for r in rows}):
        for time in ['all']+sorted({r['time'] for r in rows}, key=float):
            group = [r for r in rows if r['context']==context and (time=='all' or r['time']==time)]
            f = np.array([float(r['reference_energy']) for r in group])
            w = np.array([float(r['difference_energy']) for r in group])
            dot = np.array([float(r['dot']) for r in group])
            cosine = dot/np.sqrt(f*w)
            parallel = np.square(cosine)
            saved = np.array([float(r['parallel_energy_fraction']) for r in group])
            # The historical columns were reduced separately in FP32; their
            # identities agree to rounding accuracy, not to FP64 precision.
            np.testing.assert_allclose(parallel,saved,atol=1e-7,rtol=2e-7)
            groups.append(dict(context=context,time=time,rows=len(group),
                cosine_mean=float(cosine.mean()),cosine_min=float(cosine.min()),cosine_max=float(cosine.max()),
                mean_pointwise_parallel_fraction=float(parallel.mean()),
                mean_pointwise_orthogonal_fraction=float(1-parallel.mean()),
                pooled_pointwise_residual_fraction=float((w-dot*dot/f).sum()/w.sum()),
                positive_projection_fraction=float((dot>0).mean()),
                max_saved_fraction_rounding_error=float(np.abs(parallel-saved).max())))
    return groups, dict(file=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        rows=len(rows),raw_vectors_requeried=False,
        limitation='Three network readouts; not direct access to true scores or actual density mixtures.')


def illustration():
    a,b=.2,.6
    def n(x,m,sd):return np.exp(-.5*((x-m)/sd)**2)/(np.sqrt(2*np.pi)*sd)
    x=np.linspace(-4.5,4.5,501)
    good=.5*n(x,-1.8,.55)+.5*n(x,1.8,.55)
    error=n(x,0,1.)
    strong=(1-a)*good+a*error
    weak=(1-b)*good+b*error
    density=[dict(x=float(z),target=float(g),error=float(e),strong=float(s),weak=float(w))
             for z,g,e,s,w in zip(x,good,error,strong,weak)]
    k=a/b
    rs=np.linspace(k+1e-4,(1-a)/(1-b),501)
    gains=k/(rs-k)
    gain=[dict(ratio=float(r),exact_extra_gain=float(g),noise_end_gain=a/(b-a)) for r,g in zip(rs,gains)]
    fig,axes=plt.subplots(1,2,figsize=(10,4.4))
    axes[0].plot(x,good,label='Target',color='#215c43',lw=2)
    axes[0].plot(x,strong,label='Strong: 20% error',color='#275c83')
    axes[0].plot(x,weak,label='Weak: 60% error',color='#b45032')
    axes[0].plot(x,error,label='Shared error component',color='.5',ls='--')
    axes[0].set(xlabel='State x',ylabel='Density',title='Assumed shared contamination')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower left',bbox_to_anchor=(.055,.005),ncol=2,frameon=False,fontsize=8)
    axes[1].plot(rs,gains,color='#275c83',label='Exact extra gain')
    axes[1].axhline(a/(b-a),color='.4',ls='--',label='Gain at equal density ratio')
    axes[1].axvline(k,color='#b45032',ls=':',label='Positivity boundary')
    axes[1].set(xlabel='Strong / weak density ratio',ylabel='Extra guidance gain (log scale)',
                title='The inverse becomes singular near the boundary',yscale='log',ylim=(.15,4000))
    axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Analytic illustration only: no fitted neural distributions or image-quality evidence',fontsize=10)
    fig.tight_layout(rect=(0,.13,1,.94))
    for ext in ('png','pdf','svg'):fig.savefig(OUT/f'mixture_and_gain.{ext}',dpi=170)
    plt.close(fig)
    return density,gain


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    checks=check_algebra()
    history,provenance=historical_geometry()
    density,gain=illustration()
    tables={'density_illustration':density,'gain_illustration':gain,'historical_geometry':history}
    for name,rows in tables.items():
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    workbook_writer.OUT=OUT
    workbook_writer.workbook(tables)
    (OUT/'algebra_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    (OUT/'historical_source.json').write_text(json.dumps(provenance,indent=2)+'\n')
    manifest={str(p.relative_to(WORK)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='artifact_manifest.json'}
    manifest[str(Path(__file__).relative_to(WORK))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(checks=checks,historical_overall=[r for r in history if r['time']=='all']),indent=2))


if __name__=='__main__':main()
