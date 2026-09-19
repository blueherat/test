"""Algebra checks and source-backed reporting, separate from frozen hypothesis fit."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.guidance_pasted_20260912 import common as c
from . import mixture as m,structured_mismatch as sm

OUT=c.WORK/'docs/data/structured_mismatch_20260912'


def algebra():
    # Smooth positive periodic densities: unit checks, not evidence about images.
    x=np.arange(64)*2*np.pi/64;xx,yy=np.meshgrid(x,x,indexing='ij')
    p=1+.16*np.cos(xx)+.13*np.sin(yy)
    e=1+.22*np.cos(2*xx)+.17*np.sin(xx+yy)
    ds=.01*np.sin(2*xx-yy);dw=.013*np.cos(xx-2*yy)
    a,b,tau=.2,.6,.12;k=a/b
    freq=np.fft.fftfreq(64,d=1/64);u,v=np.meshgrid(freq,freq,indexing='ij')
    multiplier=np.exp(-.5*tau*(u*u+v*v))
    conv=lambda z:np.fft.ifft2(np.fft.fft2(z)*multiplier).real
    def grad(z):
        ft=np.fft.fft2(z)
        return np.stack([np.fft.ifft2(ft*1j*q).real for q in (u,v)])
    def score(z):return grad(z)/z
    s=(1-a)*p+a*e+ds;w=(1-b)*p+b*conv(e)+dw
    nominal=(s-k*w)/(1-k)
    delta=(a*(e-conv(e))+ds-k*dw)/(1-k)
    gamma=k*w/(s-k*w)
    # Differentiate the quotient analytically; FFT differentiation of log is unnecessary.
    h=delta/p;gradh=(grad(delta)*p-delta*grad(p))/(p*p)
    bias=gradh/(1+h)
    combined=score(s)+gamma*(score(s)-score(w))
    xi_s=np.stack([.07*np.cos(yy),.03*np.sin(xx)])
    xi_w=np.stack([.02*np.sin(xx+yy),-.05*np.cos(xx)])
    noisy=score(s)+xi_s+gamma*(score(s)+xi_s-score(w)-xi_w)
    theta=k*(1-b)/(1-a)
    left=conv(p)-theta*p
    right=(conv(s)-k*w-conv(ds)+k*dw)/(1-a)
    phi=lambda z:np.fft.fft2(z)/z.size
    observed=phi(w)-multiplier/k*phi(s)-((1-b)-(1/k-b)*multiplier)*phi(p)
    expected=phi(dw)-multiplier/k*phi(ds)
    errors=dict(density=float(np.max(np.abs(nominal-p-delta))),
        score=float(np.max(np.abs(combined-score(p)-bias))),
        additive_field=float(np.max(np.abs(noisy-score(p)-bias-xi_s+gamma*(xi_w-xi_s)))),
        eliminated_operator=float(np.max(np.abs(left-right))),
        characteristic_residual=float(np.max(np.abs(observed-expected))))
    assert min(s.min(),w.min(),nominal.min())>0
    assert max(errors.values())<1e-11,errors
    pole=np.sqrt(-2*np.log(theta)/tau)
    assert abs(np.exp(-.5*tau*pole*pole)-theta)<1e-14
    return dict(passed=True,max_absolute_errors=errors,finite_frequency_pole=pole,
        purpose='Numerical verification of identities only; no toy performance experiment.')


def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[];fits=[];bounds=[]
    for model in ('sit_small','raev2'):
        r=c.read(sm.ROOT/model/'results.json')
        assert r['source_sha256']==c.sha(Path(sm.__file__))
        assert r['protocol_sha256']==c.sha(sm.PROTOCOL)
        for row in r['rows']:
            rows.append({k:v for k,v in row.items() if k!='descriptive_2p5_97p5'}|
                dict(lower=row['descriptive_2p5_97p5'][0],upper=row['descriptive_2p5_97p5'][1],
                ratio=row['convolution_residual_mse']/row['strict_residual_mse']))
        for row in r['fits']:fits.append({k:v for k,v in row.items() if k!='optimization'})
        for track in ('ig','cfg'):
            r=c.read(m.ROOT/model/(track+'_contamination_profile')/'summary.json')
            mass=max(0.,-r['signed_event_mean']);k=r['kappa']
            bounds.append(dict(model=model,track=track,fixed_previous_kappa=k,
                observed_negative_mass=mass,
                effective_density_residual_tv_lower_bound=mass/(1-k),
                error_kernel_tv_lower_bound_if_no_additive=mass/k,
                negative_mass_descriptive_lower=max(0.,-r['signed_event_resampling_2p5_97p5'][1]),
                negative_mass_descriptive_upper=max(0.,-r['signed_event_resampling_2p5_97p5'][0])))
    tables=dict(heldout=pd.DataFrame(rows),parameters=pd.DataFrame(fits),residual_bounds=pd.DataFrame(bounds))
    for name,df in tables.items():df.to_csv(OUT/(name+'.csv'),index=False)
    tables['Sources']=pd.DataFrame([
        dict(title='Structured mismatch fixed protocol',source=str(sm.PROTOCOL),scope='New finite characteristic-function experiment'),
        dict(title='Raw source and fit artifacts',source=str(sm.ROOT),scope='Per-model results and raw-input hash manifests'),
        dict(title='Previous fixed inverse screens',source=str(m.ROOT),scope='Source-event empirical lower bounds; all 24 generation arms'),
        dict(title='Karras et al. Guiding a Diffusion Model with a Bad Version of Itself',source='https://arxiv.org/html/2406.02507v3',scope='AG motivation; not the structured-mismatch algebra'),
        dict(title='Koulischer et al. Feedback Guidance of Diffusion Models',source='https://arxiv.org/html/2506.06085v2',scope='Prior linear contamination inversion'),
        dict(title='Delaigle, Hall, Meister. On deconvolution with repeated measurements (2008)',source='https://arxiv.org/abs/0804.0713',scope='Independent repeated observations assumption; unknown-noise identification')])
    with pd.ExcelWriter(OUT/'source_data.xlsx',engine='openpyxl') as w:
        for name,df in tables.items():df.to_excel(w,sheet_name=name,index=False)
        for sh in w.book.worksheets:
            sh.freeze_panes='A2';sh.auto_filter.ref=sh.dimensions
            for col in sh.columns:sh.column_dimensions[col[0].column_letter].width=min(60,max(16,max(len(str(c.value or '')) for c in col)+2))
    q=tables['heldout'];fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    order=[('sit_small','ig'),('sit_small','cfg'),('raev2','ig'),('raev2','cfg')]
    names=['SiT IG','SiT CFG','RAEv2 IG','RAEv2 CFG']
    for i,group in enumerate(('same_directions','new_directions')):
        df=q[q.heldout==group].set_index(['model','track']).loc[order]
        axes[i].barh(np.arange(4),df.ratio,color='#737f88')
        axes[i].set_yticks(np.arange(4),names);axes[i].invert_yaxis()
        axes[i].axvline(1,color='black',ls='--',lw=1)
        axes[i].set_xlim(0,1.65);axes[i].set_title('Held-out samples, '+group.replace('_',' '))
        axes[i].set_xlabel('Convolution / strict characteristic-moment error')
        for j,r in enumerate(df.itertuples()):axes[i].text(1.59,j,f'{r.ratio:.3f}',va='center',ha='right')
    fig.suptitle('Error-only Gaussian fit: all four mixture proportions hit the upper bound',fontsize=12)
    for ext in ('png','pdf','svg'):fig.savefig(OUT/('heldout_comparison.'+ext),dpi=180)
    plt.close(fig)
    c.atomic(OUT/'algebra_checks.json',algebra())
    c.atomic(OUT/'artifact_manifest.json',dict(files={str(p.relative_to(c.WORK)):c.sha(p) for p in OUT.iterdir() if p.name!='artifact_manifest.json'},
        generator_sha256=c.sha(Path(__file__)),fit_source_sha256=c.sha(Path(sm.__file__)),visual_inspection_pending=True))
    print('Tables, chart, workbook and algebra verified',OUT,flush=True)


if __name__=='__main__':main()
