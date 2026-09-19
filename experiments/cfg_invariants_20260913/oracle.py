"""CPU-only, finite-support Gaussian-FM audit of proposed CFG invariants.

Run: python experiments/cfg_invariants_20260913/oracle.py --samples 2048 --steps 128
All targets are specified probability distributions, not perceptual image quality.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import softmax


END = .97
LAMBDA = .75


@dataclass
class Oracle:
    name: str
    points: np.ndarray
    mass: np.ndarray
    likelihood: np.ndarray
    ref_times: np.ndarray | None = None
    ref_norms: np.ndarray | None = None

    def __post_init__(self):
        self.mass = self.mass / self.mass.sum()
        self.conditional = self.mass * self.likelihood
        self.conditional /= self.conditional.sum()
        self.target = self.mass * self.likelihood ** 2
        self.target /= self.target.sum()
        self.logmass = np.log(self.mass)
        self.levels = np.unique(self.likelihood)
        self.atom_group = np.searchsorted(self.levels, self.likelihood)

    def draw(self, rng, n, weights=None, t=END):
        weights = self.mass if weights is None else weights
        ids = rng.choice(len(weights), n, p=weights)
        return t * self.points[ids] + (1-t)*rng.normal(size=(n, 2)), ids

    def fields(self, z, t):
        sigma = 1-t
        delta = z[:, None, :] - t*self.points[None, :, :]
        logp = self.logmass[None, :] - np.sum(delta*delta, axis=2)/(2*sigma*sigma)
        pu = softmax(logp, axis=1)
        q = pu @ self.likelihood
        pc = pu*self.likelihood[None, :] / q[:, None]
        h = pu @ (self.likelihood**2)
        pt = pu*(self.likelihood**2)[None, :] / h[:, None]
        du, dc, dt = pu @ self.points, pc @ self.points, pt @ self.points
        vu, vc, vt = (du-z)/sigma, (dc-z)/sigma, (dt-z)/sigma
        return vu, vc, vt, q

    def multiplier(self, arm, z, t, gap, q, extra):
        if arm == "conditional":
            return np.zeros(len(z))
        if arm.startswith("cfg"):
            return np.full(len(z), extra)
        phase = t/END
        envelope = math.sin(math.pi*phase)**2
        if arm == "time_feedback":
            return np.full(len(z), 1+LAMBDA*math.sin(2*math.pi*phase))
        if arm == "norm_feedback":
            ref = np.interp(t, self.ref_times, self.ref_norms)
            ratio = ref/(np.linalg.norm(gap, axis=1)+1e-10)
            return 1+LAMBDA*envelope*np.tanh(np.log(ratio+1e-10))
        if arm == "evidence_potential":
            return 1+LAMBDA*envelope*(1-2*q)
        raise ValueError(arm)

    def velocity(self, z, t, arm, extra=1.):
        vu, vc, vt, q = self.fields(z, t)
        if arm == "true_terminal_tilt":
            return vt, vt-vc
        gap = vc-vu
        a = self.multiplier(arm, z, t, gap, q, extra)
        effect = a[:, None]*gap
        return vc+effect, effect


def make_oracles():
    grid = np.array([(x,y) for x in [-2.,-.65,.65,2.] for y in [-1.5,0.,1.5]])
    gx, gy = grid.T
    gp = np.exp(-.14*gx**2-.5*(gy-.45*gx)**2)*(1+.12*(gy>0))
    gl = 1/(1+np.exp(-1.2*gx))
    ring, rp, rl = [], [], []
    for radius, likelihood, groupmass in [(1.,.15,.35),(2.,.5,.4),(3.,.85,.25)]:
        for angle in np.arange(8)*math.pi/4:
            ring.append((radius*math.cos(angle), .8*radius*math.sin(angle)))
            rp.append(groupmass*(1+.5*math.cos(angle)+.2*math.sin(2*angle)))
            rl.append(likelihood)
    return [Oracle("correlated_grid", grid, gp, gl),
            Oracle("anisotropic_rings", np.array(ring), np.array(rp), np.array(rl))]


def rollout(model, noise, arm, steps, extra=1., keep=False):
    z = noise.copy()
    times = np.linspace(0., END, steps+1)
    action = np.zeros(len(z))
    energy = np.zeros(len(z))
    trajectory = [z[:32].copy()] if keep else None
    for t0, t1 in zip(times[:-1], times[1:]):
        h = t1-t0
        v0, e0 = model.velocity(z, t0, arm, extra)
        pred = z+h*v0
        v1, e1 = model.velocity(pred, t1, arm, extra)
        z += h*(v0+v1)/2
        action += h*(np.linalg.norm(e0, axis=1)+np.linalg.norm(e1, axis=1))/2
        energy += h*(np.sum(e0*e0, axis=1)+np.sum(e1*e1, axis=1))/2
        if keep:
            trajectory.append(z[:32].copy())
    return z, dict(action=float(action.mean()), energy=float(energy.mean())), trajectory


def calibrate(model):
    rng = np.random.default_rng(902613)
    model.ref_times = np.linspace(0., END, 33)
    norms = []
    for t in model.ref_times:
        z, _ = model.draw(rng, 512, weights=model.conditional, t=t)
        vu, vc, _, _ = model.fields(z, t)
        norms.append(max(float(np.median(np.linalg.norm(vc-vu, axis=1))), 1e-8))
    model.ref_norms = np.array(norms)
    # Independent calibration paths; evaluation seeds are never consulted.
    noise = rng.normal(size=(384,2))
    rows = []
    for policy in ["time_feedback", "norm_feedback", "evidence_potential"]:
        _, stats, _ = rollout(model, noise, policy, 80)
        desired = stats["action"]
        low, high = .05, 3.
        _, ls, _ = rollout(model, noise, "cfg_match", 80, low)
        _, hs, _ = rollout(model, noise, "cfg_match", 80, high)
        bracketed = ls["action"] <= desired <= hs["action"]
        if not bracketed:
            raise RuntimeError(f"Calibration action not bracketed: {model.name}, {policy}")
        for _ in range(10):
            mid = (low+high)/2
            _, ms, _ = rollout(model, noise, "cfg_match", 80, mid)
            if ms["action"] < desired:
                low = mid
            else:
                high = mid
        coefficient = (low+high)/2
        _, fitted, _ = rollout(model, noise, "cfg_match", 80, coefficient)
        rows.append(dict(distribution=model.name, policy=policy,
                         extra=coefficient, desired_action=desired,
                         matched_action=fitted["action"],
                         relative_error=abs(fitted["action"]-desired)/max(desired,1e-12),
                         calibration_seed=902613, calibration_samples=384, calibration_steps=80))
    return rows


def nearest(model, z):
    return np.argmin(np.sum((z[:, None, :]-END*model.points[None, :, :])**2, axis=2), axis=1)


def sw2(a, b):
    directions = np.stack([np.cos(np.arange(32)*math.pi/32), np.sin(np.arange(32)*math.pi/32)],axis=0)
    aa, bb = np.sort(a @ directions, axis=0), np.sort(b @ directions, axis=0)
    return float(np.mean((aa-bb)**2))


def metrics(model, z, target_reference):
    ids = nearest(model,z)
    empirical = np.bincount(ids,minlength=len(model.mass))/len(ids)
    _, _, _, q = model.fields(z,END)
    result = dict(target_atom_tv=float(np.abs(empirical-model.target).sum()/2),
                  conditional_atom_tv=float(np.abs(empirical-model.conditional).sum()/2),
                  base_atom_tv=float(np.abs(empirical-model.mass).sum()/2),
                  target_sw2=sw2(z,target_reference),
                  evidence_mean=float(q.mean()),
                  log_evidence_mean=float(np.log(q).mean()),
                  atom_evidence_mean=float((empirical*model.likelihood).sum()))
    fibers, total, absent_mass = [], 0., 0.
    for group,r in enumerate(model.levels):
        mask = model.atom_group==group
        expected_cond = model.mass[mask]/model.mass[mask].sum()
        groupmass = model.target[mask].sum()
        observedmass = empirical[mask].sum()
        if observedmass:
            error = np.abs(empirical[mask]/observedmass-expected_cond).sum()/2
        else:
            error = 1.
            absent_mass += groupmass
        total += groupmass*error
        fibers.append(dict(level=float(r),count=int(np.isin(ids,np.where(mask)[0]).sum()),
                           conditional_tv=float(error),target_group_mass=float(groupmass)))
    result["within_evidence_tv"] = float(total)
    result["empty_evidence_target_mass"] = float(absent_mass)
    # Predeclared bins midway between clean evidence levels. Finite-noise
    # confidence is not constant: this disagreement quantifies the approximation.
    bins = (model.levels[1:]+model.levels[:-1])/2
    evidence_bin = np.searchsorted(bins,q)
    result["evidence_bin_atom_disagreement"] = float(np.mean(evidence_bin!=model.atom_group[ids]))
    return result, fibers, empirical


def curl_audit(model, arms):
    rng=np.random.default_rng(821309)
    rows=[]
    for t in [.1,.35,.65,.85]:
        z,_=model.draw(rng,128,t=t)
        for arm,extra in arms:
            jac=[]
            for d in range(2):
                shift=np.zeros_like(z); shift[:,d]=1e-4
                _,ep=model.velocity(z+shift,t,arm,extra)
                _,em=model.velocity(z-shift,t,arm,extra)
                jac.append((ep-em)/(2e-4)*t/(1-t))
            matrix=np.stack(jac,axis=2)
            curl=matrix[:,1,0]-matrix[:,0,1]
            rms=float(np.sqrt(np.mean(curl*curl)))
            jnorm=float(np.sqrt(np.mean(np.sum(matrix*matrix,axis=(1,2)))))
            rows.append(dict(distribution=model.name,arm=arm,time=t,
                             curl_rms=rms,jacobian_rms=jnorm,normalized_curl=rms/max(jnorm,1e-12)))
    return rows


def analytic_controls(out):
    rng=np.random.default_rng(51809)
    p=np.array([.1,.2,.3,.4]); soft=np.array([.1,.35,.6,.9]); hard=np.array([0.,1.,1.,0.])
    tilt=lambda x,l: x*l/np.sum(x*l)
    z=rng.normal(size=(20000,2)); angle=.6
    rot=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    rotated=z @ rot.T; radial=1.2*z
    times=np.array([.1,.3,.5,.7,.9]); variance=times**2+(1-times)**2
    path=np.sqrt(variance)[:,None]*np.array([1.3,-.8])[None,:]
    endpoint=path/np.sqrt(variance)[:,None]
    denoiser=times[:,None]*path/variance[:,None]
    data=dict(hard_twice_tv=float(np.abs(tilt(tilt(p,hard),hard)-tilt(p,hard)).sum()/2),
              soft_twice_tv=float(np.abs(tilt(tilt(p,soft),soft)-tilt(p,soft)).sum()/2),
              rotation_pointwise_mse=float(np.mean(np.sum((rotated-z)**2,axis=1))),
              rotation_population_kl=0.,radial_population_kl=2*(1.2**2-1-math.log(1.2**2))/2,
              rotation_sample_covariance=np.cov(rotated.T).tolist(),
              radial_sample_covariance=np.cov(radial.T).tolist(),
              remaining_endpoint_max_drift=float(np.max(np.abs(endpoint-endpoint[0]))),
              posterior_mean_endpoint_change=float(np.linalg.norm(denoiser[-1]-denoiser[0])),
              times=times.tolist(),exact_pf_path=path.tolist(),remaining_endpoint=endpoint.tolist(),
              posterior_clean_mean=denoiser.tolist())
    assert data['hard_twice_tv']<1e-12 and data['soft_twice_tv']>.01
    assert data['remaining_endpoint_max_drift']<1e-12
    (out/'analytic_controls.json').write_text(json.dumps(data,indent=2)+'\n')
    fig,ax=plt.subplots(1,3,figsize=(12,3.5))
    ax[0].scatter(z[:400,0],z[:400,1],s=3,label='original')
    ax[0].scatter(rotated[:400,0],rotated[:400,1],s=3,label='rotation')
    ax[0].set_title('Rotation preserves N(0,I)');ax[0].legend(fontsize=7)
    ax[1].scatter(radial[:400,0],radial[:400,1],s=3);ax[1].set_title('Radial expansion changes variance')
    ax[2].plot(times,endpoint[:,0],label='remaining endpoint');ax[2].plot(times,denoiser[:,0],label='posterior clean mean')
    ax[2].legend(fontsize=7);ax[2].set_title('Exact Gaussian FM trajectory')
    fig.tight_layout();fig.savefig(out/'analytic_controls.png',dpi=160);plt.close(fig)
    return data


def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def summarize(out,rows):
    summary=[]
    metric_names=['target_atom_tv','within_evidence_tv','target_sw2','evidence_mean','action','energy']
    for distribution in dict.fromkeys(r['distribution'] for r in rows):
        selected=[r for r in rows if r['distribution']==distribution]
        for arm in dict.fromkeys(r['arm'] for r in selected):
            rr=[r for r in selected if r['arm']==arm]
            line=dict(distribution=distribution,arm=arm,seeds=len(rr))
            for key in metric_names:
                vals=[r[key] for r in rr]
                line[key+'_mean']=float(np.mean(vals))
                line[key+'_min']=float(np.min(vals))
                line[key+'_max']=float(np.max(vals))
            summary.append(line)
    write_csv(out/'summary.csv',summary)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    names=['conditional','cfg_2','time_feedback','norm_feedback','evidence_potential','true_terminal_tilt']
    for ax,distribution in zip(axes,dict.fromkeys(r['distribution'] for r in rows)):
        rr={r['arm']:r for r in summary if r['distribution']==distribution}
        for i,name in enumerate(names):
            ax.scatter([i,i],[rr[name]['target_atom_tv_min'],rr[name]['target_atom_tv_max']],s=22,color='tab:blue')
            ax.plot([i-.2,i+.2],[rr[name]['target_atom_tv_mean']]*2,color='tab:blue')
        floor=rr['direct_target']
        ax.axhspan(floor['target_atom_tv_min'],floor['target_atom_tv_max'],color='gray',alpha=.2,label='direct target: two-seed range')
        ax.set_xticks(range(len(names)),names,rotation=35,ha='right',fontsize=8)
        ax.set_title(distribution);ax.set_ylabel('TV to specified atom target');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'target_error_summary.png',dpi=170);plt.close(fig)


def postprocess_existing(out, bootstrap_count=1000):
    """Only read saved endpoints: no further ODE/model evaluations."""
    started=time.time()
    rows=list(csv.DictReader((out/'metrics.csv').open()))
    keyed={(r['distribution'],r['seed'],r['arm']):r for r in rows}
    matching=[]
    for row in rows:
        if row['arm'] not in ['time_feedback','norm_feedback','evidence_potential']:
            continue
        comparison=keyed[(row['distribution'],row['seed'],'cfg_match_'+row['arm'])]
        matching.append(dict(distribution=row['distribution'],seed=row['seed'],policy=row['arm'],
                             policy_action=float(row['action']),matched_cfg_action=float(comparison['action']),
                             relative_action_error=float(row['action'])/float(comparison['action'])-1,
                             policy_target_tv=float(row['target_atom_tv']),matched_cfg_target_tv=float(comparison['target_atom_tv'])))
    write_csv(out/'action_matching_evaluation.csv',matching)
    boot=[]
    for mi,model in enumerate(make_oracles()):
        seeds=list(dict.fromkeys(r['seed'] for r in rows if r['distribution']==model.name))
        pairs=[]
        for seed in seeds:
            a=np.load(out/f'{model.name}_{seed}_evidence_potential.npz')
            b=np.load(out/f'{model.name}_{seed}_cfg_match_evidence_potential.npz')
            assert np.array_equal(a['initial_noise'],b['initial_noise'])
            pairs.append((nearest(model,a['samples']),nearest(model,b['samples'])))
        rng=np.random.default_rng(194613+mi)
        estimates=[]
        for _ in range(bootstrap_count):
            differences=[]
            for aa,bb in pairs:
                ids=rng.integers(0,len(aa),len(aa))
                pa=np.bincount(aa[ids],minlength=len(model.mass))/len(aa)
                pb=np.bincount(bb[ids],minlength=len(model.mass))/len(bb)
                differences.append((np.abs(pa-model.target).sum()-np.abs(pb-model.target).sum())/2)
            estimates.append(float(np.mean(differences)))
        observed=np.mean([float(keyed[(model.name,s,'evidence_potential')]['target_atom_tv'])-float(keyed[(model.name,s,'cfg_match_evidence_potential')]['target_atom_tv']) for s in seeds])
        boot.append(dict(distribution=model.name,comparison='evidence_potential minus matched CFG',
                         observed_tv_difference=float(observed),ci_low=float(np.quantile(estimates,.025)),
                         ci_high=float(np.quantile(estimates,.975)),bootstrap_replicates=bootstrap_count,
                         pairing='same initial Gaussian index; separate resampling within each fixed evaluation seed',
                         statistic='mean of per-seed TV differences',seeds=seeds,
                         limitation='Conditional on these two fixed oracles/seeds, not model-level or image-quality generalization.'))
    (out/'paired_bootstrap.json').write_text(json.dumps(boot,indent=2)+'\n')
    (out/'postprocess_manifest.json').write_text(json.dumps(dict(cpu_only=True,new_rollouts=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seconds=time.time()-started),indent=2)+'\n')
    print(json.dumps({'postprocess_done':True,'bootstrap':boot},indent=2),flush=True)


def plot_results(out,model,samples,metrics_rows):
    order=['direct_target','conditional','cfg_2','time_feedback','norm_feedback','evidence_potential','true_terminal_tilt']
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for ax,arm in zip(axes.flat,order):
        z=samples[arm]
        ids=nearest(model,z)
        ax.scatter(z[:700,0],z[:700,1],c=model.likelihood[ids[:700]],s=4,cmap='viridis',vmin=0,vmax=1,alpha=.6)
        ax.scatter(END*model.points[:,0],END*model.points[:,1],s=25,facecolors='none',edgecolors='black',linewidths=.7)
        row=next(r for r in metrics_rows if r['arm']==arm)
        ax.set_title(f"{arm}\nTV={row['target_atom_tv']:.3f}; fiber={row['within_evidence_tv']:.3f}",fontsize=9)
        ax.set_aspect('equal')
    axes.flat[-1].axis('off')
    fig.suptitle(model.name+' / first evaluation seed / finite support at t=.97')
    fig.tight_layout();fig.savefig(out/(model.name+'_scatter.png'),dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/oracle')
    parser.add_argument('--samples',type=int,default=2048)
    parser.add_argument('--steps',type=int,default=128)
    parser.add_argument('--seeds',type=int,nargs='+',default=[2026091301,2026091302])
    parser.add_argument('--postprocess-only',action='store_true',help='Only bootstrap and matching audits from existing saved samples.')
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.postprocess_only:
        postprocess_existing(out)
        return
    start=time.time();allrows=[];fiberrows=[];calrows=[];curlrows=[];solverrows=[];checks=[]
    for model in make_oracles():
        print('calibrating',model.name,flush=True)
        cal=calibrate(model);calrows.extend(cal)
        arms=[('conditional',0.),('cfg_2',1.),('time_feedback',1.),('norm_feedback',1.),('evidence_potential',1.),('true_terminal_tilt',1.)]
        arms += [('cfg_match_'+r['policy'],r['extra']) for r in cal]
        curlrows.extend(curl_audit(model,arms))
        dif=model.points[:,None,:]-model.points[None,:,:]
        distances=np.linalg.norm(dif,axis=2);np.fill_diagonal(distances,np.inf)
        minimum_separation=END*float(distances.min())/(1-END)
        (out/(model.name+'_definition.json')).write_text(json.dumps(dict(points=model.points.tolist(),mass=model.mass.tolist(),likelihood=model.likelihood.tolist(),conditional=model.conditional.tolist(),target=model.target.tolist(),norm_reference_times=model.ref_times.tolist(),norm_reference=model.ref_norms.tolist(),min_separation_sigma=minimum_separation),indent=2)+'\n')
        for seed_index,seed in enumerate(args.seeds):
            print('evaluation',model.name,seed,flush=True)
            rng=np.random.default_rng(seed);noise=rng.normal(size=(args.samples,2))
            reference,_=model.draw(np.random.default_rng(seed+100000),args.samples,model.target)
            direct,direct_ids=model.draw(np.random.default_rng(seed+200000),args.samples,model.target)
            confusion=float(np.mean(nearest(model,direct)!=direct_ids))
            checks.append(dict(distribution=model.name,seed=seed,direct_target_nearest_confusion=confusion,min_separation_sigma=minimum_separation))
            samplebank={'direct_target':direct};statsbank={'direct_target':dict(action=0.,energy=0.)}
            for arm,extra in arms:
                z,stats,trajectory=rollout(model,noise,arm,args.steps,extra,keep=(seed_index==0))
                samplebank[arm]=z;statsbank[arm]=stats
                if trajectory is not None:
                    np.save(out/f'{model.name}_{arm}_trajectory.npy',np.array(trajectory))
            localrows=[]
            for arm,z in samplebank.items():
                met,fibers,atom_mass=metrics(model,z,reference)
                row=dict(distribution=model.name,seed=seed,arm=arm,n=args.samples,steps=(0 if arm=='direct_target' else args.steps),**statsbank[arm],**met)
                allrows.append(row);localrows.append(row)
                for f in fibers:fiberrows.append(dict(distribution=model.name,seed=seed,arm=arm,**f))
                np.savez_compressed(out/f'{model.name}_{seed}_{arm}.npz',samples=z,atom_mass=atom_mass,initial_noise=noise)
            if seed_index==0:
                plot_results(out,model,samplebank,localrows)
                for arm,extra in [('cfg_2',1.),('norm_feedback',1.),('evidence_potential',1.),('true_terminal_tilt',1.)]:
                    coarse,_,_=rollout(model,noise[:512],arm,args.steps//2,extra)
                    fine=samplebank[arm][:512]
                    finer,_,_=rollout(model,noise[:512],arm,args.steps*2,extra)
                    solverrows.append(dict(distribution=model.name,arm=arm,n=512,coarse_steps=args.steps//2,steps=args.steps,fine_steps=args.steps*2,
                                           coarse_to_main_rms=float(np.sqrt(np.mean((coarse-fine)**2))),main_to_fine_rms=float(np.sqrt(np.mean((fine-finer)**2))),
                                           coarse_atom_disagreement=float(np.mean(nearest(model,coarse)!=nearest(model,fine))),fine_atom_disagreement=float(np.mean(nearest(model,fine)!=nearest(model,finer)))))
                pairmax=max(float(np.sqrt(np.mean((samplebank[a]-samplebank['cfg_2'])**2))) for a in ['time_feedback','norm_feedback','evidence_potential','true_terminal_tilt'])
                assert pairmax>1e-3,'Unexpected identical operators'
                checks.append(dict(distribution=model.name,max_operator_endpoint_difference=pairmax))
        write_csv(out/'metrics.csv',allrows);write_csv(out/'fiber_metrics.csv',fiberrows)
        write_csv(out/'calibration.csv',calrows);write_csv(out/'curl.csv',curlrows);write_csv(out/'solver_checks.csv',solverrows)
    analytic=analytic_controls(out)
    summarize(out,allrows)
    manifest=dict(samples=args.samples,steps=args.steps,seeds=args.seeds,cutoff=END,power=2.,feedback_lambda=LAMBDA,
                  cpu_only=True,calibration_seed=902613,seconds=time.time()-start,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  checks=checks,analytic_controls=analytic,
                  caveats=['Finite clean support; all reported distribution quality concerns the specified oracle targets, not images.',
                           'Evidence fibers use nearest clean atom labels at t=.97; exact noisy evidence is only approximately binned.',
                           'Matching coefficients use independent calibration paths; report evaluation action mismatch instead of assuming exact matching.',
                           'Two evaluation seeds are descriptive replication, not a population-level significance test.'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    postprocess_existing(out)
    print(json.dumps(dict(done=True,seconds=manifest['seconds'],rows=len(allrows),out=str(out)),indent=2),flush=True)


if __name__=='__main__':
    main()
