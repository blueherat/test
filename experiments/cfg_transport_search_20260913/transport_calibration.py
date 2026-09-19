"""Private, bounded SiT conditional transport calibration; no shared runtime edits.

prepare reads ONLY the original training cache, and does not import torch.
observe requires an explicitly selected device. fit is CPU-only.
Example:
  python -m experiments.cfg_transport_search_20260913.transport_calibration prepare
  CUDA_VISIBLE_DEVICES=0 python -m experiments.cfg_transport_search_20260913.transport_calibration observe --device cuda:0
  python -m experiments.cfg_transport_search_20260913.transport_calibration fit
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path('/home/zhoushunyu/data/eqvae/experiments/cfg_transport_search_20260913/transport_calibration')
DATA = Path('/home/zhoushunyu/data/eqvae/imagenet_sit_flow/imagenet100_cmc_sdvae')
CKPT = DATA.parent / 'runs/sit-s-2_seed0/checkpoints/step_00800000.pt'
TIMES = (.25, .5, .7)
SEED = 2026091381


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def prepare(args):
    root = args.output
    if (root / 'request.json').exists():
        raise FileExistsError('Inputs already frozen; use a different --output for a new protocol.')
    labels = np.load(DATA / 'train_labels.npy', mmap_mode='r')
    source = np.load(DATA / 'train_source_indices.npy', mmap_mode='r')
    moments = np.load(DATA / 'train_moments.npy', mmap_mode='r')
    manifest = json.loads((DATA / 'manifest.json').read_text())
    assert len(labels) == len(source) == len(moments) == manifest['splits']['train']['count']
    assert len(np.unique(source)) == len(source)
    assert manifest['source']['posterior_layout'] == 'channels 0:4 mean, channels 4:8 standard deviation'
    rng = np.random.default_rng(SEED)
    indices, splits = [], []
    for label in range(100):
        pool = rng.permutation(np.flatnonzero(labels == label))
        for split, selected in [(0, pool[:args.fit_per_class]),
                                (1, pool[args.fit_per_class:args.fit_per_class+args.holdout_per_class])]:
            indices.extend(selected.tolist()); splits.extend([split] * len(selected))
    indices = np.asarray(indices, dtype=np.int64)
    splits = np.asarray(splits, dtype=np.int8)
    assert len(np.unique(indices)) == len(indices)
    ids = np.asarray(source[indices], dtype=np.int64)
    assert not set(ids[splits == 0]) & set(ids[splits == 1])
    chosen = np.array(moments[indices])
    posterior_noise = rng.standard_normal((len(indices), 4, 32, 32), dtype=np.float32)
    clean = (chosen[:, :4] + chosen[:, 4:] * posterior_noise) * np.float32(.18215)
    noise = rng.standard_normal(clean.shape, dtype=np.float32)
    root.mkdir(parents=True, exist_ok=True)
    np.savez(root / 'inputs.npz', clean=clean, noise=noise,
             labels=np.asarray(labels[indices], dtype=np.int64), indices=indices,
             source_ids=ids, split=splits, moments=chosen, posterior_noise=posterior_noise)
    request = dict(
        seed=SEED, times=list(TIMES), horizon=.125, gamma=1., steps=2,
        check_steps=[4, 8], check_images=8, checkpoint=str(CKPT), weights='ema',
        fit_per_class=args.fit_per_class, holdout_per_class=args.holdout_per_class,
        images=len(indices), input_sha256=sha(root/'inputs.npz'),
        provenance=dict(cache=str(DATA), source_split='train',
            train_moments_declared_sha256=manifest['splits']['train']['moments_sha256'],
            labels_sha256=sha(DATA/'train_labels.npy'),
            source_indices_sha256=sha(DATA/'train_source_indices.npy'),
            cache_manifest_sha256=sha(DATA/'manifest.json'),
            selected_moments_sha256=hashlib.sha256(chosen.tobytes()).hexdigest(),
            fid_validation_loaded=False, fit_holdout_source_ids_disjoint=True),
        feature='(I-P_[gap,no_history_APG,Euler_state_secant,embedding_0.5_secant]) * bias_corrected_loop/H',
        estimator='per-time signed least squares, no fit to FID, no holdout clipping',
        main_feature_single_branch_nfe=22, dtype='float32 fields; float64 projection/statistics',
    )
    atomic(root / 'request.json', request)
    print(json.dumps(request, indent=2), flush=True)


class Fields:
    def __init__(self, device):
        import torch
        from experiments.imagenet100_sit_multiscale_models import load_sit_field_model
        from experiments.train_imagenet100_sit_flow import load_official_sit_module, DEFAULT_OFFICIAL_SIT_REPO
        module, source = load_official_sit_module(DEFAULT_OFFICIAL_SIT_REPO, verify_source=True)
        self.model, self.semantics, self.metadata = load_sit_field_model(
            checkpoint_path=CKPT, weights='ema', sit_module=module,
            source_metadata=source, device=torch.device(device))
        self.calls = 0
        self.labels = None

    def call(self, z, t, kind='conditional'):
        import torch
        from experiments.imagenet100_sit_multiscale_models import evaluate_sit_field
        self.calls += 1
        labels = self.labels if kind == 'conditional' else torch.full_like(self.labels, 100)
        return evaluate_sit_field(self.model, self.semantics, z,
                                  z.new_full((len(z),), t), labels)

    def embedding(self, z, t, scale):
        import torch
        from experiments.imagenet100_sit_static_pair import output_to_field_velocity
        self.calls += 1
        model = self.model
        times = z.new_full((len(z),), t)
        cond = model.y_embedder(self.labels, False)
        null = model.y_embedder(torch.full_like(self.labels, 100), False)
        condition = model.t_embedder(times) + (null + scale * (cond-null))
        tokens = model.x_embedder(z) + model.pos_embed
        for block in model.blocks:
            tokens = block(tokens, condition)
        output = model.unpatchify(model.final_layer(tokens, condition))
        if model.learn_sigma:
            output, _ = output.chunk(2, dim=1)
        return output_to_field_velocity(output, state=z, time_value=times, semantics=self.semantics)


def heun(z, t, end, steps, field, first=None):
    h = (end-t) / steps
    for k in range(steps):
        s = t+k*h
        one = first if k == 0 and first is not None else field(z, s)
        two = field(z+h*one, s+h)
        z = z + .5*h*(one+two)
    return z


def project(vec, columns):
    import torch
    value = vec.double().flatten(1)
    orth = []
    energies = []
    for column in columns:
        basis = column.double().flatten(1)
        for _ in range(2):
            for u in orth:
                basis = basis - (basis*u).sum(1, keepdim=True)*u
        norm = basis.norm(dim=1, keepdim=True)
        u = torch.where(norm > 1e-10, basis/norm.clamp_min(1e-20), torch.zeros_like(basis))
        orth.append(u)
        for _ in range(2):
            value = value-(value*u).sum(1, keepdim=True)*u
        energies.append(value.square().mean(1))
    return value.reshape_as(vec), torch.stack(energies, 1)


def transport_feature(fields, z, t, *, horizon=.125, gamma=1., steps=2,
                      cached=None):
    """Returns a velocity basis, not a state writeback or a guided model head."""
    import torch
    start = fields.calls
    end = t+horizon
    assert 0 <= t < end < 1
    if cached is None:
        a = fields.call(z, t)
        u = fields.call(z, t, 'null')
        g = a-u
        clean = z+(1-t)*a
        apg = g-((g.double()*clean).flatten(1).sum(1) /
                 clean.double().flatten(1).square().sum(1).clamp_min(1e-20)).to(z.dtype)[:,None,None,None]*clean
        state_secant = -(fields.call(z+horizon*(a+gamma*g), end)
                          -fields.call(z+horizon*a, end))
        embedding_secant = fields.embedding(z, t, .5)-u
        cached = dict(a=a, u=u, g=g, apg=apg, state_secant=state_secant,
                      embedding_secant=embedding_secant)
    a, g = cached['a'], cached['g']
    cond = lambda x, s: fields.call(x, s)
    high = lambda x, s: ((1+gamma)*fields.call(x, s)
                         -gamma*fields.call(x, s, 'null'))
    y_high = heun(z, t, end, steps, high, first=a+gamma*g)
    y_cond = heun(z, t, end, steps, cond, first=a)
    back_high = heun(y_high, end, t, steps, cond)
    back_cond = heun(y_cond, end, t, steps, cond)
    raw = (back_high-back_cond) / horizon
    q, energies = project(raw, [g, cached['apg'], cached['state_secant'], cached['embedding_secant']])
    scale = max(horizon, 1e-12)
    meta = dict(q=q, raw=raw.double(), energies=energies,
                self_defect=(back_cond-z).double()/scale,
                cached=cached, nfe=fields.calls-start)
    assert torch.isfinite(q).all()
    return meta


def observe(args):
    import torch
    root = args.output
    request = json.loads((root/'request.json').read_text())
    assert sha(root/'inputs.npz') == request['input_sha256']
    if (root/'observations.npz').exists():
        raise FileExistsError('Observations already exist; do not overwrite them.')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fields = Fields(args.device)
    data = np.load(root/'inputs.npz')
    rows, audits = [], []
    rng = np.random.default_rng(SEED+1)
    check = set(rng.choice(len(data['labels']), request['check_images'], replace=False).tolist())
    total_equivalent = 0
    started = time.monotonic()
    with torch.inference_mode():
        for ti, t in enumerate(request['times']):
            for first in range(0, len(data['labels']), args.batch):
                stop = min(first+args.batch, len(data['labels']))
                clean = torch.from_numpy(data['clean'][first:stop]).to(args.device)
                noise = torch.from_numpy(data['noise'][first:stop]).to(args.device)
                fields.labels = torch.from_numpy(data['labels'][first:stop]).to(args.device)
                z = t*clean+(1-t)*noise
                result = transport_feature(fields, z, t, horizon=request['horizon'],
                                           gamma=request['gamma'], steps=request['steps'])
                total_equivalent += result['nfe'] * len(z)
                assert result['nfe'] == 22
                q = result['q']
                cached = result['cached']
                residual = (clean-noise).double()-cached['a'].double()
                ms = lambda x: x.double().flatten(1).square().mean(1)
                inner = lambda x, y: (x.double()*y.double()).flatten(1).mean(1)
                values = torch.stack([ms(residual), ms(q), inner(q,residual),
                    ms(cached['g']), ms(cached['apg']), inner(q,cached['apg']),
                    ms(result['raw']), ms(result['self_defect']),
                    *result['energies'].unbind(1)], 1).cpu().numpy()
                for j, value in enumerate(values):
                    rows.append([ti, first+j, *value])
                if args.save_fields:
                    np.savez_compressed(root/f'fields_t{ti}_b{first:05d}.npz',
                        q=q.float().cpu().numpy(), raw=result['raw'].float().cpu().numpy(),
                        source_ids=data['source_ids'][first:stop], t=t)
                chosen = [j for j in range(len(z)) if first+j in check]
                if chosen:
                    idx = torch.tensor(chosen, device=args.device)
                    fields.labels = fields.labels[idx]
                    subset_cache = {k:v[idx] for k,v in cached.items()}
                    previous = q[idx]
                    for steps in request['check_steps']:
                        refined = transport_feature(fields,z[idx],t,horizon=request['horizon'],
                            gamma=request['gamma'],steps=steps,cached=subset_cache)
                        total_equivalent += refined['nfe'] * len(idx)
                        fine = refined['q']
                        cosine = inner(previous,fine) / (ms(previous)*ms(fine)).sqrt().clamp_min(1e-20)
                        rel = (ms(fine-previous) / ms(fine).clamp_min(1e-30)).sqrt()
                        for j,k in enumerate(chosen):
                            audits.append(dict(time=t, index=first+k, source_id=int(data['source_ids'][first+k]),
                                fine_steps=steps, relative_change=float(rel[j]), cosine=float(cosine[j]),
                                fine_q_rms=float(ms(fine)[j].sqrt()),
                                previous_q_rms=float(ms(previous)[j].sqrt()),
                                split=int(data['split'][first+k]),
                                fine_q2=float(ms(fine)[j]),
                                fine_q_residual=float(inner(fine,residual[idx])[j]),
                                previous_q2=float(ms(previous)[j]),
                                previous_q_residual=float(inner(previous,residual[idx])[j])))
                        previous=fine
                elapsed=time.monotonic()-started
                atomic(root/'progress.json',dict(time=t,completed_images_at_time=stop,
                    total_images=len(data['labels']),equivalent_single_sample_forwards=total_equivalent,
                    elapsed_seconds=elapsed))
                print(f't={t:.2f} images={stop}/{len(data["labels"])} elapsed={elapsed:.1f}s',flush=True)
    names=['time_index','image_index','base_risk','q2','q_residual','gap2','apg2','q_apg',
           'raw2','self_defect2','after_gap2','after_apg2','after_euler_secant2','after_embedding_secant2']
    np.savez(root/'observations.npz',rows=np.asarray(rows,dtype=np.float64),columns=np.asarray(names))
    atomic(root/'solver_audit.json',audits)
    atomic(root/'observation_manifest.json',dict(complete=True,seconds=time.monotonic()-started,
        source_sha256=sha(__file__),input_sha256=request['input_sha256'],request_sha256=sha(root/'request.json'),
        equivalent_single_sample_forwards=total_equivalent,model=fields.metadata,
        observations_sha256=sha(root/'observations.npz'),device=args.device,
        sampled_images=0,fid_computed=False,main_feature_nfe=22))


def fit(args):
    root=args.output
    request=json.loads((root/'request.json').read_text())
    manifest=json.loads((root/'observation_manifest.json').read_text())
    assert manifest['complete'] and sha(root/'observations.npz')==manifest['observations_sha256']
    inputs=np.load(root/'inputs.npz')
    data=np.load(root/'observations.npz')
    columns={name:i for i,name in enumerate(data['columns'].tolist())}
    rows=data['rows']; summaries=[]; paired=[]
    rng=np.random.default_rng(SEED+2)
    for ti,t in enumerate(request['times']):
        r=rows[rows[:,columns['time_index']]==ti]
        ids=r[:,columns['image_index']].astype(int)
        cal=inputs['split'][ids]==0; hold=~cal
        q2=r[:,columns['q2']]; qr=r[:,columns['q_residual']]
        lam=float(qr[cal].sum()/max(q2[cal].sum(),1e-30))
        delta=lam**2*q2-2*lam*qr
        delta_apg=delta+2*lam*1.25*r[:,columns['q_apg']]
        # Independent images are bootstrap units; no pixel-level pseudoreplication.
        d=delta[hold]
        samples=rng.integers(len(d),size=(2000,len(d)))
        interval=np.quantile(d[samples].mean(1),[.025,.975])
        ratios=np.abs(lam)*np.sqrt(q2/np.maximum(r[:,columns['apg2']],1e-30))
        bysplit={}
        for name,mask in [('calibration',cal),('holdout',hold)]:
            bysplit[name]=dict(base_risk=float(r[mask,columns['base_risk']].mean()),
                risk_delta=float(delta[mask].mean()),
                correction_to_unit_apg_rms=float(abs(lam)*np.sqrt(q2[mask].sum()/
                    max(r[mask,columns['apg2']].sum(),1e-30))),
                correction_to_unit_apg_quantiles=dict(zip(['p0','p50','p90','p99','p100'],
                    np.quantile(ratios[mask],[0,.5,.9,.99,1]).tolist())),
                correction_to_gap_rms=float(abs(lam)*np.sqrt(q2[mask].sum()/
                    max(r[mask,columns['gap2']].sum(),1e-30))),
                remaining_transport_fraction_rms=float(np.sqrt(q2[mask].sum()/
                    max(r[mask,columns['raw2']].sum(),1e-30))))
        summaries.append(dict(time=t,raw_coefficient=lam,
            calibration_q_rms=float(np.sqrt(q2[cal].mean())),
            normalized_coefficient=float(lam*np.sqrt(q2[cal].mean())),
            holdout_delta_ci95=interval.tolist(),
            max_delta_difference_using_no_history_apg_residual=float(np.max(np.abs(delta-delta_apg))),
            **bysplit))
        paired.append(dict(ids=ids[hold],delta=d))
    assert all(np.array_equal(paired[0]['ids'],p['ids']) for p in paired)
    jointly=np.stack([p['delta'] for p in paired]).mean(0)
    idx=rng.integers(len(jointly),size=(2000,len(jointly)))
    joint_ci=np.quantile(jointly[idx].mean(1),[.025,.975]).tolist()
    audits=json.loads((root/'solver_audit.json').read_text())
    for row in audits:
        lam=next(s['raw_coefficient'] for s in summaries if s['time']==row['time'])
        row['frozen_coefficient']=lam
        row['fine_risk_delta']=lam**2*row['fine_q2']-2*lam*row['fine_q_residual']
        row['previous_risk_delta']=lam**2*row['previous_q2']-2*lam*row['previous_q_residual']
    atomic(root/'solver_risk_audit.json',audits)
    solver={str(m):dict(median_relative_change=float(np.median([x['relative_change'] for x in audits if x['fine_steps']==m])),
        p90_relative_change=float(np.quantile([x['relative_change'] for x in audits if x['fine_steps']==m],.9)),
        min_cosine=float(min(x['cosine'] for x in audits if x['fine_steps']==m))) for m in request['check_steps']}
    summary=dict(complete=True,coefficients_frozen_from_calibration_only=True,
        holdout_clipping=False,per_time=summaries,
        joint_holdout_risk_delta=float(jointly.mean()),joint_holdout_ci95=joint_ci,
        bootstrap_replicates=2000,bootstrap_unit='independent original training image, shared across times',
        numerical_checks=solver,source_sha256=sha(__file__),observations_sha256=manifest['observations_sha256'],
        request_sha256=sha(root/'request.json'),
        interpretation='teacher-state velocity-risk evidence only; no generated-image quality claim')
    atomic(root/'calibration_summary.json',summary)
    np.savez(root/'holdout_paired.npz',image_indices=paired[0]['ids'],
             source_ids=inputs['source_ids'][paired[0]['ids']],risk_deltas=np.stack([p['delta'] for p in paired]),
             times=np.asarray(request['times']))
    print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('task',choices=['prepare','observe','fit'])
    p.add_argument('--output',type=Path,default=ROOT)
    p.add_argument('--fit-per-class',type=int,default=2)
    p.add_argument('--holdout-per-class',type=int,default=2)
    p.add_argument('--device',default='cpu')
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--save-fields',action='store_true')
    args=p.parse_args()
    globals()[args.task](args)


if __name__=='__main__':
    main()
