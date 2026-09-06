"""Fixed DC/AC raw second moments on historical FP16 latent endpoints, CPU only."""
import argparse, datetime, hashlib, json, os, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
R=HERE.parent
FEAS=R/'spectral_covariance_feasibility_v1.json'
ARMS=['real','historical_scale1','ig_1p78']
FILES=['real','scale_s1p000000','scale_s1p780000']
VIEWS=['all_5000','exclude_21_shared_sources']
COMP=['spatial_DC','spatial_AC']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def dump(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def identity(bank):
    p=bank['sample_protocol']
    assert sha(p['path'])==p['sha256']
    assert sha(Path(bank['root'])/'manifest.json')==bank['manifest_sha256']
    with np.load(p['path'],allow_pickle=False) as z:
        v={k:z[k].copy() for k in z.files}
    assert np.array_equal(v['sample_ids'],np.arange(5000))
    assert np.array_equal(v['labels'],np.arange(5000)%1000)
    assert len(np.unique(v['real_source_rows']))==5000
    for f in bank['latent_files']:
        p=Path(f['path']);s=p.stat()
        assert s.st_size==f['file_bytes'] and s.st_mtime_ns==f['mtime_ns']
        array=np.load(p,mmap_mode='r',allow_pickle=False)
        assert list(array.shape)==f['shape'] and array.dtype==np.float16 and array.flags.c_contiguous
    return v

def prepare():
    started,cpu=time.perf_counter(),time.process_time()
    assert not (HERE/'request.json').exists()
    f=json.loads(FEAS.read_text())
    banks=f['banks']
    assert [b['seed'] for b in banks]==[20260801,20260802]
    ids=[identity(b) for b in banks]
    shared=np.intersect1d(ids[0]['real_source_rows'],ids[1]['real_source_rows'])
    assert shared.tolist()==f['cross_bank_real_overlap']['shared_source_rows'] and len(shared)==21
    cohort=[]
    for b,v in zip(banks,ids):
        exclude=np.flatnonzero(np.isin(v['real_source_rows'],shared))
        counts=np.bincount(v['labels'],minlength=1000)
        kept=np.bincount(v['labels'][~np.isin(v['sample_ids'],exclude)],minlength=1000)
        assert len(exclude)==21 and np.all(counts==5) and np.all(kept>0)
        cohort.append({'seed':b['seed'],'excluded_ids':exclude.tolist(),'excluded_source_rows':v['real_source_rows'][exclude].tolist(),
                       'excluded_labels':v['labels'][exclude].tolist(),'full_class_counts':counts.tolist(),'deduplicated_class_counts':kept.tolist()})
    request={'protocol':'raev2_fixed_spatial_dc_ac_raw_energy_v1','status':'frozen_before_latent_payload_read',
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_feasibility':{'path':str(FEAS),'sha256':sha(FEAS)},
        'runner':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
        'banks':banks,'cohorts':cohort,'arms':ARMS,'views':VIEWS,'components':COMP,'channels':list(range(1024)),
        'per_image_channel_definition':{'spatial_DC':'(mean_HW X)^2',
                                        'spatial_AC':'mean_HW((X-mean_HW X)^2)',
                                        'total':'DC+AC=mean_HW(X^2)',
                                        'spatial_size':[16,16],'spatial_normalization':256},
        'arithmetic':'stored FP16 promoted to FP64; FP64 spatial mean then two-pass centered AC; no epsilon, clipping, whitening, gain fit or selected channels',
        'class_aggregation':'mean of per-image energies within each of all 1000 classes, then equal mean across classes. Full view n_c=5; deduplicated view uses remaining n_c, never global sample weights.',
        'identity_sensitivity':'Only exclude the 21 source IDs shared between banks, simultaneously from real and corresponding generated IDs in both banks. This is a prespecified statistical identity sensitivity, not generated-image selection. Preserve all-5000 primary results.',
        'residual':'generated arm energy minus real energy, retain every class/component/channel and equal-class global vector',
        'cross_bank_metrics':'For each view and each generated arm, separately for DC and AC: cosine of full 1024-dimensional raw residual vectors; exact sign agreement mean(sign(r_A)==sign(r_B)), with zero counts reported; positive/negative counts. No threshold or weighting.',
        'total_energy_ratios':'For each bank/view and each generated arm: sum_channels(E_gen)/sum_channels(E_real) separately DC, AC, and DC+AC. Also preserve raw total sums and residual vector norms.',
        'uncertainty':'No CI, independence test or significance claim. Removing shared sources does not prove statistical independence; retrospective banks and shared classes remain.',
        'interpretation':'Raw second energies of fixed orthogonal spatial DC/AC components; not centered data covariance, covariance eigenvalues, full spectrum or causal FID result. Means may be nonzero.',
        'io':'Read each of 24 FP16 latent payloads once in sequential per-shard blocks; update whole-file SHA with header then payload bytes during same pass; no second full payload hash pass.',
        'block_rows':16,'payload_bytes_expected':15728640000,
        'outputs':'Per bank NPZ: all class/component/channel energies and generated-minus-real residuals for both views; global equal-class arrays; IDs/source rows/keep masks/class counts. JSON full bank/channel summaries and cross-bank metrics.',
        'verification':'all finite values; source metadata and header identity; positive class counts; DC+AC versus direct mean X^2 on every image/channel; residual algebra and serialized artifact identities',
        'prohibitions':['no GPU/model calls','no candidate guidance or sampling','no channel selection','no coefficient/threshold/time/noise sweep','no FID'],
        'numpy_version':np.__version__}
    dump('request.json',request)
    dump('prepare_cost.json',{'wall_seconds':time.perf_counter()-started,'cpu_seconds':time.process_time()-cpu,
                             'request_sha256':sha(HERE/'request.json'),'latent_payload_bytes_read':0,'model_calls':0,'gpu_seconds':0})
    print(json.dumps({'request_sha256':sha(HERE/'request.json'),'cohorts':[{'seed':c['seed'],'removed':len(c['excluded_ids'])} for c in cohort]}),flush=True)

def bank_summary(energy,residual):
    out={}
    for vi,view in enumerate(VIEWS):
        arms={}
        for ai,arm in enumerate(ARMS):
            total=energy[vi,ai].sum(axis=1)
            arms[arm]={'energy_sum_over_channels':dict(zip(COMP,total.tolist())),
                       'energy_DC_plus_AC_sum':float(total.sum()),
                       'mean_energy_per_latent_coordinate':float(total.sum()/1024)}
            if ai:
                ref=energy[vi,0].sum(axis=1)
                arms[arm]['energy_ratio_to_real']={**{k:float(total[j]/ref[j]) for j,k in enumerate(COMP)},'DC_plus_AC':float(total.sum()/ref.sum())}
                arms[arm]['residual_vector']={k:{'l2_norm':float(np.linalg.norm(residual[vi,ai-1,j])),
                    'positive_channels':int(np.sum(residual[vi,ai-1,j]>0)),
                    'negative_channels':int(np.sum(residual[vi,ai-1,j]<0)),
                    'zero_channels':int(np.sum(residual[vi,ai-1,j]==0))} for j,k in enumerate(COMP)}
        out[view]=arms
    return out

def run():
    start,cpu=time.perf_counter(),time.process_time()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    request=json.loads((HERE/'request.json').read_text())
    assert not (HERE/'summary.json').exists()
    for x in ['source_feasibility','runner']:
        assert sha(request[x]['path'])==request[x]['sha256']
    payload_bytes=0;file_hashes=[];banks_out=[];all_global=[];all_residual=[]
    identity_error_max=0.;identity_relative_max=0.;checks=0
    for bi,bank in enumerate(request['banks']):
        bank_start=time.perf_counter();v=identity(bank);cohort=request['cohorts'][bi]
        keep=~np.isin(v['sample_ids'],cohort['excluded_ids'])
        sums=np.zeros((3,1000,2,1024),dtype=np.float64)
        excluded=np.zeros_like(sums)
        for ai,branch in enumerate(FILES):
            for rank in range(4):
                spec=next(x for x in bank['latent_files'] if x['branch']==branch and x['rank']==rank)
                p=Path(spec['path']);array=np.load(p,mmap_mode='r',allow_pickle=False)
                h=hashlib.sha256()
                with p.open('rb') as f:h.update(f.read(spec['header_bytes']))
                for offset in range(0,1250,request['block_rows']):
                    stop=min(offset+request['block_rows'],1250)
                    raw=array[offset:stop]
                    h.update(memoryview(raw).cast('B'));payload_bytes+=raw.nbytes
                    x=np.array(raw,dtype=np.float64).reshape(stop-offset,1024,256)
                    assert np.isfinite(x).all()
                    mu=x.mean(axis=-1,dtype=np.float64)
                    dc=mu*mu
                    raw_second=np.einsum('bcs,bcs->bc',x,x,optimize=False)/256.
                    x-=mu[...,None]
                    ac=np.einsum('bcs,bcs->bc',x,x,optimize=False)/256.
                    diff=np.abs(dc+ac-raw_second)
                    identity_error_max=max(identity_error_max,float(diff.max()))
                    rel=np.divide(diff,raw_second,out=np.zeros_like(diff),where=raw_second>0)
                    identity_relative_max=max(identity_relative_max,float(rel.max()))
                    checks+=diff.size
                    assert np.all(dc>=0) and np.all(ac>=0) and np.all(rel<1e-12)
                    sample_ids=rank+4*np.arange(offset,stop)
                    labels=v['labels'][sample_ids]
                    energies=np.stack([dc,ac],axis=1)
                    np.add.at(sums[ai],labels,energies)
                    removed=~keep[sample_ids]
                    np.add.at(excluded[ai],labels[removed],energies[removed])
                stat=p.stat()
                assert stat.st_size==spec['file_bytes'] and stat.st_mtime_ns==spec['mtime_ns']
                file_hashes.append({'path':str(p),'sha256':h.hexdigest(),'scope':'NPY header followed by original FP16 row bytes, complete file','bytes':spec['file_bytes']})
                print(json.dumps({'seed':bank['seed'],'branch':branch,'rank':rank,'payload_bytes_read':payload_bytes,'elapsed_seconds':time.perf_counter()-start}),flush=True)
        counts=np.stack([np.asarray(cohort['full_class_counts']),np.asarray(cohort['deduplicated_class_counts'])])
        class_energy=np.stack([sums/counts[0][None,:,None,None],(sums-excluded)/counts[1][None,:,None,None]])
        assert np.isfinite(class_energy).all() and np.all(class_energy>=0)
        class_residual=class_energy[:,1:]-class_energy[:,:1]
        global_energy=class_energy.mean(axis=2,dtype=np.float64)
        global_residual=global_energy[:,1:]-global_energy[:,:1]
        assert np.allclose(global_residual,class_residual.mean(axis=2),rtol=1e-10,atol=1e-13)
        path=HERE/f'seed{bank["seed"]}_energies.npz'
        np.savez(path,class_energy=class_energy,class_residual=class_residual,
                 global_equal_class_energy=global_energy,global_equal_class_residual=global_residual,
                 class_counts=counts,classes=np.arange(1000),channels=np.arange(1024),
                 sample_ids=v['sample_ids'],labels=v['labels'],source_rows=v['real_source_rows'],
                 deduplicated_keep=keep,excluded_ids=np.asarray(cohort['excluded_ids']),
                 views=np.asarray(VIEWS),arms=np.asarray(ARMS),residual_arms=np.asarray(ARMS[1:]),components=np.asarray(COMP),
                 class_energy_axes=np.asarray(['view','arm','class','component','channel']),
                 global_energy_axes=np.asarray(['view','arm','component','channel']))
        all_global.append(global_energy);all_residual.append(global_residual)
        banks_out.append({'seed':bank['seed'],'results':bank_summary(global_energy,global_residual),
                          'artifact':{'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size},
                          'class_counts_full':np.unique(counts[0],return_counts=True)[0].tolist(),
                          'deduplicated_samples':int(keep.sum()),'all_classes_retained':True,
                          'wall_seconds':time.perf_counter()-bank_start})
        del sums,excluded,class_energy,class_residual
    assert payload_bytes==request['payload_bytes_expected'] and len(file_hashes)==24
    cross={}
    for vi,view in enumerate(VIEWS):
        cross[view]={}
        for ai,arm in enumerate(ARMS[1:]):
            cross[view][arm]={}
            for ci,comp in enumerate(COMP):
                a,b=all_residual[0][vi,ai,ci],all_residual[1][vi,ai,ci]
                denom=float(np.linalg.norm(a)*np.linalg.norm(b))
                cross[view][arm][comp]={'residual_cosine':float(a@b/denom) if denom>0 else None,
                    'sign_agreement_fraction':float(np.mean(np.sign(a)==np.sign(b))),
                    'both_positive_channels':int(np.sum((a>0)&(b>0))),
                    'both_negative_channels':int(np.sum((a<0)&(b<0))),
                    'both_zero_channels':int(np.sum((a==0)&(b==0))),
                    'either_zero_channels':int(np.sum((a==0)|(b==0))),
                    'channel_count':1024}
    summary={'protocol':request['protocol'],'complete':True,'request_sha256':sha(HERE/'request.json'),
        'banks':banks_out,'cross_bank_residual_reproducibility':cross,'input_file_sha256':file_hashes,
        'verification':{'all_finite':True,'dc_ac_identity_image_channels_checked':checks,
            'dc_ac_identity_max_absolute_error':identity_error_max,'dc_ac_identity_max_relative_error':identity_relative_max,
            'all_original_and_deduplicated_classes_retained':True,'all_1024_channels_retained':True},
        'cost':{'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,
                'latent_payload_bytes_read':payload_bytes,'gpu_seconds':0,'model_calls':0},
        'scope':'Retrospective fixed DC/AC raw second energies, not centered covariance/eigenspectrum or guidance/FID. Source-overlap sensitivity is descriptive, not proof of independent estimators.'}
    dump('summary.json',summary)
    print(json.dumps({'complete':True,'summary':str(HERE/'summary.json'),'cost':summary['cost'],'cross_bank':cross}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','run'])
    args=parser.parse_args()
    prepare() if args.phase=='prepare' else run()
