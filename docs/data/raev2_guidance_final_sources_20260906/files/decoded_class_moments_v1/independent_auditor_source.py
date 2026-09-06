import time
wall_start, cpu_start = time.perf_counter(), time.process_time()
import json, hashlib, resource, os
from pathlib import Path
import numpy as np

OUT = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/decoded_class_moments_v1')
DEST = OUT/'independent_audit.json'
if DEST.exists(): raise FileExistsError(DEST)
def readj(path): return json.loads(Path(path).read_text())
hashed_bytes = 0
records = []
def digest(path):
    global hashed_bytes
    path = Path(path); result = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(8*1024*1024), b''):
            hashed_bytes += len(part); result.update(part)
    return result.hexdigest()
def verify(rec):
    path = Path(rec['path']); got = digest(path)
    assert got == rec['sha256'], ('hash',str(path))
    if 'bytes' in rec: assert path.stat().st_size == rec['bytes']
    if 'mtime_ns' in rec: assert path.stat().st_mtime_ns == rec['mtime_ns']
    records.append({'path':str(path),'sha256':got,'bytes':path.stat().st_size})
    return path
checks=0; scalars=0; maximum=0.; maximum_at=''; anova_max=0.
def check(value, expected, name, exact=False):
    global checks, scalars, maximum, maximum_at
    a,b=np.asarray(value),np.asarray(expected);assert a.shape==b.shape,(name,a.shape,b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all(),name
    diff=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))) if a.size else 0.
    if diff>maximum: maximum,maximum_at=diff,name
    if exact: assert np.array_equal(a,b),name
    else: np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-9,err_msg=name)
    checks+=1;scalars+=a.size

def sq(v,axis=None): return np.sum(v*v,axis=axis,dtype=np.float64)
def within_pairs(block):
    # Pairwise U-statistic, independent of producer's centered-residual route.
    return sum(sq(block[:,i]-block[:,j],axis=1) for i in range(5) for j in range(i+1,5))/20

def mean_inner_unbiased(block):
    # mean_{i != j} <x_i,x_j> estimates squared conditional mean.
    return (sq(block.sum(axis=1),axis=1)-sq(block,axis=(1,2)))/20

def describe_check(values, recorded, label):
    check(values.mean(),recorded['mean'],label+'/mean')
    check(values.std(ddof=1)/np.sqrt(len(values)),recorded['descriptive_class_sem'],label+'/sem')
    check((values<0).sum(),recorded['negative_classes'],label+'/negative',True)
    check(len(values),recorded['classes'],label+'/classes',True)

request=readj(OUT/'request.json');summary=readj(OUT/'summary.json')
assert summary['status']=='complete' and request['device']=='cpu' and request['model_calls']==0
verify(summary['request']); verify(summary['runner'])
for rec in request['source_records']:verify(rec)
raw_request=readj(verify(request['prior_raw_identity_audit']))
names=['source','reconstruction','full','ig']; stems=['source','real','scale_s1p000000','scale_s1p780000']
results=[]; feature_bytes=0; source_row_sets=[]
for bank in request['banks']:
    seed=bank['seed']; man=readj(verify(bank['manifest'])); prot_path=verify(bank['sample_protocol'])
    assert man['status']=='complete' and man['seed']==seed and man['samples']==5000
    assert man['world_size']==4 and man['sampler_steps']==100 and man['state_key']=='ema'
    assert man['precision']=='bf16' and man['ig_interval']==[.1,1.] and man['same_noise_and_labels_across_scales']
    old=next(x for x in raw_request['banks'] if x['seed']==seed)
    assert old['manifest_sha256']==bank['manifest']['sha256'] and old['protocol_sha256']==bank['sample_protocol']['sha256']
    with np.load(prot_path,allow_pickle=False) as p: prot={k:p[k].copy() for k in p.files}
    ids,labels,mask,source_rows=(prot[k] for k in ('sample_ids','labels','test_mask','real_source_rows'))
    check(ids,np.arange(5000),f'{seed}/ids',True);check(labels,ids%1000,f'{seed}/labels',True)
    check(mask,np.isin(labels,np.random.default_rng(seed+17).permutation(1000)[:200]),f'{seed}/split',True)
    assert len(np.unique(source_rows))==5000;source_row_sets.append(source_rows)
    class_ids=np.array([ids[labels==c] for c in range(1000)])
    raw_stats=verify(bank['prior_raw_class_statistics'])
    with np.load(raw_stats,allow_pickle=False) as old:
        for key,expect in [('labels',np.arange(1000)),('test_mask',mask[:1000]),('sample_ids',class_ids),('source_rows',source_rows[class_ids])]:
            check(old[key],expect,f'{seed}/raw_identity/{key}',True)
    # Load exactly the original FP32 feature shards and restore global IDs.
    recs={Path(x['path']).name:x for x in bank['features']}
    blocks=[]; arrays=[]
    for stem in stems:
        value=np.empty((5000,2048),np.float64)
        for rank in range(4):
            rec=recs[f'{stem}_rank{rank:02d}.npy'];path=verify(rec)
            a=np.load(path,allow_pickle=False)
            assert a.dtype==np.float32 and a.shape==(1250,2048) and np.isfinite(a).all()
            feature_bytes+=a.nbytes;value[rank+4*np.arange(1250)]=a
        arrays.append(value);blocks.append(value[class_ids])
    cm=np.stack([x.mean(axis=1) for x in blocks],axis=1)
    wc=np.stack([within_pairs(x) for x in blocks],axis=1)
    within_delta={};within_cross={};risk={};observed={}
    for a in range(4):
        for b in range(a+1,4):
            key=f'{names[b]}_minus_{names[a]}'
            delta=blocks[b]-blocks[a]
            wd=within_pairs(delta);within_delta[key]=wd
            within_cross[key]=(wc[:,a]+wc[:,b]-wd)/2
            observed[key]=sq(cm[:,b]-cm[:,a],axis=1)
            if (a,b) in [(0,1),(2,3)]: risk[key]=mean_inner_unbiased(delta)
            else: risk[key]=mean_inner_unbiased(blocks[a])+mean_inner_unbiased(blocks[b])-2*np.sum(cm[:,a]*cm[:,b],axis=1)
    produced=next(x for x in summary['outputs'] if x['seed']==seed)
    assert readj(OUT/f'seed{seed}_analysis.json')==produced
    stat_path=verify(produced['class_statistics'])
    with np.load(stat_path,allow_pickle=False) as z:
        check(cm,z['class_means'],f'{seed}/all_class_feature_means')
        check(wc,z['within'],f'{seed}/all_class_pairwise_within')
        check(class_ids,z['sample_ids_by_class'],f'{seed}/class_ids',True)
        check(source_rows[class_ids],z['source_rows_by_class'],f'{seed}/source_rows',True)
        for key in within_delta:
            check(within_delta[key],z[key+'_within_delta'],f'{seed}/{key}/Wdelta')
            check(within_cross[key],z[key+'_within_cross'],f'{seed}/{key}/Wcross')
    for split,take in [('train',~mask[:1000]),('heldout',mask[:1000]),('all',np.ones(1000,bool))]:
        target=produced['splits'][split];C=int(take.sum());N=5*C;alpha=5*(C-1)/(N-1)
        mu=cm[take].mean(axis=0);W=wc[take].mean(axis=0)
        B=sq(cm[take]-mu,axis=(0,2))/(C-1);Bc=B-W/5
        # Pooled trace independently from sums and squared norms, no covariance matrix.
        pooled=[]
        for a in arrays:
            v=a[take[labels]];pooled.append((sq(v)-sq(v.sum(axis=0))/N)/(N-1))
        T=np.array(pooled);anova=np.abs(T-W-alpha*Bc);anova_max=max(anova_max,float(anova.max()))
        check(T,W+alpha*Bc,f'{seed}/{split}/ANOVA')
        for i,name in enumerate(names):
            for field,v in [('pooled_trace',T[i]),('within_trace',W[i]),('centroid_scatter_observed',B[i]),('centroid_scatter_corrected',Bc[i]),('uniform_class_between_trace_corrected',(C-1)/C*Bc[i])]:
                check(v,target['branches'][name][field],f'{seed}/{split}/{name}/{field}')
        for ref,r in [('source',0),('reconstruction',1)]:
            for i,name in enumerate(names):
                if i==r:continue
                for field,v in [('pooled_trace',T),('within_trace',W),('centroid_scatter_corrected',Bc)]:
                    check(v[i]/v[r],target['ratios'][ref][name][field],f'{seed}/{split}/ratio/{ref}/{name}/{field}')
        for a in range(4):
            for b in range(a+1,4):
                key=f'{names[b]}_minus_{names[a]}';d=target['distances'][key]
                obs=observed[key][take];rr=risk[key][take]
                noise=within_delta[key][take] if (a,b) in [(0,1),(2,3)] else wc[take,a]+wc[take,b]
                describe_check(obs,d['centroid_squared_distance_observed'],f'{seed}/{split}/{key}/obs')
                describe_check(rr,d['centroid_squared_distance_corrected'],f'{seed}/{split}/{key}/risk')
                check(obs-rr,noise/5,f'{seed}/{split}/{key}/noise_identity')
                check(noise.mean()/5,d['finite_m_noise_subtraction_mean'],f'{seed}/{split}/{key}/noise')
                gd=sq(mu[b]-mu[a]);check(gd,d['global_mean_squared_distance'],f'{seed}/{split}/{key}/global')
                check(gd-noise.mean()/N,d['global_mean_squared_distance_corrected'],f'{seed}/{split}/{key}/global_corrected')
        ch=target['ig_minus_full'];delta=cm[take,3]-cm[take,2]
        for key,v in [('pooled_ig_minus_full',T[3]-T[2]),('within_ig_minus_full',W[3]-W[2]),('between_corrected_contribution_ig_minus_full',alpha*(Bc[3]-Bc[2])),('between_multiplier',alpha)]:
            check(v,ch[key],f'{seed}/{split}/{key}')
        describe_check(wc[take,3]-wc[take,2],ch['within_change_by_class'],f'{seed}/{split}/within_change')
        wfi=within_cross['ig_minus_full'][take].mean()
        bfi=np.sum((cm[take,2]-mu[2])*(cm[take,3]-mu[3]))/(C-1)-wfi/5
        check(wfi/np.sqrt(W[2]*W[3]),ch['within_full_ig_pair_alignment'],f'{seed}/{split}/within_alignment')
        check(bfi/np.sqrt(Bc[2]*Bc[3]),ch['centroid_full_ig_alignment_corrected'],f'{seed}/{split}/centroid_alignment')
        entry={'seed':seed,'split':split,'classes':C,'trace_source_recon_full_ig':T.tolist(),'within_source_recon_full_ig':W.tolist(),'corrected_between_source_recon_full_ig':Bc.tolist(),'delta_T':float(T[3]-T[2]),'delta_W':float(W[3]-W[2]),'delta_between_contribution':float(alpha*(Bc[3]-Bc[2])),'ratios_to_source':{name:{'pooled':float(T[i]/T[0]),'within':float(W[i]/W[0]),'between_corrected':float(Bc[i]/Bc[0])} for i,name in [(2,'full'),(3,'ig')]},'centroid_risk':{}}
        for ref,r in [('source',0),('reconstruction',1)]:
            diff=(risk[f'ig_minus_{ref}']-risk[f'full_minus_{ref}'])[take]
            describe_check(diff,ch['centroid_risk_ig_minus_full_relative_to_'+ref],f'{seed}/{split}/risk_change/{ref}')
            cross=2*np.sum((cm[take,2]-cm[take,r])*delta,axis=1)-2*(within_cross['ig_minus_full'][take]-wc[take,2])/5
            dr=risk['ig_minus_full'][take]
            check(diff,cross+dr,f'{seed}/{split}/risk_decomposition/{ref}')
            d=ch['centroid_risk_decomposition_relative_to_'+ref]
            describe_check(cross,d['twice_error_shift_cross_corrected'],f'{seed}/{split}/risk_cross/{ref}')
            describe_check(dr,d['shift_norm_squared_corrected'],f'{seed}/{split}/risk_shift/{ref}')
            for key,v in [('pooled',T),('within',W),('between_corrected',Bc)]:
                for label,i in [('full',2),('ig',3)]:check(abs(v[i]-v[r]),ch[key+'_absolute_error_relative_to_'+ref][label],f'{seed}/{split}/error/{ref}/{key}/{label}')
            entry['centroid_risk'][ref]={'full':float(risk[f'full_minus_{ref}'][take].mean()),'ig':float(risk[f'ig_minus_{ref}'][take].mean()),'change':float(diff.mean()),'descriptive_class_sem':float(diff.std(ddof=1)/np.sqrt(C)),'negative_classes':int((diff<0).sum())}
            entry['all_three_trace_errors_closer_to_'+ref]=bool(all(abs(v[3]-v[r])<abs(v[2]-v[r]) for v in [T,W,Bc]))
        effect=-2*np.sum(delta*(cm[take,0]-cm[take,1]),axis=1)
        actual=((risk['ig_minus_source']-risk['full_minus_source'])-(risk['ig_minus_reconstruction']-risk['full_minus_reconstruction']))[take]
        check(actual,effect,f'{seed}/{split}/reference_effect_identity')
        describe_check(effect,ch['centroid_risk_reference_effect_source_minus_reconstruction'],f'{seed}/{split}/reference_effect')
        entry['source_minus_recon_risk_change']=float(effect.mean());results.append(entry)
        print(json.dumps(entry),flush=True)

result={'protocol':'independent_raev2_decoded_class_moment_audit_v1','complete':True,'producer_summary':{'path':str(OUT/'summary.json'),'sha256':digest(OUT/'summary.json')},'auditor_source':{'path':__file__,'sha256':digest(__file__)},'numeric_checks':checks,'numeric_values_compared':scalars,'maximum_absolute_difference':maximum,'maximum_difference_at':maximum_at,'maximum_direct_ANOVA_absolute_error':anova_max,'input_hash_records':records,'results':results,'findings':['No numerical or identity mismatch found in the six seed/split combinations.','All six have positive pooled change, negative within change, and positive corrected-between contribution. All three trace errors also get closer to both original-source and reconstruction references.','Therefore pooled positive change comes from between-class expansion, but saying not within-class recovery is incorrect: Full is within-overdispersed in decoded Inception, and the within contraction also repairs that error.','Both reference choices give negative corrected centroid-risk change in all six combinations; descriptive per-class heterogeneity is retained, not a formal confidence interval.'],'boundaries':['Read actual FP32 feature caches and independently reconstruct class statistics using pairwise/U-statistic identities; no producer functions imported.','Original source rows/labels/splits align to the prior raw audit and its manifest/protocol hashes. Raw latents and original RGB/parquet metadata were not reread; prior 10000 source-label checks are reused.','Source/reconstruction share actual source images; Full/IG share generation noise. Cross-family centroid risks use independent conditional sampling assumptions.','Current feature content hashes are verified against this audit request; original manifests do not contain per-feature content hashes, so this does not independently reconstruct their historical extraction.','Original stored uint8-image Inception pipeline includes decoder, clamp/quantization and Inception; conclusions are not decoder-only or a FID decomposition.','Old banks and splits are retrospective. No new FID, guidance direction, semantic-mode or quality guarantee.'],'cost':{'cpu_only':True,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu_calls':0,'raw_latents_read':False,'feature_payload_bytes':feature_bytes,'file_bytes_hashed':hashed_bytes,'wall_seconds_from_script_start':time.perf_counter()-wall_start,'cpu_seconds_from_script_start':time.process_time()-cpu_start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'timing_excludes':'interpreter startup before first time import and final audit JSON serialization/exit'}}
tmp=DEST.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(DEST)
print(json.dumps({'audit':str(DEST),'numeric_checks':checks,'numeric_values_compared':scalars,'max_error':maximum,'max_anova':anova_max,'cost':result['cost']}),flush=True)
