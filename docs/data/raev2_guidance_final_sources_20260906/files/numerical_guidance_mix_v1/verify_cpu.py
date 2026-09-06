"""Artifact/identity and additive error-energy verification; no tensors/models loaded."""
import csv, hashlib, json, math, time
from pathlib import Path

P=Path(__file__).resolve().parent
start,cpu=time.perf_counter(),time.process_time()
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''): h.update(chunk)
    return h.hexdigest()
r=json.loads((P/'request.json').read_text())
s=json.loads((P/'summary.json').read_text())
assert s['request_sha256']==sha(P/'request.json')
for row in r['inputs']+s['outputs']: assert sha(row['path'])==row['sha256']
source=json.loads((P.parent/'normal_noise_audit_seed202609071/request.json').read_text())
ddt=Path('/home/zhoushunyu/eqvae/external/RAEv2/src/stage2/models/DDT.py')
assert sha(ddt)==source['source_sha256']['external/RAEv2/src/stage2/models/DDT.py']
rows=list(csv.DictReader((P/'per_image.csv').open()))
assert len(rows)==160
assert len({(x['step_index'],x['domain'],x['sample_id']) for x in rows})==160
assert {int(x['sample_id']) for x in rows}==set(range(8))
energy_max=0.
for x in rows:
    j=int(x['sample_id'])
    assert int(x['label'])==j and int(x['source_row'])==r['source_rows'][j]
    assert int(x['sterbenz_violations'])==0 and int(x['input_subnormal_count'])==0
    energy=sum(float(x[k])**2 for k in ('sub_error_rms','mul_error_rms','add_error_rms'))
    energy+=2*sum(float(x[k]) for k in ('sub_add_covariance','mul_add_covariance','sub_mul_covariance'))
    energy_max=max(energy_max,abs(energy-float(x['error_rms'])**2))
    assert float(x['nearest_bf16_error_rms'])<=float(x['error_rms'])+1e-15
    if float(x['t'])<.1:
        assert float(x['error_rms'])==0 and x['error_g64_cosine']==''
    for k,v in x.items():
        if k.endswith(('_cosine','_pearson')) and v: assert abs(float(v))<=1+1e-12
assert energy_max<1e-18
for j in range(8):
    t0=[x for x in rows if x['step_index']=='0' and int(x['sample_id'])==j]
    assert len(t0)==2
    assert {k:v for k,v in t0[0].items() if k!='domain'}=={k:v for k,v in t0[1].items() if k!='domain'}
result={'complete':True,'kind':'separate artifact and energy-identity check; not independent researcher review',
        'input_output_sha_checks':len(r['inputs'])+len(s['outputs'])+2,
        'rows_checked':len(rows),'error_energy_identity_max_abs':energy_max,
        'DDT_source_matches_historical_sha':sha(ddt),
        't1_domain_rows_identical':True,'nearest_rounding_rms_never_exceeds_multistep_rms':True,
        'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,
        'gpu_seconds':0,'model_calls':0,'verifier_sha256':sha(__file__)}
(P/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
