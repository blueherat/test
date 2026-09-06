"""Frozen fixed-head arithmetic diagnostic. No model import or CUDA execution."""
import argparse, csv, datetime, hashlib, json, math, os, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = Path('/home/zhoushunyu/eqvae')
SRC = ROOT / 'normal_noise_audit_seed202609071'
PARITY = ROOT / 'depth_readout_audit_seed202609071'
SCALE = 1.78

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def prepare():
    start, cpu = time.perf_counter(), time.process_time()
    if (HERE / 'request.json').exists():
        raise FileExistsError('Request already frozen')
    source = json.loads((SRC / 'request.json').read_text())
    summary = json.loads((SRC / 'summary.json').read_text())
    assert summary['complete'] and source['precision'] == 'bf16' and source['tf32']
    assert len(summary['snapshots']) == 10
    artifacts = []
    for p in [SRC/'request.json', SRC/'summary.json', SRC/'runner_source.py',
              PARITY/'request.json', PARITY/'summary.json', PARITY/'runner_source.py',
              PARITY/'native_parity.csv', REPO/'experiments/audit_raev2_endpoint_adjoint_response.py',
              Path(__file__)]:
        artifacts.append({'path': str(p), 'sha256': sha(p)})
    for row in summary['snapshots']:
        assert sha(row['path']) == row['sha256'], row['path']
        artifacts.append({'path': row['path'], 'sha256': row['sha256']})
    parity = list(csv.DictReader((PARITY/'native_parity.csv').open()))
    assert len(parity) == 20
    for row in parity:
        assert all(row[k] == 'True' for k in row if k.endswith('_bitwise'))
    request = {
        'protocol': 'raev2_fixed_native_bf16_heads_guidance_arithmetic_v1',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'frozen_before_tensor_arithmetic', 'inputs': artifacts,
        'snapshots': summary['snapshots'], 'sample_ids': list(range(8)),
        'labels': list(range(8)), 'source_rows': source['source_rows'][:8],
        'shape': [8, 1024, 16, 16], 'domains': ['teacher', 'rollout'],
        'head_dtype': 'BF16 model outputs directly promoted to stored FP32; historical depth audit confirms native/saved bitwise parity',
        'source_state_protocol': 'seed202609071 B8, TF32 true, FP32 teacher states and FP32 historical F+0.78*(F-B) rollout; not current native BF16 rollout',
        'coefficient': SCALE, 'official_interval': [.1, 1.],
        'arithmetic': {
            'bf16': 'd=RN_BF16(F-B); m=RN_BF16(float32(d)*float32(1.78)); G=RN_BF16(B+m), eager CPU PyTorch; independent integer RNE reconstruction must match',
            'fp32': 'B32+float32(1.78)*(F32-B32), separate FP32 operations',
            'fp64': 'B64+float64(1.78)*(F64-B64), separate FP64 operations',
            'inactive': 'G=F for t<0.1; no extrapolation counterfactual outside fixed official interval',
            'error': 'e=G_bf16-G_fp64',
            'decomposition': 'e_sub=1.78*(d_bf16-d64); e_mul=m_bf16-1.78*d_bf16; e_add=G_bf16-(B64+m_bf16); terms zero outside interval',
            'sterbenz': 'sufficient condition: F,B same nonzero sign and 0.5<=abs(F)/abs(B)<=2; both-zero handled separately. Count exact BF16 subtraction, conditional violations and input subnormals.',
            'relative_error': 'per-image ||e||/||F-B|| and ||e||/||G64||; epsilon-free, zero denominators return null',
            'correlations': 'per-image uncentered cosine, coordinate-centered Pearson, and projection slope <e,v>/<v,v> for v=G64,B64,d64; zero denominator returns null',
            'rounding_baseline': 'RN_BF16(G64), via FP64->BF16; descriptive lower bound on representable BF16 absolute error; no changed sampler',
        },
        'aggregation': 'retain all 160 sample-domain-time rows and 20 batch means; summarize all 9 pre-existing active snapshots per domain. Only 8 unique IDs; repeated times and teacher/rollout not independent. t=1 domains duplicate. No SEM or generalization/FID claims.',
        'verification': 'finite BF16-lattice heads; saved identity; parity metadata; SHA; eager CPU BF16 operations versus integer RNE float32 reference at all points; additive error identity',
        'prohibitions': ['no GPU/model loading', 'no generation/FID/training', 'no scale/noise/time-window scan', 'no modification of existing cache/sampler'],
        'versions': {'torch': torch.__version__, 'numpy': np.__version__},
    }
    dump(HERE/'request.json', request)
    dump(HERE/'prepare_cost.json', {'wall_seconds': time.perf_counter()-start,
         'cpu_seconds': time.process_time()-cpu, 'model_calls': 0, 'gpu_seconds': 0,
         'request_sha256': sha(HERE/'request.json')})
    print(json.dumps({'request_sha256': sha(HERE/'request.json'), 'snapshots': 10}))

def rne_bf16_from_fp32(x):
    """IEEE finite round-to-nearest-even by FP32 bit arithmetic, returned FP32."""
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & np.uint32(1))) & np.uint32(0xffff0000)
    return rounded.view(np.float32)

def ratio(x, y):
    return float(x / y) if y > 0 else None

def metric(e, v):
    ev = float(e @ v)
    e2, v2 = float(e @ e), float(v @ v)
    ec, vc = e - e.mean(), v - v.mean()
    return {'cosine': ratio(ev, math.sqrt(e2*v2)),
            'pearson': ratio(float(ec @ vc), math.sqrt(float(ec @ ec)*float(vc @ vc))),
            'slope': ratio(ev, v2)}

def run():
    start, cpu = time.perf_counter(), time.process_time()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU only'
    request = json.loads((HERE/'request.json').read_text())
    assert not (HERE/'summary.json').exists(), 'Do not overwrite completed result'
    for item in request['inputs']:
        assert sha(item['path']) == item['sha256'], item['path']
    rows, batches = [], []
    checked = 0
    decomposition_max = 0.
    t1_equal = {}
    for spec in request['snapshots']:
        state = torch.load(spec['path'], map_location='cpu', weights_only=True)
        assert state['step_index'] == spec['step_index'] and state['t'] == spec['t']
        assert state['sample_ids'].tolist() == request['sample_ids']
        assert state['labels'].tolist() == request['labels']
        active = .1 <= state['t'] <= 1.
        if state['t'] == 1.:
            t1_equal = {k: torch.equal(state['teacher'][k], state['rollout'][k])
                        for k in ('state', 'full', 'base')}
            assert all(t1_equal.values())
        for domain in request['domains']:
            full, base = state[domain]['full'], state[domain]['base']
            assert list(full.shape) == request['shape'] and base.shape == full.shape
            for x in (full, base):
                assert x.dtype == torch.float32 and bool(torch.isfinite(x).all())
                assert torch.equal(x, x.to(torch.bfloat16).float())
            f16, b16 = full.bfloat16(), base.bfloat16()
            d16 = (f16-b16)
            m16 = d16*SCALE
            g16 = b16+m16 if active else f16
            f, b = full.numpy().reshape(8,-1), base.numpy().reshape(8,-1)
            dd = d16.float().numpy().reshape(8,-1)
            mm = m16.float().numpy().reshape(8,-1)
            gg = g16.float().numpy().reshape(8,-1)
            r_d = rne_bf16_from_fp32(f-b)
            r_m = rne_bf16_from_fp32(r_d*np.float32(SCALE))
            r_g = rne_bf16_from_fp32(b+r_m) if active else f
            for x,y in ((dd,r_d), (mm,r_m), (gg,r_g)):
                assert np.array_equal(x,y), 'CPU eager arithmetic differs from integer RNE reference'
                checked += x.size
            selected = []
            for j in range(8):
                f64, b64 = f[j].astype(np.float64), b[j].astype(np.float64)
                d64 = f64-b64
                g64 = b64+SCALE*d64 if active else f64
                g32 = b[j]+np.float32(SCALE)*(f[j]-b[j]) if active else f[j]
                e = gg[j].astype(np.float64)-g64
                nearest = torch.from_numpy(g64).bfloat16().float().numpy().astype(np.float64)
                nearest_error = nearest-g64
                sub = SCALE*(dd[j].astype(np.float64)-d64) if active else np.zeros_like(e)
                mul = mm[j].astype(np.float64)-SCALE*dd[j].astype(np.float64) if active else np.zeros_like(e)
                add = gg[j].astype(np.float64)-(b64+mm[j].astype(np.float64)) if active else np.zeros_like(e)
                decomposition_max = max(decomposition_max, float(np.max(np.abs(e-(sub+mul+add)))))
                fa, ba = np.abs(f64), np.abs(b64)
                sterbenz = ((f64>0)&(b64>0) | (f64<0)&(b64<0)) & (fa<=2*ba) & (ba<=2*fa)
                bothzero = (fa==0)&(ba==0)
                exact = dd[j].astype(np.float64)==d64
                assert np.all(exact[sterbenz | bothzero])
                input_subnormal = ((fa>0)&(fa<2.**-126)) | ((ba>0)&(ba<2.**-126))
                errnorm, gapnorm, gnorm = np.linalg.norm(e), np.linalg.norm(d64), np.linalg.norm(g64)
                row = {'step_index': spec['step_index'], 't': spec['t'], 'domain': domain,
                       'sample_id': j, 'label': j, 'source_row': request['source_rows'][j],
                       'active': active, 'dimensions': len(e),
                       'gap_rms': float(np.sqrt(np.mean(d64*d64))),
                       'g64_rms': float(np.sqrt(np.mean(g64*g64))),
                       'error_rms': float(np.sqrt(np.mean(e*e))), 'error_mean': float(e.mean()),
                       'error_max_abs': float(np.max(np.abs(e))),
                       'error_over_gap_norm': ratio(errnorm, gapnorm),
                       'error_over_g64_norm': ratio(errnorm, gnorm),
                       'fp32_error_rms': float(np.sqrt(np.mean((g32.astype(np.float64)-g64)**2))),
                       'fp32_error_over_gap_norm': ratio(np.linalg.norm(g32.astype(np.float64)-g64), gapnorm),
                       'nearest_bf16_error_rms': float(np.sqrt(np.mean(nearest_error*nearest_error))),
                       'native_vs_nearest_mismatch_fraction': float(np.mean(gg[j]!=nearest)),
                       'sterbenz_fraction': float(sterbenz.mean()),
                       'both_zero_fraction': float(bothzero.mean()),
                       'subtraction_exact_fraction': float(exact.mean()),
                       'sterbenz_violations': int(np.sum(~exact & sterbenz)),
                       'input_subnormal_count': int(input_subnormal.sum()),
                       'sub_error_rms': float(np.sqrt(np.mean(sub*sub))),
                       'mul_error_rms': float(np.sqrt(np.mean(mul*mul))),
                       'add_error_rms': float(np.sqrt(np.mean(add*add))),
                       'sub_add_covariance': float(np.mean(sub*add)),
                       'mul_add_covariance': float(np.mean(mul*add)),
                       'sub_mul_covariance': float(np.mean(sub*mul))}
                for name,v in [('g64',g64), ('base',b64), ('gap',d64)]:
                    row.update({f'error_{name}_{k}':v for k,v in metric(e,v).items()})
                rows.append(row)
                selected.append(row)
            fixed = ['step_index','t','domain','active']
            batch = {k:selected[0][k] for k in fixed}
            batch['sample_count'] = 8
            batch['arithmetic_means'] = {
                k: float(np.mean([r[k] for r in selected])) if selected[0][k] is not None else None
                for k in selected[0] if k not in fixed+['sample_id','label','source_row']}
            batches.append(batch)
    assert len(rows) == 160 and decomposition_max < 1e-12
    with (HERE/'per_image.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    dump(HERE/'per_snapshot.json', batches)
    all_active = {}
    for domain in request['domains']:
        subrows = [r for r in rows if r['domain']==domain and r['active']]
        keys = [k for k in rows[0] if k not in ['step_index','t','domain','active','sample_id','label','source_row']]
        all_active[domain] = {k: {'mean': float(np.mean([r[k] for r in subrows])),
                                    'min': float(min(r[k] for r in subrows)),
                                    'max': float(max(r[k] for r in subrows))} for k in keys}
    summary = {'complete': True, 'protocol': request['protocol'],
               'request_sha256': sha(HERE/'request.json'), 'rows':len(rows), 'unique_ids':8,
               'active_rows_per_domain':72, 'independent_experiments':0,
               'all_active_per_image_descriptive':all_active,
               'verification': {'input_sha_count': len(request['inputs']),
                                'rne_values_checked':checked,
                                'decomposition_max_absolute_error':decomposition_max,
                                't1_teacher_rollout_bitwise_equal':t1_equal,
                                'native_head_bf16_lattice_and_finite':True,
                                'cached_cuda_guided_output_available':False,
                                'gpu_arithmetic_parity_checked':False},
               'cost': {'wall_seconds':time.perf_counter()-start,
                        'cpu_seconds':time.process_time()-cpu, 'gpu_seconds':0, 'model_calls':0},
               'outputs':[{'path':str(HERE/x), 'sha256':sha(HERE/x)} for x in ['per_image.csv','per_snapshot.json']],
               'scope':'Fixed historical native BF16 heads, CPU arithmetic reconstruction; TF32-on historical FP32 rollout, eight IDs only. No sampler intervention or FID inference.'}
    dump(HERE/'summary.json', summary)
    print(json.dumps({'summary':str(HERE/'summary.json'), 'cost':summary['cost'],
                      'error_over_gap_norm':{d:all_active[d]['error_over_gap_norm'] for d in all_active},
                      'sterbenz_fraction':{d:all_active[d]['sterbenz_fraction'] for d in all_active}}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['prepare','run'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    prepare() if args.phase == 'prepare' else run()
