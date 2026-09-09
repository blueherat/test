"""RAEv2 differential temporal PFR with native ongoing IG."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch
from experiments import sample_raev2_pfr_retiming as native

ROOT = Path('/home/zhoushunyu/data/eqvae/experiments/ig_pfr_differential_fourcard_20260908')
REFERENCE = ROOT.parent/'ig_condition_carrier_20260908/quality/native_ig'


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, choices=range(4), required=True)
    args = parser.parse_args()
    args.arm = 'pfr_differential'
    out = ROOT/f'rank{args.rank}'
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = native.load_config(native.DEFAULT_CONFIG)
    native.install_raev2_decoder_config_compat()
    ckhash = native.file_sha256(native.DEFAULT_CHECKPOINT)
    assert ckhash == '723c56d7fa77ace9613909f7e38cb2386b898608218dc9b52649bb373d513c9a'
    model = native.instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
    state = torch.load(native.DEFAULT_CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(state['ema'], strict=True)
    del state
    decoder = native.instantiate_from_config(cfg.stage_1).cuda().eval().requires_grad_(False)
    del decoder.encoder
    sources = [Path(__file__), Path(native.__file__), Path(inspect.getfile(native.evaluate_base_head_only)),
               Path(inspect.getfile(type(model))), native.DEFAULT_CONFIG]
    request = dict(rank=args.rank, world_size=4, arm=args.arm, seed=202609413, batch=4, samples=1000,
                   steps=100, shift=8., gamma=.78,
                   carrier='(weak_t-weak_future)-(full_t-full_future)', horizon=1/32, revision_scale=1.0, revision_interval='t>.5', strength=1.0, solver='native Euler',
                   precision='FP32 state, BF16 autocast, TF32', continued_native_ig=True,
                   checkpoint_sha256=ckhash,
                   protocol_sha256=native.file_sha256(Path('docs/RAEV2_PFR_DIFFERENTIAL_20260908_ZH.md')),
                   sources={str(p):native.file_sha256(p) for p in sources})
    if (out/'request.json').exists():
        assert json.loads((out/'request.json').read_text()) == request
    else:
        (out/'request.json').write_text(json.dumps(request,indent=2)+'\n')
    counts = dict(full_batch_calls=0, prefix_batch_calls=0)
    labels = None

    grid = native.shifted_time_grid(100,8.,torch.device('cuda'))
    rng = torch.Generator(device='cuda').manual_seed(202609413)
    nh,lh = hashlib.sha256(),hashlib.sha256()
    started = time.perf_counter()
    files = []
    request_hash = native.file_sha256(out/'request.json')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for start in range(0,1000,4):
            z = torch.randn(4,*cfg.misc.latent_size,generator=rng,device='cuda')
            labels = torch.arange(start,start+4,device='cuda')
            noise_bytes,label_bytes = z.cpu().numpy().tobytes(),labels.cpu().numpy().tobytes()
            nh.update(noise_bytes); lh.update(label_bytes)
            if (start//4)%4 != args.rank:
                continue
            path = out/f'batch{start:04d}.npz'
            if path.exists():
                with np.load(path) as saved:
                    assert str(saved['request_sha256']) == request_hash
                    assert str(saved['noise_sha256']) == hashlib.sha256(noise_bytes).hexdigest()
                    np.testing.assert_array_equal(saved['labels'],np.arange(start,start+4))
            else:
                before = counts.copy()
                batch_started = time.perf_counter()
                for t,s in zip(grid[:-1],grid[1:]):
                    times = torch.full((4,),float(t),device='cuda')
                    f,b = model(z,times,context=labels,attn_mask=None)
                    counts['full_batch_calls'] += 1
                    active = float(cfg.guidance.ig.t_min)<=float(t)<=float(cfg.guidance.ig.t_max)
                    vf = native.clean_to_velocity(f,z,times,denominator_floor=float(cfg.transport.t_eps))
                    vb = native.clean_to_velocity(b,z,times,denominator_floor=float(cfg.transport.t_eps))
                    drift = native.pfr_velocity(vf,vb,vb,guidance_scale=1.78,revision_scale=0.) if active else vf
                    if active and float(t)>.5:
                        future = float(t)-min(1/32,float(t)-.5)
                        future_times = torch.full_like(times,future)
                        ff,bb = model(z,future_times,context=labels,attn_mask=None)
                        counts['full_batch_calls'] += 1
                        vff = native.clean_to_velocity(ff,z,future_times,denominator_floor=float(cfg.transport.t_eps))
                        vbb = native.clean_to_velocity(bb,z,future_times,denominator_floor=float(cfg.transport.t_eps))
                        revision = (vb-vbb)-(vf-vff)
                        drift = drift+1.78*revision
                    z = z+(s-t)*drift
                assert torch.isfinite(z).all()
                pixels = decoder.decode(z).clamp(0,1).mul(255).permute(0,2,3,1).to('cpu',torch.uint8).numpy()
                batch_counts = {k:counts[k]-before[k] for k in counts}
                assert batch_counts == dict(full_batch_calls=189,prefix_batch_calls=0)
                temp = path.with_suffix('.tmp')
                with temp.open('wb') as stream:
                    np.savez(stream,arr_0=pixels,labels=np.arange(start,start+4),
                             noise_sha256=hashlib.sha256(noise_bytes).hexdigest(),request_sha256=request_hash,
                             seconds=time.perf_counter()-batch_started,
                             full_batch_calls=189,prefix_batch_calls=0)
                temp.replace(path)
            files.append(dict(start=start,file=path.name,sha256=native.file_sha256(path)))
            if len(files)==1 or len(files)%10==0:
                progress=dict(rank=args.rank,done=len(files)*4,seconds=time.perf_counter()-started)
                (out/'progress.json').write_text(json.dumps(progress)+'\n')
                print(json.dumps(progress),flush=True)
    reference=json.loads((REFERENCE/'summary.json').read_text())
    assert nh.hexdigest()==reference['noise_sha256'] and lh.hexdigest()==reference['label_sha256']
    result=dict(complete=True,rank=args.rank,samples=len(files)*4,files=files,
                seconds=time.perf_counter()-started,calls_this_invocation=counts,
                noise_sha256=nh.hexdigest(),label_sha256=lh.hexdigest(),request_sha256=request_hash)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True)


if __name__ == '__main__':
    main()
