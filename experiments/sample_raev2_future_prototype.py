"""Future-prototype writeback with continued native IG, fixed 1K screen."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch
from experiments import sample_raev2_pfr_retiming as native

ROOT = Path('/home/zhoushunyu/data/eqvae/experiments/ig_future_prototype_20260908')
REFERENCE = ROOT.parent/'ig_condition_carrier_20260908/quality/native_ig'
ARMS = ['full_prototype', 'ig_prototype', 'balanced_ig_prototype', 'ordinary160']


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=ARMS, required=True)
    args = parser.parse_args()
    out = ROOT/args.arm
    out.mkdir(parents=True, exist_ok=False)
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
    request = dict(arm=args.arm, seed=202609413, batch=4, samples=1000,
                   steps=160 if args.arm == 'ordinary160' else 100, shift=8., gamma=.78,
                   events=[20,40,60,80], horizon_grid_steps=4, prototype_solver='Heun', inverse_solver='Heun',
                   precision='FP32 state, BF16 autocast, TF32', continued_native_ig=True,
                   checkpoint_sha256=ckhash,
                   protocol_sha256=native.file_sha256(Path('docs/IG_FUTURE_PROTOTYPE_METHOD_20260908_ZH.md')),
                   sources={str(p):native.file_sha256(p) for p in sources})
    (out/'request.json').write_text(json.dumps(request,indent=2)+'\n')
    counts = dict(full_batch_calls=0, prefix_batch_calls=0)
    labels = None

    def field(z, t, kind):
        times = torch.full((len(z),),float(t),device='cuda')
        if kind == 'base':
            base = native.evaluate_base_head_only(model,z,times,context=labels,attn_mask=None)
            counts['prefix_batch_calls'] += 1
            return native.clean_to_velocity(base,z,times,denominator_floor=float(cfg.transport.t_eps))
        full,base = model(z,times,context=labels,attn_mask=None)
        counts['full_batch_calls'] += 1
        vf = native.clean_to_velocity(full,z,times,denominator_floor=float(cfg.transport.t_eps))
        if kind == 'full' or not float(cfg.guidance.ig.t_min)<=float(t)<=float(cfg.guidance.ig.t_max):
            return vf
        vb = native.clean_to_velocity(base,z,times,denominator_floor=float(cfg.transport.t_eps))
        return native.pfr_velocity(vf,vb,vb,guidance_scale=1.78,revision_scale=0.)

    def flow(z, grid, kind):
        for t,s in zip(grid[:-1],grid[1:]):
            v = field(z,t,kind)
            predictor = z+(s-t)*v
            z = z+(s-t)*.5*(v+field(predictor,s,kind))
        return z

    def generate(number, steps, writeback):
        nonlocal labels
        grid = native.shifted_time_grid(steps,8.,torch.device('cuda'))
        rng = torch.Generator(device='cuda').manual_seed(request['seed'])
        nh,lh = hashlib.sha256(),hashlib.sha256()
        images,records = [],[]
        counts.update(full_batch_calls=0,prefix_batch_calls=0)
        started = time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for start in range(0,number,4):
                z = torch.randn(4,*cfg.misc.latent_size,generator=rng,device='cuda')
                labels = torch.arange(start,start+4,device='cuda')%1000
                nh.update(z.cpu().numpy().tobytes()); lh.update(labels.cpu().numpy().tobytes())
                for k,(t,s) in enumerate(zip(grid[:-1],grid[1:])):
                    if writeback and k in request['events']:
                        short = grid[k:k+5]
                        target = flow(z,short,'full' if args.arm == 'full_prototype' else 'ig')
                        encoded = flow(target,short.flip(0),'base')
                        if args.arm == 'balanced_ig_prototype':
                            roundtrip = flow(flow(z,short,'base'),short.flip(0),'base')
                            updated = z+(encoded-roundtrip)
                        else:
                            updated = encoded
                        assert torch.isfinite(updated).all(), f'nonfinite writeback at {start}, {k}'
                        if start < 8:
                            records.append(dict(start=start,step=k,time=float(t),future_time=float(short[-1]),
                                                write_rms=float((updated-z).square().mean().sqrt())))
                        z = updated
                    z = z+(s-t)*field(z,t,'ig')
                assert torch.isfinite(z).all()
                images.append(decoder.decode(z).clamp(0,1).mul(255).permute(0,2,3,1).to('cpu',torch.uint8).numpy())
                if number == 1000 and (start == 0 or (start+4)%100 == 0):
                    progress = dict(arm=args.arm,done=start+4,seconds=time.perf_counter()-started)
                    (out/'progress.json').write_text(json.dumps(progress)+'\n')
                    print(json.dumps(progress),flush=True)
        return np.concatenate(images),dict(seconds=time.perf_counter()-started,calls=counts.copy(),
                                            noise_sha256=nh.hexdigest(),label_sha256=lh.hexdigest(),records=records)

    # Check the optimized weak reader on a production-precision state before using it.
    with torch.autocast('cuda',dtype=torch.bfloat16):
        gen = torch.Generator(device='cuda').manual_seed(202609501)
        z = torch.randn(4,*cfg.misc.latent_size,generator=gen,device='cuda')
        labels = torch.tensor([0,142,713,999],device='cuda')
        times = torch.tensor([1.,.9,.55,.3],device='cuda')
        _,b = model(z,times,context=labels,attn_mask=None)
        bp = native.evaluate_base_head_only(model,z,times,context=labels,attn_mask=None)
        assert torch.equal(b,bp)
    parity_images,parity = generate(8,100,False)
    np.testing.assert_array_equal(parity_images,np.load(REFERENCE/'samples.npz')['arr_0'][:8])
    smoke,smoke_result = generate(8,request['steps'],args.arm != 'ordinary160')
    np.savez(out/'smoke.npz',arr_0=smoke)
    (out/'checks.json').write_text(json.dumps(dict(prefix_bitwise=True,native_eight_pixel_parity=True,
                                                  parity=parity,smoke=smoke_result),indent=2)+'\n')
    print(json.dumps(dict(arm=args.arm,smoke_passed=True)),flush=True)
    images,result = generate(1000,request['steps'],args.arm != 'ordinary160')
    np.testing.assert_array_equal(images[:8],smoke)
    reference = json.loads((REFERENCE/'summary.json').read_text())
    for key in ['noise_sha256','label_sha256']:
        assert result[key] == reference[key]
    expected_full = 160 if args.arm == 'ordinary160' else 132
    expected_prefix = 96 if args.arm == 'balanced_ig_prototype' else (0 if args.arm == 'ordinary160' else 32)
    assert result['calls'] == dict(full_batch_calls=250*expected_full,prefix_batch_calls=250*expected_prefix)
    np.savez(out/'samples.npz',arr_0=images)
    result.update(complete=True,arm=args.arm,samples=1000,request_sha256=native.file_sha256(out/'request.json'),
                  samples_sha256=native.file_sha256(out/'samples.npz'),smoke_pixel_parity=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
