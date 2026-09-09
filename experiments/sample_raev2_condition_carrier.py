"""Paired native, shared-readout, and internal-condition guidance sampling."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import sample_raev2_pfr_retiming as native
from experiments.raev2_condition_carrier import representations, readout, adapted_condition


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = native.load_config(native.DEFAULT_CONFIG)
    native.install_raev2_decoder_config_compat()
    adapter_path = args.output.parent/'adapter.pt'
    request = dict(samples=args.samples, seed=202609413, batch=4, steps=100, gamma=.78,
                   adapter_sha256=native.file_sha256(adapter_path),
                   checkpoint_sha256=native.file_sha256(native.DEFAULT_CHECKPOINT),
                   sources={str(p):native.file_sha256(p) for p in [Path(__file__),Path(inspect.getfile(representations)),
                            Path(native.__file__),native.DEFAULT_CONFIG]},
                   arms=['native_ig','output_control','condition_carrier'], precision='BF16/TF32', continued_guidance=True)
    assert request['adapter_sha256'] == '00cb52c0fedce98fb5f1ceef9472d2ff1b392a4e2628853fa3234737517a5100'
    assert request['checkpoint_sha256'] == '723c56d7fa77ace9613909f7e38cb2386b898608218dc9b52649bb373d513c9a'
    (args.output/'request.json').write_text(json.dumps(request,indent=2)+'\n')
    adapter = {k:v.cuda() for k,v in torch.load(adapter_path,weights_only=True).items()}
    decoder = native.instantiate_from_config(cfg.stage_1).cuda().eval().requires_grad_(False)
    del decoder.encoder
    model = native.instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
    state = torch.load(native.DEFAULT_CHECKPOINT,map_location='cpu',mmap=True,weights_only=False)
    model.load_state_dict(state['ema'],strict=True)
    del state
    grid = native.shifted_time_grid(100,8.,torch.device('cuda'))
    for arm in request['arms']:
        dest = args.output/arm
        dest.mkdir()
        generator = torch.Generator(device='cuda').manual_seed(request['seed'])
        noise_hash,label_hash = hashlib.sha256(),hashlib.sha256()
        images = []
        calls = dict(encoder_samples=0,ddt_readout_samples=0,base_readout_samples=0,adapter_samples=0)
        start_time = time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for start in range(0,args.samples,4):
                count = min(4,args.samples-start)
                z = torch.randn(count,*cfg.misc.latent_size,generator=generator,device='cuda')
                labels = torch.arange(start,start+count,device='cuda')%1000
                noise_hash.update(z.cpu().numpy().tobytes())
                label_hash.update(labels.cpu().numpy().tobytes())
                for t,s in zip(grid[:-1],grid[1:]):
                    times = torch.full((count,),float(t),device='cuda')
                    active = float(cfg.guidance.ig.t_min)<=float(t)<=float(cfg.guidance.ig.t_max)
                    calls['encoder_samples'] += count
                    if arm == 'native_ig':
                        full,weak = model(z,times,context=labels,attn_mask=None)
                        calls['ddt_readout_samples'] += count
                        calls['base_readout_samples'] += count
                    else:
                        shallow,condition = representations(model,z,times,labels)
                        if active:
                            estimate = adapted_condition(shallow,adapter).to(condition.dtype)
                            calls['adapter_samples'] += count
                        if active and arm == 'condition_carrier':
                            condition = condition+.78*(condition-estimate)
                        full = readout(model,z,condition)
                        calls['ddt_readout_samples'] += count
                        if active and arm == 'output_control':
                            weak = readout(model,z,estimate)
                            calls['ddt_readout_samples'] += count
                    vf = native.clean_to_velocity(full,z,times,denominator_floor=float(cfg.transport.t_eps))
                    if active and arm != 'condition_carrier':
                        vw = native.clean_to_velocity(weak,z,times,denominator_floor=float(cfg.transport.t_eps))
                        drift = native.pfr_velocity(vf,vw,vw,guidance_scale=1.78,revision_scale=0.)
                    else:
                        drift = vf
                    z = z+(s-t)*drift
                assert torch.isfinite(z).all()
                images.append(decoder.decode(z).clamp(0,1).mul(255).permute(0,2,3,1).to('cpu',torch.uint8).numpy())
                if start == 0 or (start+count)%100 == 0:
                    print(json.dumps(dict(arm=arm,done=start+count,seconds=time.perf_counter()-start_time)),flush=True)
        np.savez(dest/'samples.npz',arr_0=np.concatenate(images))
        result = dict(complete=True,arm=arm,samples=args.samples,seconds=time.perf_counter()-start_time,
                      calls=calls,noise_sha256=noise_hash.hexdigest(),label_sha256=label_hash.hexdigest(),
                      request_sha256=native.file_sha256(args.output/'request.json'))
        (dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
