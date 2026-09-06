"""CPU-only directional check of the fixed two-mode correction.

This diagnostic is deliberately small and is not a quality/acceptance gate.
It uses one fixed cached image at each saved time, no window or gain search.
"""
import json
import os
from pathlib import Path
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'external/RAEv2/src'):sys.path.insert(0,str(p))
import torch
from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat
from experiments.sample_raev2_pfr_retiming import load_config,DEFAULT_CONFIG
from experiments.raev2_two_mode_ratio import two_mode_correction
from utils.model_utils import instantiate_from_config


def main():
    data=Path('/home/zhoushunyu/data/eqvae/experiments')
    root=data/'raev2_guidance_20260907'
    out=root/'two_mode_critic_cpu'
    out.mkdir(exist_ok=False)
    if torch.cuda.is_available():raise RuntimeError('CPU-only diagnostic')
    torch.set_num_threads(16)
    os.environ['DINOV3_CKPT_DIR']='/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3'
    os.environ['DINOV3_REPO_DIR']='/home/zhoushunyu/data/eqvae/models/RAEv2/dinov3_repo'
    install_raev2_decoder_config_compat()
    model=instantiate_from_config(load_config(DEFAULT_CONFIG).stage_1).cpu().eval().requires_grad_(False)
    probe=torch.load(data/'raev2_guidance_restart_20260906/posterior_cls_probe_v1/probe.pt',map_location='cpu',weights_only=False)
    w=probe['weight'].double();bias=float(probe['bias'])
    def logit(x):
        pixels=model.decode(x).clamp(0,1)
        inputs=model.encoder.preprocess(pixels*255)
        raw=model.encoder.model.forward_features(inputs)['x_norm_clstoken'].double()
        return (raw/raw.norm(dim=-1,keepdim=True))@w+bias
    calibration=json.loads((root/'two_mode_ratio/finite_euler_calibration.json').read_text())
    cache=data/'raev2_guidance_restart_20260906/normal_noise_audit_seed202609071/states'
    rows=[];start=time.perf_counter()
    for path in sorted(cache.glob('step_*.pt')):
        payload=torch.load(path,map_location='cpu',weights_only=False)
        t=float(payload['t']);step=int(payload['step_index'])
        f,b=[payload['rollout'][k][:1] for k in ['full','base']]
        g=(b.bfloat16()+1.78*(f.bfloat16()-b.bfloat16())).float() if t>=.1 else f
        delta=two_mode_correction(payload['rollout']['state'][:1],t,calibration)
        x=g.detach().requires_grad_(True)
        value=logit(x)
        grad,=torch.autograd.grad(value.sum(),x)
        with torch.no_grad():
            value_new=logit(g+delta)
            value_full=logit(f)
        row={'step':step,'t':t,'logit_native_clean':float(value.detach().item()),'logit_full_clean':float(value_full.item()),
            'logit_two_mode_clean':float(value_new.item()),'actual_two_mode_logit_gain':float((value_new-value.detach()).item()),
            'linear_two_mode_logit_gain':float((grad*delta).sum()),'correction_rms':float(delta.square().mean().sqrt()),
            'native_ig_rms':float((g-f).square().mean().sqrt()),'elapsed_seconds':time.perf_counter()-start}
        rows.append(row)
        (out/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    (out/'summary.json').write_text(json.dumps({'complete':True,'rows':rows,'cpu_threads':16,'cuda_used':False,
        'sample_limit':'one cached image at ten times, narrow numeric diagnostic, not a FID surrogate validation',
        'seconds':time.perf_counter()-start},indent=2)+'\n')


if __name__=='__main__':main()
