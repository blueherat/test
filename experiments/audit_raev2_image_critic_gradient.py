"""Check decoded critic gradients on fixed cached states before a FID screen."""
import json
import os
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'external/RAEv2/src'):sys.path.insert(0,str(p))
from experiments.raev2_image_critic_guidance import ImageCritic,ExchangeablePosterior,PROBE,COVARIANCE
from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat
from experiments.sample_raev2_pfr_retiming import load_config,DEFAULT_CONFIG
from experiments.raev2_training_core import file_sha256
from utils.model_utils import instantiate_from_config


@torch.no_grad()
def main():
    out=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_20260907/critic_gradient_audit')
    out.mkdir(exist_ok=False)
    os.environ['DINOV3_CKPT_DIR']='/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3'
    os.environ['DINOV3_REPO_DIR']='/home/zhoushunyu/data/eqvae/models/RAEv2/dinov3_repo'
    install_raev2_decoder_config_compat()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    decoder=instantiate_from_config(load_config(DEFAULT_CONFIG).stage_1).cuda().eval().requires_grad_(False)
    critic=ImageCritic(decoder,decoder.encoder)
    covariance=ExchangeablePosterior()
    cal=json.loads((out.parent/'guided_reverse_variance/calibration.json').read_text())['rows']
    cache=out.parent.parent/'raev2_guidance_restart_20260906/normal_noise_audit_seed202609071/states'
    records=[]
    for path in sorted(cache.glob('step_*.pt')):
        data=torch.load(path,map_location='cpu',weights_only=False)
        t=float(data['t']); step=int(data['step_index'])
        f,b=[data['rollout'][k][:1].cuda() for k in ('full','base')]
        clean=(b.bfloat16()+1.78*(f.bfloat16()-b.bfloat16())).float() if t>=.1 else f
        started=time.perf_counter()
        grad,logit=critic.gradient(clean)
        torch.cuda.synchronize()
        row={'step':step,'t':t,'gradient_rms':float(grad.square().mean().sqrt()),
             'logit':float(logit.item()),'gradient_seconds':time.perf_counter()-started,'corrections':{}}
        for mode,delta in [('isotropic',cal[step]['mse']*grad),('exchangeable',covariance.apply(grad,t,cal[step]['mse']))]:
            value=critic.logit(clean+delta)
            row['corrections'][mode]={'rms':float(delta.square().mean().sqrt()),'linear_logit_gain':float((delta*grad).sum()),
                'actual_logit_gain':float((value-logit).item()),'finite':bool(torch.isfinite(delta).all())}
        # Centered finite difference on a smooth FP32 decoder/DINO surrogate.
        direction=grad/grad.norm()
        fd=(critic.logit(clean+.01*direction)-critic.logit(clean-.01*direction))/.02
        row['directional_finite_difference']=float(fd.item())
        row['directional_autograd']=float(grad.norm())
        records.append(row)
        (out/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    (out/'summary.json').write_text(json.dumps({'complete':True,'records':records,'probe_sha256':file_sha256(PROBE),
        'covariance_sha256':file_sha256(COVARIANCE),'cache_note':'older rollout arithmetic; diagnostic only',
        'tf32':False,'precision':'FP32 decoder and DINO, FP64 feature normalization'},indent=2)+'\n')


if __name__=='__main__':main()
