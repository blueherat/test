"""Linearized Gaussian-posterior tilt by a frozen decoded-image critic."""
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import torch

DATA=Path('/home/zhoushunyu/data/eqvae/experiments')
PROBE=DATA/'raev2_guidance_restart_20260906/posterior_cls_probe_v1/probe.pt'
COVARIANCE=DATA/'raev2_guidance_20260907/exchangeable_covariance/covariance.npz'


@contextmanager
def exact_fp32():
    old_matmul=torch.backends.cuda.matmul.allow_tf32
    old_cudnn=torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        with torch.autocast('cuda',enabled=False):yield
    finally:
        torch.backends.cuda.matmul.allow_tf32=old_matmul
        torch.backends.cudnn.allow_tf32=old_cudnn


class ImageCritic:
    def __init__(self, decoder, encoder):
        self.decoder,self.encoder=decoder,encoder
        probe=torch.load(PROBE,map_location='cpu',weights_only=False)
        self.weight=probe['weight'].to(device='cuda',dtype=torch.float64)
        self.bias=float(probe['bias'])

    def logits_from_pixels(self, pixels):
        # True last CLS, matching the frozen probe; not the wrapper's pooled
        # multi-layer patch representation. No uint8 rounding in this surrogate.
        with exact_fp32():
            inputs=self.encoder.preprocess(pixels.float()*255)
            raw=self.encoder.model.forward_features(inputs)['x_norm_clstoken'].double()
            feature=raw/raw.norm(dim=-1,keepdim=True)
            return feature@self.weight+self.bias

    def logit(self, clean):
        with exact_fp32():
            pixels=self.decoder.decode(clean).float().clamp(0,1)
        return self.logits_from_pixels(pixels)

    def gradient(self, clean):
        with torch.enable_grad(),exact_fp32():
            x=clean.detach().requires_grad_(True)
            # Sum gives per-image gradients; mean would depend on microbatch.
            logit=self.logit(x)
            grad,=torch.autograd.grad(logit.sum(),x)
        return grad.detach(),logit.detach()


class ExchangeablePosterior:
    def __init__(self, device='cuda'):
        data=np.load(COVARIANCE)
        self.e0=torch.tensor(data['constant_values'],device=device,dtype=torch.float32)
        self.er=torch.tensor(data['residual_values'],device=device,dtype=torch.float32)
        self.v0=torch.tensor(data['constant_vectors'],device=device,dtype=torch.float32)
        self.vr=torch.tensor(data['residual_vectors'],device=device,dtype=torch.float32)

    def apply(self, grad, t, mse):
        a=1-t
        d0=self.e0*t*t/(t*t+a*a*self.e0)
        dr=self.er*t*t/(t*t+a*a*self.er)
        g=grad.flatten(2).transpose(1,2)
        n=g.shape[1]
        trace=(d0.sum()+(n-1)*dr.sum())/(n*len(d0))
        avg=g.mean(1,keepdim=True)
        constant=((avg@self.v0)*d0)@self.v0.T
        residual=(((g-avg)@self.vr)*dr)@self.vr.T
        return ((constant+residual)*(mse/trace)).transpose(1,2).reshape_as(grad)
