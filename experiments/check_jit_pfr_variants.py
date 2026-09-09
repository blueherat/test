"""Check historical PFR pixel parity and correction scale on four fixed inputs."""
import numpy as np
import torch
from experiments.sample_jit_pfr_variants_shard import ROOT,TRAIN,BASE,jig,atomic
@torch.inference_mode()
def main():
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 model=jig.load_source('cuda');heads=jig.Readouts().cuda().eval();heads.load_state_dict(torch.load(TRAIN/'last.pt',map_location='cpu',weights_only=False)['ema'])
 rng=torch.Generator(device='cuda').manual_seed(202609831);initial=torch.randn(4,3,256,256,device='cuda',generator=rng);y=torch.arange(4,device='cuda');grid=torch.linspace(0,1,101,device='cuda');records=[]
 def pair(z,t):
  feats,c=jig.features(model,z,t.expand(4),y,depths=(4,12));f=jig.unpatchify(model.final_layer(feats['12'],c));w=jig.unpatchify(heads.layers['4'](feats['4'],c));return jig.velocity(f,z,t.expand(4)),jig.velocity(w,z,t.expand(4))
 def weak(z,t):
  feats,c=jig.features(model,z,t.expand(4),y,depths=(4,));w=jig.unpatchify(heads.layers['4'](feats['4'],c));return jig.velocity(w,z,t.expand(4))
 def rms(x):return x.square().flatten(1).mean(1).sqrt()
 with torch.autocast('cuda',dtype=torch.bfloat16):
  for kind in ['historical_pfr','ig_audit']:
   z=initial.clone()
   for k,(t,u) in enumerate(zip(grid[:-1],grid[1:])):
    if k<50:
     f,w=pair(z,t);v=f+.3*(f-w);h=min(1/32,.5-float(t));c=1.3*(f-w);coef=(c*v).flatten(1).sum(1)/v.square().flatten(1).sum(1).clamp_min(1e-20);q=z+h*coef.clamp_min(0)[:,None,None,None]*v;wf=weak(q,t+t.new_tensor(h));delta=1.3*(w-wf)
     assert torch.equal(v+0.*delta,v)
     if kind=='ig_audit' and k in [0,5,10,20,30,40,49]:
      wt=weak(z,t+t.new_tensor(h));records.append(dict(step=k,time=float(t),projected_revision_over_ig_rms=(rms(delta)/rms(v)).tolist(),time_revision_over_ig_rms=(rms(1.3*(w-wt))/rms(v)).tolist(),shift_rms=rms(q-z).tolist(),ray_coefficient=coef.clamp_min(0).tolist()))
     if kind=='historical_pfr':v=v+delta
    else:v=jig.velocity(model(z,t.expand(4),y),z,t.expand(4))
    z=z+(u-t)*v
   pix=np.round(np.clip(((z+1)/2).cpu().numpy().transpose(0,2,3,1)*255,0,255)).astype(np.uint8)
   arm='pfr_e0.300_l0.000_m1_n1000_s202609831' if kind=='historical_pfr' else 'ig_e0.300_l0.000_m1_n1000_s202609831'
   with np.load(ROOT.parent/'jit_fine_sweep_20260909'/arm/'rank0/batch0000.npz') as old:np.testing.assert_array_equal(pix,old['arr_0'])
 atomic(ROOT/'preflight.json',dict(passed=True,historical_pfr_first4_pixel_parity=True,selected_ig_first4_pixel_parity=True,rho_zero_exact=True,records=records,note='Four-image diagnostic, not a quality result. Future clean converted at actual q,t+h; data time increases.'))
 print('PFR and IG historical pixels match; rho0 identity passed',flush=True)
if __name__=='__main__':main()
