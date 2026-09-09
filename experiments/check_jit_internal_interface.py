"""Check full-depth parity and trainable readout gradients on official CUDA implementation."""
import json
from pathlib import Path
import torch
from experiments import jit_internal_guidance as jig

torch.set_num_threads(2);torch.manual_seed(202609825)
m=jig.load_source('cuda');z=torch.randn(2,3,256,256,device='cuda');t=torch.tensor([.37,.8],device='cuda');y=torch.tensor([123,456],device='cuda')
checks={}
for enabled in [False,True]:
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=enabled):
  full=m(z,t,y);f,c=jig.features(m,z,t,y,(4,8,12));recovered=jig.unpatchify(m.final_layer(f['12'],c))
 assert torch.equal(full,recovered),(full-recovered).abs().max()
 checks['bf16' if enabled else 'fp32']=True
h=jig.Readouts().cuda()
with torch.autocast('cuda',dtype=torch.bfloat16):p=h(f,c)
loss=sum((x.float()-z).square().mean() for x in p.values());loss.backward()
assert all(torch.isfinite(v.grad).all() for v in h.parameters() if v.grad is not None)
assert all(v.grad is None for v in m.parameters())
r=dict(passed=True,full_head_bitwise_parity=checks,feature_shapes={k:list(v.shape) for k,v in f.items()},head_gradient_finite=True,backbone_frozen=True,source_parameters=sum(p.numel() for p in m.parameters()),head_parameters=sum(p.numel() for p in h.parameters()),note='Official model allocates attention bias on CUDA; CPU execution is unsupported without source changes.')
Path('experiments/results/terminal_defect_20260908/jit_transfer/implementation_check.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
