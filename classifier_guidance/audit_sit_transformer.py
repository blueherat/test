"""Verify copied SiT blocks, velocity-channel layout and cross-patch interaction."""
import argparse
from pathlib import Path

import torch

from experiments.adversarial_weak_training_20260915 import common as c
from experiments.guidance_dynamic_50k_20260915.models import Adapter, fingerprint
from .capacity_heads import make, describe
from .evaluate_sit_transformer import ROOT, VARIANTS


@torch.no_grad()
def main(args):
    _,world=c.setup();assert world==1
    adapter=Adapter('sit_small');model=adapter.model;before=fingerprint(model)
    gen=torch.Generator(device='cuda').manual_seed(2026092101)
    z=torch.randn((8,4,32,32),device='cuda',generator=gen)
    t=torch.linspace(.05,.95,8,device='cuda');y=torch.arange(8,device='cuda')
    rows=[]
    for variant in ('linear','block1','block2'):
        head=make(variant,model).cuda().eval()
        expected_keys=[f'blocks.{i}' for i in range(len(head.blocks))]
        assert not {p.data_ptr() for p in head.parameters()} & {p.data_ptr() for p in model.parameters()}
        for copy_block,source_block in zip(head.blocks,list(model.blocks)[-len(head.blocks):] if len(head.blocks) else []):
            assert type(copy_block) is type(source_block)
            for name,value in copy_block.state_dict().items():assert torch.equal(value,source_block.state_dict()[name])
            assert all(not m._forward_hooks and not m._forward_pre_hooks for m in copy_block.modules())
        for name,value in head.final.state_dict().items():assert torch.equal(value,model.final_layer.state_dict()[name])
        for bf16 in (False,True):
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
                features=adapter.features(z,t,y);tokens=features['context'];condition=features['condition']
                expected=tokens
                for source_block in list(model.blocks)[-len(head.blocks):] if len(head.blocks) else []:
                    expected=source_block(expected,condition)
                # Independent reference: official unpatchify followed by channel split.
                expected=model.unpatchify(model.final_layer(expected,condition))[:,:4]
                actual=adapter.unpatch(head(tokens,condition))
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        features=adapter.features(z,t,y);tokens=features['context'];condition=features['condition']
        perturbed=tokens.clone();perturbed[:,17]+=torch.linspace(-1,1,384,device='cuda')
        change=float((head(tokens,condition)[:,0]-head(perturbed,condition)[:,0]).abs().max())
        assert (change>0) if len(head.blocks) else (change==0)
        rows.append(dict(variant=variant,architecture=describe(head),fp32_and_bf16_native_tail_exact=True,
            velocity_channel_layout_exact=True,parameters_independent=True,other_patch_effect_on_patch0=change))
    assert fingerprint(model)==before and all(not p.requires_grad for p in model.parameters())
    data=[c.read(ROOT/v/'preflight100/first_batches.json') for v in VARIANTS]
    assert all(value==data[0] for value in data)
    for variant in VARIANTS:
        root=ROOT/variant/'preflight100';latest=c.read(root/'latest.json')
        assert latest['step']==100 and latest['strong_unchanged'] and latest['replicas_identical']
        assert all(value>0 for value in latest['parameter_deltas'].values())
        assert c.read(root/'objective_audit.json')['ordinary_fm_exact']
    result=dict(passed=True,strong_unchanged=True,first_four_global_batches_identical=True,
        variants=list(VARIANTS),native_structure_audits=rows,all_parameter_groups_updated_in_100_steps=True)
    c.atomic(args.output/'result.json',result);c.atomic(args.output/'complete.json',dict(complete=True))
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=('sit_small',),default='sit_small')
    main(p.parse_args())
