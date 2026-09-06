from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments import audit_raev2_decoder_query_mean as audit
from stage2.models.DDT import DDTDecoderBlock, DiTwDDTHeadIG
from stage2.models.model_utils import NormAttention, RoPE


def test_postrope_query_mean_is_row_constant_frobenius_projection_without_renorm():
    q=torch.randn(2,3,16,8,dtype=torch.float64,generator=torch.Generator().manual_seed(9))
    original=q.clone();projected=audit.spatial_query_mean(q)
    assert torch.equal(q,original)
    assert torch.equal(projected,q.mean(-2,keepdim=True).expand_as(q))
    assert torch.equal(projected[:,:,:1].expand_as(projected),projected)
    assert torch.allclose(audit.spatial_query_mean(projected),projected,atol=2e-16,rtol=1e-15)
    arbitrary_constant=torch.randn(2,3,1,8,dtype=q.dtype).expand_as(q)
    assert abs(float(((q-projected)*arbitrary_constant).sum()))<1e-13
    assert projected.square().sum()<q.square().sum()


@pytest.mark.parametrize('bf16',(False,True))
def test_explicit_unmodified_attention_and_block_match_native_bitwise(bf16):
    torch.manual_seed(101)
    attention=NormAttention(16,2).eval();rope=RoPE(8,16)
    block=DDTDecoderBlock(16,2,mlp_ratio=2).eval()
    x=torch.randn(2,16,16);c=torch.randn_like(x)
    context=torch.autocast('cpu',dtype=torch.bfloat16) if bf16 else nullcontext()
    with torch.no_grad(),context:
        expected=attention(x,rope)
        actual,trace=audit.explicit_attention(attention,x,rope,capture=True)
        assert actual.dtype==expected.dtype and torch.equal(actual,expected)
        expected_block=block(x,c,rope)
        actual_block,_=audit.explicit_decoder_block(block,x,c,rope,capture=True)
        assert actual_block.dtype==expected_block.dtype and torch.equal(actual_block,expected_block)
        weak,weak_trace=audit.explicit_attention(attention,x,rope,query_mean=True,capture=True)
    assert torch.equal(trace['k'],weak_trace['k']) and torch.equal(trace['v'],weak_trace['v'])
    assert torch.equal(weak_trace['q_used'],audit.spatial_query_mean(trace['q_used']))
    assert not torch.equal(weak,expected)
    with pytest.raises(ValueError,match='masks'):
        audit.explicit_attention(attention,x,rope,query_mean=True,attn_mask=torch.zeros(2,1,16,16))


def make_tiny_model():
    torch.manual_seed(37)
    model=DiTwDDTHeadIG(input_size=4,in_channels=4,patch_size=[1,1],hidden_size=[16,16],
                       depth=[28,2],num_heads=[2,2],mlp_ratio=2,base_model_depth=8,
                       cond_arch=SimpleNamespace(num_t_tokens=2,num_c_tokens=2)).eval()
    with torch.no_grad():
        # DDT initialization zeros these paths; nonzero weights exercise parity and the weak replay.
        for index in (28,29):
            nn.init.normal_(model.blocks[index].adaln_modulation[-1].weight,std=.2)
            nn.init.normal_(model.blocks[index].adaln_modulation[-1].bias,std=.1)
        for layer in (model.final_layer.linear,model.base_final_layer.linear):
            nn.init.normal_(layer.weight,std=.2)
    return model


@pytest.mark.parametrize('bf16',(False,True))
def test_shared_encoder_full_base_parity_and_weak_second_keys_are_recomputed(bf16):
    model=make_tiny_model();state=torch.randn(2,4,4,4);times=torch.tensor([1.,.2]);labels=torch.tensor([3,9])
    context=lambda:torch.autocast('cpu',dtype=torch.bfloat16) if bf16 else nullcontext()
    with torch.no_grad(),context():
        native=model(state,times,context=labels,attn_mask=None)
    second_key_inputs=[]
    handle=model.blocks[29].attn.k.register_forward_pre_hook(lambda module,inputs:second_key_inputs.append(inputs[0].detach().clone()))
    with torch.no_grad(),context():
        heads,traces=audit.shared_full_base_weak(model,state,times,labels,capture=True)
    handle.remove()
    audit.assert_native_parity(heads,native,require_bf16=bf16)
    assert len(second_key_inputs)==2  # Full and weak each run their own second-block K linear.
    assert not torch.equal(second_key_inputs[0],second_key_inputs[1])
    assert torch.equal(traces['full'][0]['k'],traces['weak'][0]['k'])
    assert torch.equal(traces['full'][0]['v'],traces['weak'][0]['v'])
    assert not torch.equal(traces['full'][1]['k'],traces['weak'][1]['k'])
    assert not torch.equal(heads['full'],heads['weak'])
    for trace in traces['weak']:
        assert torch.equal(trace['q_used'],trace['q_used'][:,:,:1].expand_as(trace['q_used']))
    bad=dict(heads);bad['full']=heads['full']+1
    with pytest.raises(AssertionError,match='parity'):
        audit.assert_native_parity(bad,native,require_bf16=bf16)


@pytest.mark.parametrize('bf16',(False,True))
def test_observed_forward_counts_cover_explicit_branches_and_detect_missing_or_extra_calls(bf16):
    model=make_tiny_model();state=torch.randn(2,4,4,4);times=torch.tensor([1.,.2]);labels=torch.tensor([3,9])
    context=lambda:torch.autocast('cpu',dtype=torch.bfloat16) if bf16 else nullcontext()
    counter=audit.ForwardCallCounter(model)
    try:
        with torch.no_grad(),context():
            heads,_=audit.shared_full_base_weak(model,state,times,labels,capture=False,counter=counter)
            # Merely entering the reference phase cannot stand in for calling the model.
            with counter.phase('native_reference'):
                pass
            with pytest.raises(AssertionError,match='observed forward events differ'):
                counter.verify_batch()
            with counter.phase('native_reference'):
                native=model(state,times,context=labels,attn_mask=None)
        result=counter.verify_batch()
        audit.assert_native_parity(heads,native,require_bf16=bf16)
        assert result['observed']=={
            'shared_encoder_passes':1,'native_model_forward_calls':1,'encoder_block_calls':56,
            'explicit_full_decoder_block_calls':2,'weak_decoder_block_calls':2,
            'native_reference_decoder_block_calls':2,
            'full_final_readout_calls_including_weak_and_reference':3,
            'base_readout_calls_including_reference':2}
        for phase in ('full_decoder_and_readout','weak_decoder_and_readout','native_reference'):
            for block in (28,29):
                assert result['observed_module_events'][phase][f'decoder_{block}_q']==1
                assert result['observed_module_events'][phase][f'decoder_{block}_mlp']==1
        # An extra actual q call fails even though no helper or model call was added.
        with torch.no_grad(),counter.phase('weak_decoder_and_readout'):
            model.blocks[28].attn.q(torch.randn(2,16,16))
        with pytest.raises(AssertionError,match='observed forward events differ'):
            counter.verify_batch()
        assert result['observed_module_events']['weak_decoder_and_readout']['decoder_28_q']==1
        counter.reset()
        assert counter.events=={}
        with pytest.raises(AssertionError,match='observed forward events differ'):
            counter.verify_batch()
    finally:
        counter.close()
    # Removal restores ordinary calls with no scope requirement or numerical change.
    with torch.no_grad(),context():
        plain_native=model(state,times,context=labels,attn_mask=None)
    assert all(torch.equal(a,b) for a,b in zip(native,plain_native))
    assert counter.events=={} and not counter.handles


def test_uniform_queries_keep_nonuniform_key_probabilities_and_reverse_kl_barycenter():
    before=torch.tensor([[[[2.,0.],[0.,0.]]]])
    used=audit.spatial_query_mean(before)
    keys=torch.tensor([[[[2.,0.],[-2.,0.]]]])
    values=torch.eye(2).reshape(1,1,2,2)
    actual=F.scaled_dot_product_attention(used.bfloat16(),keys.bfloat16(),values.bfloat16())
    trace=audit.trace_to_cpu({'q_before':before,'q_used':used,'k':keys,'v':values,
                              'sdpa_output':actual,'query_mean':True})
    rows=audit.attention_statistics_cpu(trace)
    assert len(rows)==1
    row=rows[0]
    assert row['query_rows_bitwise_equal'] and row['actual_sdpa_output_rows_bitwise_equal']
    assert row['explicit_attention_probability_row_max_abs_difference']==0
    assert row['explicit_key_nonuniform_KL_p_to_uniform_mean_query']>.2
    assert row['implemented_row_to_effective_input_reverseKL_barycenter_L1']<1e-14
    logits=(before.double()@keys.double().transpose(-2,-1))/(2**.5)
    arithmetic=logits.softmax(-1).mean(-2)
    geometric=logits.mean(-2).softmax(-1)
    assert not torch.allclose(arithmetic,geometric)


def test_effective_bf16_barycenter_gap_is_reported_not_forced_to_zero():
    before=torch.tensor([[[[1.03125,.03125],[1.046875,.04],[-.39,.62]]]])
    keys=torch.tensor([[[[1.78,.23],[-.3,2.25],[.2,-1.2]]]])
    used=audit.spatial_query_mean(before);values=torch.ones_like(keys)
    actual=F.scaled_dot_product_attention(used.bfloat16(),keys.bfloat16(),values.bfloat16())
    row=audit.attention_statistics_cpu(audit.trace_to_cpu({'q_before':before,'q_used':used,'k':keys,
             'v':values,'sdpa_output':actual,'query_mean':True}))[0]
    assert row['implemented_row_to_effective_input_reverseKL_barycenter_L1']>0
    assert row['query_rows_bitwise_equal']


def test_direction_parallel_orthogonal_decomposition_and_zero_axis_conventions():
    full=torch.tensor([[[[3.,4.]]],[[[1.,2.]]],[[[2.,3.]]]])
    base=torch.tensor([[[[1.,4.]]],[[[1.,2.]]],[[[1.,3.]]]])
    weak=torch.tensor([[[[2.,1.]]],[[[0.,1.]]],[[[2.,3.]]]])
    rows,dw,db=audit.direction_statistics_cpu({'full':full,'base':base,'weak':weak})
    assert torch.equal(torch.from_numpy(dw),full-weak) and torch.equal(torch.from_numpy(db),full-base)
    assert rows[0]['descriptive_projection_coefficient_on_internal_gap']==.5
    assert rows[0]['parallel_energy_to_internal_gap']==1
    assert rows[0]['orthogonal_energy_to_internal_gap']==9
    assert rows[0]['parallel_energy_over_internal_gap_energy']==.25
    assert rows[0]['orthogonal_energy_over_internal_gap_energy']==2.25
    assert rows[1]['internal_gap_zero'] and rows[1]['direction_cosine'] is None
    assert rows[1]['parallel_energy_to_internal_gap'] is None
    assert rows[2]['weak_gap_zero'] and rows[2]['direction_cosine'] is None
    assert rows[2]['orthogonal_energy_to_internal_gap']==0
