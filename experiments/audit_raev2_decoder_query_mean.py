#!/usr/bin/env python3
"""Fixed decoder-query-mean structure audit on existing states; never a sampler.

Only the two decoder attention query tensors are replaced by their spatial
token means, after q RMSNorm and RoPE. Encoder, keys, values, softmax, residuals,
modulation and Full readout are retained. The second weak decoder recomputes
its Q/K/V from its own first-block output.

Prepare freezes a root-supplied protocol plan and all inputs without CUDA.
Run consumes that request on one visible GPU only, after external review.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT/'external/RAEv2/src', ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.audit_raev2_proximal_calibration import atomic_json, sha256_file, write_csv
from experiments.sample_raev2_pfr_retiming import DEFAULT_CONFIG, DEFAULT_CHECKPOINT, load_config
from stage2.models.DDT import modulate

PROTOCOL = 'raev2_fixed_decoder_postrope_query_mean_structure_audit_v1'
RESTART = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906')
DEFAULT_STATES = RESTART/'normal_noise_audit_seed202609071/states'
PROTOCOL_DOCUMENT = ROOT/'docs/RAEV2_DECODER_QUERY_MEAN_PROTOCOL_20260906_ZH.md'
CONFIG_SHA256 = '3062762f2f0f12e0d4b64b074fc5b45628e5022937bbf7857cc6dc6e2720d342'
CHECKPOINT_SHA256 = '723c56d7fa77ace9613909f7e38cb2386b898608218dc9b52649bb373d513c9a'
SOURCE_FILES = (
    'experiments/audit_raev2_decoder_query_mean.py',
    'tests/test_raev2_decoder_query_mean.py',
    'experiments/sample_raev2_pfr_retiming.py',
    'experiments/audit_raev2_proximal_calibration.py',
    'external/RAEv2/src/stage2/models/DDT.py',
    'external/RAEv2/src/stage2/models/model_utils.py',
    'external/RAEv2/src/utils/model_utils.py',
)


def artifact(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256_file(path), 'bytes': path.stat().st_size}


class ForwardCallCounter:
    """Instance-local successful-forward hooks; no tensor mutation or global patch.

    Decoder blocks are invoked explicitly, so observe both their q linear and
    MLP calls instead of a block.forward hook that would miss those branches.
    Encoder blocks and final heads use their actual module forward hooks.
    """
    def __init__(self, model):
        self.current_phase = None
        self.events = {}
        self.handles = []
        modules = [('model', model), ('s_embedder', model.s_embedder),
                   ('full_readout', model.final_layer), ('base_readout', model.base_final_layer)]
        modules += [(f'encoder_block_{i:02d}', model.blocks[i]) for i in range(28)]
        for i in range(28, 30):
            modules += [(f'decoder_{i:02d}_q', model.blocks[i].attn.q),
                        (f'decoder_{i:02d}_mlp', model.blocks[i].mlp)]
        for name, module in modules:
            def observe(_module, _inputs, _output, event=name):
                if self.current_phase is None:
                    raise AssertionError(f'unscoped observed forward: {event}')
                phase = self.events.setdefault(self.current_phase, {})
                phase[event] = phase.get(event, 0) + 1
            self.handles.append(module.register_forward_hook(observe))

    @contextmanager
    def phase(self, name):
        if self.current_phase is not None:
            raise AssertionError('forward counter phases cannot be nested')
        self.current_phase = name
        try:
            yield
        finally:
            self.current_phase = None

    def reset(self):
        if self.current_phase is not None:
            raise AssertionError('cannot reset an active forward phase')
        self.events = {}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    @staticmethod
    def expected_events():
        encoder = {'s_embedder': 1, **{f'encoder_block_{i:02d}': 1 for i in range(28)}}
        decoder = {f'decoder_{i:02d}_{part}': 1 for i in (28, 29) for part in ('q', 'mlp')}
        return {'shared_encoder': encoder,
                'full_decoder_and_readout': {**decoder, 'full_readout': 1},
                'base_readout': {'base_readout': 1},
                'weak_decoder_and_readout': {**decoder, 'full_readout': 1},
                'native_reference': {'model': 1, **encoder, **decoder, 'full_readout': 1, 'base_readout': 1}}

    @staticmethod
    def totals(events):
        def count(phase, name):
            return events.get(phase, {}).get(name, 0)
        return {
            'shared_encoder_passes': count('shared_encoder', 's_embedder'),
            'native_model_forward_calls': count('native_reference', 'model'),
            'encoder_block_calls': sum(v for phase in events.values() for k, v in phase.items() if k.startswith('encoder_block_')),
            'explicit_full_decoder_block_calls': sum(count('full_decoder_and_readout', f'decoder_{i:02d}_q') for i in (28, 29)),
            'weak_decoder_block_calls': sum(count('weak_decoder_and_readout', f'decoder_{i:02d}_q') for i in (28, 29)),
            'native_reference_decoder_block_calls': sum(count('native_reference', f'decoder_{i:02d}_q') for i in (28, 29)),
            'full_final_readout_calls_including_weak_and_reference': sum(phase.get('full_readout', 0) for phase in events.values()),
            'base_readout_calls_including_reference': sum(phase.get('base_readout', 0) for phase in events.values())}

    def verify_batch(self):
        expected = self.expected_events()
        if self.events != expected:
            raise AssertionError(f'observed forward events differ: observed={self.events}, expected={expected}')
        return {'observed_module_events': {p: dict(v) for p, v in self.events.items()},
                'observed': self.totals(self.events), 'expected': self.totals(expected),
                'all_module_events_match': True}


@contextmanager
def timed_stage(costs, name, device):
    """Explicit CUDA stream-span accounting; diagnostics have separate stages."""
    if costs is None:
        yield
        return
    cuda = torch.device(device).type == 'cuda'
    if cuda:
        torch.cuda.synchronize(device)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
    started = time.perf_counter()
    yield
    if cuda:
        end.record()
        torch.cuda.synchronize(device)
    costs[name] = {'wall_seconds': time.perf_counter()-started,
                   'cuda_event_span_seconds': begin.elapsed_time(end)/1000 if cuda else None}


def spatial_query_mean(query):
    """JQ on post-RoPE [batch,head,spatial_token,head_dim]; no renormalization.

    Preserve native dtype/layout by copying the broadcast mean into a clone.
    This also retains an unmodified input for separate diagnostics.
    """
    if query.ndim != 4 or min(query.shape) <= 0:
        raise ValueError('nonempty [B,H,N,Dh] queries required')
    result = query.clone(memory_format=torch.preserve_format)
    result.copy_(query.mean(dim=-2, keepdim=True).expand_as(query))
    return result


def explicit_attention(attention, x, rope, *, query_mean=False, attn_mask=None, capture=False):
    """Same operator order as NormAttention.forward, plus only optional post-RoPE JQ."""
    if query_mean and attn_mask is not None:
        raise ValueError('fixed decoder query-mean audit forbids attention masks')
    batch, tokens, _ = x.shape
    q = attention.q(x).reshape(batch, tokens, attention.num_heads, attention.head_dim).permute(0, 2, 1, 3)
    k = attention.k(x).reshape(batch, tokens, attention.num_heads, attention.head_dim).permute(0, 2, 1, 3)
    v = attention.v(x).reshape(batch, tokens, attention.num_heads, attention.head_dim).permute(0, 2, 1, 3)
    q = attention.q_norm(q)
    k = attention.k_norm(k)
    q, k = rope(q), rope(k)
    q_before = q
    if query_mean:
        q = spatial_query_mean(q)
    # The production output ALWAYS uses the original SDPA operator.
    raw_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    out = raw_output.permute(0, 2, 1, 3).reshape(batch, tokens, attention.dim)
    result = attention.proj(out)
    trace = ({'q_before': q_before.detach(), 'q_used': q.detach(), 'k': k.detach(),
              'v': v.detach(), 'sdpa_output': raw_output.detach(),
              'query_mean': query_mean} if capture else None)
    return result, trace


def explicit_decoder_block(block, x, conditioning, rope, *, query_mean=False, capture=False):
    """Copy DDTDecoderBlock's native modulation/residual/MLP ordering exactly."""
    modulation = block.adaln_modulation(conditioning)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
    update, trace = explicit_attention(block.attn, modulate(block.norm1(x), shift_msa, scale_msa),
                                       rope, query_mean=query_mean, capture=capture)
    x = x + gate_msa*update
    x = x + gate_mlp*block.mlp(modulate(block.norm2(x), shift_mlp, scale_mlp))
    return x, trace


@torch.no_grad()
def shared_full_base_weak(model, state, times, labels, *, capture=True, costs=None, counter=None):
    """One native encoder, original Full then Base, followed by weak decoder replay."""
    if model.training or (model.num_enc_blocks, model.num_dec_blocks, model.base_model_depth) != (28, 2, 8):
        raise ValueError('frozen eval-mode DDT with 28 encoder/2 decoder/base depth8 required')
    if model.use_cfg_conds:
        raise ValueError('extra CFG tokens are outside the frozen structure')
    conditions = {'context': labels, 'attn_mask': None}
    count_phase = lambda name: counter.phase(name) if counter is not None else nullcontext()
    with timed_stage(costs, 'shared_encoder', state.device), count_phase('shared_encoder'):
        seq, t_emb_base = model._build_sequence(state, times, conditions)
        mask = model._build_attn_mask(seq, conditions)
        h8 = None
        for index in range(model.num_enc_blocks):
            seq = model.blocks[index](seq, model.enc_rope, mask)
            if index+1 == model.base_model_depth:
                h8 = seq[:, :model.s_embedder.num_patches, :]
    with timed_stage(costs, 'full_preparation', state.device), count_phase('full_preparation'):
        conditioning = model.s_projector(F.silu(t_emb_base + seq[:, :model.s_embedder.num_patches, :]))
        original_tokens = model.x_embedder(state)
    traces = {'full': [], 'weak': []}
    with timed_stage(costs, 'full_decoder_and_readout', state.device), count_phase('full_decoder_and_readout'):
        x = original_tokens
        for index in range(2):
            x, trace = explicit_decoder_block(model.blocks[28+index], x, conditioning, model.dec_rope,
                                               query_mean=False, capture=capture)
            if capture:
                traces['full'].append(trace)
        full = model.unpatchify(model.final_layer(x, conditioning), model.x_patch_size)
    # Preserve original DiTwDDTHeadIG.forward order through both native heads.
    with timed_stage(costs, 'base_readout', state.device), count_phase('base_readout'):
        base = F.silu(t_emb_base+h8)
        base = model.unpatchify(model.base_final_layer(base, base), model.s_patch_size)
    with timed_stage(costs, 'weak_decoder_and_readout', state.device), count_phase('weak_decoder_and_readout'):
        x = original_tokens
        for index in range(2):
            # This x is the WEAK previous output, never the full branch's tokens.
            x, trace = explicit_decoder_block(model.blocks[28+index], x, conditioning, model.dec_rope,
                                               query_mean=True, capture=capture)
            if capture:
                traces['weak'].append(trace)
        weak = model.unpatchify(model.final_layer(x, conditioning), model.x_patch_size)
    return {'full': full, 'base': base, 'weak': weak}, traces


def assert_native_parity(heads, native, *, require_bf16=True):
    result = {}
    for name, expected in zip(('full', 'base'), native):
        actual = heads[name]
        identical = actual.dtype == expected.dtype and torch.equal(actual, expected)
        result[f'{name}_native_dtype'] = str(expected.dtype)
        result[f'{name}_helper_dtype'] = str(actual.dtype)
        result[f'{name}_native_bitwise'] = identical
        result[f'{name}_native_max_abs_difference'] = float((actual.double()-expected.double()).abs().max())
        if not identical or (require_bf16 and expected.dtype != torch.bfloat16):
            raise AssertionError(f'current native {name} parity failed: {result}')
    if require_bf16 and heads['weak'].dtype != torch.bfloat16:
        raise AssertionError('weak final head must retain native BF16 dtype')
    return result


def trace_to_cpu(trace):
    result = {'query_mean': trace['query_mean'], 'native_dtypes': {}}
    for key in ('q_before', 'q_used', 'k', 'v', 'sdpa_output'):
        value = trace[key]
        result['native_dtypes'][key] = str(value.dtype)
        result[key] = value.detach().float().cpu()
    return result


def attention_statistics_cpu(trace):
    """Explicit probabilities of effective BF16 Q/K, not fused-kernel probabilities.

    All diagnostic arithmetic is CPU FP64 after the fixed BF16 input cast.
    The reverse-KL reference uses the SAME weak-block input Q/K, not full-block2.
    Its small implementation gap need not vanish: mean then BF16 cast differs
    from averaging already-cast queries, and the native mean is finite precision.
    """
    if any(trace[k].device.type != 'cpu' for k in ('q_before','q_used','k','sdpa_output')):
        raise ValueError('attention diagnostics must run on captured CPU tensors')
    before = trace['q_before'].bfloat16().double()
    used = trace['q_used'].bfloat16().double()
    keys = trace['k'].bfloat16().double()
    output = trace['sdpa_output'].double()
    n, dim = used.shape[-2:]
    records = []
    for image in range(len(used)):
        logp = F.log_softmax((used[image] @ keys[image].transpose(-2,-1))/math.sqrt(dim), dim=-1)
        p = logp.exp()
        row_probability_error = (p-p[:, :1]).abs().amax(dim=(-2,-1))
        key_kl = (p*(logp+math.log(n))).sum(-1).mean(-1)
        q_difference = trace['q_used'][image]-trace['q_used'][image,:, :1]
        q_error = q_difference.abs().amax(dim=(-2,-1))
        original_q_error = (trace['q_before'][image]-trace['q_before'][image,:, :1]).abs().amax(dim=(-2,-1))
        out_error = (output[image]-output[image,:, :1]).abs().amax(dim=(-2,-1))
        if trace['query_mean']:
            original_logp = F.log_softmax((before[image] @ keys[image].transpose(-2,-1))/math.sqrt(dim),dim=-1)
            log_barycenter = F.log_softmax(original_logp.mean(-2),dim=-1)
            barycenter_l1 = (p[:,0]-log_barycenter.exp()).abs().sum(-1)
            barycenter_kl = (p[:,0]*(logp[:,0]-log_barycenter)).sum(-1)
        for head in range(used.shape[1]):
            records.append({'image_in_batch':image,'head_index':head,
                'query_rows_bitwise_equal':bool(q_error[head] == 0),
                'query_row_max_abs_difference':float(q_error[head]),
                'original_query_row_max_abs_difference':float(original_q_error[head]),
                'effective_bf16_query_rows_bitwise_equal':bool(torch.equal(used[image,head],used[image,head,:1].expand_as(used[image,head]))),
                'actual_sdpa_output_rows_bitwise_equal':bool(out_error[head] == 0),
                'actual_sdpa_output_row_max_abs_difference':float(out_error[head]),
                'explicit_attention_probability_row_max_abs_difference':float(row_probability_error[head]),
                'explicit_key_nonuniform_KL_p_to_uniform_mean_query':float(key_kl[head]),
                'implemented_row_to_effective_input_reverseKL_barycenter_L1':float(barycenter_l1[head]) if trace['query_mean'] else None,
                'implemented_row_KL_to_effective_input_reverseKL_barycenter':float(barycenter_kl[head]) if trace['query_mean'] else None})
    return records


def direction_statistics_cpu(heads):
    full, base, weak = (heads[k].float().cpu() for k in ('full','base','weak'))
    delta_weak = full-weak
    delta_base = full-base
    values = {k:v.double().flatten(1) for k,v in [('full',full),('base',base),('weak',weak),('F_minus_W',delta_weak),('F_minus_B',delta_base)]}
    norms = {k:v.square().sum(1).sqrt() for k,v in values.items()}
    dot = (values['F_minus_W']*values['F_minus_B']).sum(1)
    records=[]
    for i in range(len(full)):
        nw,nb = float(norms['F_minus_W'][i]),float(norms['F_minus_B'][i])
        coefficient = float(dot[i])/(nb*nb) if nb>0 else None
        parallel = coefficient*values['F_minus_B'][i] if nb>0 else None
        parallel_energy = float(parallel.square().sum()) if nb>0 else None
        orthogonal_energy = float((values['F_minus_W'][i]-parallel).square().sum()) if nb>0 else None
        row={f'{k}_norm':float(v[i]) for k,v in norms.items()}
        row.update(F_minus_W_rms=nw/math.sqrt(full[0].numel()),F_minus_B_rms=nb/math.sqrt(full[0].numel()),
                   direction_cosine=float(dot[i])/(nw*nb) if nw>0 and nb>0 else None,
                   weak_gap_over_internal_gap=nw/nb if nb>0 else None,
                   descriptive_projection_coefficient_on_internal_gap=coefficient,
                   parallel_energy_to_internal_gap=parallel_energy,
                   orthogonal_energy_to_internal_gap=orthogonal_energy,
                   parallel_energy_over_internal_gap_energy=parallel_energy/(nb*nb) if nb>0 else None,
                   orthogonal_energy_over_internal_gap_energy=orthogonal_energy/(nb*nb) if nb>0 else None,
                   weak_gap_zero=nw==0,internal_gap_zero=nb==0)
        records.append(row)
    return records, delta_weak.numpy(), delta_base.numpy()


def prepare(args):
    started,cpu=time.perf_counter(),time.process_time()
    out=args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('refusing to overwrite a prepared or completed audit')
    if args.protocol_plan is None:
        raise ValueError('prepare requires the root-frozen --protocol-plan')
    source_request=json.loads((args.state_dir.parent/'request.json').read_text())
    source_summary=json.loads((args.state_dir.parent/'summary.json').read_text())
    if not source_summary['complete'] or len(source_summary['snapshots'])!=10:
        raise ValueError('complete original ten snapshots required')
    config_identity,checkpoint_identity=artifact(args.config),artifact(args.checkpoint)
    if config_identity['sha256']!=CONFIG_SHA256 or checkpoint_identity['sha256']!=CHECKPOINT_SHA256:
        raise ValueError('requires current frozen official configuration and EMA checkpoint')
    snapshots=[]
    for old in source_summary['snapshots']:
        path=args.state_dir/Path(old['path']).name
        record=artifact(path)
        if record['sha256']!=old['sha256']:
            raise ValueError('historical state snapshot changed')
        snapshots.append({**record,'step_index':old['step_index'],'t':old['t']})
    plan=artifact(args.protocol_plan)
    protocol_document=artifact(PROTOCOL_DOCUMENT)
    out.mkdir(parents=True,exist_ok=True)
    sources={}
    for relative in SOURCE_FILES:
        p=out/'sources'/relative;p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,p);sources[relative]=artifact(p)
    shutil.copy2(args.protocol_plan,out/f'root_protocol_plan{args.protocol_plan.suffix}')
    shutil.copy2(PROTOCOL_DOCUMENT,out/'frozen_protocol.md')
    request={'protocol':PROTOCOL,'root_protocol_plan':plan,'protocol_document':protocol_document,
        'config':config_identity,'checkpoint':checkpoint_identity,
        'sources':sources,'snapshots':snapshots,'state_source_request':artifact(args.state_dir.parent/'request.json'),
        'state_source_summary':artifact(args.state_dir.parent/'summary.json'),
        'sample_ids':list(range(8)),'labels':list(range(8)),'source_rows':source_request['source_rows'][:8],
        'domains':['teacher','rollout'],'batches':20,'rows':160,'latent_shape':[1024,16,16],
        'architecture':{'encoder_blocks':28,'base_depth':8,'decoder_blocks':2,'decoder_spatial_tokens':256},
        'operator':'both decoder blocks: q Linear -> q RMSNorm -> RoPE -> mean over spatial queries broadcast; no q renorm; K/V untouched as operators, recomputed at each weak block input; original SDPA and same final Full readout',
        'native_order':'one shared encoder -> unmodified explicit Full decoder/readout -> original Base -> replay weak two-block decoder/readout',
        'parity':'each B8 state: separate current model.forward must match explicit Full/Base dtype and bitwise; BF16 heads required. Historical saved heads are not parity targets.',
        'precision':'FP32 weights/states, native BF16 CUDA autocast, TF32 off; query mean in post-RoPE native dtype; FP32 saved F-W/F-B; CPU FP64 norms/logits/softmax',
        'attention_audit':'all image/head/block/branch values; actual SDPA output row equality; explicit CPU FP64 probabilities computed from effective BF16 Q/K, not claimed as fused internal probabilities; nonuniform key KL and finite-precision weak-to-reverseKL-barycenter gap',
        'theoretical_boundary':'uniform query rows imply identical attention rows, not uniform keys or spatially uniform final output. Residual/token modulation/MLP remain. Mean and BF16 cast do not commute; exact barycenter theorem applies in real arithmetic. No weak/full target-error compatibility or quality guarantee.',
        'cohort_boundary':'only8 unique IDs/classes repeated over10 historical time points and2 domains; t1 domains duplicate; old state trajectories used TF32on and historical FP32 guidance; no current rollout reproduced',
        'output_vectors':['F_minus_W.npy','F_minus_B.npy'],'output_dtype':'float32','output_shape':[160,1024,16,16],
        'output_order':'snapshot ascending original step, teacher then rollout, IDs0..7',
        'zero_norm_convention':'undefined direction cosine/norm ratio -> null; zero flags retained',
        'direction_decomposition':'project F-W onto current F-B; retain descriptive projection coefficient, parallel/orthogonal squared norms and normalization by ||F-B||²; never used as a gain. If F-B=0 all axis-dependent quantities are null.',
        'forward_count_method':'instance-local successful-forward hooks on model, s_embedder, every encoder block, decoder q and MLP, Full/Base readouts; exact phase/module events asserted for each B8 input',
        'expected_forward_module_events_per_batch':ForwardCallCounter.expected_events(),
        'no_sampling':True,'no_decode':True,'no_fid':True,'no_new_noise':True,'no_gain_or_window':True,
        'torch_version':str(torch.__version__),'expected_forward_counts':{
            'shared_encoder_passes':20,'native_model_forward_calls':20,'encoder_block_calls':1120,
            'explicit_full_decoder_block_calls':40,'weak_decoder_block_calls':40,'native_reference_decoder_block_calls':40,
            'full_final_readout_calls_including_weak_and_reference':60,'base_readout_calls_including_reference':40}}
    atomic_json(out/'request.json',request)
    atomic_json(out/'prepare_cost.json',{'wall_seconds':time.perf_counter()-started,'cpu_seconds':time.process_time()-cpu,
                'request_sha256':sha256_file(out/'request.json'),'model_calls':0,'gpu_seconds':0})
    atomic_json(out/'progress.json',{'status':'prepared_no_gpu','complete':False})
    print(json.dumps({'prepared':True,'request':artifact(out/'request.json')}),flush=True)


def run(args):
    started,cpu=time.perf_counter(),time.process_time();out=args.output_dir
    run_started_utc=datetime.now(timezone.utc).isoformat()
    request=json.loads((out/'request.json').read_text())
    if (out/'summary.json').exists() or (out/'run_started.json').exists():
        raise FileExistsError('refusing to rerun/overwrite a started audit')
    if request['protocol']!=PROTOCOL:
        raise ValueError('wrong frozen audit protocol')
    for name,record in request['sources'].items():
        if sha256_file(ROOT/name)!=record['sha256'] or sha256_file(Path(record['path']))!=record['sha256']:
            raise ValueError(f'source changed since prepare: {name}')
    for record in [request[k] for k in ('root_protocol_plan','protocol_document','config','checkpoint','state_source_request','state_source_summary')]+request['snapshots']:
        if sha256_file(Path(record['path']))!=record['sha256']:
            raise ValueError(f'input changed since prepare: {record["path"]}')
    atomic_json(out/'run_started.json',{'run_started_utc':run_started_utc,'request':artifact(out/'request.json'),
                'root_admission':'external; runner never launches itself'})
    counter=None
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise RuntimeError('exactly one visible CUDA GPU required')
        device=torch.device('cuda:0');torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.set_num_threads(4)
        torch.cuda.reset_peak_memory_stats(device)
        from utils.model_utils import instantiate_from_config
        config=load_config(Path(request['config']['path']))
        loading={}
        with timed_stage(loading,'model_load',device):
            model=instantiate_from_config(config.stage_2).to(device).eval().requires_grad_(False)
            checkpoint=torch.load(request['checkpoint']['path'],map_location='cpu',mmap=True,weights_only=False)
            model.load_state_dict(checkpoint['ema'],strict=True)
            checkpoint_step=int(checkpoint['step']);del checkpoint
        gc.collect()
        counter=ForwardCallCounter(model)
        vectors_w=np.lib.format.open_memmap(out/'F_minus_W.npy',mode='w+',dtype=np.float32,shape=tuple(request['output_shape']))
        vectors_b=np.lib.format.open_memmap(out/'F_minus_B.npy',mode='w+',dtype=np.float32,shape=tuple(request['output_shape']))
        rows=[];attention_rows=[];parity_rows=[];batch_costs=[];offset=0;t1_duplicate=None
        for snapshot in request['snapshots']:
            cache=torch.load(snapshot['path'],map_location='cpu',weights_only=True)
            if (cache['sample_ids'].tolist()!=request['sample_ids'] or cache['labels'].tolist()!=request['labels']
                    or cache['step_index']!=snapshot['step_index'] or cache['t']!=snapshot['t']):
                raise ValueError('cached state identity differs from frozen cohort')
            if snapshot['t']==1.:
                t1_duplicate=torch.equal(cache['teacher']['state'],cache['rollout']['state'])
                if not t1_duplicate:raise ValueError('original t1 domains must be identical')
            for domain in request['domains']:
                costs={};identity={'step_index':snapshot['step_index'],'t':snapshot['t'],'domain':domain}
                counter.reset()
                with timed_stage(costs,'state_transfer',device):
                    state=cache[domain]['state'].to(device)
                    if state.shape!=(8,1024,16,16) or state.dtype!=torch.float32:
                        raise ValueError('unexpected state shape/dtype')
                    labels=cache['labels'].to(device)
                    times=torch.full((8,),snapshot['t'],device=device,dtype=torch.float32)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    heads,traces=shared_full_base_weak(model,state,times,labels,capture=True,costs=costs,counter=counter)
                    with timed_stage(costs,'native_reference_forward',device),counter.phase('native_reference'):
                        native=model(state,times,context=labels,attn_mask=None)
                with timed_stage(costs,'forward_count_verification',torch.device('cpu')):
                    forward_counts=counter.verify_batch()
                with timed_stage(costs,'native_parity_and_trace_transfer',device):
                    parity=assert_native_parity(heads,native)
                    parity['first_decoder_keys_bitwise_same_full_weak']=torch.equal(traces['full'][0]['k'],traces['weak'][0]['k'])
                    parity['first_decoder_values_bitwise_same_full_weak']=torch.equal(traces['full'][0]['v'],traces['weak'][0]['v'])
                    if not parity['first_decoder_keys_bitwise_same_full_weak'] or not parity['first_decoder_values_bitwise_same_full_weak']:
                        raise AssertionError('first-block K/V changed despite shared exact input')
                    heads_cpu={k:v.float().cpu() for k,v in heads.items()}
                    traces_cpu={branch:[trace_to_cpu(t) for t in branch_traces] for branch,branch_traces in traces.items()}
                    trace_dtypes={branch:[t['native_dtypes'] for t in ts] for branch,ts in traces_cpu.items()}
                del heads,traces,native
                with timed_stage(costs,'direction_and_attention_statistics_cpu',torch.device('cpu')):
                    if any(not bool(torch.isfinite(v).all()) for v in heads_cpu.values()):
                        raise FloatingPointError('nonfinite Full/Base/weak head')
                    directions,dw,db=direction_statistics_cpu(heads_cpu)
                    for i,row in enumerate(directions):
                        rows.append({**identity,'sample_id':i,'label':i,'source_row':request['source_rows'][i],**row})
                    for branch in ('full','weak'):
                        for block,trace in enumerate(traces_cpu[branch]):
                            for row in attention_statistics_cpu(trace):
                                image=row.pop('image_in_batch')
                                attention_rows.append({**identity,'branch':branch,'decoder_block':block,'sample_id':image,'label':image,**row})
                with timed_stage(costs,'output_write',torch.device('cpu')):
                    vectors_w[offset:offset+8]=dw;vectors_b[offset:offset+8]=db;offset+=8
                    vectors_w.flush();vectors_b.flush()
                    parity_rows.append({**identity,**parity})
                    batch_costs.append({**identity,'stages':costs.copy(),'trace_native_dtypes':trace_dtypes,
                                        'forward_counts':forward_counts})
                    write_csv(out/'per_image_directions.csv',rows)
                    write_csv(out/'per_head_attention.csv',attention_rows)
                    write_csv(out/'native_parity.csv',parity_rows)
                    atomic_json(out/'progress.json',{'complete':False,'state_batches':len(parity_rows),'rows':offset,
                                'last':identity,'wall_seconds':time.perf_counter()-started})
                # Capture output timer after its context exits; no recursive cost dict.
                batch_costs[-1]['stages']=costs.copy()
                print(json.dumps({'state_batches':len(parity_rows),'rows':offset,'last':identity}),flush=True)
                del heads_cpu,traces_cpu,dw,db,state
        if offset!=160 or len(parity_rows)!=20 or len(attention_rows)!=10240:
            raise AssertionError('incomplete fixed cohort/head audit')
        del vectors_w,vectors_b
        stage_totals={}
        observed_forward_counts={name:0 for name in request['expected_forward_counts']}
        for batch in batch_costs:
            for name,value in batch['stages'].items():
                stage_totals[name]=stage_totals.get(name,0.)+value['wall_seconds']
            for name,value in batch['forward_counts']['observed'].items():
                observed_forward_counts[name]+=value
        if observed_forward_counts!=request['expected_forward_counts']:
            raise AssertionError('aggregate observed forward counts differ from frozen expected counts')
        artifacts=[artifact(out/name) for name in ('F_minus_W.npy','F_minus_B.npy','per_image_directions.csv','per_head_attention.csv','native_parity.csv')]
        atomic_json(out/'batch_costs.json',batch_costs)
        summary={'complete':True,'protocol':PROTOCOL,'request':artifact(out/'request.json'),'rows':160,'unique_ids':8,
                 'run_started_utc':run_started_utc,
                 'attention_head_rows':len(attention_rows),'native_Full_Base_bitwise_parity_all':True,
                 't1_teacher_rollout_state_duplicate':t1_duplicate,'checkpoint_step':checkpoint_step,
                 'cuda_device':torch.cuda.get_device_name(device),'observed_forward_counts':observed_forward_counts,
                 'expected_forward_counts':request['expected_forward_counts'],
                 'forward_count_method':request['forward_count_method'],'all_batch_forward_module_events_match':True,
                 'loading':loading,'stage_wall_seconds':stage_totals,'batch_costs':artifact(out/'batch_costs.json'),
                 'outputs':artifacts,'wall_seconds_before_summary':time.perf_counter()-started,
                 'cpu_seconds_before_summary':time.process_time()-cpu,
                 'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),
                 'no_sampling':True,'no_fid':True,'no_decoder_images':True,'no_new_noise':True,
                 'interpretation':'fixed structural/attention information deletion audit; no weak error compatibility, new guidance or quality guarantee; repeated historical states are not independent experiments'}
        atomic_json(out/'summary.json',summary)
        atomic_json(out/'progress.json',{'complete':True,'summary':artifact(out/'summary.json')})
        print(json.dumps(summary),flush=True)
    except BaseException as error:
        atomic_json(out/'failure.json',{'complete':False,'run_started_utc':run_started_utc,'error':f'{type(error).__name__}: {error}',
                    'wall_seconds':time.perf_counter()-started})
        raise
    finally:
        if counter is not None:
            counter.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prepare','run'),required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--protocol-plan',type=Path,default=PROTOCOL_DOCUMENT)
    parser.add_argument('--state-dir',type=Path,default=DEFAULT_STATES)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    args=parser.parse_args()
    for name in ('output_dir','protocol_plan','state_dir','config','checkpoint'):
        value=getattr(args,name)
        if value is not None:setattr(args,name,value.expanduser().resolve())
    prepare(args) if args.mode=='prepare' else run(args)


if __name__=='__main__':
    main()
