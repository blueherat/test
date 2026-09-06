#!/usr/bin/env python3
"""Reproducible native-IG controls and Gaussian-channel stochastic guidance."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT/'external/RAEv2/src'):
    sys.path.insert(0, str(p))

from experiments.raev2_ancestral_guidance import ancestral_step
from experiments.raev2_transport_projection import projected_guidance
from experiments.raev2_stochastic_weak import SharedPrefixWeak
from experiments.raev2_two_mode_ratio import two_mode_correction
from experiments.raev2_semantic_complement import semantic_correction
from experiments.raev2_paired_ratio_model import PairedRatioCritic
from experiments.raev2_image_critic_guidance import ImageCritic, ExchangeablePosterior, PROBE, COVARIANCE, exact_fp32
from experiments.raev2_stage1_compat import install_raev2_decoder_config_compat
from experiments.sample_raev2_pfr_retiming import DEFAULT_CONFIG, DEFAULT_CHECKPOINT, load_config, shifted_time_grid
from experiments.raev2_training_core import file_sha256
from utils.model_utils import instantiate_from_config


def put(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def seed_for(seed, batch, namespace):
    return int.from_bytes(hashlib.sha256(f'raev2-ag-20260907:{namespace}:{seed}:{batch}'.encode()).digest()[:8], 'little') % (2**63)


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--modes', nargs='+', default=['official', 'piecewise', 'ancestral', 'partial'], choices=['official','piecewise','ancestral','partial','full','calibrated','velocity_projection','noise_projection','stochastic_weak','mean_weak','critic_isotropic','critic_exchangeable','two_mode','semantic_add','semantic_orthogonal','paired_ratio','paired_ratio_calibrated'])
    p.add_argument('--variance-calibration', type=Path, default=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_20260907/guided_reverse_variance/calibration.json'))
    p.add_argument('--samples', type=int, default=1000)
    p.add_argument('--seed', type=int, default=202609071)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument('--parity', action='store_true')
    args = p.parse_args()
    if args.samples % args.batch or not 0 <= args.shard < args.shards or args.steps <= 0:
        p.error('invalid complete-batch partition or steps')
    return args


def native_clean(full, base, current, mode):
    active = current >= (.5 if mode == 'piecewise' else .1)
    return (base+1.78*(full-base) if active and mode != 'full' else full).float()


@torch.no_grad()
def main():
    args = get_args()
    out = args.output.resolve()/f'shard{args.shard}'
    out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('DINOV3_CKPT_DIR', '/home/zhoushunyu/data/eqvae/models/RAEv2/encoders/dinov3')
    install_raev2_decoder_config_compat()
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = load_config(args.config)
    decoder = instantiate_from_config(cfg.stage_1).cuda().eval().requires_grad_(False)
    critic=ImageCritic(decoder,decoder.encoder) if any(m.startswith('critic_') for m in args.modes) else None
    covariance=ExchangeablePosterior() if 'critic_exchangeable' in args.modes else None
    del decoder.encoder
    torch.cuda.empty_cache()
    model = instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
    ckpt = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(ckpt['ema'], strict=True)
    del ckpt
    paired_ratio = None
    paired_temperature = None
    if any(m.startswith('paired_ratio') for m in args.modes):
        paired_path = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_20260907/paired_ratio_fit/critic.pt')
        paired_checkpoint = torch.load(paired_path, map_location='cpu', weights_only=False)
        if args.steps != 100 or not paired_checkpoint['validation']['entry_condition_passed']:
            raise ValueError('paired ratio requires the fixed 100-step protocol and held-out entry condition')
        paired_ratio = PairedRatioCritic(paired_checkpoint['state_dict']['class_features'])
        paired_ratio.load_state_dict(paired_checkpoint['state_dict'], strict=True)
        paired_ratio = paired_ratio.cuda().eval().requires_grad_(False)
        if 'paired_ratio_calibrated' in args.modes:
            temperature_path = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_20260907/paired_ratio_calibration.json')
            paired_temperature = json.loads(temperature_path.read_text())
            if not paired_temperature['entry_condition_passed'] or paired_temperature['fid_used_for_parameter_fit']:
                raise ValueError('temperature requires held-out probability calibration')
            if paired_temperature['checkpoint_sha256'] != file_sha256(paired_path):
                raise ValueError('temperature and critic mismatch')
    weak_model = SharedPrefixWeak(model) if any(m in args.modes for m in ('stochastic_weak','mean_weak')) else None
    shift = math.sqrt((cfg.misc.time_dist_shift_dim or math.prod(cfg.misc.latent_size))/cfg.misc.time_dist_shift_base)
    grid = shifted_time_grid(args.steps,shift,torch.device('cuda')).cpu().tolist()
    calibration = None
    if 'calibrated' in args.modes or critic is not None:
        calibration=json.loads(args.variance_calibration.read_text())
        if not calibration['complete'] or calibration['fid_used_for_fit'] or args.steps!=100:
            raise ValueError('requires the complete, fixed 100-time forward-pair calibration')
        if max(abs(t-row['time']) for t,row in zip(grid[:-1],calibration['rows']))>2e-7:
            raise ValueError('calibration time grid mismatch')
    ratio_calibration=None
    if 'two_mode' in args.modes:
        ratio_path=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_20260907/two_mode_ratio/finite_euler_calibration.json')
        ratio_calibration=json.loads(ratio_path.read_text())
        if args.steps!=100 or not ratio_calibration.get('finite_euler_calibration') or not ratio_calibration['complete']:
            raise ValueError('two-mode guidance requires the fixed finite-Euler calibration')
        if len(grid)!=len(ratio_calibration['time_grid']) or max(abs(a-b) for a,b in zip(grid,ratio_calibration['time_grid']))>2e-7:
            raise ValueError('two-mode time grid mismatch')
    request = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    request.update(protocol='raev2_gaussian_channel_guidance_v1', checkpoint_sha256=file_sha256(args.checkpoint),
                   config_sha256=file_sha256(args.config), state_key='ema', time_grid=grid,
                   torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                   source_sha256={p:file_sha256(ROOT/p) for p in ['experiments/sample_raev2_ancestral_guidance.py','experiments/raev2_ancestral_guidance.py','experiments/raev2_transport_projection.py','experiments/raev2_stochastic_weak.py','experiments/raev2_image_critic_guidance.py','experiments/raev2_two_mode_ratio.py','experiments/raev2_semantic_complement.py','experiments/raev2_semantic_quality_guidance.py','experiments/raev2_paired_ratio_model.py']},
                   decoder_sha256=file_sha256(Path(cfg.stage_1.params['pretrained_decoder_path'])),
                   stats_sha256=file_sha256(Path(cfg.stage_1.params['normalization_stat_path'])))
    if calibration is not None:
        if calibration['baseline_checkpoint']['sha256']!=request['checkpoint_sha256'] or calibration['config']['sha256']!=request['config_sha256']:
            raise ValueError('calibration source identity mismatch')
        request['variance_calibration_sha256']=file_sha256(args.variance_calibration)
        request['variance_formula']='q^2 E||X-G_native||^2 / D; q=(t-s)/t; no fitted strength'
    if weak_model is not None:
        request['weak_protocol']={'eligible_blocks_zero_indexed':list(range(model.base_model_depth,model.num_enc_blocks)),
            'skip_count':1,'shared_mask_per_batch':True,'strength':.25,'time_interval':[.1,1.],
            'mean_attenuation':1-1/(model.num_enc_blocks-model.base_model_depth)}
    if critic is not None:
        request['critic']={'probe_sha256':file_sha256(PROBE),'covariance_sha256':file_sha256(COVARIANCE) if covariance else None,
            'gradient_precision':'FP32 decoder and DINO, TF32 off, FP64 unit CLS','continuous_pixels':True,
            'strength':1.,'all_100_times':True,'gradient_microbatch':args.batch}
    if ratio_calibration is not None:
        request['two_mode']={'calibration_sha256':file_sha256(ratio_path),'strength':1.,'all_100_times':True,
            'real_effective_prior':ratio_calibration['real'],'generated_effective_prior':ratio_calibration['generated']}
    if any(m.startswith('semantic_') for m in args.modes):
        request['semantic_complement']={'additional_cfg_strength':.15,'null_label':1000,
            'activity':'same original IG interval [.1,1]','conditional_layout':'original B8; null in a separate B8 forward',
            'base_anchor':'unchanged native BF16 IG; correction computed in FP32',
            'model_calls_note':'sample_model_calls includes conditional and null; extra_sample_unconditional_calls is a subset'}
    if paired_ratio is not None:
        request['paired_ratio']={'checkpoint_sha256':file_sha256(paired_path),
            'training_plan':paired_checkpoint['plan'], 'held_out_validation':paired_checkpoint['validation'],
            'strength':1., 'all_100_times':True, 'formula':'native G + t^2 grad_z f',
            'gradient_precision':'FP32 model and input, TF32 off',
            'trainable_critic_parameters':sum(p.numel() for p in paired_ratio.parameters())}
        if paired_temperature is not None:
            request['paired_ratio']['temperature_calibration'] = paired_temperature
            request['paired_ratio']['temperature_sha256'] = file_sha256(temperature_path)
    put(out/'request.json',request)
    batch_ids = list(range(args.shard,args.samples//args.batch,args.shards))
    # Shared model/decoder warmup; excluded from per-image timings and disclosed.
    t0 = time.perf_counter()
    dummy = torch.zeros(args.batch,*cfg.misc.latent_size,device='cuda')
    times = torch.full((args.batch,),.8,device='cuda')
    labels = torch.arange(args.batch,device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        f,b = model(dummy,times,context=labels,attn_mask=None)
        if weak_model is not None and not torch.equal(f,weak_model.weak(dummy,times)):
            raise AssertionError('shared-prefix full readout parity failed')
        decoder.decode(f.float())
        if args.parity:
            from utils.guidance_utils import forward_with_internalguidance
            official = forward_with_internalguidance(model,torch.cat([dummy,dummy]),torch.cat([times,times]),
                    ig_scale=1.78,ig_interval=(.1,1.),context=torch.cat([labels,labels]),attn_mask=None)[:args.batch]
            if not torch.equal(official.float(),native_clean(f,b,.8,'official')):
                raise AssertionError('native official IG parity failed')
    torch.cuda.synchronize()
    put(out/'warmup.json',{'seconds':time.perf_counter()-t0,'native_parity_passed':args.parity})
    for mode in args.modes:
        target=out/mode
        target.mkdir()
        records=[]
        all_images=[]
        all_ids=[]
        trajectory_seconds=decode_seconds=0.
        weak_calls=0
        semantic_calls=0
        critic_calls=0
        ratio_calls=0
        critic_records=[]
        started=time.perf_counter()
        for batch_id in batch_ids:
            ids=np.arange(batch_id*args.batch,(batch_id+1)*args.batch)
            rng=torch.Generator(device='cuda').manual_seed(seed_for(args.seed,batch_id,'initial'))
            refresh=torch.Generator(device='cuda').manual_seed(seed_for(args.seed,batch_id,'refresh'))
            masks=np.random.default_rng(seed_for(args.seed,batch_id,'weak_masks')).integers(model.base_model_depth,model.num_enc_blocks,size=args.steps)
            state=torch.randn(args.batch,*cfg.misc.latent_size,device='cuda',generator=rng)
            labels=torch.from_numpy(ids%1000).cuda()
            record={'batch':batch_id,'noise_sha256':digest(state),'labels_sha256':digest(labels)}
            torch.cuda.synchronize()
            start=time.perf_counter()
            for step,(t,s) in enumerate(zip(grid[:-1],grid[1:])):
                times=torch.full((args.batch,),t,device='cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    full,base=model(state,times,context=labels,attn_mask=None)
                    clean=native_clean(full,base,t,mode)
                    if mode in ('stochastic_weak','mean_weak') and t>=.1:
                        weak=weak_model.weak(state,times,skip=int(masks[step]) if mode=='stochastic_weak' else None,
                            attenuation=1-1/(model.num_enc_blocks-model.base_model_depth) if mode=='mean_weak' else None)
                        clean=clean+.25*(full.float()-weak.float())
                        weak_calls+=1
                    if mode.startswith('semantic_') and t>=.1:
                        null_full,_=model(state,times,context=torch.full_like(labels,1000),attn_mask=None)
                        clean=clean+semantic_correction(full,base,null_full,orthogonal=mode=='semantic_orthogonal')
                        semantic_calls+=1
                if mode=='two_mode':
                    clean=clean+two_mode_correction(state,t,ratio_calibration)
                if mode.startswith('paired_ratio'):
                    with exact_fp32():
                        correction=paired_ratio.clean_correction(state,times,labels)
                        if mode=='paired_ratio_calibrated':
                            correction=paired_temperature['alpha']*correction
                    clean=clean+correction
                    ratio_calls+=1
                    if batch_id==batch_ids[0]:
                        critic_records.append({'step':step,'time':t,
                            'correction_rms':float(correction.square().mean().sqrt())})
                if mode in ('velocity_projection','noise_projection'):
                    clean=projected_guidance(state,full,base,clean,t,coordinate=mode.split('_')[0])
                if mode.startswith('critic_'):
                    grad,logit=critic.gradient(clean)
                    correction=calibration['rows'][step]['mse']*grad if mode=='critic_isotropic' else covariance.apply(grad,t,calibration['rows'][step]['mse'])
                    clean=clean+correction
                    critic_calls+=1
                    if batch_id==batch_ids[0]:
                        critic_records.append({'step':step,'time':t,'mean_logit':float(logit.mean()),
                            'gradient_rms':float(grad.square().mean().sqrt()),'correction_rms':float(correction.square().mean().sqrt())})
                if mode=='calibrated':
                    noise=torch.randn(state.shape,device='cuda',generator=refresh)
                    euler=state-(t-s)*((state-clean)/t)
                    state=euler+((t-s)/t)*math.sqrt(calibration['rows'][step]['mse'])*noise
                elif mode in ('ancestral','partial'):
                    noise=torch.randn(state.shape,device='cuda',generator=refresh)
                    state=ancestral_step(state,clean,t,s,noise,eta=1. if mode=='ancestral' else .5)
                else:
                    state=state-(t-s)*((state-clean)/max(t,float(cfg.transport.t_eps)))
            torch.cuda.synchronize()
            trajectory_seconds+=time.perf_counter()-start
            if not torch.isfinite(state).all():
                raise FloatingPointError(f'nonfinite {mode} batch{batch_id}')
            start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                decoded=decoder.decode(state)
                pixels=decoded.clamp(0,1).mul(255).permute(0,2,3,1).to('cpu',torch.uint8).numpy()
            torch.cuda.synchronize()
            decode_seconds+=time.perf_counter()-start
            all_images.append(pixels)
            all_ids.extend(ids.tolist())
            records.append(record)
            if batch_id==batch_ids[0]:
                from PIL import Image
                Image.fromarray(np.concatenate(list(pixels[:4]),axis=1)).save(target/'preview.png')
            put(target/'progress.json',{'samples':len(all_ids),'target':len(batch_ids)*args.batch,'elapsed':time.perf_counter()-started})
            print(json.dumps({'mode':mode,'shard':args.shard,'samples':len(all_ids),'seconds':round(time.perf_counter()-started,2)}),flush=True)
        images=np.concatenate(all_images)
        np.savez(target/'samples.npz',images,ids=np.array(all_ids,dtype=np.int64))
        put(target/'summary.json',{'complete':True,'mode':mode,'samples':len(all_ids),'sample_sha256':file_sha256(target/'samples.npz'),
            'trajectory_seconds':trajectory_seconds,'decode_seconds':decode_seconds,'total_seconds':time.perf_counter()-started,
            'model_calls':len(batch_ids)*args.steps+semantic_calls,'sample_model_calls':len(all_ids)*args.steps+semantic_calls*args.batch,'decoder_calls':len(batch_ids),
            'extra_unconditional_calls':semantic_calls,'extra_sample_unconditional_calls':semantic_calls*args.batch,
            'extra_weak_continuations':weak_calls,'extra_sample_weak_continuations':weak_calls*args.batch,
            'critic_backward_calls':critic_calls,'sample_critic_backward_calls':critic_calls*args.batch,'critic_first_batch':critic_records,
            'ratio_backward_calls':ratio_calls,'sample_ratio_backward_calls':ratio_calls*args.batch,
            'initial_noise':records,'max_memory_allocated':torch.cuda.max_memory_allocated()})


if __name__=='__main__':
    main()
