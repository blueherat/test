import os
from pathlib import Path
from experiments.guidance_loss_50k_20260914.config import ARMS, WEAK_LOSSES, atomic, read, sha, array_sha, RequestedStop
from experiments.guidance_strength_sweep_20260915.planning import refine

WORK = Path(__file__).resolve().parents[2]
EXPS = Path('/home/zhoushunyu/data/eqvae/experiments')
ROOT = EXPS / 'guidance_dynamic_50k_20260915'
MODULE = 'experiments.guidance_dynamic_50k_20260915'
PYTHON = '/home/zhoushunyu/miniconda3/envs/myenv/bin/python'
PROTOCOL = WORK / 'docs/GUIDANCE_DYNAMIC_50K_20260915_ZH.md'
REPORT = WORK / 'docs/GUIDANCE_DYNAMIC_RESULTS_20260915_ZH.md'
SIT_DATA = EXPS.parent / 'imagenet_sit_flow/imagenet100_cmc_sdvae'
JIT_DATA = Path('/data/shared/imagenet-1k/random_access_v1')
SIT_PRIOR = EXPS.parent / 'imagenet_sit_flow/multiscale_guidance_study_v1/runs/depth4_v/run_config.json'
JIT_PRIOR = EXPS / 'jit_internal_readouts_20260908/request.json'
MODELS = ('sit_small', 'jit')
METHODS = ('real','guided_weak','self','mixture','gaussian','excess','shuffled','contrast_s','contrast_m',
    'contrast_weak','covariance','contrast_null','real_residual','weakmix_fm','weakmix_contrast',
    'ig_residual','ig_contrast','cfg_residual','cfg_contrast')
STEPS, GLOBAL_BATCH, WORLD = 50000, 256, 4
SEARCH_STEPS = (16, 8, 4, 1)
COARSE = tuple(range(0,81,16))


def model_root(model):
    assert model in MODELS
    return ROOT/model


def settings(model):
    return dict(classes=100 if model=='sit_small' else 1000,
        shape=(4,32,32) if model=='sit_small' else (3,256,256),
        width=384 if model=='sit_small' else 768, patch=2 if model=='sit_small' else 16,
        seed=0 if model=='sit_small' else 202609822,
        alpha=.8 if model=='sit_small' else .3,
        cfg_extra=1.25 if model=='sit_small' else 2.,
        sample_batch=8 if model=='sit_small' else 4,
        endpoint_batch=64 if model=='sit_small' else 32,
        validation_count=5000 if model=='sit_small' else 1000)


def point_key(method,tick):
    assert method in (*METHODS,'strong','native','cfg_native') and 0<=tick<=80
    return f'{method}__c{tick:04d}'


def parse_point(point):
    method,raw=point.rsplit('__c',1)
    tick=int(raw)
    assert point_key(method,tick)==point
    return method,tick


def default_tick(model,method):
    return round(settings(model)['alpha']*40) if ARMS[method]['loss'] in WEAK_LOSSES else 40


def check_stop():
    if (ROOT/'STOP_AFTER_CURRENT').exists():
        raise RequestedStop('Dynamic retraining: save the current optimizer step or sampling batch')


_verified = {}
def verify():
    request=read(ROOT/'request.json')
    for group in ('sources','assets'):
        for filename,digest in request[group].items():
            p=Path(filename);stat=p.stat();key=(filename,stat.st_size,stat.st_mtime_ns)
            if _verified.get(key)!=digest:
                assert sha(p)==digest, (group,filename)
                _verified[key]=digest
    return request


def prepare():
    import numpy as np
    from .data import prepare_indices
    assert not (ROOT/'request.json').exists()
    ROOT.mkdir(parents=True,exist_ok=True)
    sit=read(SIT_PRIOR)['config'];jit=read(JIT_PRIOR)
    assert sit['max_steps']==50000 and sit['global_batch_size']==jit['global_batch']==256
    assert sit['learning_rate']==jit['lr']==1e-4 and sit['ema_decay']==jit['ema']==.9999
    assert sit['weight_decay']==0 and sit['precision']=='bf16'
    datasets=prepare_indices()
    from experiments.guidance_pasted_20260912 import common as common
    from experiments import jit_internal_guidance as jig
    sources=set(Path(__file__).parent.glob('*.py'))
    old=read(EXPS/'guidance_loss_50k_20260914/request.json')
    sources.update(map(Path,old['sources']))
    sources.update([PROTOCOL,WORK/'experiments/train_imagenet100_sit_flow.py',
        WORK/'experiments/train_imagenet100_sit_frozen_internal_v_head.py',
        WORK/'experiments/train_jit_internal_readouts.py',WORK/'experiments/jit_internal_guidance.py',
        WORK/'experiments/raev2_training_core.py',jig.REPO/'model_jit.py',jig.REPO/'util/model_util.py',
        WORK/'experiments/compute_adm_fid.py',WORK/'experiments/evaluate_raev2_official_samples.py'])
    assets={p:sha(p) for p in common.asset_paths('sit_small')}
    assets.update({str(p):sha(p) for p in (jig.CHECKPOINT,EXPS/'jit_internal_readouts_20260908/last.pt',
        SIT_PRIOR,JIT_PRIOR,SIT_DATA/'manifest.json',JIT_DATA/'manifest.json')})
    for bank in (EXPS/'guidance_loss_50k_20260914/sit_small/screen1000/inputs',
                 EXPS/'jit_readout_transfer_20260913/screen_1000/inputs'):
        assets.update({str(bank/name):sha(bank/name) for name in ('noise.npy','labels.npy')})
    for data in datasets.values(): assets.update(data['files'])
    # Freeze the complete real data, not only a small endpoint subset.
    manifest=read(SIT_DATA/'manifest.json')
    for split in ('train','validation'):
        for kind in ('moments','labels','source_indices'):
            p=SIT_DATA/f'{split}_{kind}.npy'
            digest=sha(p);assert digest==manifest['splits'][split][kind+'_sha256']
            assets[str(p)]=digest
    request=dict(models=list(MODELS),methods=list(METHODS),steps=STEPS,global_batch=GLOBAL_BATCH,
        world_size=WORLD,lr=1e-4,weight_decay=0.,betas=[.9,.999],ema=.9999,precision='bf16',
        source_frozen=True,head='depth4 Context MLP; new initialization and full-data normalization',
        real_data=datasets,generated_data='user-selected full-size frozen sample banks; new time and noise every batch',
        jit_generated_storage='lossless PNG packing of the model\'s ordinary uint8 output images; dynamic flip/time/noise',
        generated_cache_retention='after the last consumer, retire only newly generated bulk banks; retain seeds, hashes, manifests and all model/quality results',
        validation='original held-out split; never used for training, classifier fitting, or early stopping',
        sequence='one idea: SiT train50K -> four-level 1K search -> top2 extend5K -> JiT same -> next idea',
        strength_steps=[.4,.2,.1,.025],initial_range=[0.,2.],connected_intervals=True,
        top2_extension=dict(select='lowest two valid 1K FIDs; ties prefer smaller coefficient',
            reuse_first_1000=True,additional_samples=4000,total_samples_per_selected_coefficient=5000,
            note='5K contains the 1K selection images; not an independent holdout'),
        no_historical_head_reuse=True,original_requests_preserved=True,
        old_protocol_results='historical small-bank results only; all trainable heads and auxiliary heads reset',
        sources={str(p.resolve()):sha(p) for p in sorted(sources)},assets={str(p):h for p,h in assets.items()})
    atomic(ROOT/'request.json',request)
    return request
