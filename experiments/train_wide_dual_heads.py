"""Remove the D512/H128 output bottleneck using the original training protocol."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from terminal_defect_spiral import spiral,core,sha
from subspace_gate_distribution import make_distribution


def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(a.source_config.read_text())
    if a.curvature is not None:
        cfg['curvature']=a.curvature
    cfg.update(hidden_dim=512,model_ids=['D0_xeps','D2_velocity','D4_safe'],
               output_root=str(a.output.parent),device=a.device,device_resolved=a.device)
    cfg['data_protocol']=f"v4/v10 continuous spiral; unit-RMS curvature={cfg['curvature']} embedding"
    core.save_json(a.output/'config.json',cfg)
    torch.set_num_threads(4)
    core.set_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(a.device)
    dist=make_distribution(cfg,device)
    suite=core.build_model_suite(ambient_dim=cfg['ambient_dim'],hidden_dim=cfg['hidden_dim'],
        depth=cfg['depth'],time_dim=cfg['time_dim'],mode_dim=cfg['mode_dim'],model_ids=cfg['model_ids'],
        lr=cfg['lr'],weight_decay=cfg['weight_decay'],seed=core.stable_seed(cfg['seed'],cfg['ambient_dim'],113),device=device)
    prior_steps=0
    train_steps=cfg['train_steps']
    train_seed=core.stable_seed(cfg['seed'],cfg['ambient_dim'],127)
    if a.resume is not None:
        previous=torch.load(a.resume,map_location=device,weights_only=False)
        prior_steps=int(previous['steps'])
        for name,model in suite.models.items():
            model.load_state_dict(previous['models'][name],strict=True)
            suite.optimizers[name].load_state_dict(previous['optimizers'][name])
        train_steps=a.additional_steps
        train_seed=core.stable_seed(cfg['seed'],cfg['ambient_dim'],127,prior_steps)
        cfg.update(train_steps=prior_steps+train_steps,continuation_steps=train_steps,
                   continuation_data_seed=train_seed,resume_checkpoint=str(a.resume))
        core.save_json(a.output/'config.json',cfg)
    sources=a.output/'source';sources.mkdir()
    for p in [Path(__file__),Path(spiral.__file__),Path(core.__file__),Path(spiral.v4.__file__),Path(spiral.v10.__file__)]:
        (sources/p.name).write_bytes(p.read_bytes())
    core.save_json(a.output/'manifest.json',{'source_config_sha256':sha(a.source_config),
        'torch':torch.__version__,'claim':'width/maturity/curvature control according to config; not matched compute',
        'curvature':cfg['curvature'],'prior_steps':prior_steps,'additional_steps':train_steps,
        'parameters':{k:sum(p.numel() for p in m.parameters()) for k,m in suite.models.items()}})
    history=core.train_models(suite=suite,distribution=dist,steps=train_steps,batch_size=cfg['batch_size'],
        t_min=cfg['train_t_min'],t_max=cfg['train_t_max'],denominator_floor=cfg['denominator_floor'],
        consistency_weight=cfg['consistency_weight'],grad_clip=cfg['grad_clip'],log_every=cfg['log_every'],
        seed=train_seed,checkpoint_path=a.output/'checkpoint.pt')
    if prior_steps:
        saved=torch.load(a.output/'checkpoint.pt',map_location='cpu',weights_only=False)
        saved.update(steps=prior_steps+train_steps,prior_steps=prior_steps,resume_sha256=sha(a.resume))
        core.atomic_torch_save(saved,a.output/'checkpoint.pt')
    core.save_csv(a.output/'train_history.csv',history)
    core.save_json(a.output/'complete.json',{'complete':True,'checkpoint_sha256':sha(a.output/'checkpoint.pt')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',required=True)
    p.add_argument('--resume',type=Path)
    p.add_argument('--additional-steps',type=int,default=15000)
    p.add_argument('--curvature',type=float)
    run(p.parse_args())
