"""One-GPU preflight, then 30K joint GAN updates without unattended pilot stalls."""
import json
from pathlib import Path
import subprocess
import sys

from experiments.adversarial_weak_training_20260915 import common as c

ROOT=Path('/home/zhoushunyu/data/eqvae/projects/classifier_guidance/sit_joint_gan_20260922')
HEAD=Path('/home/zhoushunyu/data/eqvae/projects/classifier_guidance/sit_transformer_capacity_20260921/block1/training_50k/checkpoint_050000.pt')


def main():
    assert c.read(ROOT/'audit_v1/audit.json')['passed']
    request=c.read(ROOT/'audit_v1/request.json')
    for name in ('sit_joint.py','sampler.py','sit_transformer_heads.py'):
        source=str(c.WORK/'classifier_guidance'/name)
        assert request['sources'][source]==c.sha(source),f'Unaudited change: {source}'
    for phase,updates,warmup in (('preflight',4,4),('training_30k',30000,128)):
        if phase=='training_30k':
            rows=[json.loads(line) for line in (ROOT/'preflight/train.jsonl').read_text().splitlines()]
            training=[r for r in rows if r['phase']=='training']
            assert len(training)==4
            for row in training:
                for key in ('weak_update_norm','coefficient_update_norm','d_update_norm','weak_gan_gradient_norm'):
                    assert row[key]>0,(key,row)
                assert row['coefficient_nonzero_gradients']==64
                assert row['g_loss_saturation']==0
            c.atomic(ROOT/'preflight_passed.json',dict(passed=True,training_updates=4,
                mean_seconds=sum(r['seconds'] for r in training)/len(training),
                peak_allocated_gib=max(r['peak_allocated_gib'] for r in training),
                peak_reserved_gib=max(r['peak_reserved_gib'] for r in training),
                last=training[-1],updated_utc=c.now()))
        command=[sys.executable,'-u','-m','classifier_guidance.launch','--gpus','1','--task','sit-joint',
            '--output',str(ROOT/phase),'--','--head-checkpoint',str(HEAD),'--updates',str(updates),
            '--warmup',str(warmup),'--global-batch','32','--microbatch','8','--coefficient','.75',
            '--lr-w','1e-6','--lr-a','2e-4','--norm-weight','.1','--save-every','300']
        c.atomic(ROOT/'pipeline_progress.json',dict(phase=phase,command=command,updated_utc=c.now()))
        subprocess.run(command,cwd=c.WORK,check=True)
    c.atomic(ROOT/'pipeline_complete.json',dict(complete=True,updated_utc=c.now()))


if __name__=='__main__':main()
