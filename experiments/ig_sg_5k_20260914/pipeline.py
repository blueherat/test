"""Respect requested order: finish post-trained heads, then SG tuning and 5K."""
import os,subprocess,sys,time
from experiments.ig_sg_5k_20260914 import grid as r
from experiments.raev2_shallow_ig_20260914 import core as heads


def main():
    while not (heads.ROOT/'fivek_report_complete.json').exists():time.sleep(5)
    assert heads.c.read(heads.ROOT/'fivek_report_complete.json')['complete']
    r.c.atomic(r.ROOT/'pipeline_manifest.json',dict(source_sha256=r.c.sha(__file__),
        head_report_complete=r.c.sha(heads.ROOT/'fivek_report_complete.json'),order=list(r.STAGES)))
    for phase in r.STAGES:
        done=r.ROOT/phase/'controller_complete.json'
        if done.exists() and r.c.read(done)['complete']:continue
        for action in ('prepare','controller'):
            subprocess.run([sys.executable,'-u','-m',r.MODULE,action,'--phase',phase],check=True,cwd=r.c.WORK)
    r.c.atomic(r.ROOT/'pipeline_complete.json',dict(complete=True,models=3,final_arms=15,final_new_samples=75000))
    subprocess.run([sys.executable,'-u','-m','experiments.ig_sg_5k_20260914.report'],check=True,cwd=r.c.WORK,
        env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    print('SG search, all 15 independent 5K arms, and report finished',flush=True)

if __name__=='__main__':main()
