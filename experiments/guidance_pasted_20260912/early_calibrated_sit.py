"""Use the idle SiT GPU while RAE source paths run; preserve raw worker ordering."""
import os
import signal
import subprocess
import time
from . import common as c,ig,calibrate as cal
from .run_ig import aggregate


def main():
    root=c.ROOT/'sit_small'/ig.STAGE
    while not all((root/s['arm']/'summary.json').exists() for s in ig.CONFIGS):time.sleep(3)
    state=c.read(c.ROOT/'ig_status.json')
    if state['phase']!='sit_quality_and_rae_sources':return
    if (c.ROOT/'raev2'/ig.DATA/'complete.json').exists():return
    pid=state['pid'];cmd=open(f'/proc/{pid}/cmdline','rb').read().split(b'\0')
    assert b'experiments.guidance_pasted_20260912.run_ig' in cmd
    # The raw RAE source child continues; only its parent's next GPU allocation waits.
    os.kill(pid,signal.SIGSTOP);job=None;log=None
    c.atomic(c.ROOT/'early_calibrated_sit.json',dict(phase='running',pid=os.getpid(),paused_raw_controller=pid))
    try:
        cal.configure();out=c.ROOT/'sit_small'/cal.STAGE
        log=(out/'worker0.log').open('a')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        job=subprocess.Popen([c.PYTHON,'-u','-m','experiments.guidance_pasted_20260912.calibrate','--model','sit_small',
            '--rank','0','--world','1','--parent-pid',str(os.getpid())],env=env,cwd=c.WORK,stdout=log,stderr=subprocess.STDOUT)
        while job.poll() is None:aggregate('sit_small',1);time.sleep(3)
        if job.returncode:raise RuntimeError('Early calibrated SiT worker failed')
        aggregate('sit_small',1)
        from .evaluate import watch
        watch('sit_small',cal.STAGE,[s['arm'] for s in cal.CONFIGS])
        c.atomic(c.ROOT/'early_calibrated_sit.json',dict(phase='complete',pid=os.getpid(),paused_raw_controller=pid))
    finally:
        if job is not None and job.poll() is None:job.terminate();job.wait()
        if log is not None:log.close()
        os.kill(pid,signal.SIGCONT)


if __name__=='__main__':main()
