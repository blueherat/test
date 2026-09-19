"""Detached supervisor: resume head5K sweep, then SG5K sweep, and write reports."""
import datetime,fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
WORK=Path('/home/zhoushunyu/eqvae')
PYTHON='/home/zhoushunyu/miniconda3/envs/myenv/bin/python'
ROOT=Path('/home/zhoushunyu/data/eqvae/experiments')
HEAD=ROOT/'raev2_shallow_ig_20260914'
SG=ROOT/'ig_sg_5k_20260914'
STATE=HEAD/'tmux_status.json'
ENV=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
CHILDREN=[]
STREAMS=[]


def state(status,**extra):
    value=dict(status=status,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        supervisor_pid=os.getpid(),session='guidance5k_0914',selection_samples=5000,final_samples=0,independent_validation=False,coefficient_range=[1,2],coarse_step=.2,fine_step=.05,**extra)
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(STATE)
    print(value,flush=True)


def spawn(module,args,log):
    stream=Path(log).open('a',buffering=1);STREAMS.append(stream)
    process=subprocess.Popen([PYTHON,'-u','-m',module,*args],cwd=WORK,env=ENV,
        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    CHILDREN.append(process);return process


def stop(process):
    # Successful processes have no remaining children; avoid stale process-group IDs.
    if process.poll()==0:return
    # Each process owns a separate session/group, including its GPU worker children.
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:pass
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    process.wait()


def run(module,args,log,stage):
    process=spawn(module,args,log);state('running',stage=stage,pipeline_pid=process.pid,log=str(log))
    code=process.wait()
    if code:
        stop(process)
        raise RuntimeError(f'{stage} exited with status {code}; inspect {log}')


def interrupted(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')


def main():
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    with (HEAD/'tmux_supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (SG/'pause_requested.json').exists():
            state('paused',stage='sg_5k_scan_refine_same_bank_controls',pause_request=str(SG/'pause_requested.json'),note='Paused by user; preserve requests and batches until explicit resume.')
            subprocess.run([PYTHON,'-c','from experiments.raev2_shallow_ig_20260914.live_report import write; write()'],cwd=WORK,env=ENV,check=False)
            return
        state('starting',note='Resume existing batches and frozen requests; no automatic retry of scientific or numerical errors')
        try:
            spawn('experiments.raev2_shallow_ig_20260914.live_report',[],HEAD/'live_report.log')
            if not (HEAD/'fivek_report_complete.json').exists():
                run('experiments.raev2_shallow_ig_20260914.finish_135',[],HEAD/'fivek_pipeline.log','mlp8_finish_w1_35_and_report')
            if not (SG/'report_complete.json').exists():
                run('experiments.ig_sg_5k_20260914.selection_only',['pipeline'],ROOT/'ig_sg_5k_pipeline_20260914.log','sg_5k_scan_refine_same_bank_controls')
            for marker in (HEAD/'fivek_report_complete.json',SG/'report_complete.json'):
                assert json.loads(marker.read_text())['complete']
            state('complete',head_report=str(WORK/'docs/RAEV2_SHALLOW_IG_5K_SELECTION_RESULTS_20260914_ZH.md'),
                sg_report=str(WORK/'docs/IG_SG_5K_RESULTS_20260914_ZH.md'))
        except BaseException as error:
            state('failed',error=repr(error));traceback.print_exc();raise
        finally:
            for process in reversed(CHILDREN):stop(process)
            for stream in STREAMS:stream.close()
            # Final refresh of the provisional table, without touching scientific outputs.
            subprocess.run([PYTHON,'-c','from experiments.raev2_shallow_ig_20260914.live_report import write; write()'],cwd=WORK,env=ENV,check=False)

if __name__=='__main__':main()
