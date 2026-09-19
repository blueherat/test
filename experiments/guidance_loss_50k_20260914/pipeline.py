"""CPU-only persistent supervisor, admitting one genuinely idle GPU at a time."""
import argparse
import datetime
import fcntl
import os
import signal
import subprocess
import traceback
from . import config as k
from . import planning
from experiments.weak_reference_loss_20260914 import idle

idle.k=k
idle.PREDECESSORS=(*idle.PREDECESSORS,b'experiments.weak_reference_loss_20260914.pipeline')
CURRENT=None


def status(phase,**extra):
    k.atomic(k.ROOT/'status.json',dict(phase=phase,pid=os.getpid(),
        updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra))


def metrics(stage):
    result={}
    for path in sorted((k.ROOT/k.MODEL/stage).glob('*/metrics.json')):
        row=k.read(path)
        assert row['request_sha256']==k.sha(k.ROOT/'request.json')
        result[row['arm']]=row
    return result


def report():
    state=k.read(k.ROOT/'status.json') if (k.ROOT/'status.json').exists() else dict(phase='prepared')
    ledger=k.read(k.ROOT/'jobs.json') if (k.ROOT/'jobs.json').exists() else {}
    lines=['**Guidance loss 全候选 50K 队列**','',
        f"状态：{state['phase']}。所有新向量头、每折来源概率头和混合均值头固定训练50,000步。",'',
        '[实施协议](GUIDANCE_LOSS_50K_PROTOCOL_20260914_ZH.md) · [理论记录](IG_GUIDANCE_INFORMATION_AND_CONTRASTIVE_LOSS_20260914_ZH.md)','',
        '|训练头|目标|已训练步数|状态|','|---|---|---:|---|']
    for arm,spec in k.ARMS.items():
        root=k.ROOT/'training'/arm
        progress=k.read(root/'progress.json') if (root/'progress.json').exists() else {}
        row=ledger.get('train_'+arm,{})
        lines.append(f"|{arm}|{spec['loss']}|{progress.get('step',0)}|{row.get('state','queued')}|")
    lines+=['','|阶段|方法|图数|FID↓|IS↑|完整前向/新头/原弱头调用|','|---|---|---:|---:|---:|---|']
    for stage in ('screen1000','confirm5000'):
        for arm,row in metrics(stage).items():
            lines.append(f"|{stage}|{arm}|{row['primary_samples']}|{row['fid']:.4f}|{row['inception_score']:.4f}|"
                         f"{row['full_calls_per_output']}/{row['head_calls_per_output']}/{row.get('native_head_calls_per_output',0)}|")
    lines+=['','历史Context对照为原3K权重；新real头提供50K匹配训练对照。独立5K只对每个计算预算组的最佳通过者及相应对照进行。',
        '',f"原始记录：{k.ROOT}。任务失败会阻止其依赖项，其他独立实验继续；不会把尚未运行的实验当成完成。",'',
        '验证loss固定在每类最后2个端点及固定新噪声上；所有头训练只使用其余18个端点。训练不依据验证loss提前结束或挑选checkpoint。','']
    temporary=k.REPORT.with_suffix('.tmp')
    temporary.write_text('\n'.join(lines));temporary.replace(k.REPORT)


def run(job):
    global CURRENT
    k.check_stop();k.verify()
    lease=None
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    resource={}
    if job['gpu']:
        gpu,lease=idle.wait_gpu(lambda phase,**extra:status(phase,next_job=job['id'],**extra))
        env['CUDA_VISIBLE_DEVICES']=gpu['uuid']
        resource=dict(gpu_index=gpu['index'],gpu_uuid=gpu['uuid'])
    command=[k.PYTHON,'-u','-m',k.MODULE+'.worker',job['action']]
    for key in ('arm','source','fold','stage'):
        if key in job:command+=['--'+key,str(job[key])]
    log=k.ROOT/'logs'/(job['id']+'.log')
    log.parent.mkdir(exist_ok=True)
    try:
        with log.open('a') as stream:
            CURRENT=subprocess.Popen(command,cwd=k.WORK,env=env,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            status('running',job=job['id'],child_pid=CURRENT.pid,log=str(log),**resource)
            report()
            code=CURRENT.wait()
        if code==75:raise k.RequestedStop(job['id'])
        if code:raise RuntimeError(f"{job['id']} exited {code}; {log}")
        assert planning.verify_output(job),('missing output',job['id'])
    finally:
        if CURRENT is not None and CURRENT.poll() is None:
            os.killpg(CURRENT.pid,signal.SIGTERM)
            try:CURRENT.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(CURRENT.pid,signal.SIGKILL);CURRENT.wait()
        CURRENT=None
        if lease is not None:lease.close()


def select():
    rows=metrics('screen1000')
    decisions={}
    for candidate,controls in k.candidate_controls().items():
        if any(arm not in rows for arm in [candidate,*controls]):
            decisions[candidate]=dict(passed=False,reason='incomplete candidate or controls')
            continue
        best=min(controls,key=lambda arm:rows[arm]['fid'])
        native='cfg_native' if k.spec(candidate)['base']=='cfg' else 'native'
        margin=rows[best]['fid']-rows[candidate]['fid']
        passed=margin>=1 and rows[candidate]['inception_score']>=.9*rows[native]['inception_score']
        if candidate=='excess' and k.read(k.ROOT/'weights/complete.json')['degenerate']:passed=False
        decisions[candidate]=dict(passed=passed,fid_margin=margin,best_control=best,controls=controls)
    confirmations=[]
    for budget in (128,224):
        pool=[arm for arm,d in decisions.items() if d['passed'] and k.inference_counts(arm)['full']==budget]
        if not pool:continue
        winner=min(pool,key=lambda arm:rows[arm]['fid'])
        native='native' if budget==128 else 'cfg_native'
        confirmations.extend([winner,decisions[winner]['best_control'],native])
    value=dict(candidates=decisions,confirmation_arms=list(dict.fromkeys(confirmations)),
        request_sha256=k.sha(k.ROOT/'request.json'))
    k.atomic(k.ROOT/'decision.json',value)
    return value


def main():
    k.verify()
    cpu=k.read(k.ROOT/'cpu_preflight.json')
    assert cpu['passed'] and cpu['request_sha256']==k.sha(k.ROOT/'request.json')
    with (k.ROOT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=planning.jobs()
        k.atomic(k.ROOT/'plan.json',jobs)
        ledger=k.read(k.ROOT/'jobs.json') if (k.ROOT/'jobs.json').exists() else {}
        status('queued',jobs=len(jobs),new_head_steps=k.STEPS)
        report()
        try:
            for job in jobs:
                k.check_stop()
                if any(ledger.get(dep,{}).get('state')!='complete' for dep in job['depends']):
                    ledger[job['id']]=dict(state='blocked',dependencies=job['depends'])
                else:
                    try:
                        if not planning.verify_output(job):
                            ledger[job['id']]=dict(state='running')
                            k.atomic(k.ROOT/'jobs.json',ledger)
                            run(job)
                        ledger[job['id']]=dict(state='complete',output=str(planning.output(job)))
                    except k.RequestedStop:raise
                    except Exception as error:
                        ledger[job['id']]=dict(state='failed',error=repr(error))
                        traceback.print_exc()
                k.atomic(k.ROOT/'jobs.json',ledger);report()
            decision=select()
            for arm in decision['confirmation_arms']:
                for action in ('sample','evaluate'):
                    job=dict(id=action+'_'+arm+'_confirm5000',action=action,arm=arm,stage='confirm5000',gpu=action=='sample')
                    try:
                        if not planning.verify_output(job):run(job)
                        ledger[job['id']]=dict(state='complete',output=str(planning.output(job)))
                    except k.RequestedStop:raise
                    except Exception as error:
                        ledger[job['id']]=dict(state='failed',error=repr(error))
                        traceback.print_exc()
                        break
                    finally:k.atomic(k.ROOT/'jobs.json',ledger)
            failed=[name for name,row in ledger.items() if row['state']!='complete']
            status('complete_with_failures' if failed else 'complete',unfinished=failed,
                   confirmation_arms=decision['confirmation_arms'])
            k.atomic(k.ROOT/'completion.json',dict(complete=not failed,unfinished=failed,decision=decision,
                request_sha256=k.sha(k.ROOT/'request.json')))
            report()
        except k.RequestedStop as error:
            status('paused',reason=str(error));report()
        except BaseException as error:
            status('failed',error=repr(error));report()
            raise


if __name__=='__main__':
    def interrupted(signum,frame):
        raise KeyboardInterrupt(f'Signal {signum}; stopping only this queue child')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    parser=argparse.ArgumentParser();parser.add_argument('--report',action='store_true')
    if parser.parse_args().report:report()
    else:main()
