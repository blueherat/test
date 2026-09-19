"""Persistent single-GPU queue, with CPU evaluation and bounded confirmation."""
import argparse
import datetime
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

from . import config as k
from . import idle

CURRENT=None


def status(phase,**extra):
    k.atomic(k.ROOT/'status.json',dict(phase=phase,pid=os.getpid(),
        updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra))


def result_rows(stage):
    rows=[]
    for path in sorted((k.ROOT/k.MODEL/stage).glob('*/metrics.json')):
        row=k.read(path)
        assert row['request_sha256']==k.sha(k.ROOT/'request.json')
        assert row['samples_sha256']==k.sha(path.parent/'samples.npz')
        rows.append(row)
    return rows


def report():
    rows=result_rows('screen1000')
    confirmation=result_rows('confirm5000')
    state=k.read(k.ROOT/'status.json') if (k.ROOT/'status.json').exists() else {'phase':'prepared'}
    lines=['# 参考分布训练目标：执行与结果','',
        f"执行状态：`{state['phase']}`。当前完成1K评估{len(rows)}组、独立5K评估{len(confirmation)}组。",'',
        '这是固定目标比较，不是参数扫描。新方法只有完成生成评估后才能判断质量。', '',
        '[冻结协议](WEAK_REFERENCE_LOSS_PROTOCOL_20260914_ZH.md) · [目标推导](WEAK_REFERENCE_LOSS_DESIGN_20260914_ZH.md)', '',
        '|阶段|方法|样本数|FID↓|IS↑|完整/额外前缀调用|',
        '|---|---|---:|---:|---:|---|']
    for name,group in [('筛选',rows),('独立确认',confirmation)]:
        for row in group:
            lines.append(f"|{name}|{row['arm']}|{row['primary_samples']}|{row['fid']:.4f}|{row['inception_score']:.4f}|128/0|")
    if not rows:
        lines+=['','当前尚无新目标的图像质量结果。GPU预检也必须在空闲后完成。']
    decision=k.ROOT/'decision.json'
    if decision.exists():
        value=k.read(decision)
        lines+=['',f"筛选晋级：{value['selected'] or '无'}。资源门槛为比指定全部对照低至少1 FID，且IS不低于原IG的90%；不代表统计显著性。"]
    weights=k.ROOT/'weights/complete.json'
    if weights.exists():
        value=k.read(weights)
        lines+=['',f"离线权重标准差：{value['normalized_weight_std']:.6g}；退化标记：{value['degenerate']}。该比值只在固定表征空间估计。"]
    lines+=['',f"原始产物：`{k.ROOT}`。训练和采样均可按同一请求校验恢复，STOP_AFTER_CURRENT仅作用于本实验。",
        '', '来源bank为各2K连续latent，所有新头固定3K；原有Context控制来自既有3K训练。所有新采样均为独立于训练bank的新噪声、类平衡配对。',
        '', '采样和解码墙钟时间保存在原始batch中，不当作经过专门预热控制的速度基准。', '']
    (k.WORK/'docs/WEAK_REFERENCE_LOSS_RESULTS_20260914_ZH.md').write_text('\n'.join(lines))
    k.atomic(k.ROOT/'metrics_index.json',dict(screen=rows,confirmation=confirmation))


def run(action,arm=None,stage=None,gpu=True):
    global CURRENT
    k.check_stop()
    k.verify()
    label='_'.join(x for x in (action,stage,arm) if x)
    lease=None
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    if gpu:
        row,lease=idle.wait_gpu(lambda phase,**extra:status(phase,next_job=label,**extra))
        env['CUDA_VISIBLE_DEVICES']=row['uuid']
        resource=dict(gpu_index=row['index'],gpu_uuid=row['uuid'])
    else:
        env['CUDA_VISIBLE_DEVICES']=''
        resource=dict(cpu_only=True)
    command=[k.PYTHON,'-u','-m','experiments.weak_reference_loss_20260914.runner',action]
    if arm:command+=['--arm',arm]
    if stage:command+=['--stage',stage]
    log=k.ROOT/'logs'/f'{label}.log'
    log.parent.mkdir(exist_ok=True)
    try:
        with log.open('a') as stream:
            CURRENT=subprocess.Popen(command,cwd=k.WORK,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            status('running',job=label,child_pid=CURRENT.pid,log=str(log),**resource)
            report()
            code=CURRENT.wait()
        if code==75:
            raise k.RequestedStop(label+' stopped by STOP_AFTER_CURRENT')
        if code:
            raise RuntimeError(f'{label} exited {code}; inspect {log}')
    finally:
        if CURRENT is not None and CURRENT.poll() is None:
            os.killpg(CURRENT.pid,signal.SIGTERM)
            try:CURRENT.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(CURRENT.pid,signal.SIGKILL)
                CURRENT.wait()
        CURRENT=None
        if lease is not None:lease.close()


def ensure_train(arm):
    out=k.ROOT/'training'/arm
    if not (out/'complete.json').exists():
        run('train',arm)
    done=k.read(out/'complete.json')
    assert done['complete'] and done['request_sha256']==k.sha(k.ROOT/'request.json')
    assert done['head_sha256']==k.sha(out/'head.pt')


def ensure_quality(arm,stage='screen1000'):
    out=k.ROOT/k.MODEL/stage/arm
    if not (out/'summary.json').exists():run('sample',arm,stage)
    summary=k.read(out/'summary.json')
    assert summary['complete'] and summary['request_sha256']==k.sha(k.ROOT/'request.json')
    assert summary['samples_sha256']==k.sha(out/'samples.npz')
    if not (out/'metrics.json').exists():run('evaluate',arm,stage,gpu=False)
    report()


def select():
    rows={row['arm']:row for row in result_rows('screen1000')}
    criteria={'mixture':['real','self','gaussian','gaussian_half','context','native']}
    if 'excess' in rows:
        criteria['excess']=['real','self','context','native','shuffled','gaussian','gaussian_half','mixture']
    decisions={}
    for candidate,controls in criteria.items():
        margin=min(rows[arm]['fid'] for arm in controls)-rows[candidate]['fid']
        passed=margin>=1 and rows[candidate]['inception_score']>=.9*rows['native']['inception_score']
        decisions[candidate]=dict(passed=passed,fid_margin=margin,controls=controls)
    eligible=[arm for arm,value in decisions.items() if value['passed']]
    selected=min(eligible,key=lambda arm:rows[arm]['fid']) if eligible else None
    confirmation=[]
    if selected:
        best_control=min(decisions[selected]['controls'],key=lambda arm:rows[arm]['fid'])
        confirmation=list(dict.fromkeys([selected,best_control,'native']))
    value=dict(selected=selected,criteria=decisions,confirmation_arms=confirmation,
        screen_seed=k.SCREEN_SEED,confirmation_seed=k.CONFIRM_SEED,
        request_sha256=k.sha(k.ROOT/'request.json'))
    path=k.ROOT/'decision.json'
    if path.exists():assert k.read(path)==value
    else:k.atomic(path,value)
    return value


def main():
    k.verify()
    cpu=k.read(k.ROOT/'cpu_preflight.json')
    assert cpu['passed'] and cpu['request_sha256']==k.sha(k.ROOT/'request.json')
    with (k.ROOT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            status('queued',note='Wait for existing head->SG queue, then one idle GPU; no automatic retry after failures')
            report()
            if not (k.ROOT/'gpu_preflight.json').exists():run('gpu_preflight')
            gpu=k.read(k.ROOT/'gpu_preflight.json')
            assert gpu['passed'] and gpu['request_sha256']==k.sha(k.ROOT/'request.json')
            for arm in ('real','self','gaussian','mixture'):ensure_train(arm)
            for arm in k.SCREEN_ARMS[:7]:ensure_quality(arm)
            if not (k.ROOT/'weights/features_complete.json').exists():run('features')
            if not (k.ROOT/'weights/complete.json').exists():run('weights',gpu=False)
            if not k.read(k.ROOT/'weights/complete.json')['degenerate']:
                for arm in ('excess','shuffled'):
                    ensure_train(arm)
                    ensure_quality(arm)
            decision=select()
            report()
            for arm in decision['confirmation_arms']:ensure_quality(arm,'confirm5000')
            status('complete',selected=decision['selected'],confirmation_arms=decision['confirmation_arms'])
            report()
            k.atomic(k.ROOT/'complete.json',dict(complete=True,decision=decision,
                screen_metrics=len(result_rows('screen1000')),confirmation_metrics=len(result_rows('confirm5000')),
                request_sha256=k.sha(k.ROOT/'request.json')))
        except k.RequestedStop as error:
            status('paused',reason=str(error))
            report()
        except BaseException as error:
            status('failed',error=repr(error))
            report()
            traceback.print_exc()
            raise


if __name__=='__main__':
    def interrupted(signum,frame):
        raise KeyboardInterrupt(f'Signal {signum}; only this queue child will be stopped')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',action='store_true')
    args=parser.parse_args()
    if args.report:report()
    else:main()
