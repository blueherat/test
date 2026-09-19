"""Parallel dispatch amendment; the original 50K scientific request stays frozen."""
import argparse
import datetime
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from experiments.guidance_loss_50k_20260914 import config as k, planning
from experiments.weak_reference_loss_20260914 import idle

MODULE = 'experiments.guidance_loss_parallel_20260914'
PROTOCOL = k.WORK / 'docs/GUIDANCE_PARALLEL_1K_20260914_ZH.md'
AMENDMENT = k.ROOT / 'dispatch_request.json'
GPU_LIMIT, CPU_LIMIT = 4, 1
TERMINAL = {'complete', 'failed', 'blocked'}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def start_ticks(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def plan():
    # Latest user instruction reinstates every CFG arm. No confirm5000 jobs.
    result = planning.jobs()
    assert len({j['id'] for j in result}) == len(result)
    known = set()
    for job in result:
        assert set(job['depends']) <= known, job
        assert job.get('stage', 'screen1000') == 'screen1000'
        known.add(job['id'])
    return result


def classify(jobs, ledger):
    """A pending dependency is different from a failed one in a parallel DAG."""
    ready, blocked = [], []
    for job in jobs:
        if ledger.get(job['id'], {}).get('state', 'queued') != 'queued':
            continue
        states = [ledger.get(dep, {}).get('state', 'queued') for dep in job['depends']]
        if any(state in ('failed', 'blocked') for state in states):
            blocked.append(job)
        elif all(state == 'complete' for state in states):
            ready.append(job)
    return ready, blocked


def prepare():
    k.verify()
    cpu = k.read(k.ROOT / 'cpu_preflight.json')
    assert cpu['passed'] and cpu['request_sha256'] == k.sha(k.ROOT / 'request.json')
    # Create the shared bank once, before multiple sampling processes can race.
    from experiments.guidance_loss_50k_20260914 import components as x
    bank = x.legacy.prepare_bank('screen1000', 1000, k.SCREEN_SEED)
    value = dict(scientific_request_sha256=k.sha(k.ROOT / 'request.json'),
        max_gpu_workers=GPU_LIMIT, max_cpu_workers=CPU_LIMIT,
        gpu_policy='one job per idle GPU; nonblocking DAG; no preemption',
        idle_observations=2, idle_interval_seconds=5,
        cfg_enabled=True, evaluation_models=['sit_small', 'jit'],
        quality_samples_per_configuration=1000, confirm5000_enabled=False,
        new_head_steps=50000, sit_jobs=plan(),
        sampling_inputs={str(bank / 'complete.json'): k.sha(bank / 'complete.json')},
        sources={str(p): k.sha(p) for p in (Path(__file__).resolve(), PROTOCOL)},
        preserved_request=True,
        jit_policy='SiT screening then transfer promising losses; JiT quality comparisons use 1000 images',
        endpoint_bank_note='Existing 2000 clean training endpoints are training data, not FID quality samples')
    if AMENDMENT.exists():
        assert k.read(AMENDMENT) == value, 'Dispatch amendment changed; archive it explicitly before replacement'
    else:
        k.atomic(AMENDMENT, value)
    return value


def verify_dispatch():
    value = k.read(AMENDMENT)
    assert value['scientific_request_sha256'] == k.sha(k.ROOT / 'request.json')
    for group in ('sources', 'sampling_inputs'):
        for path, digest in value[group].items():
            assert k.sha(path) == digest, (group, path)
    return value


def report():
    state = k.read(k.ROOT / 'status.json') if (k.ROOT / 'status.json').exists() else {}
    ledger = k.read(k.ROOT / 'jobs.json') if (k.ROOT / 'jobs.json').exists() else {}
    lines = ['**Guidance loss：四卡并行、每配置 1K**', '',
        f"状态：{state.get('phase', 'prepared')}；GPU任务 {state.get('active_gpu_workers', 0)}/4，CPU任务 {state.get('active_cpu_workers', 0)}/1。", '',
        'CFG 按用户最新指示保留；弱头和辅助头均训练50K。质量比较仅限SiT、JiT，每个配置1K；不自动追加5K。', '',
        '[当前执行修订](GUIDANCE_PARALLEL_1K_20260914_ZH.md) · [原训练协议](GUIDANCE_LOSS_50K_PROTOCOL_20260914_ZH.md)', '',
        '|SiT训练头|loss|已训练步数|状态|', '|---|---|---:|---|']
    for arm, spec in k.ARMS.items():
        root = k.ROOT / 'training' / arm
        p = k.read(root / 'progress.json') if (root / 'progress.json').exists() else {}
        row = ledger.get('train_' + arm, {})
        step = 50000 if row.get('state') == 'complete' else p.get('step', 0)
        lines.append(f"|{arm}|{spec['loss']}|{step}|{row.get('state', 'queued')}|")
    lines += ['', '|运行任务|GPU|PID|', '|---|---|---:|']
    for job, row in ledger.items():
        if row.get('state') == 'running':
            lines.append(f"|{job}|{row.get('gpu_index', 'CPU')}|{row.get('child_pid', '')}|")
    lines += ['', '|模型|方法|图数|FID↓|IS↑|', '|---|---|---:|---:|---:|']
    for path in sorted((k.ROOT / k.MODEL / 'screen1000').glob('*/metrics.json')):
        row = k.read(path)
        lines.append(f"|SiT|{row['arm']}|{row['primary_samples']}|{row['fid']:.4f}|{row['inception_score']:.4f}|")
    lines += ['', 'JiT的新loss迁移在SiT筛选后进行，采样预算同样是1K；不把尚未实施的迁移写成已运行。', '',
        '2000张连续端点训练数据与1K质量采样分开计数。已有完成证明与断点沿用原请求哈希，切换调度不重置优化器或训练步数。', '',
        f'原始记录：{k.ROOT}', '']
    temp = k.REPORT.with_suffix('.tmp')
    temp.write_text('\n'.join(lines))
    temp.replace(k.REPORT)


class Supervisor:
    def __init__(self, jobs):
        self.jobs = jobs
        self.ledger = k.read(k.ROOT / 'jobs.json') if (k.ROOT / 'jobs.json').exists() else {}
        self.active = {}
        self.streak = {}
        self.gpus = []
        self.last_query = -1e10
        self.last_report = -1e10
        self.query_error = None
        # Never duplicate an orphan worker after an unclean supervisor restart.
        for job in jobs:
            row = self.ledger.get(job['id'], {})
            pid = row.get('child_pid')
            if row.get('state') == 'running' and pid and start_ticks(pid) == row.get('process_start_ticks'):
                raise RuntimeError(f"Recorded worker still alive: {job['id']} PID {pid}; drain before restarting")
            if planning.verify_output(job):
                self.ledger[job['id']] = dict(state='complete', output=str(planning.output(job)))
            elif row.get('state') in ('failed', 'blocked'):
                self.ledger[job['id']] = row
            else:
                self.ledger[job['id']] = dict(state='queued')

    def publish(self, phase=None, **extra):
        gpu_count = sum(bool(v['job']['gpu']) for v in self.active.values())
        k.atomic(k.ROOT / 'jobs.json', self.ledger)
        state = dict(phase=phase or ('running' if self.active else 'waiting_gpu'), pid=os.getpid(),
            updated_utc=now(), active_gpu_workers=gpu_count,
            active_cpu_workers=len(self.active) - gpu_count, max_gpu_workers=GPU_LIMIT,
            active_jobs=[dict(job=name, **self.ledger[name]) for name in self.active],
            gpus=self.gpus, query_error=self.query_error, cfg_enabled=True,
            quality_samples_per_configuration=1000, new_head_steps=50000,
            dispatch_sha256=k.sha(AMENDMENT), **extra)
        k.atomic(k.ROOT / 'status.json', state)
        if phase is not None or time.monotonic() - self.last_report >= 15:
            report()
            self.last_report = time.monotonic()

    def refresh_gpus(self):
        if time.monotonic() - self.last_query < 5:
            return
        self.last_query = time.monotonic()
        try:
            self.gpus = idle.gpu_snapshot()
            self.query_error = None
        except (OSError, subprocess.SubprocessError) as error:
            self.gpus, self.streak = [], {}
            self.query_error = repr(error)
            return
        owned = {v.get('gpu_uuid') for v in self.active.values()}
        for gpu in self.gpus:
            uuid = gpu['uuid']
            self.streak[uuid] = self.streak.get(uuid, 0) + 1 if uuid not in owned and idle.eligible(gpu) else 0
            gpu['idle_observations'] = self.streak[uuid]

    def acquire(self):
        if sum(v['job']['gpu'] for v in self.active.values()) >= GPU_LIMIT:
            return None
        owned = {v.get('gpu_uuid') for v in self.active.values()}
        for gpu in self.gpus:
            if gpu['uuid'] in owned or self.streak.get(gpu['uuid'], 0) < 2:
                continue
            lease = Path('/tmp', f"eqvae_idle_{gpu['uuid']}.lock").open('a')
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                current = {g['uuid']: g for g in idle.gpu_snapshot()}.get(gpu['uuid'])
                if current and idle.eligible(current):
                    return current, lease
            except (BlockingIOError, OSError, subprocess.SubprocessError):
                pass
            lease.close()
        return None

    def launch(self, job, gpu=None, lease=None):
        try:
            k.check_stop()
            verify_dispatch()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else gpu['uuid'],
                OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
            command = [k.PYTHON, '-u', '-m', k.MODULE + '.worker', job['action']]
            for key in ('arm', 'source', 'fold', 'stage'):
                if key in job:
                    command += ['--' + key, str(job[key])]
            log = k.ROOT / 'logs' / (job['id'] + '.log')
            log.parent.mkdir(exist_ok=True)
            with log.open('a') as stream:
                child = subprocess.Popen(command, cwd=k.WORK, env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            resource = {} if gpu is None else dict(gpu_index=gpu['index'], gpu_uuid=gpu['uuid'])
            self.active[job['id']] = dict(job=job, process=child, lease=lease, **resource)
            self.ledger[job['id']] = dict(state='running', child_pid=child.pid,
                process_start_ticks=start_ticks(child.pid), started_utc=now(), log=str(log), **resource)
            print('Started', job['id'], child.pid, resource, flush=True)
            self.publish()
        except BaseException:
            if job['id'] not in self.active and lease is not None:
                lease.close()
            raise

    def reap(self):
        for name, value in list(self.active.items()):
            code = value['process'].poll()
            if code is None:
                continue
            job = value['job']
            previous = self.ledger[name]
            try:
                if code == 75:
                    self.ledger[name] = dict(previous, state='queued', last_exit_code=75)
                else:
                    if code:
                        raise RuntimeError(f"Worker exited {code}; {previous['log']}")
                    assert planning.verify_output(job), ('Missing valid receipt', name)
                    self.ledger[name] = dict(previous, state='complete', finished_utc=now(), output=str(planning.output(job)))
            except Exception as error:
                self.ledger[name] = dict(previous, state='failed', finished_utc=now(), error=repr(error))
                traceback.print_exc()
            finally:
                if value['lease'] is not None:
                    value['lease'].close()
                del self.active[name]
            print('Finished', name, self.ledger[name]['state'], flush=True)

    def run(self):
        self.publish('queued')
        while True:
            self.reap()
            stopping = (k.ROOT / 'STOP_AFTER_CURRENT').exists()
            if stopping:
                self.publish('draining' if self.active else 'paused')
                if not self.active:
                    return
                time.sleep(2)
                continue
            ready, blocked = classify(self.jobs, self.ledger)
            for job in blocked:
                self.ledger[job['id']] = dict(state='blocked', dependencies=job['depends'])
            self.refresh_gpus()
            for job in ready:
                if (k.ROOT / 'STOP_AFTER_CURRENT').exists():
                    break
                if job['gpu']:
                    assignment = self.acquire()
                    if assignment is None:
                        continue
                    self.launch(job, *assignment)
                elif sum(not v['job']['gpu'] for v in self.active.values()) < CPU_LIMIT:
                    self.launch(job)
            if not self.active and all(self.ledger[j['id']]['state'] in TERMINAL for j in self.jobs):
                unfinished = [j['id'] for j in self.jobs if self.ledger[j['id']]['state'] != 'complete']
                k.atomic(k.ROOT / 'completion.json', dict(complete=not unfinished, unfinished=unfinished,
                    request_sha256=k.sha(k.ROOT / 'request.json'), dispatch_sha256=k.sha(AMENDMENT),
                    quality_samples_per_configuration=1000, confirm5000_enabled=False))
                self.publish('complete_with_failures' if unfinished else 'complete', unfinished=unfinished)
                return
            self.publish()
            time.sleep(2)


def main():
    def stop(signum, frame):
        (k.ROOT / 'STOP_AFTER_CURRENT').write_text(f'Parallel supervisor received signal {signum}; save checkpoints\n')
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    with (k.ROOT / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_dispatch()
        k.verify()
        jobs = plan()
        k.atomic(k.ROOT / 'plan.json', jobs)
        Supervisor(jobs).run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.report:
        report()
    else:
        main()
