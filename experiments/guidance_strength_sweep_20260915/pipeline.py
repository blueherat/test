"""One idea at a time; parallel coefficients within its connected adaptive search."""
import argparse
import fcntl
import os
import signal
import time
from . import config as k, planning
from experiments import guidance_loss_parallel_20260914 as dispatch


def report():
    state = k.read(k.ROOT / 'status.json') if (k.ROOT / 'status.json').exists() else {}
    searches = k.read(k.ROOT / 'searches.json') if (k.ROOT / 'searches.json').exists() else {}
    lines = ['**Guidance 系数扫描：完成一个idea的训练与评估，再做下一个**', '',
        f"当前：{state.get('current_idea', 'preparing')}；阶段：{state.get('current_step', '')}；状态：{state.get('phase', 'prepared')}。", '',
        '步长0.4 → 0.2 → 0.1 → 0.025；两段入选区间之间全部补齐；每个系数1K。旧MLP与原生IG暂不扫描。', '',
        '[完整评估规则](GUIDANCE_STRENGTH_EVALUATION_20260915_ZH.md)', '',
        '|方法|状态|最佳系数|选择集FID↓|已评估系数数|', '|---|---|---:|---:|---:|']
    for method in k.METHODS:
        search = searches.get(method, {})
        best = search.get('best', {})
        fid = f"{best['fid']:.4f}" if 'fid' in best else ''
        lines.append(f"|{method}|{search.get('state', 'queued')}|{best.get('coefficient', '')}|{fid}|{search.get('evaluated_points', '')}|")
    current = state.get('current_idea')
    if current in searches:
        lines += ['', f'当前 {current} 的结果：', '', '|系数|FID↓|状态|', '|---:|---:|---|']
        points = {t for stage in searches[current].get('stages', []) for t in stage['ticks']}
        default = 32 if k.old.spec(current)['loss'] in k.old.WEAK_LOSSES else 40
        points.add(default)
        for tick in sorted(points):
            point = k.canonical(current, tick)
            path = planning.point_metrics(point)
            row = k.read(path) if path.exists() else {}
            fid = f"{row['fid']:.4f}" if row.get('fid') is not None else ''
            label = '复用' if row.get('reused') else '完成' if row.get('valid') else '数值无效' if row.get('valid') is False else '等待'
            lines.append(f'|{k.value(tick):g}|{fid}|{label}|')
    lines += ['', '下一个idea只会在当前idea全部四级评价结束后启动。未完成训练保留原优化器、EMA、随机状态与50K总预算。',
        '最低FID来自同一组1K调参样本；最优点在搜索边界时明确标记。JiT迁移尚未执行，未来采用同一评估规则。', '', f'原始记录：{k.ROOT}', '']
    temp = k.REPORT.with_suffix('.tmp')
    temp.write_text('\n'.join(lines))
    temp.replace(k.REPORT)


def install():
    # Reuse the tested lease/reaping implementation without editing any frozen file.
    dispatch.k = k
    dispatch.planning = planning
    dispatch.AMENDMENT = k.ROOT / 'request.json'
    dispatch.verify_dispatch = k.verify
    dispatch.report = report
    dispatch.GPU_LIMIT = 4
    dispatch.CPU_LIMIT = 4


class Supervisor(dispatch.Supervisor):
    def __init__(self):
        self.searches = k.read(k.ROOT / 'searches.json') if (k.ROOT / 'searches.json').exists() else {}
        self.current = None
        super().__init__(planning.initial_jobs())
        self.by_id = {job['id']: job for job in self.jobs}
        for method, search in self.searches.items():
            for stage in search.get('stages', []):
                for tick in stage['ticks']:
                    self.add_point(k.canonical(method, tick))
            if search.get('stages'):
                self.add_point(k.key(method, self.default_tick(method)))

    @staticmethod
    def default_tick(method):
        return 32 if k.old.spec(method)['loss'] in k.old.WEAK_LOSSES else 40

    def add_point(self, point):
        for job in planning.point_jobs(point):
            name = job['id']
            if name in self.by_id:
                continue
            row = self.ledger.get(name, {})
            pid = row.get('child_pid')
            if row.get('state') == 'running' and pid and dispatch.start_ticks(pid) == row.get('process_start_ticks'):
                raise RuntimeError(f'Previous search worker still alive: {name} {pid}')
            self.jobs.append(job)
            self.by_id[name] = job
            if planning.verify_output(job):
                self.ledger[name] = dict(state='complete', output=str(planning.output(job)))
            elif row.get('state') in ('failed', 'blocked'):
                self.ledger[name] = row
            else:
                self.ledger[name] = dict(state='queued')

    def save_searches(self):
        k.atomic(k.ROOT / 'searches.json', self.searches)
        k.atomic(k.ROOT / 'plan.json', self.jobs)

    def rows(self, method, ticks):
        result = []
        for tick in ticks:
            point = k.canonical(method, tick)
            metric = k.read(planning.point_metrics(point))
            result.append(dict(tick=tick, coefficient=k.value(tick), point=point,
                valid=metric.get('valid', True), fid=metric.get('fid'), inception_score=metric.get('inception_score'),
                metric_sha256=k.sha(planning.point_metrics(point))))
        return result

    def finish_search(self, method, reason=None):
        search = self.searches[method]
        ticks = sorted({self.default_tick(method), *[t for stage in search['stages'] for t in stage['ticks']]})
        rows = self.rows(method, ticks)
        valid = [row for row in rows if row['valid'] and row['fid'] is not None]
        search.update(state='complete', evaluated_points=len(rows), reason=reason,
            best=min(valid, key=lambda row: (row['fid'], row['tick'])) if valid else {},
            final_rows=rows, finished_utc=dispatch.now())
        best = search['best']
        search['best_at_initial_boundary'] = bool(best and best['tick'] in (0, 80))
        self.save_searches()
        print('Idea fully evaluated', method, search['best'], flush=True)

    def advance(self):
        for method in k.METHODS:
            search = self.searches.setdefault(method, dict(state='queued', stages=[]))
            if search['state'] in ('complete', 'failed'):
                continue
            self.current = method
            if not search['stages']:
                search.update(state='running', started_utc=dispatch.now())
                search['stages'].append(dict(step_ticks=k.STEPS[0], ticks=list(k.COARSE)))
                for tick in k.COARSE:
                    self.add_point(k.canonical(method, tick))
                self.add_point(k.key(method, self.default_tick(method)))
                self.save_searches()
            stage = search['stages'][-1]
            points = [k.canonical(method, tick) for tick in stage['ticks']]
            points.append(k.key(method, self.default_tick(method)))
            states = [self.ledger[planning.point_job_id(point)]['state'] for point in points]
            if any(state in ('failed', 'blocked') for state in states):
                search.update(state='failed', reason='Required training/sampling/evaluation failed', finished_utc=dispatch.now())
                self.save_searches()
                if self.active:
                    return method
                continue
            if not all(state == 'complete' for state in states):
                return method
            if self.active:
                return method
            stage['complete'] = True
            stage['rows'] = self.rows(method, stage['ticks'])
            if stage['step_ticks'] == k.STEPS[-1]:
                self.finish_search(method)
                continue
            next_step = k.STEPS[k.STEPS.index(stage['step_ticks']) + 1]
            intervals, ticks = planning.refine(stage['rows'], stage['step_ticks'], next_step)
            if not ticks:
                self.finish_search(method, 'No valid adjacent interval remains to refine')
                continue
            search['stages'].append(dict(step_ticks=next_step, ticks=ticks,
                selected_intervals=intervals, connected_range=[ticks[0], ticks[-1]],
                selected_from_step=stage['step_ticks']))
            for tick in ticks:
                self.add_point(k.canonical(method, tick))
            self.save_searches()
            return method
        self.current = None
        return None

    def allowed_jobs(self, method):
        needed = {'sweep_preflight', f'evaluate_{method}_screen1000'}
        search = self.searches[method]
        points = {k.canonical(method, t) for stage in search['stages'] for t in stage['ticks']}
        points.add(k.key(method, self.default_tick(method)))
        for point in points:
            needed.update(job['id'] for job in planning.point_jobs(point))
        pending = list(needed)
        while pending:
            for dep in self.by_id[pending.pop()]['depends']:
                if dep not in needed:
                    needed.add(dep)
                    pending.append(dep)
        return needed

    def publish(self, phase=None, **extra):
        current = self.current
        stage = self.searches.get(current, {}).get('stages', [])
        super().publish(phase, current_idea=current,
            current_step=k.value(stage[-1]['step_ticks']) if stage else None,
            execution='one idea at a time', **extra)
        # Keep the familiar queue pointer truthful for future status requests.
        state = k.read(k.ROOT / 'status.json')
        k.atomic(k.old.ROOT / 'status.json', dict(state, managed_by=str(k.ROOT),
            active_report=str(k.REPORT), preserved_scientific_request_sha256=k.sha(k.old.ROOT / 'request.json')))
        k.atomic(k.old.ROOT / 'jobs.json', {job['id']: self.ledger[job['id']]
            for job in self.jobs if job.get('engine') == 'original'})

    def run(self):
        self.publish('queued')
        while True:
            self.reap()
            stopping = (k.ROOT / 'STOP_AFTER_CURRENT').exists() or (k.old.ROOT / 'STOP_AFTER_CURRENT').exists()
            if stopping:
                self.publish('draining' if self.active else 'paused')
                if not self.active:
                    return
                time.sleep(2)
                continue
            if self.ledger['sweep_preflight']['state'] in ('failed', 'blocked'):
                self.publish('failed', reason='New sampling interface failed its GPU checks')
                return
            if self.current and self.searches.get(self.current, {}).get('state') == 'failed' and self.active:
                self.publish('draining_failed_idea')
                time.sleep(2)
                continue
            method = self.advance()
            if method is None and not self.active:
                failed = [m for m, row in self.searches.items() if row['state'] != 'complete']
                k.atomic(k.ROOT / 'completion.json', dict(complete=not failed, failed_ideas=failed,
                    request_sha256=k.sha(k.ROOT / 'request.json'), samples_per_point=1000))
                self.publish('complete_with_failures' if failed else 'complete')
                return
            allowed = self.allowed_jobs(method)
            scoped = [job for job in self.jobs if job['id'] in allowed]
            ready, blocked = dispatch.classify(scoped, self.ledger)
            for job in blocked:
                self.ledger[job['id']] = dict(state='blocked', dependencies=job['depends'])
            self.refresh_gpus()
            ready.sort(key=lambda job: (job.get('engine') != 'sweep', job['gpu']))
            for job in ready:
                if (k.ROOT / 'STOP_AFTER_CURRENT').exists() or (k.old.ROOT / 'STOP_AFTER_CURRENT').exists():
                    break
                if job['gpu']:
                    assignment = self.acquire()
                    if assignment is None:
                        continue
                    self.launch(job, *assignment)
                elif sum(not value['job']['gpu'] for value in self.active.values()) < 4:
                    self.launch(job)
            self.publish()
            time.sleep(2)


def main():
    install()
    def stop(signum, frame):
        message = f'Strength supervisor received signal {signum}; save progress\n'
        (k.ROOT / 'STOP_AFTER_CURRENT').write_text(message)
        (k.old.ROOT / 'STOP_AFTER_CURRENT').write_text(message)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    with (k.ROOT / 'controller.lock').open('a') as lock, (k.old.ROOT / 'controller.lock').open('a') as original_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(original_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        k.verify()
        k.old.verify()
        assert k.read(k.ROOT / 'cpu_preflight.json')['passed']
        Supervisor().run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        k.prepare()
    elif args.report:
        report()
    else:
        main()
