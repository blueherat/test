from . import install, k, AMENDMENT
import os
import subprocess


def main():
    install()
    from experiments.guidance_dynamic_50k_20260915 import pipeline as original
    old_publish = original.Supervisor.publish
    old_add = original.Supervisor.add
    old_advance = original.Supervisor.advance
    all_models = tuple(k.MODELS)
    training_actions = ('normalize', 'check', 'train', 'nuisance')

    def add(self, action, model, output, depends=(), gpus=0, **params):
        if action in training_actions:
            gpus = k.WORLD
        return old_add(self, action, model, output, depends, gpus, **params)

    def basics(self, model):
        root = k.model_root(model)
        inputs = self.add('inputs', model, root / 'quality_inputs/complete.json')
        normalize = self.add('normalize', model, root / 'calibration/complete.json', gpus=k.WORLD)
        return self.add('check_runtime2', model, root / 'checks_runtime2/complete.json',
                        depends=[normalize, inputs], gpus=k.WORLD)

    def advance(self):
        # Keep the existing per-idea search and extension state machine, but exhaust
        # one model before admitting the other. No completed SiT point is rerun.
        for model in all_models:
            if all(self.searches.get(model + '/' + method, {}).get('phase') == 'complete'
                   for method in k.METHODS):
                continue
            previous = k.MODELS
            k.MODELS = (model,)
            try:
                allowed = old_advance(self)
            finally:
                k.MODELS = previous
            if allowed:
                return allowed
        return set()

    def launch(self, job, assignment=None):
        devices, leases = assignment or ([], [])
        k.verify()
        k.check_stop()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in devices),
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   PYTHONUNBUFFERED='1', TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
        action = 'check' if job['action'] == 'check_runtime2' else job['action']
        command = [k.PYTHON, '-u']
        if action in training_actions:
            assert len(devices) == k.WORLD
            command += ['-m', 'torch.distributed.run', '--standalone',
                        '--nproc-per-node=' + str(k.WORLD), '-m', k.MODULE + '.worker']
        else:
            command += ['-m', k.MODULE + '.worker']
        command += [action, '--model', job['model']]
        for name, value in job['params'].items():
            command += ['--' + name, str(value)]
        log = k.ROOT / 'logs' / (job['id'] + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a') as stream:
            child = subprocess.Popen(command, cwd=k.WORK, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        self.active[job['id']] = dict(job=job, process=child, leases=leases,
                                     gpu_uuids=[g['uuid'] for g in devices])
        self.ledger[job['id']] = dict(state='running', pid=child.pid,
            process_start_ticks=original.process_ticks(child.pid),
            gpu_indices=[g['index'] for g in devices], gpu_uuids=[g['uuid'] for g in devices],
            log=str(log), started_utc=original.now(), runtime_amendment_sha256=k.sha(AMENDMENT))
        print('Started', job['id'], child.pid, self.ledger[job['id']]['gpu_indices'], flush=True)

    def publish(self, phase=None):
        previous = k.MODELS
        k.MODELS = all_models
        try:
            old_publish(self, phase)
        finally:
            k.MODELS = previous
        state = k.read(k.ROOT / 'status.json')
        state['runtime_amendment_sha256'] = k.sha(AMENDMENT)
        state['sequence'] = 'all SiT ideas, then all JiT ideas'
        state['training_world_size'] = k.WORLD
        state['models'] = list(all_models)
        ready = [name for name, job in self.jobs.items()
                 if self.ledger[name]['state'] == 'queued' and
                 all(self.ledger[d]['state'] == 'complete' for d in job['depends'])]
        state['ready_jobs'] = ready
        k.atomic(k.ROOT / 'status.json', state)
        text = k.REPORT.read_text()
        lines = text.splitlines()
        lines[0] = '**完整数据动态采样重训：先 SiT 全部 idea，再 JiT 全部 idea；当前两卡训练，全局 batch 256**'
        lines[2] = (f"队列状态：{state['phase']}。当前 {state['current_model']} / {state['current_idea']}："
                    f"{state['current_phase']}。GPU任务占用 {state['active_gpu_workers']}/4。")
        if ready and not self.active:
            lines[2] += ' 待启动：' + ', '.join(ready) + '（等待所需空闲资源）。'
        lines[4] = '每个模型内部：一个 idea 训练 50K → 四级 1K 扫描 → 前两名各扩展 5K → 下一个 idea；SiT 全部完成后进入 JiT。'
        tmp = k.REPORT.with_suffix('.tmp')
        tmp.write_text('\n'.join(lines) + '\n')
        tmp.replace(k.REPORT)

    original.Supervisor.publish = publish
    original.Supervisor.add = add
    original.Supervisor.basics = basics
    original.Supervisor.advance = advance
    original.Supervisor.launch = launch
    original.main()


if __name__ == '__main__':
    main()
