import argparse
import datetime
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
from . import config as k
from .sampling import canonical,point_root
from experiments.weak_reference_loss_20260914 import idle


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def process_ticks(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (FileNotFoundError,ProcessLookupError):return None


class Supervisor:
    def __init__(self):
        self.jobs={};self.ledger=k.read(k.ROOT/'jobs.json') if (k.ROOT/'jobs.json').exists() else {}
        self.searches=k.read(k.ROOT/'searches.json') if (k.ROOT/'searches.json').exists() else {}
        self.active={};self.gpus=[];self.streak={};self.last_query=0.;self.last_report=0.;self.current=None
        for name,row in self.ledger.items():
            if row.get('state')=='running' and process_ticks(row['pid'])==row['process_start_ticks']:
                raise RuntimeError(f'Existing worker is still alive: {name} {row["pid"]}')
    def add(self,action,model,output,depends=(),gpus=0,**params):
        name='__'.join([action,model,*[str(v) for v in params.values()]])
        job=dict(id=name,action=action,model=model,params=params,depends=list(depends),gpus=gpus,output=str(output))
        if name not in self.jobs:
            self.jobs[name]=job
            if self.valid(job):self.ledger[name]=dict(state='complete',output=str(output))
            elif self.ledger.get(name,{}).get('state') not in ('failed','blocked'):self.ledger[name]=dict(state='queued')
        else:assert self.jobs[name]==job,name
        return name
    def valid(self,job):
        path=Path(job['output'])
        if not path.exists():return False
        row=k.read(path)
        assert row['request_sha256']==k.sha(k.ROOT/'request.json'),path
        if not row.get('complete',row.get('passed',False)):return False
        if 'head_sha256' in row:assert k.sha(path.parent/'head.pt')==row['head_sha256']
        if 'samples_sha256' in row:assert k.sha(row['samples_path'])==row['samples_sha256']
        if job['action'] in ('train','nuisance'):assert row['steps']==k.STEPS
        return True
    def basics(self,model):
        root=k.model_root(model)
        inputs=self.add('inputs',model,root/'quality_inputs/complete.json')
        normalize=self.add('normalize',model,root/'calibration/complete.json',gpus=4)
        check=self.add('check',model,root/'checks/complete.json',depends=[normalize,inputs],gpus=4)
        return check
    def endpoints(self,model,source):
        if source=='weakmix':return self.endpoints(model,'strong')+self.endpoints(model,'weak')
        deps=[self.basics(model)]
        if source=='weak':deps.append(self.training(model,'self'))
        completed=[]
        for split in ('train','validation'):
            root=k.model_root(model)/'endpoints'/source/split
            shards=[self.add('generate',model,root/f'rank{rank}.json',depends=deps,gpus=1,
                source=source,split=split,rank=rank) for rank in range(4)]
            completed.append(self.add('endpoints_collect',model,root/'complete.json',depends=shards,source=source,split=split))
        return completed
    def nuisance(self,model,source):
        deps=self.endpoints(model,source);jobs=[]
        for fold in (0,1):
            jobs.append(self.add('nuisance',model,k.model_root(model)/'nuisance'/source/f'fold{fold}/complete.json',
                depends=deps,gpus=4,source=source,fold=fold))
        return jobs
    def weights(self,model):
        root=k.model_root(model)/'weights';deps=self.endpoints(model,'strong')
        parts=[self.add('features',model,root/f'features_rank{rank}.json',depends=deps,gpus=1,rank=rank) for rank in range(4)]
        return self.add('weights',model,root/'complete.json',depends=parts)
    def training(self,model,arm):
        spec=k.ARMS[arm];deps=[self.basics(model)]
        if spec['source']!='real':deps.extend(self.endpoints(model,spec['source']))
        if spec['loss'] in ('contrast','contrast_weak','covariance'):deps.extend(self.nuisance(model,spec['source']))
        if spec.get('weight'):deps.append(self.weights(model))
        return self.add('train',model,k.model_root(model)/'training'/arm/'complete.json',depends=deps,gpus=4,arm=arm)
    def quality(self,model,method,tick,n):
        point=canonical(model,method,tick);root=point_root(model,point)/f'n{n}'
        deps=[self.basics(model)]
        if k.parse_point(point)[0] in k.ARMS:deps.append(self.training(model,k.parse_point(point)[0]))
        if n==5000:deps.append(self.quality(model,method,tick,1000))
        shards=[self.add('sample',model,root/f'rank{rank}.json',depends=deps,gpus=1,point=point,n=n,rank=rank) for rank in range(4)]
        merged=self.add('collect',model,root/'summary.json',depends=shards,point=point,n=n)
        return self.add('evaluate',model,root/'metrics.json',depends=[merged],gpus=int(model=='jit'),point=point,n=n)
    def closure(self,goals):
        result=set(goals);pending=list(goals)
        while pending:
            for dep in self.jobs[pending.pop()]['depends']:
                if dep not in result:result.add(dep);pending.append(dep)
        return result
    def rows(self,model,method,ticks,n=1000):
        rows=[]
        for tick in ticks:
            point=canonical(model,method,tick);path=point_root(model,point)/f'n{n}'/'metrics.json'
            metric=k.read(path)
            rows.append(dict(tick=tick,coefficient=tick/40,fid=metric.get('fid'),
                inception_score=metric.get('inception_score'),valid=metric.get('valid',True),
                point=point,n=n,metrics_path=str(path),metrics_sha256=k.sha(path)))
        return rows
    def retire_generated_cache(self,model):
        for source in ('strong','weak','ig','cfg'):
            consumers=[method for method in k.METHODS if k.ARMS[method]['source']==source or
                (source in ('strong','weak') and k.ARMS[method]['source']=='weakmix')]
            if not consumers or not all(self.searches.get(model+'/'+m,{}).get('phase')=='complete' for m in consumers):continue
            root=k.model_root(model)/'endpoints'/source
            if not root.exists() or (root/'bulk_retired.json').exists():continue
            files=[]
            for split in ('train','validation'):
                if not (root/split/'complete.json').exists():continue
                files.extend((root/split/'batches').glob('*.npz'))
                files.extend((root/split/'batches').glob('*.bin'))
                if (root/split/'clean.npy').exists():files.append(root/split/'clean.npy')
            inventory={str(p):p.stat().st_size for p in files}
            k.atomic(root/'bulk_retirement_plan.json',dict(files=inventory,all_consumers_complete=consumers,
                recoverable_from_frozen_seeds_and_models=True,recorded_utc=now()))
            for path in files:
                assert path.resolve().is_relative_to(k.ROOT.resolve())
                path.unlink()
            k.atomic(root/'bulk_retired.json',dict(bytes_reclaimed=sum(inventory.values()),files=len(files),
                retained='all generation receipts, seeds, hashes, model heads and quality samples',finished_utc=now()))
    def advance(self):
        for method in k.METHODS:
            for model in k.MODELS:
                key=model+'/'+method;row=self.searches.setdefault(key,dict(phase='training',stages=[]))
                if row['phase']=='complete':continue
                self.current=(model,method);row.setdefault('started_utc',now())
                train=self.training(model,method)
                if self.ledger[train]['state']!='complete':return self.closure([train])
                if not row['stages']:
                    row['phase']='search';row['stages'].append(dict(step=16,ticks=list(k.COARSE)))
                if row['phase']=='search':
                    stage=row['stages'][-1]
                    ticks=sorted({*stage['ticks'],k.default_tick(model,method)})
                    goals=[self.quality(model,method,tick,1000) for tick in ticks]
                    if not all(self.ledger[job]['state']=='complete' for job in goals):return self.closure(goals)
                    stage['complete']=True;stage['rows']=self.rows(model,method,stage['ticks'])
                    if stage['step']!=1:
                        next_step=k.SEARCH_STEPS[k.SEARCH_STEPS.index(stage['step'])+1]
                        intervals,newticks=k.refine(stage['rows'],stage['step'],next_step)
                        if newticks:
                            row['stages'].append(dict(step=next_step,ticks=newticks,
                                selected_intervals=intervals,connected_range=[newticks[0],newticks[-1]]))
                            return self.advance()
                    all_ticks=sorted({k.default_tick(model,method),*[t for stage in row['stages'] for t in stage['ticks']]})
                    allrows=self.rows(model,method,all_ticks)
                    valid=sorted([x for x in allrows if x['valid'] and x['fid'] is not None],key=lambda x:(x['fid'],x['tick']))
                    row.update(phase='extend5k',screen_rows=allrows,selected=valid[:2],selection_locked_utc=now())
                    assert valid,'No valid coefficient in the complete search'
                if row['phase']=='extend5k':
                    goals=[self.quality(model,method,point['tick'],5000) for point in row['selected']]
                    if not all(self.ledger[job]['state']=='complete' for job in goals):return self.closure(goals)
                    row.update(phase='complete',results5k=self.rows(model,method,[p['tick'] for p in row['selected']],5000),finished_utc=now())
                    self.publish()
                    self.retire_generated_cache(model)
        self.current=None;return set()
    def refresh_gpus(self):
        if time.monotonic()-self.last_query<5:return
        self.last_query=time.monotonic();self.gpus=idle.gpu_snapshot()
        owned={uuid for value in self.active.values() for uuid in value.get('gpu_uuids',[])}
        for gpu in self.gpus:
            uuid=gpu['uuid'];self.streak[uuid]=self.streak.get(uuid,0)+1 if uuid not in owned and idle.eligible(gpu) else 0
    def acquire(self,n):
        if sum(value['job']['gpus'] for value in self.active.values())+n>4:return None
        owned={uuid for value in self.active.values() for uuid in value.get('gpu_uuids',[])}
        candidates=[g for g in self.gpus if g['uuid'] not in owned and self.streak.get(g['uuid'],0)>=2]
        chosen=[];leases=[]
        try:
            for gpu in candidates:
                lease=Path('/tmp',f'eqvae_idle_{gpu["uuid"]}.lock').open('a')
                try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:lease.close();continue
                current={g['uuid']:g for g in idle.gpu_snapshot()}.get(gpu['uuid'])
                if current and idle.eligible(current):chosen.append(current);leases.append(lease)
                else:lease.close()
                if len(chosen)==n:return chosen,leases
        finally:
            if len(chosen)<n:
                for lease in leases:lease.close()
        return None
    def launch(self,job,assignment=None):
        devices,leases=assignment or ([],[]);k.verify();k.check_stop()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in devices),OMP_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
        command=[k.PYTHON,'-u']
        if job['gpus']==4:command+=['-m','torch.distributed.run','--standalone','--nproc-per-node=4','-m',k.MODULE+'.worker']
        else:command+=['-m',k.MODULE+'.worker']
        command += [job['action'],'--model',job['model']]
        for name,value in job['params'].items():command+=['--'+name,str(value)]
        log=k.ROOT/'logs'/(job['id']+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as stream:
            child=subprocess.Popen(command,cwd=k.WORK,env=env,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        self.active[job['id']]=dict(job=job,process=child,leases=leases,gpu_uuids=[g['uuid'] for g in devices])
        self.ledger[job['id']]=dict(state='running',pid=child.pid,process_start_ticks=process_ticks(child.pid),
            gpu_indices=[g['index'] for g in devices],gpu_uuids=[g['uuid'] for g in devices],log=str(log),started_utc=now())
        print('Started',job['id'],child.pid,self.ledger[job['id']]['gpu_indices'],flush=True)
    def reap(self):
        for name,value in list(self.active.items()):
            code=value['process'].poll()
            if code is None:continue
            previous=self.ledger[name]
            try:
                if self.valid(value['job']) and code==0:self.ledger[name]=dict(previous,state='complete',finished_utc=now())
                elif code==75 or ((k.ROOT/'STOP_AFTER_CURRENT').exists() and code==0):self.ledger[name]=dict(previous,state='queued',last_exit_code=code)
                else:self.ledger[name]=dict(previous,state='failed',exit_code=code,finished_utc=now())
            finally:
                for lease in value['leases']:lease.close()
                del self.active[name]
            print('Finished',name,self.ledger[name]['state'],flush=True)
    def publish(self,phase=None):
        model,method=self.current or (None,None)
        row=self.searches.get((model or '')+'/'+(method or ''),{})
        state=dict(phase=phase or ('running' if self.active else 'waiting'),pid=os.getpid(),updated_utc=now(),
            current_idea=method,current_model=model,current_phase=row.get('phase'),
            current_step=row.get('stages',[{}])[-1].get('step',0)/40 if row.get('stages') else None,
            active_gpu_workers=sum(v['job']['gpus'] for v in self.active.values()),
            active_cpu_workers=sum(v['job']['gpus']==0 for v in self.active.values()),
            active_jobs=[dict(job=name,**self.ledger[name]) for name in self.active],
            complete_model_idea_pairs=sum(r['phase']=='complete' for r in self.searches.values()),
            request_sha256=k.sha(k.ROOT/'request.json'),training_steps=50000,global_batch=256,
            models=list(k.MODELS),quality='four-stage 1K; top two coefficients extend to 5K',gpus=self.gpus)
        k.atomic(k.ROOT/'status.json',state);k.atomic(k.ROOT/'jobs.json',self.ledger)
        k.atomic(k.ROOT/'plan.json',list(self.jobs.values()));k.atomic(k.ROOT/'searches.json',self.searches)
        if time.monotonic()-self.last_report>15 or phase:
            lines=['**完整数据动态采样重训：SiT / JiT，各50K，四卡训练与采样**','',
                f'当前 {model} / {method}：{row.get("phase",phase)}。GPU任务占用 {state["active_gpu_workers"]}/4。','',
                '每个idea：SiT训练与四级1K扫描、前两名各扩展5K；JiT执行同样流程；然后下一个idea。','',
                '|idea|模型|状态|1K最佳系数|1K最佳FID|5K结果|','|---|---|---|---:|---:|---|']
            for idea in k.METHODS:
                for model in k.MODELS:
                    r=self.searches.get(model+'/'+idea,{});best=r.get('selected',[{}])[0]
                    five='; '.join(f"{p['coefficient']:g}: {p['fid']:.4f}" for p in r.get('results5k',[]) if p.get('fid') is not None)
                    lines.append(f"|{idea}|{model}|{r.get('phase','queued')}|{best.get('coefficient','')}|{best.get('fid','')}|{five}|")
            lines+=['','旧固定小bank结果保留为历史；本队列不复用旧训练头或旧系数选择。5K包含原1K，不是独立留出集。',
                '',f'数据与训练协议：[说明]({k.PROTOCOL.name})。原始记录：{k.ROOT}','']
            tmp=k.REPORT.with_suffix('.tmp');tmp.write_text('\n'.join(lines));tmp.replace(k.REPORT);self.last_report=time.monotonic()
    def run(self):
        self.publish('preparing')
        while True:
            self.reap()
            if (k.ROOT/'STOP_AFTER_CURRENT').exists():
                self.publish('draining' if self.active else 'paused')
                if not self.active:return
                time.sleep(2);continue
            allowed=self.advance()
            failed=[name for name in allowed if self.ledger[name]['state'] in ('failed','blocked')]
            if failed:
                k.atomic(k.ROOT/'failure.json',dict(jobs=failed,recorded_utc=now()))
                (k.ROOT/'STOP_AFTER_CURRENT').write_text('Required job failed; save other workers before repair.\n')
                self.publish('draining_failure');continue
            if not allowed and not self.active:
                k.atomic(k.ROOT/'completion.json',dict(complete=True,request_sha256=k.sha(k.ROOT/'request.json')))
                self.publish('complete');return
            self.refresh_gpus()
            ready=[self.jobs[name] for name in self.jobs if name in allowed and self.ledger[name]['state']=='queued'
                and all(self.ledger[dep]['state']=='complete' for dep in self.jobs[name]['depends'])]
            ready.sort(key=lambda j:(j['gpus'] not in (0,4),j['gpus']))
            for job in ready:
                if (k.ROOT/'STOP_AFTER_CURRENT').exists():break
                if job['gpus']:
                    assignment=self.acquire(job['gpus'])
                    if assignment:self.launch(job,assignment)
                elif sum(v['job']['gpus']==0 for v in self.active.values())<4:self.launch(job)
            self.publish();time.sleep(2)


def main():
    def stop(signum,frame):(k.ROOT/'STOP_AFTER_CURRENT').write_text(f'Controller signal {signum}; save and drain.\n')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (k.ROOT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);k.verify()
        assert k.read(k.ROOT/'cpu_checks.json')['passed']
        Supervisor().run()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    if a.prepare:
        k.prepare()
        from .checks import cpu
        cpu()
    else:main()
