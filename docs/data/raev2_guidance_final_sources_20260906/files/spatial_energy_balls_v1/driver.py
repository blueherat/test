import argparse, datetime, hashlib, json, os, subprocess, time
from pathlib import Path

def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def put(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False))
    tmp.replace(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',type=Path,required=True);args=ap.parse_args()
    request_path=args.jobs.resolve(); jobs=json.loads(request_path.read_text());root=request_path.parent
    summary_path=root/(request_path.stem+'_execution.json')
    if summary_path.exists():raise FileExistsError(summary_path)
    for rec in jobs['fixed_files']:
        if hashlib.sha256(Path(rec['path']).read_bytes()).hexdigest()!=rec['sha256']:
            raise RuntimeError('frozen source mismatch: '+rec['path'])
    logs=root/'driver_logs';logs.mkdir(exist_ok=True)
    summary={'request':str(request_path),'request_sha256':hashlib.sha256(request_path.read_bytes()).hexdigest(),'started_utc':stamp(),'driver_pid':os.getpid(),'complete':False,'jobs':[]}
    started=time.perf_counter();live=[]
    for job in jobs['jobs']:
        log_path=logs/(job['name']+'.log')
        if log_path.exists():raise FileExistsError(log_path)
        env=os.environ.copy();env.update(job.get('env',{}));env['PYTHONUNBUFFERED']='1'
        stream=log_path.open('wb');begin=time.perf_counter()
        proc=subprocess.Popen(job['argv'],cwd=jobs['cwd'],env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        rec={'name':job['name'],'argv':job['argv'],'env_overrides':job.get('env',{}),'pid':proc.pid,'started_utc':stamp(),'log':str(log_path),'exit_code':None}
        summary['jobs'].append(rec);live.append((proc,stream,begin,rec));put(summary_path,summary)
    while live:
        remaining=[]
        for proc,stream,begin,rec in live:
            code=proc.poll()
            if code is None:remaining.append((proc,stream,begin,rec));continue
            stream.close();rec.update(exit_code=code,finished_utc=stamp(),outer_wall_seconds=time.perf_counter()-begin)
            put(summary_path,summary)
        live=remaining
        if live:time.sleep(1)
    summary.update(complete=all(j['exit_code']==0 for j in summary['jobs']),finished_utc=stamp(),driver_wall_seconds=time.perf_counter()-started)
    put(summary_path,summary)
    if not summary['complete']:raise SystemExit(1)
if __name__=='__main__':main()
