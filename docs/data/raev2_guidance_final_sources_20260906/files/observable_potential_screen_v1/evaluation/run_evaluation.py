import json,os,subprocess,time,resource
from pathlib import Path
out=Path(__file__).resolve().parent
request=json.loads((out/"request.json").read_text())
start=time.perf_counter(); before=resource.getrusage(resource.RUSAGE_CHILDREN)
with (out/"process.log").open("w") as log:
    result=subprocess.run(request["command"],env={**os.environ,"CUDA_VISIBLE_DEVICES":"0","OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4","OPENBLAS_NUM_THREADS":"4"},stdout=log,stderr=subprocess.STDOUT)
elapsed=time.perf_counter()-start; after=resource.getrusage(resource.RUSAGE_CHILDREN)
summary={"complete":result.returncode==0,"returncode":result.returncode,"evaluator_process_wall_seconds_including_imports_and_exit":elapsed,"evaluator_child_cpu_seconds":after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,"input_images":2000,"gpu_used":True,"sampling_performed":False}
(out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary),flush=True)
if result.returncode: raise RuntimeError((out/"process.log").read_text())
print((out/"metrics.json").read_text(),flush=True)
