"""Wait for frozen four-shard completion, then evaluate the sole new candidate."""
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path('/home/zhoushunyu/data/eqvae/experiments/ig_fk_path_fourcard_20260908')
while True:
    statuses=[]
    for rank in range(4):
        d=root/f'rank{rank}'
        summary=d/'summary.json'
        statuses.append(dict(rank=rank,samples=4*len(list(d.glob('batch*.npz'))),
                             complete=summary.exists() and json.loads(summary.read_text()).get('complete',False)))
    print(json.dumps(dict(time=time.time(),ranks=statuses)),flush=True)
    if all(s['complete'] for s in statuses): break
    time.sleep(45)
subprocess.run([sys.executable,'-m','experiments.evaluate_raev2_fk_path_shards'],check=True)
subprocess.run([sys.executable,'-m','experiments.summarize_raev2_fk_path'],check=True)
print('Candidate sampling, evaluation, and diagnostic summary complete.',flush=True)
