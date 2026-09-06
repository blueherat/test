"""Post-hoc CPU rounding of frozen controls at stored baseline successors.

This does not replay a model or reconstruct a controlled trajectory. Raw BF16
casting is a coordinate diagnostic, not the complete backbone computation.
"""
import time
START_WALL, START_CPU = time.perf_counter(), time.process_time()
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

P = Path(__file__).resolve().parent
torch.set_num_threads(4)
def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

request = json.loads((P/'request.json').read_text())
control = json.loads((P/'frozen_control.json').read_text())
h = -np.diff(np.asarray(request['time_grid']))
rows = []
for identity in control['collect_summaries']:
    assert digest(identity['path']) == identity['sha256']
    saved = json.loads(Path(identity['path']).read_text())
    for key in ('states','adjoints'):
        assert digest(saved[key]['path']) == saved[key]['sha256']
    states = np.load(saved['states']['path'], mmap_mode='r')
    adjoints = np.load(saved['adjoints']['path'], mmap_mode='r')
    for k, width in enumerate(h):
        state = torch.from_numpy(np.array(states[k+1],copy=True))
        a = torch.from_numpy(np.array(adjoints[k],copy=True))
        increment = float(width)*(a*control['lambda'])
        changed = state+increment
        implemented = changed.double()-state.double()
        intended = increment.double()
        rows.append({'global_id':saved['global_id'],'step':k,'time':request['time_grid'][k],
              'fp32_add_unchanged_fraction':float((changed==state).double().mean()),
              'raw_bf16_cast_unchanged_fraction':float((changed.bfloat16()==state.bfloat16()).double().mean()),
              'intended_increment_squared_norm':float(intended.square().sum()),
              'rounding_error_squared_norm':float((implemented-intended).square().sum()),
              'implemented_increment_squared_norm':float(implemented.square().sum())})
summary = {'posthoc':True,'no_model_or_gpu_calls':True,
           'request_sha256':digest(P/'request.json'), 'frozen_control_sha256':digest(P/'frozen_control.json'),
           'source_sha256':digest(__file__), 'rows':rows,
           'per_image':[{ 'global_id':i,
                 **{key:float(np.mean([r[key] for r in rows if r['global_id']==i])) for key in
                    ('fp32_add_unchanged_fraction','raw_bf16_cast_unchanged_fraction')},
                 'aggregate_rounding_error_over_intended_energy':
                 sum(r['rounding_error_squared_norm'] for r in rows if r['global_id']==i)/
                 sum(r['intended_increment_squared_norm'] for r in rows if r['global_id']==i)}
                 for i in request['global_ids_and_labels']],
           'timing':{'wall_seconds':time.perf_counter()-START_WALL,'cpu_seconds':time.process_time()-START_CPU,
                     'scope':'After import time, includes subsequent imports and reads; excludes final result write/exit.'},
           'limits':['Arithmetic at stored baseline successor, not actual controlled states.',
                     'Raw BF16 cast comparison is not a model Jacobian or attribution of response sign.',
                     'Coordinate fractions do not measure relative effect on endpoint features.',
                     'No control adjustment, replay, training or FID follows from this diagnostic.']}
with (P/'rounding_diagnostic.json').open('x') as f: json.dump(summary,f,indent=2)
print(json.dumps({'per_image':summary['per_image'],'timing':summary['timing']}))
