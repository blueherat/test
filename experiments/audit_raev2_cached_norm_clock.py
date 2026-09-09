"""Apply a fixed real-data norm scale to existing native IG trajectory caches."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch


def main():
    started = time.perf_counter()
    output = Path('experiments/results/terminal_defect_20260908')
    calibration = json.loads((output/'raev2_norm_time_information.json').read_text())
    mean = calibration['train_clean_norm_mean']
    var = calibration['train_clean_norm_variance']
    dim = calibration['dimension']
    base = torch.linspace(1, 0, 101, dtype=torch.float32)
    grid = (8*base/(1+7*base)).numpy()
    root = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/endpoint_adjoint_response_v1/collect')
    rows, inputs = [], {}
    for label in [0,142,285,428,570,713,856,999]:
        directory = root/f'id{label:04d}'
        summary = json.loads((directory/'summary.json').read_text())
        file = directory/'states.npy'
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        assert summary['complete'] and digest==summary['states']['sha256']
        states = np.load(file, mmap_mode='r')
        assert states.shape == (101,1024,16,16)
        inputs[str(file)] = digest
        for step, t in enumerate(grid[:-1]):
            raw = np.asarray(states[step])
            assert hashlib.sha256(raw.tobytes()).hexdigest()==summary['state_sha256_by_step'][step]
            if t <= .5:
                continue
            t = float(t)
            r = max(.5,t-1/32)
            observed = float(np.square(raw.astype(np.float64)).mean())
            predicted = (mean+np.sqrt(max((1+mean)*observed-mean,0)))/(1+mean)
            def moments(u):
                return (1-u)**2*mean+u*u, (1-u)**4*var+(4*(1-u)**2*u*u*mean+2*u**4)/dim
            mt, vt = moments(t)
            mr, vr = moments(r)
            rows.append(dict(label=label,step=step,noise_time=t,future_time=r,norm=observed,
                             norm_implied_time=float(predicted),
                             current_standardized_norm=(observed-mt)/np.sqrt(vt),
                             future_standardized_norm=(observed-mr)/np.sqrt(vr)))
    events = [int(np.argmin(np.abs(grid-t))) for t in [1.,.95,.9,.75,.6]]
    summary = []
    for step in events:
        selected = [r for r in rows if r['step']==step]
        summary.append(dict(step=step,noise_time=float(grid[step]),
                            median_implied_time=float(np.median([r['norm_implied_time'] for r in selected])),
                            median_current_z=float(np.median([r['current_standardized_norm'] for r in selected])),
                            median_future_z=float(np.median([r['future_standardized_norm'] for r in selected]))))
    result = dict(complete=True,rows=rows,summary=summary,inputs=inputs,
                  seconds=time.perf_counter()-started,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Eight historical native FP32 IG trajectories; norm scale from separate real training latents. No model queries or causal quality claim.')
    with (output/'raev2_cached_norm_clock.json').open('x') as f:
        json.dump(result,f,indent=2)
    print(json.dumps(dict(complete=True,summary=summary,seconds=result['seconds']),indent=2))


if __name__=='__main__':
    main()
