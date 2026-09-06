"""Invert the existing 100-step Gaussian Euler map for two observed energies."""
import hashlib
import json
from pathlib import Path
from experiments.raev2_two_mode_ratio import fit_euler_prior_variance,euler_terminal_variance

DATA=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_20260907')


def main():
    source=DATA/'two_mode_ratio/calibration.json'
    grid_source=DATA/'ancestral_screen1k/shard0/request.json'
    old=json.loads(source.read_text())
    grid=json.loads(grid_source.read_text())['time_grid']
    if len(grid)!=101 or grid[0]!=1 or grid[-1]!=0 or min(grid[:-1])<.05:
        raise ValueError('requires the fixed 100-step grid without an active floor')
    out=source.with_name('finite_euler_calibration.json')
    if out.exists():raise FileExistsError(out)
    result={**old,'raw_second_moments':{k:old[k] for k in ['real','generated']},'real':{},'generated':{},
        'finite_euler_calibration':True,'time_grid':grid,'sampling_steps':100,
        'raw_calibration_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'grid_source_sha256':hashlib.sha256(grid_source.read_bytes()).hexdigest(),'inversion_checks':[]}
    for side in ['real','generated']:
        for mode in ['dc_variance','ac_variance']:
            target=old[side][mode]
            fitted=fit_euler_prior_variance(target,grid)
            actual=euler_terminal_variance(fitted,grid)
            if abs(actual/target-1)>1e-10:raise ValueError('root did not match target')
            result[side][mode]=fitted
            result['inversion_checks'].append({'side':side,'mode':mode,'target':target,'effective_prior':fitted,
                'finite_endpoint':actual,'uncorrected_endpoint':euler_terminal_variance(target,grid)})
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['inversion_checks'],indent=2))


if __name__=='__main__':main()
