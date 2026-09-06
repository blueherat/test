"""All-time descriptive audit of the three failed small actual-law fits.

Does not alter any entry gate or fit. Fixed t-bin summaries include every row.
"""
import json
from pathlib import Path
import numpy as np
from experiments.summarize_raev2_guidance_20260907 import DATA, ROOT, sha


def main():
    audit = {'complete': True, 'source_sha256': sha(Path(__file__).resolve()),
             'purpose': 'descriptive time-distribution audit, not a changed fit entry gate',
             'independent_heldout_ids': 1000,
             'validation_query_allocation': '125 native B8 batches: first26 indices twice, remaining73 once, shuffled before generation',
             'weight': 'w_i=N/(99*n_time_i); estimate=mean(w_i*loss_i)',
             'cluster_se': 'sqrt(B/(B-1)*sum_b [sum_i_in_b w_i*(loss_i-estimate)/N]^2), B=125 native batches',
             'uncertainty_limit': 'descriptive weighted cluster sandwich under this fixed time/class design, not an exact confidence interval',
             'models': {}}
    for name in ['actual_ratio_fit', 'prefix_ratio_fit', 'prefix_ratio_rb_fit']:
        path = DATA/name/'validation_records.npz'
        data = np.load(path)
        if 'records' in data:
            ids = data['records'][:, 0].astype(int)
            times, loss = data['records'][:, 1], data['records'][:, 2]
        else:
            ids, times, loss = data['ids'], data['times'], data['loss']
        unique = np.unique(times)
        assert len(unique) == 99 and len(ids) == 1000
        weights = np.empty(len(ids))
        for value in unique:
            mask = times == value
            weights[mask] = len(ids)/(99*mask.sum())
        estimate = float((weights*loss).mean())
        batches = np.unique(ids//8)
        influence = np.array([(weights[ids//8 == batch]*(loss[ids//8 == batch]-estimate)).sum()
                              for batch in batches])/len(ids)
        se = float(np.sqrt(len(batches)/(len(batches)-1)*(influence**2).sum()))
        bins = []
        for lo, hi in [(0, .5), (.5, .9), (.9, 1.)]:
            mask = (times >= lo) & (times < hi)
            bins.append({'t_min': lo, 't_max_exclusive': hi, 'n': int(mask.sum()), 'mean_loss': float(loss[mask].mean())})
        audit['models'][name] = {'validation_records_sha256': sha(path),
            'original_unweighted_loss': float(loss.mean()), 'uniform99_time_loss': estimate,
            'native_batch_cluster_se': se, 'all_fixed_time_bins': bins, 'gate_changed': False}
    out = ROOT/'experiments/results/raev2_guidance_20260907/actual_ratio_small_time_audit.json'
    out.write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({name: row['uniform99_time_loss'] for name, row in audit['models'].items()}))


if __name__ == '__main__':
    main()
