"""Inspect retained RAEv2 MLP loss logs without updating any model weights."""
from pathlib import Path
import csv
import hashlib
import json
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


WORK = Path(__file__).resolve().parents[1]
TRAIN = Path('/home/zhoushunyu/data/eqvae/experiments/guidance_distribution_20260912/raev2/input_local_training')
OUT = WORK / 'docs/data/raev2_context_mlp_loss_20260913'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary = json.loads((TRAIN / 'summary.json').read_text())
    request = json.loads((TRAIN / 'request.json').read_text())
    source = WORK / 'experiments/guidance_distribution_20260912/local_head.py'
    assert sha(source) == request['sources'][str(source)]
    assert sha(TRAIN / 'head.pt') == summary['head_sha256']
    assert sha(TRAIN / 'request.json') == summary['request_sha256']
    rows = summary['history']
    assert [r['step'] for r in rows] == [1, *range(100, 3001, 100)]
    assert summary['steps'] == 3000 and summary['batch'] == 8
    assert all(np.isfinite(r[k]) and r[k] >= 0 for r in rows for k in ('context', 'local'))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'recorded_batches.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'local', 'context'])
        writer.writeheader()
        writer.writerows(rows)
    blocks = []
    for high in range(500, 3001, 500):
        selected = [r for r in rows if high - 500 < r['step'] <= high and r['step'] != 1]
        assert len(selected) == 5
        values = [r['context'] for r in selected]
        avg = statistics.fmean(values)
        assert abs(avg - float(np.mean(values))) < 1e-14
        blocks.append(dict(first_logged_step=selected[0]['step'], last_logged_step=high,
                           recorded_batches=5, mean_logged_step=statistics.fmean(r['step'] for r in selected),
                           context_mean=avg, context_batch_sd=statistics.stdev(values)))
    with (OUT / 'block_means.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(blocks[0]))
        writer.writeheader()
        writer.writerows(blocks)
    late_fraction = 1 - blocks[-1]['context_mean'] / blocks[-2]['context_mean']
    penultimate_fraction = 1 - blocks[-2]['context_mean'] / blocks[-3]['context_mean']
    analysis = dict(
        model='raev2', head='context', training_steps=3000, training_batch=8,
        source_history_points=len(rows), log_grain='Single online-model training batch before its update; every 100 steps, plus step 1.',
        block_mean_grain='Arithmetic mean of five recorded batches; not the mean of all 500 training steps.',
        blocks=blocks, last_block_relative_decrease=late_fraction,
        previous_block_relative_decrease=penultimate_fraction,
        final_ema_validation_mse=summary['validation_mse']['context'],
        validation_evaluations=1, validation_batches=32, validation_batch=8,
        validation_draws=256, validation_unique_cases_unknown=True,
        historical_validation_curve_available=False,
        interpretation='Late logged training loss is flatter, but convergence is not established. Sparse changing batches and one final EMA validation evaluation do not establish a validation plateau.',
        training_performed=False, generated_images=0,
        source_hashes={str(p):sha(p) for p in (source, TRAIN/'summary.json', TRAIN/'request.json', TRAIN/'head.pt')},
    )
    (OUT / 'analysis.json').write_text(json.dumps(analysis, indent=2) + '\n')
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False,
                         'svg.fonttype':'none', 'pdf.fonttype':42})
    fig, ax = plt.subplots(figsize=(10.8, 5.7))
    fig.subplots_adjust(left=.085, right=.975, top=.80, bottom=.25)
    ax.plot([r['step'] for r in rows], [r['context'] for r in rows],
            color='#9BA6AD', marker='o', markersize=3.6, linewidth=1.1,
            label='One recorded training batch (batch 8)')
    ax.plot([r['mean_logged_step'] for r in blocks], [r['context_mean'] for r in blocks],
            color='#21677F', marker='s', markersize=6, linewidth=2,
            label='Mean of 5 logged batches per 500-step block')
    for row, offset in ((blocks[-2], 15), (blocks[-1], -24)):
        ax.annotate(f"{row['context_mean']:.4f}", (row['mean_logged_step'], row['context_mean']),
                    textcoords='offset points', xytext=(0, offset), ha='center', color='#21677F')
    ax.set(xlim=(0, 3050), ylim=(0, 1.08), xlabel='Optimizer step', ylabel='Clean-latent MSE')
    ax.set_xticks(np.arange(0, 3001, 500))
    ax.grid(axis='y', color='#E6E8E9', linewidth=.65)
    ax.legend(loc='upper right', frameon=False, fontsize=10)
    fig.text(.085, .94, 'RAEv2 Context MLP: recorded training loss', fontsize=17, weight='bold')
    fig.text(.085, .875, '3,000 steps  |  batch 8  |  fixed learning rate 0.0003', fontsize=11)
    fig.text(.085, .13, 'Training points are single batches, not 100-step averages; block means use only 5 logged batches.', fontsize=9.6)
    fig.text(.085, .085, f"Only one validation evaluation: final EMA MSE = {summary['validation_mse']['context']:.4f} (32 batches x 8 draws).", fontsize=9.6)
    fig.text(.085, .04, 'No historical validation curve or uncertainty interval is available. Source: retained input_local_training logs.', fontsize=9.6)
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / f'context_loss.{ext}', dpi=170)
    plt.close(fig)
    manifest = dict(generator=str(Path(__file__).resolve()), generator_sha256=sha(Path(__file__)),
                    files={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'manifest.json'},
                    visual_review_pending=True)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(last_block_relative_decrease=late_fraction,
                         previous_block_relative_decrease=penultimate_fraction,
                         final_ema_validation_mse=summary['validation_mse']['context'],
                         output=str(OUT)), indent=2))


if __name__ == '__main__':
    main()
