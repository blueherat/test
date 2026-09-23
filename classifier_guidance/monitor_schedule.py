"""Read-only health monitoring of a continued signed-schedule GAN experiment.

Heuristic alerts flag symptoms for review; they neither prove quality nor change
training. All measurements come from actual training updates, not a fixed probe
dataset. The monitor uses no GPU and writes only reports below the run root.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def history(root, filename):
    rows = {}
    for path in sorted(root.glob(f'training*/{filename}')):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # The worker may be appending its last line.
            rows[row['step']] = row
    return [rows[key] for key in sorted(rows)]


def stats(rows, keys):
    out = {}
    for key in keys:
        values = np.asarray([r[key] for r in rows if key in r], dtype=float)
        if len(values):
            out[key] = dict(min=float(values.min()), median=float(np.median(values)),
                            max=float(values.max()))
    return out


def inspect(root):
    rows = history(root, 'train.jsonl')
    training = [r for r in rows if r['phase']=='training']
    probes = [r for r in history(root, 'diagnostics.jsonl') if 'head_gradient' in r]
    if not training:
        return None
    last = training[-1]
    window = training[-100:]
    coefficients = np.asarray([r['coefficients'] for r in training])
    last_values = coefficients[-1]
    middle = len(last_values)//2
    keys = ('d_ce', 'r1', 'g_loss', 'real_accuracy', 'fake_accuracy',
            'schedule_gradient_norm', 'gradient_first_half_norm',
            'gradient_second_half_norm', 'endpoint_rms', 'seconds')
    diagnostic_keys = ('d_gradient_norm', 'd_update_norm', 'd_relative_update_norm',
        'd_clip_applied', 'real_probability_mean', 'fake_probability_after_d_mean',
        'd_real_saturated_fraction', 'd_fake_saturated_fraction',
        'g_logit_gradient_mean', 'g_loss_saturated_fraction',
        'g_feature_gradient_norm_mean', 'g_endpoint_gradient_norm_mean',
        'head_update_norm', 'head_zero_gradient_fraction',
        'head_near_zero_gradient_fraction', 'head_clip_applied')
    finite = bool(np.isfinite(coefficients).all())
    finite = finite and all(np.isfinite(r[k]) for r in rows for k in keys if k in r)
    finite = finite and all(np.isfinite(r[k]).all() for r in probes
                           for k in (*diagnostic_keys, 'head_gradient', 'head_update') if k in r)
    alerts = []
    if not finite:
        alerts.append('Nonfinite training values detected.')
    if len(window)>=100 and np.median([r['schedule_gradient_norm'] for r in window])<1e-8:
        alerts.append('Median coefficient gradient below 1e-8 over 100 updates; investigate signal loss.')
    if len(training)>=200:
        baseline = np.median([r['schedule_gradient_norm'] for r in training[:100]])
        if np.median([r['schedule_gradient_norm'] for r in window])<baseline*.01:
            alerts.append('Median coefficient gradient below 1% of the first-100-update level; inspect the cause.')
    changes = np.diff(coefficients[-101:], axis=0)
    if len(changes)>=100 and not (changes!=0).any():
        alerts.append('All coefficients unchanged for 100 updates.')
    recent_probes = probes[-20:]
    if len(recent_probes)>=20:
        if all(r['d_update_norm']==0 for r in recent_probes):
            alerts.append('Discriminator weights unchanged in all 20 recent probes.')
        if np.median([r['g_feature_gradient_norm_mean'] for r in recent_probes])<1e-8:
            alerts.append('Median feedback feature gradient below 1e-8 in 20 probes.')
        if np.median([r['g_loss_saturated_fraction'] for r in recent_probes])>.95:
            alerts.append('Generator loss derivative saturated on >95% of examples in 20 probes.')
        if all(np.median([r[k] for r in recent_probes])>.95
               for k in ('d_real_saturated_fraction','d_fake_saturated_fraction')):
            alerts.append('Discriminator probabilities extremely confident on both classes in 20 probes; inspect feedback gradients.')
        if np.median([r['head_near_zero_gradient_fraction'] for r in recent_probes])>.5:
            alerts.append('Over half of coefficients have near-zero gradients in 20 probes.')
        if np.mean([r['d_clip_applied'] for r in recent_probes])>.9:
            alerts.append('Discriminator gradient clipped in >90% of recent probes; review stability.')
        if np.mean([r['head_clip_applied'] for r in recent_probes])>.9:
            alerts.append('Coefficient gradient clipped in >90% of recent probes; review stability.')
    pipeline_path = root/'pipeline_status.json'
    pipeline = json.loads(pipeline_path.read_text()) if pipeline_path.exists() else {}
    age = (datetime.now(timezone.utc)-datetime.fromisoformat(last['updated_utc'])).total_seconds()
    if pipeline.get('phase')=='schedule' and age>300:
        alerts.append('No new training update for over 300 seconds.')
    if pipeline.get('phase')=='failed':
        alerts.append('Training/evaluation pipeline reported failure.')
    result = dict(updated_utc=datetime.now(timezone.utc).isoformat(),
        target_coefficient_updates=30000, coefficient_updates=last['coefficient_updates'],
        step=last['step'], training_complete=last['coefficient_updates']>=30000,
        pipeline_phase=pipeline.get('phase'), last_update_age_seconds=age,
        first100=stats(training[:100], keys), last100=stats(window, keys),
        recent_diagnostics=stats(recent_probes, diagnostic_keys),
        diagnostic_probe_count=len(probes), latest_diagnostic=probes[-1] if probes else None,
        coefficient_min=float(last_values.min()), coefficient_max=float(last_values.max()),
        negative_count=int((last_values<0).sum()),
        boundary_left=float(last_values[middle-1]), boundary_right=float(last_values[middle]),
        boundary_abs_jump=float(abs(last_values[middle]-last_values[middle-1])),
        minimum_updates_per_coefficient_last100=int((changes!=0).sum(0).min()) if len(changes) else 0,
        compared_coefficient_transitions=len(changes), all_observed_values_finite=bool(finite),
        remaining_training_hours=max(0,30000-last['coefficient_updates'])*np.median([r['seconds'] for r in window])/3600,
        alerts=alerts, interpretation='Nonzero gradients and updates establish activity, not image quality or GAN convergence.')
    if len(recent_probes)>1 and 'head_gradient' in recent_probes[-1]:
        gradients = np.asarray([r['head_gradient'] for r in recent_probes])
        left, right = gradients[:,middle-1], gradients[:,middle]
        denominator = np.linalg.norm(left)*np.linalg.norm(right)
        result['boundary_gradient_cosine_over_probes'] = float(np.dot(left,right)/denominator) if denominator else None
    # Invalid telemetry must not prevent a persistent alert file from being written.
    def sanitize(value):
        if isinstance(value, dict): return {k:sanitize(v) for k,v in value.items()}
        if isinstance(value, list): return [sanitize(v) for v in value]
        if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
        return value
    result = sanitize(result)
    atomic(root/'gan_health.json', result)
    atomic(root/'gan_alerts.json', dict(updated_utc=result['updated_utc'], alerts=alerts))
    return result, training, probes


def plot(root, result, training, probes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values = np.asarray(training[-1]['coefficients'])
    initial = np.r_[np.full(len(values)//2, .6), np.zeros(len(values)//2)]
    updates = np.asarray([r['coefficient_updates'] for r in training])
    curves = np.asarray([r['coefficients'] for r in training])
    middle = len(values)//2
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    axes[0,0].stairs(initial, np.linspace(0,1,len(values)+1), baseline=None, label='Initial', color='gray', linestyle='--')
    axes[0,0].stairs(values, np.linspace(0,1,len(values)+1), baseline=None, label='Current raw', color='#1764ab')
    axes[0,0].axvline(.5, color='gray', alpha=.4)
    axes[0,0].set(xlabel='Sampling time: noise 0 to image 1', ylabel='Signed coefficient',
                  title=f"Schedule after {result['coefficient_updates']:,} updates")
    axes[0,0].legend()
    axes[0,1].plot(updates, np.abs(curves[:,middle]-curves[:,middle-1]))
    axes[0,1].axhline(.6, color='gray', linestyle='--', label='Initial gap')
    axes[0,1].set(xlabel='Coefficient update', ylabel='Absolute adjacent coefficient gap', title='Boundary at t=0.5')
    axes[0,1].legend()
    for key,label in [('schedule_gradient_norm','Total before clipping'),('gradient_first_half_norm','First half after clipping'),('gradient_second_half_norm','Second half after clipping')]:
        axes[1,0].plot(updates, [r[key] for r in training], label=label, linewidth=.7, alpha=.7)
    axes[1,0].set(yscale='log', xlabel='Coefficient update', ylabel='Gradient L2 norm', title='Gradient reaches both halves')
    axes[1,0].legend(fontsize=8)
    for key,label in [('d_ce','D logistic (without R1)'),('g_loss','Coefficient non-saturating loss')]:
        values_loss = np.array([r[key] for r in training])
        width=min(50,len(values_loss))
        axes[1,1].plot(updates[width-1:], np.convolve(values_loss,np.ones(width)/width,mode='valid'), label=label)
    axes[1,1].set(xlabel='Coefficient update', ylabel='50-update moving mean', title='Losses: activity does not establish quality')
    axes[1,1].legend(fontsize=8)
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.savefig(root/'schedule.png', dpi=150);fig.savefig(root/'schedule.pdf');plt.close(fig)
    with (root/'coefficients.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['solver_step','time_left','initial','current_raw','coefficient_update'])
        writer.writerows((i,i/len(values),initial[i],v,result['coefficient_updates']) for i,v in enumerate(values))
    if not probes: return
    x=[r['coefficient_updates'] for r in probes]
    fig, axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    groups=[(('real_probability_mean','Real before D update'),('fake_probability_after_d_mean','Fake after D update')),
            (('d_gradient_norm','D gradient before clipping'),('d_update_norm','D parameter update')),
            (('g_feature_gradient_norm_mean','Feature gradient per image'),('g_endpoint_gradient_norm_mean','Latent gradient per image')),
            (('g_logit_gradient_mean','Loss derivative magnitude'),('head_update_norm','Coefficient update norm'))]
    titles=['Discriminator probabilities','Discriminator activity','Backpropagation reaches generated endpoint','Loss signal and coefficient movement']
    for ax,group,title in zip(axes.flat,groups,titles):
        for key,label in group:ax.plot(x,[r[key] for r in probes],label=label,linewidth=1)
        ax.set(xlabel='Coefficient update',title=title);ax.grid(alpha=.2);ax.legend(fontsize=8)
        if ax is not axes[0,0]: ax.set_yscale('log')
    axes[0,0].set_ylim(0,1)
    fig.savefig(root/'gan_health.png',dpi=150);fig.savefig(root/'gan_health.pdf');plt.close(fig)


def main(args):
    last_plot=0
    while True:
        result=inspect(args.root)
        if result:
            health,training,probes=result
            done=health['pipeline_phase'] in ('complete','failed')
            if not args.watch or done or time.monotonic()-last_plot>=args.plot_every:
                plot(args.root,health,training,probes);last_plot=time.monotonic()
            print(json.dumps({k:health[k] for k in ('updated_utc','coefficient_updates','diagnostic_probe_count','boundary_abs_jump','alerts')}),flush=True)
            if done:break
        if not args.watch:break
        time.sleep(args.watch)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--watch',type=float,default=0,help='Polling interval in seconds; 0 runs once')
    parser.add_argument('--plot-every',type=float,default=300)
    main(parser.parse_args())
