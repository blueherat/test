"""Conservative, rerunnable workspace organization; no experiment is deleted.

Moves old root-level result trees into archive/results, retaining relative
compatibility links. Downloaded literature and new large evidence files move to
external storage. Tracked evidence is retained in Git at its new location.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/home/zhoushunyu/data/eqvae')
OUT = ROOT / 'archive/manifests/20260919'
RESULTS = (
    'checkpoint_reference_long_study_v1', 'checkpoint_reference_schedule_fid1k_v1',
    'discriminator_ag_transport_v4', 'dual_target_closed_loop_spiral_toy_v1',
    'external_v180_temporal_utility_fid1k_v1', 'frequency_axis_screen_v1_smoke',
    'ig_ablation', 'internal_head_gamma_schedule_sweep_v4',
    'official_scale_metric_suite_seed20260805', 'prediction_target_rank_operator_toy_v1',
    'prediction_target_toy_v10_final_full_mechanism', 'prediction_target_toy_v3',
    'prediction_target_toy_v4_constant_norm', 'prediction_target_toy_v4_curved_screen',
    'prediction_target_toy_v4_direct_loss', 'prediction_target_toy_v4_main',
    'prediction_target_toy_v4_multiregime_screen', 'prediction_target_toy_v4_reverse_from_v',
    'spectral_ig_mechanism_v3_formal_1k', 'v800_v270_v400_gamma_search_fid1k_v2',
    'all_experiment_figures.pdf', 'figures_manifest.tsv',
)


def git_paths(*args):
    return [os.fsdecode(p) for p in subprocess.check_output(
        ['git', 'ls-files', *args, '-z'], cwd=ROOT).split(b'\0') if p]


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def move_link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        assert source.resolve() == target.resolve() and target.exists(), source
        return
    if target.exists():
        raise FileExistsError(target)
    shutil.move(str(source), str(target))
    source.symlink_to(os.path.relpath(target, source.parent), target_is_directory=target.is_dir())


def main(apply):
    if (OUT / 'complete.json').exists():
        print('Organization already complete; see', OUT)
        return
    tracked = git_paths()
    untracked = git_paths('--others', '--exclude-standard')
    plan = [(ROOT / name, ROOT / 'archive/results' / name) for name in RESULTS]
    plan += [(ROOT / 'readings', DATA / 'library/readings_20260919')]
    plan += [(ROOT / p, DATA / 'archive/large_evidence_20260919' / p)
             for p in untracked if p.startswith('docs/') and (ROOT / p).is_file()
             and not (ROOT / p).is_symlink() and (ROOT / p).stat().st_size > 2 * 1024**2]
    print(json.dumps([{'from': str(a), 'to': str(b)} for a, b in plan], indent=2))
    if not apply:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    # Recovery record is written before any movement. No git reset or clean.
    (OUT / 'moves.json').write_text(json.dumps([
        {'from': str(a.relative_to(ROOT)), 'to': str(b),
         'directory': a.is_dir(), 'sha256': digest(a) if a.is_file() else None}
        for a, b in plan], indent=2) + '\n')
    (OUT / 'preexisting_git_status.txt').write_bytes(subprocess.check_output(
        ['git', 'status', '--porcelain=v1'], cwd=ROOT))
    for source, target in plan:
        move_link(source, target)
    renamed = []
    for name in tracked:
        top = name.split('/')[0]
        if top in RESULTS:
            renamed.append('archive/results/' + name)
    # Only the files already tracked are staged from the large historic trees.
    staging = Path('/tmp/eqvae_organized_tracked_20260919.paths')
    staging.write_bytes(b''.join(os.fsencode(p) + b'\0' for p in renamed))
    (OUT / 'complete.json').write_text(json.dumps(dict(
        complete=True, moves=len(plan), deleted_experiments=0,
        tracked_relocations=len(renamed), staging_list=str(staging)), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    main(parser.parse_args().apply)
