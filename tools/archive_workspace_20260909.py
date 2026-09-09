"""Inventory research files and preserve portable metadata; never move raw assets."""
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/data/workspace_archive_20260909'
EXTERNAL = Path('/home/zhoushunyu/data/eqvae/experiments')
CACHES = {'.git', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', '.ipynb_checkpoints'}
RUNS = [
    'jit_internal_readouts_20260908', 'jit_transfer_20260908',
    'jit_ordered_20260909', 'jit_fine_sweep_20260909', 'jit_pfr_variants_20260909',
    'small_sit_best_config_lifting_20260909', 'small_sit_best_lifting_sweep_20260909',
    'small_sit_lifting_20260909', 'small_sit_two_stage_ig_20260909',
    'small_sit_two_stage_ig_strength_20260909',
    'ig_capacity_lifting_fourcard_20260908', 'ig_capacity_lifting_constant_fourcard_20260908',
]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = []
    count = size = 0
    symlinks = []
    with (OUT / 'workspace_files.jsonl').open('w') as dest:
        for directory, dirs, files in os.walk(ROOT, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in CACHES and Path(directory) / d != OUT)
            paths = sorted(files + [d for d in dirs if (Path(directory) / d).is_symlink()])
            for name in paths:
                p = Path(directory) / name
                rel = str(p.relative_to(ROOT))
                st = p.lstat()
                rec = {'path': rel, 'bytes': st.st_size}
                if p.is_symlink():
                    rec.update(kind='symlink', target=os.readlink(p), target_exists=p.exists())
                    symlinks.append(rec)
                elif p.is_file():
                    vendor = p.relative_to(ROOT).parts[0] in {'external', 'research_repos'}
                    rec.update(kind='file', sha256=digest(p))
                    # All ordinary unignored files will be added by git add -A.
                    # Also retain small research evidence currently hidden by broad ignores.
                    if (not vendor and st.st_size <= 2 * 1024 * 1024
                            and p.suffix not in {'.pyc', '.pyo', '.pt', '.pth', '.ckpt', '.safetensors'}):
                        candidates.append(rel)
                    count += 1
                    size += st.st_size
                else:
                    rec['kind'] = 'special'
                dest.write(json.dumps(rec, ensure_ascii=False) + '\n')
    snapshots, metrics, missing = [], [], []
    keep = {'result.json', 'fid.json', 'metrics.json', 'complete.json', 'comparison.json',
            'selected_ig.json', 'request.json', 'preflight.json', 'input_preflight.json',
            'iteration_check.json', 'math_check.json', 'status.json'}
    for run in RUNS:
        base = EXTERNAL / run
        if not base.exists():
            missing.append(str(base))
            continue
        for p in sorted(base.rglob('*.json')):
            if p.name not in keep and not p.name.startswith('validation'):
                continue
            # Per-shard requests carry source hashes and exact frozen settings.
            data = json.loads(p.read_text())
            snapshots.append({'source': str(p), 'sha256': digest(p), 'data': data})
            if p.name == 'result.json' and isinstance(data, dict):
                row = {k: data.get(k) for k in ['arm', 'method', 'early', 'late', 'iterations',
                       'samples', 'seed', 'fid', 'noise_sha256', 'label_sha256',
                       'sample_sha256', 'sum_batch_gpu_seconds']}
                row.update(run=run, source=str(p))
                if row['arm'] is None:
                    row['arm'] = p.parent.name
                metrics.append(row)
        result_list = base / 'results.json'
        if result_list.exists():
            snapshots.append({'source': str(result_list), 'sha256': digest(result_list),
                              'data': json.loads(result_list.read_text())})
    dump(OUT / 'external_metadata.json', snapshots)
    if metrics:
        with (OUT / 'result_records.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(metrics)
    dump(OUT / 'symlinks.json', symlinks)
    dump(OUT / 'summary.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'workspace': str(ROOT),
        'regular_files': count, 'regular_bytes': size, 'symlinks': len(symlinks),
        'external_metadata_records': len(snapshots), 'result_records': len(metrics),
        'absent_optional_run_roots': missing,
        'excluded': sorted(CACHES) + ['this generated archive directory'],
        'limitations': 'Symlink target trees are not traversed. External run metadata is copied, '
                      'not raw samples/checkpoints. Workspace regular-file hashes are computed '
                      'at snapshot time, not a backup or a claim of successful experiments.',
    })
    # A NUL-delimited local staging list avoids shell path interpretation.
    staging = Path('/tmp/eqvae_archive_small_files_20260909.paths')
    staging.write_bytes(b''.join(os.fsencode(p) + b'\0' for p in candidates))
    print(json.dumps({'files': count, 'bytes': size, 'metadata': len(snapshots),
                      'result_records': len(metrics), 'absent_roots': missing,
                      'small_file_staging_list': str(staging)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
