#!/usr/bin/env python3
"""Copy bounded metadata only; finalize after the root's explicit closeout signal.

Snapshots exclude the three late review/annotation artifacts and final_results.
Finalization never changes prior copies and requires both final reviews to exist.
No image/model/feature arrays are opened, no experiments or evaluation are run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[2]
RESTART = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906')
EXTENSION = RESTART / 'scale_extension_5k_v1'
DOCUMENT = ROOT / 'docs/RAEV2_MILD_NEGATIVE_5K_EXTENSION_PROTOCOL_20260907_ZH.md'
LATE = {'merge_execution.json', 'final_sampling_cost_independent_review.json',
        'evaluation_preparation_v1/final_fid_independent_review.json'}
RECOVERY_SOURCE = Path('evaluation_preparation_v1/final_fid_independent_review_v1_source.py')
FINAL_REVIEW_IDENTITIES = {
    'final_sampling_cost_independent_review.json': '49da2e34ca97800c491f88649ce9ebc435e191be01b286836d4a2cb7c1cb2d7e',
    'evaluation_preparation_v1/final_fid_independent_review.json': '8c661a400fdd9caade497d0e8242f9b84e6f7112a15f2cc001bd196e38404bf0',
    'evaluation_preparation_v1/final_fid_independent_review_v1.json': 'b0dc6b051af782ccee53d4358c36c02231088c886f2ab7e22e261a8bd37654e5',
    str(RECOVERY_SOURCE): 'a6d23843116a0e17a7673fd618d7b680d9630710f6bf6f1e315772e59ad0eb27',
}
SUFFIXES = {'.json', '.csv'}
MAX_FILE = 8 * 1024 * 1024
MAX_TOTAL = 128 * 1024 * 1024


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def collect(final):
    plan = json.loads((EXTENSION / 'plan.json').read_text())
    if plan.get('protocol') != 'raev2_mild_negative_fixed_1k_to_5k_extension_v1':
        raise ValueError('unexpected root plan')
    for index in range(1, 5):
        execution = json.loads((EXTENSION / f'cohort_{index}/execution.json').read_text())
        if execution.get('complete') is not True or execution.get('terminal') is not True or any(j.get('exit_code') != 0 for j in execution['jobs']):
            raise ValueError('all original sampling workers must have completed successfully')
    for family in ('legacy', 'native_global', 'reflection'):
        execution = json.loads((EXTENSION / f'evaluation_preparation_v1/{family}_evaluation_execution.json').read_text())
        if execution.get('complete') is not True:
            raise ValueError('all three family evaluations must be complete')
    sources = {}

    def add(path, relative, category):
        path = Path(path)
        if path.suffix not in SUFFIXES and path not in (DOCUMENT, EXTENSION / RECOVERY_SOURCE):
            raise ValueError(f'non-metadata copy forbidden: {path}')
        if not path.is_file() or path.stat().st_size > MAX_FILE:
            raise ValueError(f'missing/oversized metadata: {path}')
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('unsafe output path')
        key = str(path.resolve())
        if key not in sources:
            sources[key] = {'source': path, 'relative': Path(relative), 'category': category}

    for path in sorted(EXTENSION.rglob('*')):
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        rel = path.relative_to(EXTENSION)
        if 'sources' in rel.parts:
            continue
        if not path.resolve().is_relative_to(EXTENSION.resolve()):
            raise ValueError(f'extension metadata symlink escapes root: {path}')
        if not final and (str(rel) in LATE or rel.parts[0] == 'final_results_v1'):
            continue
        add(path, Path('extension') / rel, 'completed_extension_metadata')
    if final:
        for name in LATE:
            if not (EXTENSION / name).is_file():
                raise ValueError(f'root finalization signal requires final artifact: {name}')
        if not (EXTENSION / 'final_results_v1/summary.json').is_file():
            raise ValueError('final result summary is missing')
        for relative, expected in FINAL_REVIEW_IDENTITIES.items():
            if digest((EXTENSION / relative).read_bytes()) != expected:
                raise ValueError(f'root-approved final/recovery review identity changed: {relative}')
        add(EXTENSION / RECOVERY_SOURCE, Path('extension') / RECOVERY_SOURCE, 'initial_fid_reviewer_source_recovery_snapshot')
    add(DOCUMENT, Path('protocol') / DOCUMENT.name, 'frozen_extension_protocol')
    for arm, row in plan['historical_1k'].items():
        directory = Path(row['directory'])
        if not directory.resolve().is_relative_to(RESTART.resolve()) or directory.resolve().is_relative_to(EXTENSION.resolve()):
            raise ValueError('unexpected historical directory')
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix in SUFFIXES:
                add(path, Path('historical') / arm / path.name, 'historical_1k_arm_metadata')
    for dirname in ('proximal_seed202609066', 'energy_ball_seed202609066', 'spatial_energy_balls_v1', 'affine_reflection_v1'):
        directory = RESTART / dirname
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix in SUFFIXES:
                add(path, Path('historical_context') / dirname / path.name, 'historical_1k_context_and_metrics')
        if dirname in ('spatial_energy_balls_v1', 'affine_reflection_v1'):
            for path in sorted((directory / 'parity16').iterdir()):
                if path.is_file() and path.suffix in SUFFIXES:
                    add(path, Path('historical_context') / dirname / 'parity16' / path.name, 'historical_native_parity_metadata')
    if sum(row['source'].stat().st_size for row in sources.values()) > MAX_TOTAL:
        raise ValueError('metadata package exceeds bounded total')
    return plan, sorted(sources.values(), key=lambda row: str(row['relative']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('snapshot', 'finalize'))
    parser.add_argument('--root-finalization-approved', action='store_true')
    args = parser.parse_args()
    final = args.mode == 'finalize'
    if final and not args.root_finalization_approved:
        parser.error('finalize requires the root closeout signal and --root-finalization-approved')
    if (OUTPUT / 'portable_manifest.json').exists():
        raise FileExistsError('package already finalized; no overwrite')
    plan, selected = collect(final)
    records, json_documents = [], []
    for row in selected:
        path, rel = row['source'], row['relative']
        data = path.read_bytes()
        if path.suffix == '.json':
            json_documents.append((path, json.loads(data)))
        target = OUTPUT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != data:
                raise ValueError(f'previously copied source changed: {path}')
        else:
            with target.open('xb') as stream:
                stream.write(data)
        copied = target.read_bytes()
        if copied != data:
            raise ValueError(f'copy mismatch: {path}')
        records.append({'source_path': str(path), 'resolved_source_path': str(path.resolve()),
                        'sha256': digest(data), 'size_bytes': len(data),
                        'copy_relative_path': str(rel), 'category': row['category'],
                        'verification': 'source and copied bytes both read; exact byte equality and SHA-256 verified'})
    current_paths = {record['copy_relative_path'] for record in records}
    previous = OUTPUT / 'draft_inventory.json'
    if previous.exists():
        prior = json.loads(previous.read_text())
        if not {r['copy_relative_path'] for r in prior['files']} <= current_paths:
            raise ValueError('final selection dropped a previously copied artifact')
    manifest = {'created_at_utc': utc(), 'complete': final,
                'source_root': str(EXTENSION), 'root_plan_sha256': digest((EXTENSION / 'plan.json').read_bytes()),
                'copied_file_count': len(records), 'copied_bytes': sum(r['size_bytes'] for r in records),
                'files': records, 'no_large_arrays_models_images_or_logs_copied': True,
                'archive_scope': 'All JSON/CSV under the completed extension except nested frozen source copies; exact historical 1K arm metadata, top-level historical metrics/context and native parity metadata; frozen protocol; explicitly root-approved initial FID reviewer Python recovery snapshot. Arrays/model assets remain external.',
                'pending_until_finalization': [] if final else sorted([*LATE, 'final_results_v1/*']),
                'exporter': {'path': str(Path(__file__)), 'sha256': digest(Path(__file__).read_bytes())}}
    if final:
        assets = {}

        def walk(value, source, pointer=''):
            if isinstance(value, dict):
                path, sha = value.get('path'), value.get('sha256')
                if isinstance(path, str) and isinstance(sha, str) and len(sha) == 64 and Path(path).suffix.lower() in {'.npz','.npy','.pt','.pth','.safetensors','.bin','.png','.jpg','.jpeg'}:
                    key = (path, sha)
                    record = assets.setdefault(key, {'recorded_path': path, 'recorded_sha256': sha,
                                                    'recorded_size_bytes': value.get('size_bytes'),
                                                    'first_metadata_source': {'path': str(source), 'json_pointer': pointer},
                                                    'metadata_reference_count': 0})
                    record['metadata_reference_count'] += 1
                for key, child in value.items():
                    walk(child, source, pointer + '/' + str(key).replace('~','~0').replace('/','~1'))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    walk(child, source, pointer + '/' + str(index))
        for path, value in json_documents:
            walk(value, path)
        asset_index = {'scope': 'Recorded file identities extracted from copied metadata; large asset bytes were NOT reread or copied by this exporter. Raw tensor hashes without file paths are not promoted to file SHA.',
                       'records': list(assets.values())}
        write_json(OUTPUT / 'recorded_large_asset_identities.json', asset_index)
        logs = [{'source_path': str(path), 'size_bytes_at_index': path.stat().st_size,
                 'sha256': None, 'verification': 'path/stat only; log bytes not copied or hashed'}
                for path in sorted(EXTENSION.rglob('*.log')) if path.is_file()]
        write_json(OUTPUT / 'raw_log_paths.json', {'scope': 'Original extension process logs only; not duplicated into Git.', 'logs': logs})
        manifest['generated_indexes'] = [{'copy_relative_path': name, 'sha256': digest((OUTPUT/name).read_bytes()),
                                          'size_bytes': (OUTPUT/name).stat().st_size}
                                         for name in ('recorded_large_asset_identities.json','raw_log_paths.json')]
        manifest['package_support_files'] = [{'copy_relative_path': name, 'sha256': digest((OUTPUT/name).read_bytes()),
                                             'size_bytes': (OUTPUT/name).stat().st_size}
                                            for name in ('README.md','export_portable.py')]
        if previous.exists():
            previous.unlink()
        write_json(OUTPUT / 'portable_manifest.json', manifest)
    else:
        write_json(previous, manifest)
    print(json.dumps({'complete': final, 'copied_file_count': len(records), 'copied_bytes': manifest['copied_bytes'],
                      'manifest': str(OUTPUT / ('portable_manifest.json' if final else 'draft_inventory.json')),
                      'no_experiments_evaluation_or_large_asset_reads': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
