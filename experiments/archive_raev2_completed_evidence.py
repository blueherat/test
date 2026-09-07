"""Index complete research datasets and copy compact evidence into Git storage.

Large numerical arrays and image archives remain at their original data paths.
Every indexed file is actually hashed. Pass only immutable completed datasets;
an unfinished dataset is rejected, including an unfinished 5K execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--dataset', action='append', required=True, help='name=absolute_completed_directory')
    args = parser.parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    names = []
    for item in args.dataset:
        name, raw = item.split('=', 1)
        assert name and '/' not in name and name not in ('.', '..')
        source, target = Path(raw).resolve(), destination/name
        marks = [p for p in (source/'completion.json', source/'execution.json', source/'analysis.json') if p.exists()]
        assert marks and all(json.loads(p.read_text()).get('complete') is True for p in marks), f'incomplete: {source}'
        target.mkdir(exist_ok=True)
        records = []
        for path in sorted(source.rglob('*')):
            if not path.is_file() or path.suffix in ('.tmp', '.pyc'):
                continue
            relative, size = path.relative_to(source), path.stat().st_size
            compact = path.suffix in ('.json', '.csv', '.py', '.yaml', '.md', '.log') or path.name in ('finite_retention.png', 'finite_retention.pdf')
            compact = compact and size <= 8*1024**2
            record = {'relative_path': str(relative), 'data_path': str(path), 'bytes': size,
                      'sha256': sha(path), 'copied_to_git_evidence': compact}
            if compact:
                copied = target/relative
                copied.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy2(path, copied)
                assert sha(copied) == record['sha256']
            records.append(record)
        manifest = {'complete': True, 'dataset': name, 'data_directory': str(source),
            'file_count': len(records), 'total_bytes': sum(r['bytes'] for r in records),
            'copied_bytes': sum(r['bytes'] for r in records if r['copied_to_git_evidence']),
            'files': records, 'archiver_source_sha256': sha(Path(__file__).resolve())}
        (target/'archive_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        names.append({'dataset': name, 'file_count': len(records), 'bytes': manifest['total_bytes'], 'copied_bytes': manifest['copied_bytes']})
    print(json.dumps(names, indent=2))


if __name__ == '__main__':
    main()
