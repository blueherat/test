#!/usr/bin/env python3
"""Copy the completed Round 2's bounded numerical evidence into Git-sized data."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
RESTART = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906')
OUTPUT = ROOT / 'docs/data/raev2_pressure_innovation_20260906'


def identity(path):
    data = path.read_bytes()
    return dict(path=str(path), resolved_path=str(path.resolve()),
                sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def main():
    audit = RESTART / 'pressure_innovation_v1'
    sources = {name: audit / name for name in
               ('request.json', 'summary.json', 'per_time.csv',
                'channel_moments.npz', 'gaussian_counterexample.json')}
    sources['independent_review.json'] = RESTART / 'pressure_innovation_review_v1/independent_review.json'
    sources['producer_request.json'] = RESTART / 'paired_bridge_v1/validate/request.json'
    sources['producer_summary.json'] = RESTART / 'paired_bridge_v1/validate/summary.json'
    assert json.loads(sources['summary.json'].read_text())['complete']
    assert json.loads(sources['independent_review.json'].read_text())['complete']
    records = {name: identity(path) for name, path in sources.items()}
    assert sum(record['bytes'] for record in records.values()) < 2_000_000
    OUTPUT.mkdir(exist_ok=False)
    for name, path in sources.items():
        shutil.copyfile(path, OUTPUT / name)
        assert identity(OUTPUT / name)['sha256'] == records[name]['sha256']
        records[name]['package_path'] = name
    readme = (
        '# RAEv2 final Round 2: pressure and symmetric innovation\n\n'
        'Completed CPU-only mechanism audit; no new training, GPU sampling or image FID. '
        'The current variants were not admitted to training.\n\n'
        'Eight original small outputs are copied byte-for-byte. `per_time.csv` covers every '
        'fixed time; `channel_moments.npz` retains all 100 × 1024 channel differences. '
        '`independent_review.json` records an independent source-cache recomputation, '
        'including 50-digit Decimal accumulation.\n\n'
        '`request.json` and the independent review identify upstream source files by path '
        'and SHA-256. Original 10K teacher rows (3.74 MB) and the full moment cache (6.56 MB) '
        'remain in the external experiment directory and are not copied here. Thus the '
        'package supports inspection of the reported aggregates and provenance, while '
        'rerunning the source-cache review requires those originals. Original residual '
        'and W/Y tensors were not saved by the producer; their absence is not repaired '
        'by this export.\n\n'
        'Population moment identities, this finite cohort, and the analytic Gaussian '
        'counterexample have separate scopes; none is a generated-image quality result. '
        'The Gaussian number is a one-dimensional exact example with an identity decoder.\n\n'
        'See [Chinese result record](../../RAEV2_PRESSURE_INNOVATION_RESULTS_20260906_ZH.md).\n'
    )
    (OUTPUT / 'README.md').write_text(readme)
    manifest = dict(
        version=1, round=2, created_utc=datetime.now(timezone.utc).isoformat(),
        exporter=identity(Path(__file__).resolve()), source_root=str(RESTART),
        scope='Completed fixed-cache mechanism decision; no additional experiment round.',
        copied_file_count=len(records), copied_bytes=sum(record['bytes'] for record in records.values()),
        files=records, readme=identity(OUTPUT / 'README.md'),
        documentation={name: identity(ROOT / 'docs' / name) for name in
                       ('RAEV2_PRESSURE_INNOVATION_PROTOCOL_20260906_ZH.md',
                        'RAEV2_PRESSURE_INNOVATION_RESULTS_20260906_ZH.md')},
        omitted_sources='Original teacher CSV and moments cache retained externally; SHA identities in '
                        'copied request/review. No model weights, source tensors, images or paper PDFs copied.',
    )
    dest = OUTPUT / 'manifest.json'
    dest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(manifest=identity(dest), files=len(records), bytes=manifest['copied_bytes']), indent=2))


if __name__ == '__main__':
    main()
