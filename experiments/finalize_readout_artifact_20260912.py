"""Reconcile rendered reports after the listed PNGs have been inspected manually."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import openpyxl
from experiments.guidance_pasted_20260912 import common as c


def main(case):
    names = {
        'normalization': ('ig_readout_normalization_20260912', 'IG_READOUT_NORMALIZATION_RESULTS_20260912_ZH.md'),
        '5k': ('context_reference_5k_20260912', 'CONTEXT_REFERENCE_5K_RESULTS_20260912_ZH.md')}
    name, filename = names[case]
    out = c.WORK / 'docs/data' / name
    report = c.WORK / 'docs' / filename
    manifest = c.read(out / 'artifact_manifest.json')
    for p, value in manifest['files'].items():
        assert c.sha(c.WORK / p) == value, p
    assert c.sha(report) == manifest['report_sha256']
    wb = openpyxl.load_workbook(out / 'source_data.xlsx', read_only=True, data_only=True)
    reconciled = []
    for sheet in wb.worksheets:
        path = out / (sheet.title + '.csv')
        if not path.exists():
            continue
        with path.open() as stream:
            expected = list(csv.reader(stream))
        actual = list(sheet.values)
        assert len(actual) == len(expected), sheet.title
        for a, b in zip(actual, expected):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                if x is None:
                    assert y == ''
                elif isinstance(x, bool):
                    assert str(x) == y
                elif isinstance(x, (float, int)):
                    assert math.isclose(x, float(y), rel_tol=1e-12, abs_tol=1e-10), (x, y)
                else:
                    assert str(x) == y, (x, y)
        reconciled.append(sheet.title)
    wb.close()
    links = []
    for url in re.findall(r'\]\(([^)]+)\)', report.read_text()):
        if url.startswith(('https://', 'http://', '#')):
            continue
        assert (report.parent / url).exists(), url
        links.append(url)
    verification = c.read(out / 'verification.json')
    assert verification['passed']
    verification.update(visual_inspection_pending=False,
        visual_inspection=dict(passed=True, inspected_figures={p.name: c.sha(p) for p in out.glob('*.png')},
            notes='All listed figures were displayed and inspected before this finalizer; readable labels and fixed unselected sample order.'),
        workbook_tables_reconciled=reconciled, local_links_verified=links)
    supplemental = {}
    raw = c.EXPS / name
    for filename in ('inference_benchmark.json', 'direct_readout_verification.json', 'replacement_api_verification.json'):
        path = raw / 'sit_small' / filename
        if path.exists():
            value = c.read(path)
            assert value.get('complete', value.get('passed', False))
            for source, digest in value.get('sources', {}).items():
                assert c.sha(source) == digest
            supplemental[str(path)] = c.sha(path)
    verification['supplemental_artifacts_sha256'] = supplemental
    c.atomic(out / 'verification.json', verification)
    manifest.update(files={str(p.relative_to(c.WORK)): c.sha(p) for p in out.iterdir() if p.name != 'artifact_manifest.json'},
        finalizer_sha256=c.sha(Path(__file__)))
    c.atomic(out / 'artifact_manifest.json', manifest)
    print('Final verification passed', case, reconciled, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=('normalization', '5k'), required=True)
    parser.add_argument('--visuals-reviewed', action='store_true', required=True)
    args = parser.parse_args()
    main(args.case)
