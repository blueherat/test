"""Verify compact copies, large-file availability and final closeout consistency."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    manifests = sorted(bundle.glob('*/archive_manifest.json'))
    assert len(manifests) == 5
    copied = indexed = total_bytes = copied_bytes = 0
    for path in manifests:
        manifest = read(path)
        assert manifest['complete'] and len(manifest['files']) == manifest['file_count']
        for row in manifest['files']:
            original = Path(row['data_path'])
            assert original.is_file() and original.stat().st_size == row['bytes'], original
            indexed += 1
            total_bytes += row['bytes']
            if row['copied_to_git_evidence']:
                copy = path.parent/row['relative_path']
                assert copy.stat().st_size == row['bytes'] and sha(copy) == row['sha256'], copy
                copied += 1
                copied_bytes += row['bytes']
    context = bundle/'round2_run_context'
    for row in read(context/'manifest.json')['files']:
        assert sha(context/row['file']) == row['sha256']
    summary, state = read(bundle/'closeout.json'), read(bundle/'state.json')
    assert summary['complete'] and summary['completed_additional_idea_rounds'] == 3
    assert state['rounds_completed'] == 3 and state['new_idea_rounds_remaining'] == 0
    assert state['research_expansion_stopped'] and not state['goal_3_percent_achieved']
    assert summary['stop_new_research'] and not summary['discovery_point_reaches_three_percent']
    assert not summary['quality_goal_certified']
    for entry in summary['source_audits'].values():
        assert sha(Path(entry['path'])) == entry['sha256']
    for row in summary['paired5k_rows']:
        gain = 100*(1-row['fid']/summary['baseline_fid'])
        assert abs(gain-row['improvement_percent']) < 1e-12
        assert row['samples'] == 5000 and row['seed'] == 202609072
    figure = read(bundle/'figure_manifest.json')
    assert figure['source_summary_sha256'] == sha(bundle/'closeout.json')
    for name, digest in figure['outputs'].items():
        assert sha(bundle/name) == digest
    final = read(bundle/'round3_full_read_after_write/completion.json')
    assert final['complete'] and all(not row['verified_alive'] for row in final['process_exit_checks'])
    result = {'complete': True, 'indexed_datasets': len(manifests), 'indexed_files': indexed,
        'indexed_bytes': total_bytes, 'copied_files_sha_verified': copied,
        'copied_bytes_sha_verified': copied_bytes, 'all_indexed_data_paths_and_sizes_verified': True,
        'large_file_sha_scope': 'Large files were actually hashed by the completed dataset archiver; this check verifies current existence and size, not a second large-file hash pass.',
        'all_summary_source_audits_sha_verified': True, 'summary_and_plot_consistent': True,
        'three_round_limit_and_unmet_goal_consistent': True,
        'final_process_exit_record_verified': True,
        'verifier_source_sha256': sha(Path(__file__).resolve())}
    (bundle/'archive_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
