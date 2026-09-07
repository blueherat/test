"""Verify the entire archived research inventory against immutable Git blobs.

This checks what a commit contains, independently of working-tree existence.
It does not sample images, change methods, or claim that missing external assets
were archived. Defaults to the commit that closed all quality experiments.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = 'experiments/results/raev2_guidance_20260907/workspace_research_inventory.json'
DEFAULT_COMMIT = '573e939'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', default=DEFAULT_COMMIT)
    args = parser.parse_args()
    started = time.monotonic()
    commit = git('rev-parse', '--verify', args.commit + '^{commit}').decode().strip()
    manifest_bytes = git('show', f'{commit}:{INVENTORY}')
    manifest = json.loads(manifest_bytes)
    assert manifest['complete'] and manifest['all_regular_listed_files_fully_hashed']
    records = manifest['files']
    assert len(records) == manifest['file_count']
    assert hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest() == manifest['file_records_sha256']
    names = {row['path'] for row in records}
    assert len(names) == len(records) and INVENTORY not in names
    tree_names = {p.decode() for p in git('ls-tree', '-r', '--name-only', '-z', commit).split(b'\0') if p}
    assert tree_names == names | {INVENTORY}, 'Commit contains different files than the full inventory'
    total_bytes = 0
    # A regular input file avoids a pipe deadlock while Git emits large blobs.
    with tempfile.TemporaryFile() as queries:
        for row in records:
            assert '\n' not in row['path']
            queries.write(f"{commit}:{row['path']}\n".encode())
        queries.seek(0)
        with subprocess.Popen(['git', 'cat-file', '--batch'], cwd=ROOT,
                              stdin=queries, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as child:
            try:
                for row in records:
                    header = child.stdout.readline().split()
                    assert len(header) == 3 and header[1] == b'blob', (row['path'], header)
                    remaining = int(header[2])
                    assert remaining == row['bytes'], row['path']
                    total_bytes += remaining
                    digest = hashlib.sha256()
                    while remaining:
                        block = child.stdout.read(min(8 << 20, remaining))
                        assert block, 'Unexpected EOF from Git'
                        digest.update(block)
                        remaining -= len(block)
                    assert child.stdout.read(1) == b'\n'
                    expected = row.get('sha256', row.get('symlink_text_sha256'))
                    assert digest.hexdigest() == expected, row['path']
                assert child.stdout.read() == b''
                error = child.stderr.read().decode(errors='replace')
                assert child.wait() == 0, error
            finally:
                if child.poll() is None:
                    child.kill()
    assert total_bytes == manifest['total_bytes']
    goal_path = 'experiments/results/raev2_guidance_20260907/final_goal_requirement_audit.json'
    goal_bytes = git('show', f'{commit}:{goal_path}')
    goal = json.loads(goal_bytes)
    assert goal['complete'] and not goal['goal_achieved']
    assert goal['quality_research_closed'] and goal['quality_rows'] == 30
    assert all(row['best_improvement_vs_official_percent'] < 3 for row in goal['summaries'].values())
    review_path = 'experiments/results/raev2_guidance_20260907/workspace_reference_review.json'
    review = json.loads(git('show', f'{commit}:{review_path}'))
    absent = [r for r in review['rows'] if r.get('nearby_smoke_directories_not_accepted_as_replacement')]
    assert len(absent) == 8 and review['unreviewed_warnings'] == 0
    output = {
        'complete': True, 'goal_achieved': False, 'created_unix': time.time(),
        'verified_commit': commit, 'verified_root_tree': git('rev-parse', commit + '^{tree}').decode().strip(),
        'git_archival_of_quality_closeout_verified': True,
        'verification_scope': 'Every inventory file fully read from Git blobs, plus exact commit tree membership and the inventory blob itself',
        'inventory_blob_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
        'inventory_file_records_sha256': manifest['file_records_sha256'],
        'file_count_excluding_inventory_itself': len(records),
        'commit_file_count': len(tree_names), 'fully_verified_payload_bytes_excluding_inventory': total_bytes,
        'document_count': manifest['document_count'], 'goal_requirement_audit_blob_sha256': hashlib.sha256(goal_bytes).hexdigest(),
        'goal_still_unmet': True, 'best1k_improvement_percent': goal['summaries']['1000']['best_improvement_vs_official_percent'],
        'best5k_improvement_percent': goal['summaries']['5000']['best_improvement_vs_official_percent'],
        'eight_historical_external_raw_asset_gaps_remain_disclosed': True,
        'does_not_certify_all_external_assets_or_remote_backup': True,
        'no_training_sampling_or_new_parameter_changes': True,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'elapsed_seconds': time.monotonic() - started,
    }
    path = ROOT / 'experiments/results/raev2_guidance_20260907/round7_git_commit_verification.json'
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
