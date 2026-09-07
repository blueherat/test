"""Record one manual metadata-compatibility recovery; never resample images."""
import json
from pathlib import Path
import time
from experiments.continue_raev2_guidance_after_5k import process_identity
from experiments.summarize_raev2_guidance_20260907 import DATA, ROOT, sha


def main():
    folder = DATA/'final8_legacy5k_continuation'
    path = folder/'state.json'
    state = json.loads(path.read_text())
    assert state['stage'] == 'failed_requires_inspection_no_restart' and not state['complete']
    assert state['error'] == 'audit.log: child failed (1); inspect, no automatic repeat'
    assert process_identity(state['pid']) is None
    execution_path = DATA/'final8_legacy_confirm5k/execution.json'
    execution = json.loads(execution_path.read_text())
    assert execution['complete'] and all(process_identity(j['pid']) is None for j in execution['jobs'])
    audit_path = ROOT/'experiments/results/raev2_guidance_20260907/final8_legacy5k_audit.json'
    audit = json.loads(audit_path.read_text())
    assert audit['complete'] and audit['paired_inputs_and_all_merged_pixels_verified'] and audit['source_snapshots_verified']
    assert audit['execution_sha256'] == sha(execution_path)
    assert audit['audit_source_sha256'] == sha(ROOT/'experiments/audit_raev2_final8_legacy5k.py')
    assert {r['mode'] for r in audit['rows']} == {'paired_ratio_calibrated', 'calibrated'}
    assert not (folder/'state.failed_metadata_annotation.json').exists()
    (folder/'state.failed_metadata_annotation.json').write_bytes(path.read_bytes())
    state.update(complete=True, stage='legacy5k_complete_results_require_review', results=audit['rows'],
        recovery={'reason': audit['legacy_request_compatibility'], 'sampling_repeated': False,
                  'original_failure_sha256': sha(folder/'state.failed_metadata_annotation.json'),
                  'revised_audit_sha256': sha(audit_path), 'manual_recovery_source_sha256': sha(Path(__file__).resolve()),
                  'completed_unix': time.time()})
    state.pop('error')
    path.write_text(json.dumps(state, indent=2)+'\n')
    (ROOT/'experiments/results/raev2_guidance_20260907/final8_legacy5k_execution.json').write_text(json.dumps(state, indent=2)+'\n')
    print([(r['mode'], r['fid'], r['improvement_vs_official_percent'], r['inference_cost_ratio']) for r in audit['rows']])


if __name__ == '__main__':
    main()
