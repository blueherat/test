"""Reconcile finalized JiT artifacts; does not change the ongoing research goal."""
from pathlib import Path
import time
from experiments.jit_readout_transfer_20260913 import common as c


def check_project(name, stage, images):
    root = c.EXPS / name
    status = c.read(root / 'status.json')
    assert status['phase'] == 'complete' and status['images'] == images
    c.verify(root / stage / 'request.json')
    audit = c.read(root / stage / 'audit.json')
    assert audit['passed'] and audit['max_fid_error'] < .002
    output = c.WORK / 'docs/data' / name
    verification = c.read(output / 'verification.json')
    assert verification['passed'] and not verification['visual_inspection_pending']
    assert verification['samples'] == images and verification['workbook_tables_reconciled']
    manifest = c.read(output / 'artifact_manifest.json')
    for path, digest in manifest['files'].items():
        assert c.sha(c.WORK / path) == digest
    for path, digest in verification['source_files'].items():
        assert c.sha(path) == digest
    report = c.WORK / 'docs' / ('JIT_READOUT_CONFIRM_RESULTS_20260913_ZH.md' if stage == 'confirm_5000'
                              else 'JIT_READOUT_CFG_APPLICATION_RESULTS_20260913_ZH.md')
    assert c.sha(report) == manifest['report_sha256']
    assert c.sha(c.WORK / 'experiments' / name / 'report.py') == manifest['generator_sha256']
    pids = [status['pid']] + [x['pid'] for x in c.read(root / 'workers.json')['children']]
    process_checks = []
    for pid in pids:
        path = Path('/proc') / str(pid) / 'cmdline'
        command = path.read_bytes().replace(b'\0', b' ').decode(errors='replace') if path.exists() else ''
        assert name not in command, (pid, command)
        process_checks.append(dict(pid=pid, experiment_process_ended=True))
    return dict(name=name, images=images, audit_sha256=c.sha(root / stage / 'audit.json'),
        report=str(report), report_sha256=c.sha(report), manifest_sha256=c.sha(output / 'artifact_manifest.json'),
        process_checks=process_checks, decision=c.read(root / 'decision.json'))


def main():
    confirm = check_project('jit_readout_confirm_20260913', 'confirm_5000', 20000)
    projects = [confirm]
    if confirm['decision']['passes_ig_5k_gate']:
        projects.append(check_project('jit_readout_cfg_application_20260913', 'screen_1000', 4000))
    api = c.read(c.EXPS / 'jit_readout_confirm_20260913/replacement_api_verification.json')
    assert api['passed'] and api['samples_replayed'] == 8 and api['new_quality_samples'] == 0
    for path, digest in api['sources'].items():
        assert c.sha(path) == digest
    prior = c.read(c.EXPS / 'readout_followup_completion_20260913.json')
    for name, digest in prior['old_queue_stop_markers'].items():
        assert c.sha(c.EXPS / name / 'STOP_AFTER_CURRENT') == digest
    assert (c.EXPS / 'jit_prefix_transfer_20260912/STOP_AFTER_CURRENT').exists()
    result = dict(passed=True, utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        projects=projects, new_quality_images=sum(x['images'] for x in projects), replayed_images=8,
        old_queue_stop_markers=prior['old_queue_stop_markers'], independent_prefix_route_still_stopped=True,
        api_verification_sha256=c.sha(c.EXPS / 'jit_readout_confirm_20260913/replacement_api_verification.json'),
        research_note_sha256=c.sha(c.WORK / 'docs/CFG_RESEARCH_SCREEN_AFTER_JIT_20260913_ZH.md'),
        generator_sha256=c.sha(Path(__file__)), goal_complete=False)
    c.atomic(c.EXPS / 'jit_readout_confirm_20260913/completion_verification.json', result)
    print('Completed and reconciled', result['new_quality_images'], 'new JiT quality images; goal remains active', flush=True)


if __name__ == '__main__':
    main()
