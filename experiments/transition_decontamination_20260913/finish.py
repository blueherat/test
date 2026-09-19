"""Verify finalized artifacts and stopped workers for this bounded experiment."""
from pathlib import Path
from . import core as m

c=m.c


def main():
    root=m.ROOT
    status=c.read(root/'status.json')
    assert status['phase']=='complete' and status['images']==4800
    for track in m.TRACKS:m.verify(track)
    out=c.WORK/'docs/data/transition_decontamination_20260913'
    verification=c.read(out/'verification.json')
    assert verification['passed'] and not verification['visual_inspection_pending']
    assert verification['workbook_tables_reconciled'] and verification['samples']==4800
    manifest=c.read(out/'artifact_manifest.json')
    for path,digest in manifest['files'].items():assert c.sha(c.WORK/path)==digest,path
    report=c.WORK/'docs/TRANSITION_DECONTAMINATION_RESULTS_20260913_ZH.md'
    assert c.sha(report)==manifest['report_sha256']
    assert c.sha(Path(__file__).with_name('report.py'))==manifest['generator_sha256']
    for path,digest in verification['source_files'].items():assert c.sha(path)==digest,path
    workers=c.read(root/'workers.json')
    pids=[workers['parent']]+[r['pid'] for r in workers['children']]
    process_checks=[]
    for pid in pids:
        p=Path('/proc')/str(pid)/'cmdline'
        command=p.read_bytes().decode(errors='replace') if p.exists() else ''
        assert 'experiments.transition_decontamination_20260913' not in command,(pid,command)
        process_checks.append(dict(pid=pid,experiment_process_ended=True))
    old=c.read(c.EXPS/'readout_followup_completion_20260913.json')
    for name,digest in old['old_queue_stop_markers'].items():
        assert c.sha(c.EXPS/name/'STOP_AFTER_CURRENT')==digest,name
    assert (c.EXPS/'jit_prefix_transfer_20260912/STOP_AFTER_CURRENT').exists()
    result=dict(passed=True,new_quality_images=4800,source_manifest_verified=True,
        process_checks=process_checks,old_queue_stop_markers=old['old_queue_stop_markers'],
        independent_prefix_route_still_stopped=True,decision=c.read(root/'decision.json'),
        report_sha256=c.sha(report),artifact_manifest_sha256=c.sha(out/'artifact_manifest.json'),
        generator_sha256=c.sha(Path(__file__)),goal_complete=False)
    c.atomic(root/'completion_verification.json',result)
    print('4800 images, numeric checks, source workbook, figures and worker completion verified',flush=True)


if __name__=='__main__':main()
