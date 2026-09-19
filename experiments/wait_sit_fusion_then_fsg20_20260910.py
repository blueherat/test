"""Resume the frozen FSG queue only after the entire fusion pipeline finishes.

This external scheduler corrects the stage-completion race without changing
either experiment's frozen sampling, evaluation, or selection implementation.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

WORK = Path('/home/zhoushunyu/eqvae')
BASE = Path('/home/zhoushunyu/data/eqvae/experiments')
ROOT = BASE / 'sit_fsg_followup_20ideas_20260910'
DEPENDENCY = BASE / 'sit_guidance_fusion_20260910'
RECORD = ROOT / 'serial_queue_correction_20260910.json'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def pipeline_finished(status):
    return (status.get('phase') == 'complete'
            and status.get('no_further_sampling_queued') is True
            and 'stage' not in status)


def flag_signature(path):
    stat = path.stat()
    return dict(inode=stat.st_ino, mtime_ns=stat.st_mtime_ns,
                size=stat.st_size, sha256=sha(path))


def verify_frozen(record):
    for name, digest in record['request_hashes'].items():
        path = Path(name)
        assert sha(path) == digest, path
        for source, expected in read(path)['sources'].items():
            assert sha(Path(source)) == expected, source
    path = ROOT / 'fsg_screen_1k/results.json'
    assert read(path) == record['preserved_results'], 'Existing results changed during pause'


def annotate_final_timing():
    # The final marker is written after all report writers have finished.
    warning = ('> 计时修正（2026-09-10）：`ig_native_03` 的采样期间与 FSG 后续队列'
               '短暂共用 GPU。保留其原始质量结果与耗时记录，但不能用该耗时计算公平的'
               '相对速度；Full/prefix 调用数仍有效。详见 '
               '`sit_guidance_fusion_20260910/timing_overlap_20260910.json`。\n\n')
    for path in (WORK / 'docs/SIT_GUIDANCE_FUSION_CONFIRMATION_RESULTS_20260910_ZH.md',
                 DEPENDENCY / 'selected_5k/report.md'):
        if path.exists():
            text = path.read_text()
            if warning not in text:
                first, rest = text.split('\n', 1)
                path.write_text(first + '\n\n' + warning + rest)


def main():
    record = read(RECORD)
    verify_frozen(record)
    # The same lock used by the frozen controller prevents a second live queue.
    with (ROOT / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = None
        while True:
            dependency = read(DEPENDENCY / 'status.json')
            if pipeline_finished(dependency):
                stage = read(DEPENDENCY / 'selected_5k/status.json')
                assert stage['phase'] == 'complete'
                assert stage['completed'] == stage['total'] == 7
                assert (DEPENDENCY / 'selected_5k/analysis_audit.json').exists()
                break
            if dependency.get('phase') == 'failed':
                atomic(ROOT / 'status.json', dict(phase='dependency_failed',
                       controller_pid=os.getpid(), dependency=dependency))
                raise RuntimeError(f'Fusion dependency failed: {dependency}')
            status = dict(phase='waiting_for_fusion_final', controller_pid=os.getpid(),
                          completed=len(record['preserved_results']), total=256,
                          dependency=dependency, scheduler=str(Path(__file__).resolve()),
                          required_marker='complete + no_further_sampling_queued=true',
                          updated_unix=time.time())
            atomic(ROOT / 'status.json', status)
            atomic(ROOT / 'serial_gate_status.json', status)
            progress = (dependency.get('stage'), dependency.get('phase'),
                        dependency.get('completed'), dependency.get('arm'))
            if progress != previous:
                print(json.dumps(status), flush=True)
                previous = progress
            time.sleep(10)
        verify_frozen(record)
        annotate_final_timing()
        flag = ROOT / 'STOP_AFTER_CURRENT'
        assert flag_signature(flag) == record['owned_stop_flag'], 'Pause flag changed; preserving it'
        flag.unlink()
        atomic(ROOT / 'serial_gate_status.json', dict(phase='releasing',
               controller_pid=os.getpid(), dependency=dependency, released_unix=time.time(),
               preserved_arms=[row['arm'] for row in record['preserved_results']]))
    # No GPU/model import occurs in this process before the final marker.
    os.chdir(WORK)
    os.execv(sys.executable, [sys.executable, '-u', '-m',
                             'experiments.run_sit_fsg_20ideas_20260910', '--pipeline'])


if __name__ == '__main__':
    main()
