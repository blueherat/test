"""Finish the numerical audit and report after the existing controller exits."""
import os
from pathlib import Path
import subprocess
import time
import traceback

from experiments.guidance_pasted_20260912 import common as c
from . import core as m


def main():
    workers = c.read(m.ROOT/'workers.json')
    parent = workers['parent']
    state_path = m.ROOT/'postprocess_status.json'
    c.atomic(state_path,dict(pid=os.getpid(),phase='waiting_for_existing_controller',controller=parent))
    try:
        while True:
            path = Path('/proc')/str(parent)/'cmdline'
            cmd = path.read_bytes().decode(errors='replace') if path.exists() else ''
            running = 'experiments.raev2_context_5k_20260914.run' in cmd
            state = c.read(m.ROOT/'status.json')
            if not running:
                assert state['phase']=='complete',state
                break
            time.sleep(10)
        c.atomic(state_path,dict(pid=os.getpid(),phase='auditing_and_reporting',controller=parent))
        subprocess.run([c.PYTHON,'-u','-m','experiments.raev2_context_5k_20260914.report'],
            cwd=c.WORK,check=True)
        out=c.WORK/'docs/data/raev2_context_5k_20260914'
        verification=c.read(out/'verification.json')
        assert verification['passed'] and verification['new_quality_images']==10000
        c.atomic(state_path,dict(pid=os.getpid(),phase='complete',quality_images=10000,
            numerical_audit_passed=True,report=str(c.WORK/'docs/RAEV2_CONTEXT_5K_RESULTS_20260914_ZH.md'),
            visual_review_pending=verification['visual_review_pending']))
        print('Both 5K arms, metrics, timing, numerical audit and report are complete.',flush=True)
    except BaseException:
        c.atomic(state_path,dict(pid=os.getpid(),phase='failed',error=traceback.format_exc()))
        raise


if __name__=='__main__':main()
