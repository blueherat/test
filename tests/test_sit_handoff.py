import json
from pathlib import Path
import sys
import time

import psutil

from classifier_guidance.sit_joint_timed_handoff import run_bounded


def test_deadline_terminates_only_owned_process_tree(tmp_path):
    marker=tmp_path/'child.json'
    script=("import json,subprocess,sys,time; from pathlib import Path; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            f"Path({str(marker)!r}).write_text(json.dumps(dict(pid=p.pid))); time.sleep(60)")
    begin=time.monotonic()
    result=run_bounded([sys.executable,'-c',script],cutoff=time.time()+1.,grace=.1)
    assert result['forced'] and result['launched'] and time.monotonic()-begin<5
    pid=json.loads(marker.read_text())['pid']
    if psutil.pid_exists(pid):assert psutil.Process(pid).status()==psutil.STATUS_ZOMBIE


def test_expired_deadline_does_not_start_another_job(tmp_path):
    marker=tmp_path/'unexpected'
    result=run_bounded([sys.executable,'-c',f'from pathlib import Path; Path({str(marker)!r}).touch()'],
                       cutoff=time.time()-1)
    assert result['forced'] and not result['launched'] and not marker.exists()
