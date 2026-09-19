"""Run this one fixed application experiment only after the independent IG gate."""
import time
from . import core as m
from experiments.jit_readout_transfer_20260913 import common as c


def main():
    while True:
        status = c.read(m.CONFIRM / 'status.json')
        if status['phase'] == 'failed':
            raise RuntimeError('Independent 5K controller failed; no application samples started')
        if status['phase'] == 'complete':
            break
        time.sleep(5)
    if not c.read(m.CONFIRM / 'decision.json')['passes_ig_5k_gate']:
        c.atomic(m.ROOT / 'status.json', dict(phase='skipped', images=0,
            reason='Independent 5K IG confirmation did not pass'))
        print('Application experiment skipped after failed 5K gate', flush=True)
        return
    check = c.read(m.ROOT / 'implementation_check.json')
    assert check['passed'] and check['arms'] == list(m.ARMS)
    for path, digest in check['sources'].items():
        assert c.sha(path) == digest
    from .run import main as run
    run()


if __name__ == '__main__':
    main()
