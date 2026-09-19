import argparse
import fcntl
import signal
from . import config as k, planning
from experiments.guidance_strength_sweep_20260915 import pipeline as previous


def install():
    previous.k = k
    previous.planning = planning
    previous.install()
    previous.dispatch.AMENDMENT = k.AMENDMENT


class Supervisor(previous.Supervisor):
    def publish(self, phase=None, **extra):
        super().publish(phase, sampling_shards_per_point=k.SHARDS,
            gpu_policy='four GPUs share original batches within the current idea', **extra)

    def advance(self):
        failed = [planning.check_id(rank) for rank in range(k.SHARDS)
            if self.ledger[planning.check_id(rank)]['state'] in ('failed', 'blocked')]
        if failed:
            self.publish('failed', reason='Four-GPU replay checks failed', failed_checks=failed)
            raise RuntimeError(f'Four-GPU checks failed: {failed}')
        return super().advance()


def cpu_check():
    starts = [start for rank in range(k.SHARDS) for start in k.batch_starts(rank)]
    assert len(starts) == len(set(starts)) == 125
    assert sorted(starts) == list(range(0, 1000, 8))
    samples = [i for start in starts for i in range(start, min(start + 8, 1000))]
    assert sorted(samples) == list(range(1000))
    jobs = planning.point_jobs(k.key('guided_weak', 40))
    assert sum(job['gpu'] for job in jobs) == 4
    assert jobs[-2]['action'] == 'sweep_assemble' and not jobs[-2]['gpu']
    assert set(jobs[-2]['depends']) == {job['id'] for job in jobs[:4]}
    assert jobs[-1]['depends'] == [jobs[-2]['id']]
    assert planning.point_job_id(k.key('guided_weak', 40)) == jobs[-1]['id']
    # Existing seriality and connected-interval selection are inherited unchanged.
    intervals, ticks = planning.refine([
        dict(tick=tick, fid=fid, valid=True) for tick, fid in
        ((0, 10.), (16, 11.), (32, 100.), (48, 11.), (64, 10.), (80, 100.))], 16, 8)
    assert len(intervals) == 2 and ticks == list(range(0, 65, 8))
    k.atomic(k.ROOT / 'four_gpu_cpu_check.json', dict(passed=True,
        dispatch_sha256=k.sha(k.AMENDMENT), no_missing_or_duplicate_samples=True,
        original_batch_boundaries=True, connected_refinement_preserved=True))
    print('Four-GPU partition and DAG checks passed', flush=True)


def main():
    install()
    def stop(signum, frame):
        (k.ROOT / 'STOP_AFTER_CURRENT').write_text(f'Four-GPU controller signal {signum}\n')
        (k.old.ROOT / 'STOP_AFTER_CURRENT').write_text(f'Four-GPU controller signal {signum}\n')
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    with (k.ROOT / 'controller.lock').open('a') as lock, (k.old.ROOT / 'controller.lock').open('a') as old_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(old_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        k.verify()
        k.old.verify()
        assert k.read(k.ROOT / 'four_gpu_cpu_check.json')['passed']
        Supervisor().run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        k.prepare()
        cpu_check()
    else:
        main()
