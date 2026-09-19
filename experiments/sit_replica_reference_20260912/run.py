import fcntl
from . import catalog as c, train, pipeline
from experiments.sit_strong_reference_20260912.run import group as previous_group
from experiments.sit_strong_reference_20260912 import run as group_module


def group(tasks,phase):
    prior=group_module.c
    try:
        group_module.c=c;previous_group(tasks,phase)
    finally:group_module.c=prior


def main():
    c.ROOT.mkdir(parents=True,exist_ok=True)
    with (c.ROOT/'orchestration.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (c.ROOT/'STOP_AFTER_CURRENT').exists():return
        train.prepare()
        group([(r,['-m','experiments.sit_replica_reference_20260912.train','--train',method]) for r,method in enumerate(c.METHODS)],'training')
        pipeline.pipeline()


if __name__=='__main__':main()
