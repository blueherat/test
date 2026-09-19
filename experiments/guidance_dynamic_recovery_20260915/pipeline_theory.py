"""Entry point for the frozen operator-matched follow-up amendment."""
from experiments.guided_weak_deployed_20260915 import install
from . import pipeline


if __name__ == '__main__':
    pipeline.install = install
    pipeline.main()

