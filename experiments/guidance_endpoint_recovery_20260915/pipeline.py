from . import install
from experiments.guidance_dynamic_recovery_20260915 import pipeline


if __name__ == '__main__':
    pipeline.install = install
    pipeline.main()
