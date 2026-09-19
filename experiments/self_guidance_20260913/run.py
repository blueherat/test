"""Reuse the frozen paired SiT image/FID runner in an isolated output root."""
from experiments.cfg_transport_search_20260913 import runner

if __name__ == '__main__':
    runner.ROOT = runner.c.EXPS / 'self_guidance_20260913'
    runner.MODULE = 'experiments.self_guidance_20260913.run'
    runner.main()
