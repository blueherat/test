"""Isolated frozen SiT paired experiment runner."""
from experiments.cfg_transport_search_20260913 import runner

if __name__ == '__main__':
    runner.ROOT = runner.c.EXPS / 'weak_reference_20260914'
    runner.MODULE = 'experiments.weak_reference_20260914.run'
    runner.main()
