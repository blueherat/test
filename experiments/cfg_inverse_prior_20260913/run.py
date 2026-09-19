"""Reuse the paired evaluator without resuming any parked search branch."""
from experiments.cfg_transport_search_20260913 import runner
from .fit import ROOT

if __name__=='__main__':
    runner.ROOT=ROOT/'generation'
    runner.MODULE='experiments.cfg_inverse_prior_20260913.run'
    runner.main()
