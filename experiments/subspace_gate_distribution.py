"""Explicit geometry factory; curved settings deliberately have no Bayes oracle."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_dual_target_closed_loop_spiral_toy as spiral


class CurvedSpiralDistribution(spiral.ContinuousSpiralDistribution):
    def __init__(self,ambient_dim,*,curvature,**kwargs):
        super().__init__(ambient_dim,curvature=0.,**kwargs)
        self.curvature=curvature
        self.embedding=spiral.v4.CurvedEmbedding(ambient_dim,curvature=curvature,
            frequency_scale=kwargs['frequency_scale'],seed=kwargs['embedding_seed'],
            device=kwargs['device'],scale_mode=kwargs['scale_mode'])
        self.basis=self.embedding.Q[:,:2]
        self.scale=float(self.embedding.global_scale)

    def bayes_clean(self,*args,**kwargs):
        raise NotImplementedError('No exact Bayes oracle implemented for nonlinear embedding')

    def bayes_velocity(self,*args,**kwargs):
        raise NotImplementedError('No exact Bayes oracle implemented for nonlinear embedding')


def make_distribution(cfg,device):
    cls=CurvedSpiralDistribution if cfg['curvature'] else spiral.ContinuousSpiralDistribution
    return cls(cfg['ambient_dim'],data_jitter=cfg['data_jitter'],quadrature_points=cfg['quadrature_points'],
        locator_points=cfg['locator_points'],frequency_scale=cfg['frequency_scale'],
        embedding_seed=spiral.core.stable_seed(cfg['seed'],cfg['ambient_dim'],71),device=device,
        scale_mode=cfg['scale_mode'],curvature=cfg['curvature'],bayes_batch_chunk=cfg['bayes_batch_chunk'])
