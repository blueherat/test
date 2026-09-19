"""Two retained mechanisms; three refinements and one control per mechanism."""
from copy import deepcopy
from experiments.recursive_guidance_20260913 import catalog as prior

IDEAS = [deepcopy(row) for row in prior.IDEAS if row['id'] in (1, 7)]
IDEAS[0]['title'] = '回灌一阶增益门控：局部系数精炼'
IDEAS[1]['title'] = '回灌方向反向补偿：分离方向与幅度'


def configurations():
    def cfg(arm, key, source, strength, coefficient=None, role='candidate', **parameters):
        if coefficient is not None:
            parameters['probe_coefficient'] = coefficient
        idea = 1 if source == 'cfg' else 7
        return dict(arm=arm, family=f'focus_i{idea:02d}' if role == 'candidate' else arm,
                    key=key, source=source, strength=strength,
                    theta=(.5 if source == 'cfg' else .65) if key != 'native' else 0.,
                    role=role, idea_id=idea if role == 'candidate' else None,
                    reference='native' if source == 'cfg' else 'mlp', solver='heun64',
                    cutoff=.75 if source == 'cfg' else .5, parameters=parameters)
    # Alternate mechanisms; show the decisive controls early as well.
    return [
        cfg('cfg_gain_l2', 'cfg_cycle_gain', 'cfg', 1.25, 2.),
        cfg('ig_rotation_norm_preserved', 'ig_cycle_extrapolate', 'ig', .4, .5, preserve_norm=True),
        cfg('cfg_native_a1p24', 'native', 'cfg', 1.24, role='control'),
        cfg('ig_amplitude_only', 'ig_cycle_extrapolate', 'ig', .4, .5, role='control', norm_only=True),
        cfg('cfg_gain_l0p5', 'cfg_cycle_gain', 'cfg', 1.25, .5),
        cfg('ig_extrapolate_l0p25', 'ig_cycle_extrapolate', 'ig', .4, .25),
        cfg('cfg_gain_l4', 'cfg_cycle_gain', 'cfg', 1.25, 4.),
        cfg('ig_extrapolate_l0p75', 'ig_cycle_extrapolate', 'ig', .4, .75),
    ]
