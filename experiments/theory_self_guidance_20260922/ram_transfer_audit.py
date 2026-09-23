"""CPU counterexamples for transferring RAM to a restricted guidance field.

Analytic Gaussian laws, independent quadrature, and finite probability tables.
No image model, RAM training, GPU work, or empirical speedup is involved.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/eqvae_self_guidance_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import quad
from scipy.linalg import expm


OUT = Path(__file__).resolve().parents[2] / "docs/research/self_guidance_ram_20260923"
J = np.array([[0., -1.], [1., 0.]])


def noising_coefficient(t, clean_variance):
    variance = (1-t)**2 * clean_variance + t*t
    return (t-(1-t)*clean_variance) / variance


def conditional_ram(t, state, clean_variance):
    """Data-to-noise convention of RAM; reward = -clean^2/2."""
    variance = (1-t)**2 * clean_variance + t*t
    mean = (1-t) * clean_variance / variance * state
    posterior_variance = t*t * clean_variance / variance
    nodes, weights = hermgauss(32)
    clean = mean + np.sqrt(2 * posterior_variance) * nodes
    target = (state-clean) / t
    velocity = noising_coefficient(t, clean_variance) * state
    return float(weights @ (-.5 * clean**2 * (target-velocity)) / np.sqrt(np.pi))


def gaussian_ram_fixed_point():
    # Ref N(0,1), reward -x^2/2: exact terminal KL optimum is N(0,1/2).
    t, x, clean_variance = .5, 1., .5
    displacement = (noising_coefficient(t, clean_variance)
                    - noising_coefficient(t, 1.)) * x
    ram_right = conditional_ram(t, x, clean_variance)

    def tilt_derivative(lam):
        variance = 1 / (1 + lam)
        marginal = (1-t)**2 * variance + t*t
        return t * (1-t) * variance**2 / marginal**2 * x

    exact_integral = quad(tilt_derivative, 0, 1, epsabs=1e-13)[0]
    assert abs(displacement - 2/3) < 1e-13
    assert abs(ram_right - 4/9) < 1e-13
    assert abs(exact_integral-displacement) < 1e-13
    return dict(
        reference_variance=1., reward="-x^2/2", true_KL_optimum_variance=.5,
        noising_time=t, state=x, exact_velocity_displacement=displacement,
        RAM_reward_covariance=ram_right, RAM_fixed_point_residual=displacement-ram_right,
        exact_tilt_path_integral=exact_integral,
    )


def gauge_and_restricted_family():
    # Noise-to-data convention here. canonical base has identity endpoint.
    omega = .4
    covariance_error = 0.
    for t in np.linspace(0, 1, 101):
        marginal = (1-t)**2+t*t
        flow = np.sqrt(marginal) * expm(omega*t*J)
        covariance_error = max(covariance_error,
                               float(np.max(np.abs(flow @ flow.T - marginal*np.eye(2)))))
    shifts = [dict(constant_reward=c,
                   half_RAM_semigradient=(4/3)*(1+c)*omega,
                   endpoint_reward_gradient=0., endpoint_KL_gradient=0.)
              for c in [0., 1., -1., -2.]]

    # A one-parameter family couples removable rotation to endpoint scale:
    # v_theta=(b(t)+theta(theta-1))*x + rotation*theta*J*x.
    # At theta=1 it generates the exact base endpoint but RAM moves it.
    theta, rotation, learning_rate = 1., .7, .1
    ram_gradient = (4/3) * rotation**2
    after = theta - learning_rate * ram_gradient

    def endpoint_kl(value):
        log_scale = value*(value-1)
        return np.expm1(2*log_scale)-2*log_scale

    before_kl, after_kl = endpoint_kl(theta), endpoint_kl(after)
    assert before_kl == 0 and after_kl > 0
    return dict(
        all_marginal_covariance_error=covariance_error,
        rotation_coefficient=omega, reward_shift_examples=shifts,
        restricted_family=dict(
            velocity="(b(t)+theta*(theta-1))*x + 0.7*theta*J*x",
            initial_theta=theta, constant_reward=0.,
            half_RAM_gradient=ram_gradient, learning_rate=learning_rate,
            updated_theta=after, initial_endpoint_KL=float(before_kl),
            updated_endpoint_KL=float(after_kl),
            interpretation="Canonical reference regression can leave an endpoint-optimal member of a restricted family.",
        ),
    )


def feature_and_refresh():
    base = np.array([[.45, .05], [.10, .40]])
    data = np.array([[.10, .20], [.60, .10]])
    q = base * (data.sum(1) / base.sum(1))[:, None]
    kl = lambda first, second: float(np.sum(first*np.log(first/second)))
    conditional_residual = sum(
        data.sum(1)[i] * kl(data[i]/data.sum(1)[i], base[i]/base.sum(1)[i])
        for i in range(2)
    )
    assert np.allclose(q.sum(1), data.sum(1))
    assert abs(kl(data, q)-conditional_residual) < 1e-13
    refresh_base = np.array([.8, .2])
    refresh_data = np.array([.2, .8])
    sequence = [refresh_base.copy()]
    for _ in range(4):
        updated = refresh_base * refresh_data / sequence[-1]
        sequence.append(updated / updated.sum())
    assert np.allclose(sequence[2], refresh_base)
    fixed_point = np.sqrt(refresh_base * refresh_data)
    fixed_point /= fixed_point.sum()
    return dict(
        feature_target=dict(base=base.tolist(), data=data.tolist(), optimal=q.tolist(),
                            target_feature_marginal=data.sum(1).tolist(),
                            matched_feature_marginal=q.sum(1).tolist(),
                            full_data_KL_to_optimal=kl(data,q),
                            conditional_KL_identity=conditional_residual),
        fixed_base_current_discriminator_refresh=dict(
            base=refresh_base.tolist(), data=refresh_data.tolist(),
            exact_best_response_sequence=[value.tolist() for value in sequence],
            fixed_point=fixed_point.tolist(),
        ),
    )


def rlg_scaling():
    weight = 2.
    integral = quad(lambda t: (1-weight)*noising_coefficient(t,1.)
                    + weight*noising_coefficient(t,.5), 1, 0, epsabs=1e-13)[0]
    actual_variance = float(np.exp(2*integral))
    assert abs(actual_variance-.25) < 1e-13
    return dict(reference_variance=1., unit_reward_tilt_variance=.5,
                guidance_weight=weight, mixed_velocity_endpoint_variance=actual_variance,
                stronger_reward_true_tilt_variance=1/(1+weight))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(
        kind="CPU analytic transfer audits; no image experiment",
        gaussian_RAM_fixed_point=gaussian_ram_fixed_point(),
        gauge_and_restricted_family=gauge_and_restricted_family(),
        feature_and_refresh=feature_and_refresh(),
        RLG_velocity_scaling=rlg_scaling(),
    )
    (OUT/"ram_transfer_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    times=np.linspace(.02,.98,200)
    displacements=np.array([noising_coefficient(t,.5)-noising_coefficient(t,1.) for t in times])
    ram=np.array([conditional_ram(t,1.,.5) for t in times])
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.8),constrained_layout=True)
    axes[0].plot(times,displacements,label="Exact KL-optimum displacement",linewidth=2)
    axes[0].plot(times,ram,label="RAM right-endpoint target",linewidth=2)
    axes[0].set(xlabel="Data-to-noise time",ylabel="Velocity coefficient at x=1",
                title="RAM target need not fix the KL optimum")
    axes[0].legend(fontsize=8)
    rewards=np.linspace(-2,1,101)
    axes[1].plot(rewards,(4/3)*(1+rewards)*.4,label="RAM semigradient",linewidth=2)
    axes[1].axhline(0,color="black",linewidth=.8,label="Endpoint gradient")
    axes[1].set(xlabel="Constant reward offset",ylabel="Rotation-parameter gradient",
                title="Reward offsets expose velocity inconsistency")
    axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(OUT/"ram_transfer_audit.png",dpi=180)
    fig.savefig(OUT/"ram_transfer_audit.pdf")
    plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
