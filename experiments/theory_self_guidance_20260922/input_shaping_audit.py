"""CPU check of an illustrative signed input shaper, not a diffusion experiment."""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "docs/research/self_guidance_schedule_shape_20260923"


def main():
    omega = 2 * np.pi
    times = np.array([0., 1/6, 1/3])
    signed_weights = np.array([1., -1., 1.])
    modal_residual = np.sum(signed_weights * np.exp(-1j * omega * times))
    positive_times = np.array([0., .5])
    positive_weights = np.array([.5, .5])
    positive_residual = np.sum(positive_weights * np.exp(-1j * omega * positive_times))
    assert abs(modal_residual) < 1e-12
    assert abs(positive_residual) < 1e-12
    assert abs(signed_weights.sum() - 1) < 1e-12

    # Classical robustness diagnostic: cancellation at one frequency alone
    # does not make the shaper insensitive to errors in the estimated mode.
    offsets = [-.1, -.05, 0., .05, .1]
    sensitivity = [
        dict(relative_frequency_error=offset,
             signed_residual=float(abs(np.sum(
                 signed_weights * np.exp(-1j * omega * (1+offset) * times)))),
             positive_residual=float(abs(np.sum(
                 positive_weights * np.exp(-1j * omega * (1+offset) * positive_times)))))
        for offset in offsets
    ]
    result = dict(
        kind="Analytic input-shaping illustration; no fit to JiT or SiT",
        mode_frequency=omega, mode_period=1.,
        signed_shaper=dict(
            times=times.tolist(), weights=signed_weights.tolist(),
            dc_gain=float(signed_weights.sum()), modal_residual=float(abs(modal_residual)),
            duration=float(times[-1])),
        positive_shaper=dict(
            times=positive_times.tolist(), weights=positive_weights.tolist(),
            dc_gain=float(positive_weights.sum()), modal_residual=float(abs(positive_residual)),
            duration=float(positive_times[-1])),
        sensitivity=sensitivity,
        caveat="A control-input filter with unit DC gain and one modal zero. "
               "It does not demonstrate that a learned guidance curve suppresses this mode.",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "input_shaping_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
