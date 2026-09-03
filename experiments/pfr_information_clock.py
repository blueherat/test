"""Information-coordinate horizons for PFR counterfactual queries."""

from __future__ import annotations

import math


INFORMATION_CLOCKS = ("raw_t", "log_snr", "snr")


def _validate_time(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must lie in [0, 1)")
    return value


def _snr(time_value: float) -> float:
    odds = time_value / (1.0 - time_value)
    return odds * odds


def _time_from_snr(snr: float) -> float:
    root = math.sqrt(max(float(snr), 0.0))
    return root / (1.0 + root)


def _log_snr(time_value: float) -> float:
    if time_value == 0.0:
        return -math.inf
    return 2.0 * (math.log(time_value) - math.log1p(-time_value))


def _time_from_log_snr(log_snr: float) -> float:
    if log_snr == -math.inf:
        return 0.0
    log_odds = 0.5 * float(log_snr)
    if log_odds >= 0.0:
        return 1.0 / (1.0 + math.exp(-log_odds))
    exponential = math.exp(log_odds)
    return exponential / (1.0 + exponential)


def matched_information_horizon(
    time_value: float,
    *,
    clock: str,
    anchor_time: float,
    anchor_horizon: float,
    intervention_time: float,
) -> float:
    """Return a flow-time step matched at one common anchor.

    ``snr`` is the additive precision of the normalized Gaussian channel

        z_t / (1 - t) = t / (1 - t) * x + eps.

    ``log_snr`` instead advances by a fixed multiplicative noise scale.  Both
    are invariant to a relabeling of the underlying flow-time coordinate.
    """

    if clock not in INFORMATION_CLOCKS:
        raise ValueError(f"unknown information clock: {clock}")
    time_value = _validate_time(time_value, "time_value")
    anchor_time = _validate_time(anchor_time, "anchor_time")
    intervention_time = _validate_time(intervention_time, "intervention_time")
    anchor_horizon = float(anchor_horizon)
    if not math.isfinite(anchor_horizon) or anchor_horizon <= 0.0:
        raise ValueError("anchor_horizon must be positive and finite")
    if not anchor_time < anchor_time + anchor_horizon <= intervention_time:
        raise ValueError("the anchor query must end by intervention_time")
    if time_value >= intervention_time:
        return 0.0

    if clock == "raw_t":
        proposal = time_value + anchor_horizon
    elif clock == "snr":
        information_increment = _snr(anchor_time + anchor_horizon) - _snr(
            anchor_time
        )
        proposal = _time_from_snr(_snr(time_value) + information_increment)
    else:
        log_scale_increment = _log_snr(
            anchor_time + anchor_horizon
        ) - _log_snr(anchor_time)
        proposal = _time_from_log_snr(
            _log_snr(time_value) + log_scale_increment
        )

    return max(0.0, min(proposal, intervention_time) - time_value)
