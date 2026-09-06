"""Reject histories that cannot implement the recorded first-success sampler."""

import copy
import math

import pytest

from experiments.merge_raev2_accept_d_guidance import validate_history


def proposal(class_id, attempt, batch, slot, probability, uniform):
    return {
        "class_id": class_id,
        "label": class_id,
        "attempt": attempt,
        "proposal_batch": batch,
        "slot": slot,
        "D": probability,
        "log_D": math.log(probability),
        "uniform": uniform,
        "accepted": uniform < probability,
        "noise_source": "initial_bank" if attempt == 1 else "retry_generator",
    }


@pytest.fixture
def history():
    return [
        proposal(0, 1, 0, 0, 0.4, 0.8),
        proposal(1, 1, 0, 1, 0.7, 0.2),
        proposal(0, 2, 1, 0, 0.4, 0.2),
    ]


def test_first_proposal_and_first_success_are_distinct_when_rejected(history):
    result = validate_history(history, [0, 1])
    assert result[0][0] is history[0]
    assert result[0][-1] is history[2]
    assert result[1][0] is result[1][-1]
    assert {class_id: len(trials) for class_id, trials in result.items()} == {0: 2, 1: 1}


@pytest.mark.parametrize("attempt", [0, 1, 3])
def test_reject_nonconsecutive_attempts(history, attempt):
    history[-1]["attempt"] = attempt
    with pytest.raises(ValueError, match="nonconsecutive attempts"):
        validate_history(history, [0, 1])


def test_reject_out_of_order_attempts(history):
    with pytest.raises(ValueError, match="nonconsecutive attempts"):
        validate_history([history[2], history[1], history[0]], [0, 1])


def test_reject_continuation_after_an_earlier_acceptance(history):
    # Keep each Bernoulli event valid; the failure must be the stopping rule.
    history[0]["uniform"] = 0.1
    history[0]["accepted"] = True
    with pytest.raises(ValueError, match="not first acceptance"):
        validate_history(history, [0, 1])


def test_reject_unfinished_class(history):
    history[-1]["uniform"] = 0.9
    history[-1]["accepted"] = False
    with pytest.raises(ValueError, match="not first acceptance"):
        validate_history(history, [0, 1])


@pytest.mark.parametrize("position", [0, 1, 2])
def test_reject_tampered_bernoulli_decision(history, position):
    history[position]["accepted"] = not history[position]["accepted"]
    with pytest.raises(ValueError, match="Bernoulli decision mismatch"):
        validate_history(history, [0, 1])


def test_equal_uniform_and_probability_is_rejection(history):
    history[0]["uniform"] = history[0]["D"]
    assert not history[0]["accepted"]
    assert len(validate_history(history, [0, 1])[0]) == 2
    history[0]["accepted"] = True
    with pytest.raises(ValueError, match="Bernoulli decision mismatch"):
        validate_history(history, [0, 1])


@pytest.mark.parametrize("class_id", [0, 1])
def test_reject_missing_class(history, class_id):
    records = [record for record in history if record["class_id"] != class_id]
    with pytest.raises(ValueError, match="incomplete class coverage"):
        validate_history(records, [0, 1])


def test_reject_empty_cohort_history():
    with pytest.raises(ValueError, match="incomplete class coverage"):
        validate_history([], list(range(1000)))


def test_reject_duplicate_batch_slot(history):
    records = [history[0], copy.deepcopy(history[0]), *history[1:]]
    with pytest.raises(ValueError, match="duplicate proposal batch slot"):
        validate_history(records, [0, 1])


def test_reject_same_class_twice_in_one_proposal_batch(history):
    # Slots and attempts are distinct, but one class cannot be live twice.
    history[-1]["proposal_batch"] = 0
    history[-1]["slot"] = 2
    with pytest.raises(ValueError, match="duplicate class in proposal batch"):
        validate_history(history, [0, 1])


@pytest.mark.parametrize("position,source", [(0, "retry_generator"), (2, "initial_bank")])
def test_reject_wrong_initial_or_retry_noise_source(history, position, source):
    history[position]["noise_source"] = source
    with pytest.raises(ValueError, match="paired noise bank|retry source mismatch"):
        validate_history(history, [0, 1])


@pytest.mark.parametrize("key,value,error", [
    ("D", 0.0, "invalid probability/uniform"),
    ("D", 1.1, "invalid probability/uniform"),
    ("uniform", -0.1, "invalid probability/uniform"),
    ("uniform", 1.0, "invalid probability/uniform"),
    ("uniform", float("nan"), "nonfinite acceptance evidence"),
    ("log_D", float("inf"), "nonfinite acceptance evidence"),
    ("log_D", -0.1, "D/log_D mismatch"),
    ("label", 9, "unexpected class/label"),
    ("class_id", 9, "unexpected class/label"),
    ("slot", 8, "invalid proposal batch slot"),
    ("proposal_batch", -1, "invalid proposal batch slot"),
])
def test_reject_invalid_proposal_evidence(history, key, value, error):
    history[0][key] = value
    with pytest.raises(ValueError, match=error):
        validate_history(history, [0, 1])
