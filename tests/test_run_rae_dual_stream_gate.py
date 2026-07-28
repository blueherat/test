from __future__ import annotations

import json

from experiments.run_rae_dual_stream_gate import analyze_results, branch_name


def test_dual_stream_gate_accepts_two_consistent_seeds(tmp_path) -> None:
    for seed in (3407, 4211):
        for mode, paired_mse, shuffled_mse in (
            ("paired", 0.70, 0.85),
            ("shuffled", 0.90, 0.91),
        ):
            path = tmp_path / branch_name(seed, mode, 2000)
            path.mkdir()
            (path / "result.json").write_text(
                json.dumps(
                    {
                        "paired_context_normalized_mse": paired_mse,
                        "shuffled_context_normalized_mse": shuffled_mse,
                        "context_usage_gain": 1.0 - paired_mse / shuffled_mse,
                    }
                ),
                encoding="utf-8",
            )
    result = analyze_results(tmp_path)
    assert result["gate_pass"] is True
    assert result["decision"] == "proceed_to_full_dual_stream_generation"
