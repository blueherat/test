import json
from pathlib import Path
import sys

import pytest
import torch

RAE_ROOT = Path(__file__).resolve().parents[1] / "external" / "RAE"
RAE_SRC = RAE_ROOT / "src"
if str(RAE_SRC) not in sys.path:
    sys.path.insert(0, str(RAE_SRC))

from experiments.evaluate_rae_strict_lpl_generation import (
    RAE_ROOT as EVALUATOR_RAE_ROOT,
    assert_paired_sampling_audits,
    label_balance_metadata,
    load_sampling_audits,
    reject_partial_sampling_folder,
    validate_strict_sampling_protocol,
)
from experiments.evaluate_rae_spectral_generation import sample_folder_name
from sample_ddp import build_label_sampler


def test_strict_sampling_protocol_accepts_existing_contract() -> None:
    assert (EVALUATOR_RAE_ROOT / "src/sample_ddp.py").exists()
    validate_strict_sampling_protocol(4, 4)


def test_label_balance_metadata_records_exact_balance_after_tail_trim() -> None:
    assert label_balance_metadata(5000, 4, 4) == {
        "class_balance_exact": True,
        "samples_generated_before_trim": 5008,
    }


def test_interleaved_rank_label_pool_preserves_exact_trimmed_balance() -> None:
    labels_by_index = torch.empty(5008, dtype=torch.long)
    for rank in range(4):
        sampler = build_label_sampler(
            "equal",
            num_classes=1000,
            num_fid_samples=5000,
            total_samples=5008,
            samples_needed_this_device=1252,
            batch_size=4,
            device=torch.device("cpu"),
            rank=rank,
            iterations=313,
            seed=20260715,
        )
        for step in range(313):
            labels = sampler(step)
            local_start = step * 4
            for offset, label in enumerate(labels):
                labels_by_index[(local_start + offset) * 4 + rank] = label
    counts = torch.bincount(labels_by_index[:5000], minlength=1000)
    assert torch.equal(counts, torch.full((1000,), 5, dtype=torch.long))


def _sampling_audit(rank: int) -> dict[str, object]:
    return {
        "protocol": "interleaved-labels-v2",
        "rank": rank,
        "global_seed": 123,
        "initial_cuda_rng_state_sha256": f"initial-{rank}",
        "first_noise_sha256": f"noise-{rank}",
        "first_label_sha256": f"label-{rank}",
        "first_labels": [rank],
        "iterations": 313,
        "final_cuda_rng_state_sha256": f"final-{rank}",
    }


def test_sampling_audit_load_and_pair_check(tmp_path: Path) -> None:
    folders = []
    for branch in ("official", "flow", "lpl"):
        folder = tmp_path / branch
        folder.mkdir()
        folders.append(folder)
        for rank in range(4):
            (folder / f"sampling_audit_rank{rank}.json").write_text(
                json.dumps(_sampling_audit(rank)),
                encoding="utf-8",
            )
    audits = {folder.name: load_sampling_audits(folder, 4) for folder in folders}
    assert_paired_sampling_audits(audits)
    audits["lpl"][2]["first_noise_sha256"] = "different"
    with pytest.raises(ValueError, match="sampling audit mismatch"):
        assert_paired_sampling_audits(audits)


def test_sampling_seed_is_explicit_in_output_folder() -> None:
    assert (
        sample_folder_name(5000, 2000, 50, sampling_seed=123)
        == "fixed_seed123_5000_step2000_labels-interleaved-v3-provenance"
    )
    assert label_balance_metadata(50_000, 4, 4) == {
        "class_balance_exact": True,
        "samples_generated_before_trim": 50_000,
    }


@pytest.mark.parametrize("processes, batch", [(2, 8), (4, 1), (1, 16)])
def test_strict_sampling_protocol_rejects_different_noise_pairing(
    processes: int,
    batch: int,
) -> None:
    with pytest.raises(ValueError, match="strict paired evaluation"):
        validate_strict_sampling_protocol(processes, batch)


def test_strict_sampling_rejects_partial_legacy_resume(tmp_path: Path) -> None:
    folder = (
        tmp_path
        / "generation"
        / "fixed_seed20260715_5000_step2000_labels-interleaved-v3-provenance"
    )
    folder.mkdir(parents=True)
    (folder / "000000.png").write_bytes(b"partial")

    with pytest.raises(RuntimeError, match="partial PNG"):
        reject_partial_sampling_folder(
            tmp_path,
            endpoint=2000,
            sample_count=5000,
            steps=50,
        )


def test_strict_sampling_allows_completed_archive(tmp_path: Path) -> None:
    folder = (
        tmp_path
        / "generation"
        / "fixed_seed20260715_5000_step2000_labels-interleaved-v3-provenance"
    )
    folder.mkdir(parents=True)
    (folder / "000000.png").write_bytes(b"complete")
    folder.with_suffix(".npz").write_bytes(b"archive")

    reject_partial_sampling_folder(
        tmp_path,
        endpoint=2000,
        sample_count=5000,
        steps=50,
    )
