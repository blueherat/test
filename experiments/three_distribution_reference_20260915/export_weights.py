"""Export endpoint weights from out-of-fold three-source classifier logits.

This is an offline interface, not a classifier trainer or a job launcher.
The input NPZ contains logits [N,3], endpoint_ids [N], labels [N], and
excluded_fold [N]. Metadata records source_order, source_priors, split,
input_law, critic_observation, and weak_snapshot {path, sha256, step}.
For two-fold cross-fitting, excluded_fold must equal endpoint_ids % 2.

critic_observation is 'endpoint' or 'endpoint_features'. The latter is an
explicit approximation and is not presented as exact endpoint density EM.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .objectives import (
    SOURCE_ORDER,
    endpoint_responsibility,
    normalize_endpoint_weights,
    reference_fraction,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(input_path, metadata_path, output_path, guidance_anchor):
    input_path, metadata_path, output_path = map(Path, (input_path, metadata_path, output_path))
    receipt_path = output_path.with_suffix(".json")
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("use a new round-specific output; existing weights are immutable")
    metadata = json.loads(metadata_path.read_text())
    if tuple(metadata["source_order"]) != SOURCE_ORDER:
        raise ValueError("source ordering must be real, strong, weak_snapshot")
    if metadata["split"] != "train" or metadata["input_law"] != "strong_endpoints":
        raise ValueError("normalization requires the complete strong TRAINING endpoint bank")
    observation = metadata["critic_observation"]
    if observation not in ("endpoint", "endpoint_features"):
        raise ValueError("noisy states and sampler trajectory states are not endpoint posteriors")
    snapshot = metadata["weak_snapshot"]
    if sha256(snapshot["path"]) != snapshot["sha256"]:
        raise ValueError("weak checkpoint no longer matches the classifier's snapshot")
    if not isinstance(snapshot["step"], int) or snapshot["step"] < 0:
        raise ValueError("record the actual weak snapshot training step")
    with np.load(input_path, allow_pickle=False) as value:
        logits, ids, labels, fold = [value[key] for key in
                                     ("logits", "endpoint_ids", "labels", "excluded_fold")]
    count = len(logits)
    if count != metadata["complete_training_bank_count"]:
        raise ValueError("input does not contain the complete training endpoint bank")
    if ids.shape != (count,) or not np.array_equal(ids, np.arange(count)):
        raise ValueError("endpoint IDs must be complete and ordered from zero")
    if fold.shape != (count,) or not np.array_equal(fold, ids % 2):
        raise ValueError("each endpoint must be evaluated by a classifier excluding its fold")
    if labels.shape != (count,):
        raise ValueError("one conditional label is required per endpoint")
    kappa = reference_fraction(guidance_anchor)
    raw = endpoint_responsibility(torch.as_tensor(logits), kappa, metadata["source_priors"]).cpu().numpy()
    weights, normalizers = normalize_endpoint_weights(raw, labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, weights=weights.astype(np.float32), raw=raw,
                 endpoint_ids=ids, labels=labels)
    temporary.replace(output_path)
    report = dict(
        objective="KL(G || (1-kappa) P + kappa Q_snapshot), endpoint EM surrogate",
        guidance_anchor=guidance_anchor,
        kappa=kappa,
        source_order=SOURCE_ORDER,
        source_priors=metadata["source_priors"],
        weak_snapshot=snapshot,
        critic_observation=observation,
        exact_endpoint_posterior_assumed=observation == "endpoint",
        finite_classifier_is_approximate=True,
        endpoint_count=count,
        class_normalizers=normalizers,
        raw_min=float(raw.min()), raw_max=float(raw.max()),
        normalized_max=float(weights.max()),
        input_sha256=sha256(input_path), metadata_sha256=sha256(metadata_path),
        weights_sha256=sha256(output_path),
        classifier_needed_at_guided_inference=False,
        generated_image_evaluation_performed=False,
    )
    temporary = receipt_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(receipt_path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--guidance-anchor", required=True, type=float)
    args = parser.parse_args()
    report = export(args.input, args.metadata, args.output, args.guidance_anchor)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
