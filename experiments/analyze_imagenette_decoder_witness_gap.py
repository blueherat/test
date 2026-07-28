"""Post-hoc audit of distribution mismatch before and after the frozen decoder.

This analysis does not alter the preregistered decoder-amplification gates.  It
uses grouped train/test splits so empirical/prior outputs sharing a pixel-noise
index cannot leak across the classifier boundary.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_imagenette_latent_prior_tradeoff import (  # noqa: E402
    condition_embeddings,
    load_run_config,
)
from experiments.imagenette_latent_prior_tradeoff import (  # noqa: E402
    IMAGENETTE_SYNSET_TO_IMAGENET_INDEX,
    INTERFACE_DIM,
    OrthogonalLatentInterface,
    ResNet18Evaluator,
    build_prior,
    fixed_orthogonal_basis,
    load_frozen_models,
    sample_prior_coordinates,
)
from experiments.mnist_spectral_rollout_toy import configure_fp32  # noqa: E402


def grouped_domain_split(
    real: np.ndarray,
    generated: np.ndarray,
    *,
    seed: int,
    test_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if real.shape != generated.shape or real.ndim != 2:
        raise ValueError("domain samples must have equal rank-two shapes")
    generator = np.random.default_rng(int(seed))
    order = generator.permutation(len(real))
    test_count = max(1, int(round(len(real) * float(test_fraction))))
    test_index = order[:test_count]
    train_index = order[test_count:]

    def combine(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        features = np.concatenate([real[indices], generated[indices]], axis=0)
        labels = np.concatenate(
            [
                np.zeros(len(indices), dtype=np.int64),
                np.ones(len(indices), dtype=np.int64),
            ]
        )
        return features, labels

    train_x, train_y = combine(train_index)
    test_x, test_y = combine(test_index)
    return train_x, train_y, test_x, test_y


def domain_classifier_metrics(
    real: torch.Tensor,
    generated: torch.Tensor,
    *,
    seed: int,
    prefix: str,
) -> dict[str, float]:
    train_x, train_y, test_x, test_y = grouped_domain_split(
        real.double().numpy(), generated.double().numpy(), seed=seed
    )
    classifiers = {
        "linear": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2_000, random_state=int(seed)),
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                alpha=1e-3,
                batch_size=128,
                learning_rate_init=1e-3,
                max_iter=300,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=int(seed),
            ),
        ),
    }
    result = {}
    for name, classifier in classifiers.items():
        classifier.fit(train_x, train_y)
        probability = classifier.predict_proba(test_x)[:, 1]
        prediction = probability >= 0.5
        result[f"{prefix}_{name}_accuracy"] = float(
            accuracy_score(test_y, prediction)
        )
        result[f"{prefix}_{name}_auc"] = float(roc_auc_score(test_y, probability))
    return result


def class_distribution_metrics(
    empirical_features: torch.Tensor,
    prior_features: torch.Tensor,
    evaluator: ResNet18Evaluator,
    class_to_idx: dict[str, int],
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    imagenet_indices = torch.empty(len(class_to_idx), dtype=torch.long)
    for synset, local_index in class_to_idx.items():
        imagenet_indices[local_index] = IMAGENETTE_SYNSET_TO_IMAGENET_INDEX[synset]
    weight = evaluator.classifier.weight.detach().cpu()[imagenet_indices]
    bias = evaluator.classifier.bias.detach().cpu()[imagenet_indices]
    empirical_logits = empirical_features @ weight.T + bias
    prior_logits = prior_features @ weight.T + bias
    empirical_prediction = empirical_logits.argmax(dim=1)
    prior_prediction = prior_logits.argmax(dim=1)
    empirical_histogram = torch.bincount(
        empirical_prediction, minlength=len(class_to_idx)
    ).float()
    prior_histogram = torch.bincount(
        prior_prediction, minlength=len(class_to_idx)
    ).float()
    empirical_histogram /= empirical_histogram.sum()
    prior_histogram /= prior_histogram.sum()
    empirical_probability = empirical_logits.softmax(dim=1).mean(dim=0)
    prior_probability = prior_logits.softmax(dim=1).mean(dim=0)
    return (
        {
            "output_argmax_class_tv": float(
                0.5 * (prior_histogram - empirical_histogram).abs().sum()
            ),
            "output_probability_class_tv": float(
                0.5 * (prior_probability - empirical_probability).abs().sum()
            ),
            "output_empirical_class_entropy": float(
                -(empirical_histogram * empirical_histogram.clamp_min(1e-12).log()).sum()
            ),
            "output_prior_class_entropy": float(
                -(prior_histogram * prior_histogram.clamp_min(1e-12).log()).sum()
            ),
        },
        empirical_prediction,
        prior_prediction,
    )


def condition_to_output_probe(
    empirical_condition: torch.Tensor,
    prior_condition: torch.Tensor,
    empirical_class: torch.Tensor,
    prior_class: torch.Tensor,
    *,
    seed: int,
) -> dict[str, float]:
    generator = np.random.default_rng(int(seed))
    order = generator.permutation(len(empirical_condition))
    test_count = int(round(0.25 * len(order)))
    test_index = order[:test_count]
    train_index = order[test_count:]
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2_000, random_state=int(seed)),
    )
    empirical_numpy = empirical_condition.double().numpy()
    empirical_labels = empirical_class.numpy()
    classifier.fit(empirical_numpy[train_index], empirical_labels[train_index])
    heldout_prediction = classifier.predict(empirical_numpy[test_index])
    heldout_accuracy = accuracy_score(
        empirical_labels[test_index], heldout_prediction
    )
    heldout_balanced = balanced_accuracy_score(
        empirical_labels[test_index], heldout_prediction
    )
    classifier.fit(empirical_numpy, empirical_labels)
    predicted_empirical = classifier.predict(empirical_numpy)
    predicted_prior = classifier.predict(prior_condition.double().numpy())
    class_count = int(max(empirical_class.max(), prior_class.max())) + 1

    def histogram(labels: np.ndarray | torch.Tensor) -> np.ndarray:
        values = np.asarray(labels, dtype=np.int64)
        counts = np.bincount(values, minlength=class_count).astype(np.float64)
        return counts / counts.sum()

    predicted_tv = 0.5 * np.abs(
        histogram(predicted_prior) - histogram(predicted_empirical)
    ).sum()
    actual_tv = 0.5 * np.abs(
        histogram(prior_class) - histogram(empirical_class)
    ).sum()
    return {
        "condition_to_output_linear_accuracy": float(heldout_accuracy),
        "condition_to_output_linear_balanced_accuracy": float(heldout_balanced),
        "condition_probe_predicted_class_tv": float(predicted_tv),
        "condition_probe_actual_class_tv": float(actual_tv),
    }


@torch.no_grad()
def analyze_run(run: Path, device_name: str, overwrite: bool = False) -> Path:
    output = run / "decoder_witness_gap_posthoc.json"
    if output.is_file() and not overwrite:
        print(f"decoder witness audit already complete: {output}", flush=True)
        return output
    config = load_run_config(run, device_name)
    configure_fp32(config.prior_seed)
    device = torch.device(device_name)
    _encoder, decoder, frozen = load_frozen_models(config, device)
    cache = torch.load(run / "latent_cache.pt", map_location="cpu", weights_only=True)
    prior_state = torch.load(run / "prior_state.pt", map_location="cpu", weights_only=True)
    prior = build_prior(config, device)
    prior.load_state_dict(prior_state["prior_ema"])
    prior.eval()
    interface = OrthogonalLatentInterface(
        config.latent_dim,
        fixed_orthogonal_basis(INTERFACE_DIM, config.basis_seed),
    ).to(device)
    count = int(config.quality_count)
    empirical_indices = torch.randint(
        len(cache["train_latent"]),
        (count,),
        generator=torch.Generator(device="cpu").manual_seed(config.prior_seed + 1_101),
    )
    empirical_latent = cache["train_latent"][empirical_indices]
    prior_latent = sample_prior_coordinates(
        prior,
        interface,
        count,
        config.prior_ode_steps,
        seed=config.prior_seed + 1_201,
        batch_size=config.prior_batch_size,
    )
    empirical_condition = condition_embeddings(
        decoder, empirical_latent, batch_size=config.eval_batch_size
    )
    prior_condition = condition_embeddings(
        decoder, prior_latent, batch_size=config.eval_batch_size
    )
    empirical_features = torch.load(
        run / "features_empirical.pt", map_location="cpu", weights_only=True
    )["generated_features"]
    prior_features = torch.load(
        run / "features_prior.pt", map_location="cpu", weights_only=True
    )["generated_features"]
    if not (
        len(empirical_condition)
        == len(prior_condition)
        == len(empirical_features)
        == len(prior_features)
        == count
    ):
        raise RuntimeError("witness audit sample counts differ")
    evaluator = ResNet18Evaluator().eval()
    class_metrics, empirical_class, prior_class = class_distribution_metrics(
        empirical_features, prior_features, evaluator, frozen["class_to_idx"]
    )
    summary = json.loads((run / "summary.json").read_text())
    payload: dict[str, float | int | str] = {
        "latent_dim": int(config.latent_dim),
        "frozen_seed": int(config.frozen_seed),
        "count": count,
        "modeling_gap": float(summary["modeling_gap"]),
        **domain_classifier_metrics(
            empirical_condition,
            prior_condition,
            seed=config.prior_seed + 5_101,
            prefix="condition_domain",
        ),
        **domain_classifier_metrics(
            empirical_features,
            prior_features,
            seed=config.prior_seed + 5_101,
            prefix="decoded_feature_domain",
        ),
        **class_metrics,
        **condition_to_output_probe(
            empirical_condition,
            prior_condition,
            empirical_class,
            prior_class,
            seed=config.prior_seed + 5_301,
        ),
    }
    payload["mlp_witness_auc_gain"] = float(
        payload["decoded_feature_domain_mlp_auc"]
        - payload["condition_domain_mlp_auc"]
    )
    if not all(
        math.isfinite(float(value))
        for value in payload.values()
        if isinstance(value, (int, float))
    ):
        raise FloatingPointError("non-finite decoder witness metric")
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    return analyze_run(args.run, args.device, args.overwrite)


if __name__ == "__main__":
    main()
