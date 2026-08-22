"""Joint cumulative readouts on a frozen ImageNet-100 SiT backbone."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

try:
    from experiments.imagenet100_sit_internal_v_head import (
        create_internal_velocity_head,
        embed_sit_inputs,
        freeze_source_model,
        unpatchify_channels,
    )
except ModuleNotFoundError:
    from imagenet100_sit_internal_v_head import (
        create_internal_velocity_head,
        embed_sit_inputs,
        freeze_source_model,
        unpatchify_channels,
    )


DEFAULT_DEPTHS = (4, 6, 8, 10, 12)


def validate_depths(source: nn.Module, depths: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(depth) for depth in depths)
    if not result or tuple(sorted(set(result))) != result:
        raise ValueError("depths must be unique and increasing")
    if result[0] < 1 or result[-1] > len(source.blocks):
        raise ValueError("depths lie outside the source backbone")
    return result


class FrozenMultiDepthPrefix(nn.Module):
    """Return paired hidden states while keeping the source strictly frozen."""

    def __init__(self, source: nn.Module, depths: Sequence[int]) -> None:
        super().__init__()
        self.source = freeze_source_model(source)
        self.depths = validate_depths(source, depths)

    def train(self, mode: bool = True):
        super().train(False)
        self.source.eval()
        return self

    def forward(
        self,
        state: torch.Tensor,
        time_value: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        tokens, conditioning = embed_sit_inputs(
            self.source, state, time_value, labels
        )
        features: list[torch.Tensor] = []
        requested = set(self.depths)
        for depth, block in enumerate(self.source.blocks, start=1):
            tokens = block(tokens, conditioning)
            if depth in requested:
                features.append(tokens)
            if depth >= self.depths[-1]:
                break
        if len(features) != len(self.depths):
            raise RuntimeError("not all requested hidden states were produced")
        return tuple(features), conditioning


class CumulativeReadoutStack(nn.Module):
    """Predict a base field followed by additive depth-wise innovations."""

    def __init__(
        self,
        heads: nn.ModuleDict,
        *,
        depths: Sequence[int],
        source: nn.Module,
        latent_channels: int,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.depths = tuple(int(depth) for depth in depths)
        self.latent_channels = int(latent_channels)
        self.patch_size = int(source.x_embedder.patch_size[0])
        if tuple(heads) != tuple(f"d{depth}" for depth in self.depths):
            raise ValueError("head names do not match cumulative depths")

    def _unpatchify(self, projected: torch.Tensor) -> torch.Tensor:
        if projected.ndim != 3:
            raise ValueError("projected readout must have shape [B,T,F]")
        side = int(projected.shape[1] ** 0.5)
        if side * side != projected.shape[1]:
            raise ValueError("token count must form a square grid")
        expected = self.patch_size**2 * self.latent_channels
        if projected.shape[-1] != expected:
            raise ValueError("readout emitted an unexpected feature dimension")
        values = projected.reshape(
            len(projected),
            side,
            side,
            self.patch_size,
            self.patch_size,
            self.latent_channels,
        )
        values = torch.einsum("nhwpqc->nchpwq", values)
        return values.reshape(
            len(projected),
            self.latent_channels,
            side * self.patch_size,
            side * self.patch_size,
        )

    def forward(
        self,
        features: tuple[torch.Tensor, ...],
        conditioning: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        if len(features) != len(self.depths):
            raise ValueError("feature count does not match cumulative depths")
        cumulative: torch.Tensor | None = None
        outputs: list[torch.Tensor] = []
        innovations: list[torch.Tensor] = []
        for depth, hidden in zip(self.depths, features, strict=True):
            projected = self.heads[f"d{depth}"](hidden, conditioning)
            innovation = self._unpatchify(projected)
            cumulative = innovation if cumulative is None else cumulative + innovation
            innovations.append(innovation)
            outputs.append(cumulative)
        return tuple(outputs), tuple(innovations)


def create_joint_cumulative_parts(
    sit_module,
    source: nn.Module,
    *,
    depths: Sequence[int] = DEFAULT_DEPTHS,
    latent_channels: int = 4,
) -> tuple[FrozenMultiDepthPrefix, CumulativeReadoutStack]:
    checked_depths = validate_depths(source, depths)
    heads = nn.ModuleDict(
        {
            f"d{depth}": create_internal_velocity_head(
                sit_module, source, latent_channels=latent_channels
            )
            for depth in checked_depths
        }
    )
    prefix = FrozenMultiDepthPrefix(source, checked_depths)
    readouts = CumulativeReadoutStack(
        heads,
        depths=checked_depths,
        source=source,
        latent_channels=latent_channels,
    )
    return prefix, readouts


def source_velocity_from_final_features(
    source: nn.Module,
    final_features: torch.Tensor,
    conditioning: torch.Tensor,
    *,
    latent_channels: int,
) -> torch.Tensor:
    projected = source.final_layer(final_features, conditioning)
    output = source.unpatchify(projected)
    return output[:, :latent_channels]


def sequence_losses(
    outputs: tuple[torch.Tensor, ...],
    target: torch.Tensor,
    *,
    monotonic_weight: float,
    contraction_ratio: float,
) -> dict[str, torch.Tensor]:
    """Joint deep supervision plus an optional per-sample contraction penalty."""

    if not outputs or any(output.shape != target.shape for output in outputs):
        raise ValueError("all cumulative outputs must match the target")
    if monotonic_weight < 0:
        raise ValueError("monotonic_weight must be non-negative")
    if not 0 < contraction_ratio <= 1:
        raise ValueError("contraction_ratio must lie in (0,1]")
    per_sample = torch.stack(
        [
            (output.float() - target.float()).square().flatten(1).mean(1)
            for output in outputs
        ],
        dim=1,
    )
    supervised = per_sample.mean()
    if len(outputs) == 1:
        monotonic = supervised.new_zeros(())
        strict_fraction = supervised.new_ones(())
    else:
        allowed = contraction_ratio**2 * per_sample[:, :-1]
        monotonic = torch.relu(per_sample[:, 1:] - allowed).mean()
        strict_fraction = (per_sample[:, 1:] < per_sample[:, :-1]).all(dim=1).float().mean()
    return {
        "optimized": supervised + float(monotonic_weight) * monotonic,
        "supervised": supervised,
        "monotonic": monotonic,
        "strict_monotonic_fraction": strict_fraction,
        "per_sample_mse": per_sample,
    }


def direct_head_outputs(
    readouts: CumulativeReadoutStack,
    features: tuple[torch.Tensor, ...],
    conditioning: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Return non-cumulative head outputs for architecture diagnostics."""

    values = []
    for depth, hidden in zip(readouts.depths, features, strict=True):
        projected = readouts.heads[f"d{depth}"](hidden, conditioning)
        values.append(readouts._unpatchify(projected))
    return tuple(values)


def select_joint_cumulative_field(
    strong: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
    innovations: tuple[torch.Tensor, ...],
    *,
    mode: str,
    gamma: float,
    stage_index: int | None = None,
) -> torch.Tensor:
    """Select the source, final cumulative, or last-innovation extrapolated field."""

    if not outputs or len(outputs) != len(innovations):
        raise ValueError("outputs and innovations must be non-empty and aligned")
    final = outputs[-1]
    if strong.shape != final.shape or any(value.shape != final.shape for value in innovations):
        raise ValueError("all candidate fields must have the same shape")
    if mode == "strong":
        return strong
    if mode == "stage":
        if stage_index is None or not 0 <= stage_index < len(outputs):
            raise ValueError("stage mode requires a valid stage_index")
        return outputs[stage_index]
    if mode == "final":
        return final
    if mode == "last_extrapolation":
        if gamma < 0:
            raise ValueError("last-innovation extrapolation requires gamma >= 0")
        return final + float(gamma) * innovations[-1]
    raise ValueError(f"unsupported joint cumulative field mode: {mode!r}")
