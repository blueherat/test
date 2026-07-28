"""Auditable RAEv2 continuation helpers.

RAEv2's ImageNet model predicts the clean latent directly.  This module keeps
that convention explicit so the older RAE velocity-prediction code cannot be
accidentally reused.
"""

from __future__ import annotations

import bisect
import hashlib
import io
import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


REQUIRED_STAGE2_KEYS = frozenset(
    {"step", "epoch", "model", "ema", "optimizer", "scheduler"}
)


def predicted_clean_latent(
    model_output: torch.Tensor,
    *,
    prediction: str,
    noisy_latent: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    """Convert a Stage-2 primary output to its predicted clean latent."""

    if prediction == "x":
        return model_output
    if prediction == "velocity":
        scale = time.reshape((time.shape[0],) + (1,) * (noisy_latent.ndim - 1))
        return noisy_latent - scale * model_output
    raise ValueError(f"unsupported transport prediction type: {prediction!r}")


def split_internal_guidance_output(
    model_output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return ``(full, base)`` for either ordinary or IG Stage-2 models."""

    if isinstance(model_output, tuple):
        if len(model_output) != 2:
            raise ValueError(f"expected a two-output IG model, got {len(model_output)} outputs")
        return model_output[0], model_output[1]
    return model_output, None


def official_flow_loss_map(
    transport: Any,
    model_output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    *,
    target_velocity: torch.Tensor,
    noisy_latent: torch.Tensor,
    time: torch.Tensor,
    base_model_coeff: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reproduce RAEv2's primary plus internal-guidance base loss exactly."""

    primary, base = split_internal_guidance_output(model_output)
    primary_map = transport.compute_loss(primary, target_velocity, noisy_latent, time)
    total = primary_map
    details = {"primary": primary_map}
    if base is not None:
        base_map = transport.compute_loss(base, target_velocity, noisy_latent, time)
        total = total + float(base_model_coeff) * base_map
        details["base"] = base_map
    return total, details


def validate_full_stage2_checkpoint(checkpoint: dict[str, Any]) -> None:
    """Reject model-only or otherwise incomplete RAEv2 source checkpoints."""

    missing = REQUIRED_STAGE2_KEYS.difference(checkpoint)
    if missing:
        raise ValueError(f"RAEv2 continuation checkpoint is missing keys: {sorted(missing)}")
    if not isinstance(checkpoint["model"], dict) or not isinstance(checkpoint["ema"], dict):
        raise TypeError("checkpoint model/ema entries must be state dictionaries")
    if not isinstance(checkpoint["optimizer"], dict):
        raise TypeError("checkpoint optimizer entry must be a state dictionary")
    if int(checkpoint["step"]) < 0 or int(checkpoint["epoch"]) < 0:
        raise ValueError("checkpoint step and epoch must be non-negative")


def synchronize_loaded_gmuon_param_groups(
    optimizer: Any,
    loaded_state: dict[str, Any],
) -> dict[str, Any]:
    """Repair and audit GMuon's public parameter-group aliases after loading.

    The GMuon version pinned by RAEv2 exposes ``param_groups`` through an
    internal ``_combined_param_groups`` list. PyTorch's generic
    ``Optimizer.load_state_dict`` replaces ``_muon_param_groups`` but does not
    refresh that alias. Without this repair, optimization uses the restored
    learning rate while logging, scheduling, and the next checkpoint can use
    the constructor learning rate instead.
    """

    muon = getattr(optimizer, "_muon", None)
    adamw = getattr(optimizer, "_adamw", None)
    if muon is None or adamw is None or set(loaded_state) != {"muon", "adamw"}:
        return {
            "composite_gmuon": False,
            "aliases_repaired": False,
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
        }

    current_muon_groups = getattr(muon, "_muon_param_groups", None)
    if current_muon_groups is None:
        raise RuntimeError("loaded GMuon has no _muon_param_groups")
    saved_muon_groups = loaded_state["muon"]["param_groups"]
    if len(current_muon_groups) != len(saved_muon_groups):
        raise RuntimeError(
            "loaded GMuon parameter-group count mismatch: "
            f"current={len(current_muon_groups)}, checkpoint={len(saved_muon_groups)}"
        )

    # Optimizer.load_state_dict writes ``param_groups`` through __dict__ during
    # __setstate__. GMuon's property masks that field, so its real internal
    # groups retain constructor hyperparameters. Rebuild those groups with the
    # current Parameter objects and the checkpoint's saved hyperparameters.
    restored_muon_groups = []
    for current, saved in zip(current_muon_groups, saved_muon_groups):
        restored = dict(saved)
        restored["params"] = current["params"]
        restored_muon_groups.append(restored)
    muon._muon_param_groups = restored_muon_groups
    muon.__dict__.pop("param_groups", None)

    combined_groups = list(restored_muon_groups)
    scalar_optimizer = getattr(muon, "scalar_optimizer", None)
    if scalar_optimizer is not None:
        combined_groups.extend(scalar_optimizer.param_groups)
    muon._combined_param_groups = combined_groups
    optimizer.param_groups = list(muon.param_groups) + list(adamw.param_groups)

    expected_lrs = [
        float(group["lr"])
        for name in ("muon", "adamw")
        for group in loaded_state[name]["param_groups"]
    ]
    public_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    internal_lrs = [
        float(group["lr"])
        for group in list(restored_muon_groups) + list(adamw.param_groups)
    ]
    resaved_state = optimizer.state_dict()
    resaved_lrs = [
        float(group["lr"])
        for name in ("muon", "adamw")
        for group in resaved_state[name]["param_groups"]
    ]
    expected_state_entries = [
        len(loaded_state["muon"]["state"]),
        len(loaded_state["adamw"]["state"]),
    ]
    actual_state_entries = [len(muon.state), len(adamw.state)]
    if (
        public_lrs != expected_lrs
        or internal_lrs != expected_lrs
        or resaved_lrs != expected_lrs
        or actual_state_entries != expected_state_entries
    ):
        raise RuntimeError(
            "optimizer restore mismatch after GMuon group repair: "
            f"checkpoint={expected_lrs}, public={public_lrs}, "
            f"internal={internal_lrs}, resaved={resaved_lrs}, "
            f"checkpoint_state_entries={expected_state_entries}, "
            f"actual_state_entries={actual_state_entries}"
        )
    return {
        "composite_gmuon": True,
        "aliases_repaired": True,
        "learning_rates": public_lrs,
        "resaved_learning_rates": resaved_lrs,
        "muon_state_entries": len(muon.state),
        "adamw_state_entries": len(adamw.state),
    }


def infer_source_steps_per_epoch(source_step: int, source_epoch: int) -> int:
    """Infer the scheduler's original epoch length from an epoch-boundary checkpoint."""

    if source_epoch <= 0 or source_step <= 0:
        raise ValueError("source step and epoch must both be positive")
    quotient, remainder = divmod(int(source_step), int(source_epoch))
    if remainder:
        raise ValueError(
            "source checkpoint is not on an epoch boundary; pass the original "
            "steps-per-epoch explicitly instead of inferring it"
        )
    return quotient


def branch_epoch(
    source_epoch: int,
    branch_update: int,
    source_steps_per_epoch: int,
) -> int:
    """Map branch updates onto the source scheduler's epoch convention."""

    if source_steps_per_epoch <= 0:
        raise ValueError("source_steps_per_epoch must be positive")
    return int(source_epoch) + int(branch_update) // int(source_steps_per_epoch)


def tensor_fingerprint(value: torch.Tensor) -> str:
    array = value.detach().to(device="cpu").contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def file_sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def center_crop_arr(image: Image.Image, image_size: int) -> Image.Image:
    """ADM center crop used by the official RAE/DiT ImageNet pipeline."""

    while min(*image.size) >= 2 * image_size:
        image = image.resize(tuple(x // 2 for x in image.size), resample=Image.Resampling.BOX)
    scale = image_size / min(*image.size)
    image = image.resize(
        tuple(round(x * scale) for x in image.size),
        resample=Image.Resampling.BICUBIC,
    )
    array = np.asarray(image)
    crop_y = (array.shape[0] - image_size) // 2
    crop_x = (array.shape[1] - image_size) // 2
    return Image.fromarray(
        array[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


class DeterministicImageNetParquet(Dataset):
    """ImageNet parquet reader with index-deterministic augmentation.

    Flow and LPL are run as separate jobs.  Deriving the horizontal flip from
    the source row index gives both jobs the exact same image stream without
    depending on worker RNG state.
    """

    def __init__(
        self,
        root: Path,
        *,
        split: str = "train",
        image_size: int = 256,
        augmentation_seed: int = 0,
        horizontal_flip: bool = True,
        row_group_cache_size: int = 4,
    ) -> None:
        self.root = Path(root).expanduser()
        self.data_dir = self.root / "data" if (self.root / "data").exists() else self.root
        split_alias = {"val": "validation", "valid": "validation"}
        self.split = split_alias.get(str(split).lower(), str(split).lower())
        self.image_size = int(image_size)
        self.augmentation_seed = int(augmentation_seed)
        self.horizontal_flip = bool(horizontal_flip)
        self.files = sorted(self.data_dir.glob(f"{self.split}-*.parquet"))
        if not self.files:
            raise FileNotFoundError(
                f"no {self.split} parquet shards found under {self.data_dir}"
            )

        import pyarrow.parquet as pq

        self._pq = pq
        self._offsets = [0]
        for path in self.files:
            self._offsets.append(
                self._offsets[-1] + int(pq.ParquetFile(path).metadata.num_rows)
            )
        self._pf_cache: dict[Path, Any] = {}
        self._row_group_offsets: dict[int, list[int]] = {}
        self._row_group_cache: OrderedDict[tuple[int, int], Any] = OrderedDict()
        self._row_group_cache_size = int(row_group_cache_size)

    def __len__(self) -> int:
        return self._offsets[-1]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_pq"] = None
        state["_pf_cache"] = {}
        state["_row_group_offsets"] = {}
        state["_row_group_cache"] = OrderedDict()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        import pyarrow.parquet as pq

        self._pq = pq

    def _parquet_file(self, file_index: int):
        path = self.files[file_index]
        if path not in self._pf_cache:
            self._pf_cache[path] = self._pq.ParquetFile(path)
        return self._pf_cache[path]

    def _row_group_for(self, file_index: int, local_index: int) -> tuple[int, int]:
        if file_index not in self._row_group_offsets:
            parquet_file = self._parquet_file(file_index)
            offsets = [0]
            for row_group in range(parquet_file.num_row_groups):
                offsets.append(
                    offsets[-1] + parquet_file.metadata.row_group(row_group).num_rows
                )
            self._row_group_offsets[file_index] = offsets
        offsets = self._row_group_offsets[file_index]
        row_group = bisect.bisect_right(offsets, local_index) - 1
        return row_group, local_index - offsets[row_group]

    def _row_group_table(self, file_index: int, row_group: int):
        key = (int(file_index), int(row_group))
        if key not in self._row_group_cache:
            self._row_group_cache[key] = self._parquet_file(file_index).read_row_group(
                row_group, columns=["image", "label"]
            )
            while len(self._row_group_cache) > self._row_group_cache_size:
                self._row_group_cache.popitem(last=False)
        self._row_group_cache.move_to_end(key)
        return self._row_group_cache[key]

    def _should_flip(self, index: int) -> bool:
        if not self.horizontal_flip:
            return False
        rng = random.Random((self.augmentation_seed << 32) ^ int(index))
        return rng.random() < 0.5

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        file_index = bisect.bisect_right(self._offsets, index) - 1
        local_index = index - self._offsets[file_index]
        row_group, row_in_group = self._row_group_for(file_index, local_index)
        row = (
            self._row_group_table(file_index, row_group)
            .slice(row_in_group, 1)
            .to_pydict()
        )
        image_info = row["image"][0]
        image_bytes = image_info.get("bytes")
        if image_bytes is not None:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            image_path = image_info.get("path")
            if image_path is None:
                raise ValueError(f"parquet row {index} has neither image bytes nor path")
            image = Image.open(self.data_dir / image_path).convert("RGB")

        image = center_crop_arr(image, self.image_size)
        if self._should_flip(index):
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        array = np.asarray(image, dtype=np.float32)
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).div_(255.0)
        return tensor, int(row["label"][0]), index
