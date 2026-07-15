from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.dinov2_token_diagnostics import (
    configure_fp32,
    load_baseline_adapter,
    load_named_dataset,
    pick_dataset_images,
    split_indices,
)


EPS = 1e-12
BLUE = "#2563EB"
ORANGE = "#EA580C"
GOLD = "#CA8A04"
OLIVE = "#657A30"
PINK = "#BE185D"
INK = "#1F2937"
MID_GREY = "#6B7280"
LIGHT_GREY = "#D1D5DB"
PALETTE = (BLUE, ORANGE, GOLD, OLIVE, PINK, MID_GREY)
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GaugeSpec:
    name: str
    kind: str = "identity"
    strength: float = 0.0
    radius: int = 1
    shift_y: int = 0
    shift_x: int = 0
    seed: int = 0


@dataclass
class CodecDataConfig:
    dataset_name: str = "imagenet_parquet"
    data_root: str = "/data/shared"
    dataset_path: str = "/data/shared/imagenet-1k"
    train_split: str = "train"
    val_split: str = "validation"
    train_count: int = 32
    val_count: int = 16
    image_size: int = 256
    model_key: str = "rae_dinov2"
    rae_repo_path: str = "external/RAE"
    device: str = "cuda:0"
    encode_batch_size: int = 8
    seed: int = 0
    posterior: str = "mode"


@dataclass
class CodecDataBundle:
    config: CodecDataConfig
    adapter: object
    train_images: torch.Tensor
    val_images: torch.Tensor
    train_latents: torch.Tensor
    val_latents: torch.Tensor
    latent_scale: float
    train_indices: List[int]
    val_indices: List[int]

    @property
    def device(self) -> torch.device:
        return torch.device(self.config.device if torch.cuda.is_available() else "cpu")


class OrthogonalGauge:
    """Low-capacity, exactly invertible orthogonal map on BCHW latents."""

    def __init__(self, spec: GaugeSpec):
        self.spec = spec

    def __repr__(self) -> str:
        return f"OrthogonalGauge({self.spec})"

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self._apply(z, inverse=False)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return self._apply(y, inverse=True)

    def _apply(self, x: torch.Tensor, inverse: bool) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"gauge expects BCHW, got shape={tuple(x.shape)}")
        kind = self.spec.kind.strip().lower()
        if kind == "identity":
            return x
        if kind in {"roll", "translation"}:
            sign = -1 if inverse else 1
            return torch.roll(
                x,
                shifts=(sign * self.spec.shift_y, sign * self.spec.shift_x),
                dims=(-2, -1),
            )
        if kind in {"channel", "channel_givens", "givens"}:
            angle = -self.spec.strength if inverse else self.spec.strength
            return self._channel_givens(x, angle)
        if kind in {"allpass", "fourier_allpass", "phase"}:
            strength = -self.spec.strength if inverse else self.spec.strength
            return self._fourier_allpass(x, strength, self.spec.radius)
        if kind in {"haar", "block_haar"}:
            return self._haar_2x2(x, inverse=inverse)
        raise ValueError(f"unknown gauge kind: {self.spec.kind}")

    def _channel_givens(self, x: torch.Tensor, angle: float) -> torch.Tensor:
        channels = x.shape[1]
        if channels < 2:
            return x
        generator = torch.Generator(device="cpu").manual_seed(int(self.spec.seed))
        order = torch.randperm(channels, generator=generator)
        usable = (channels // 2) * 2
        first = order[:usable:2].to(x.device)
        second = order[1:usable:2].to(x.device)
        cosine = math.cos(float(angle))
        sine = math.sin(float(angle))
        out = x.clone()
        left = x[:, first]
        right = x[:, second]
        out[:, first] = cosine * left - sine * right
        out[:, second] = sine * left + cosine * right
        return out

    @staticmethod
    def _fourier_allpass(x: torch.Tensor, strength: float, radius: int) -> torch.Tensor:
        height, width = x.shape[-2:]
        radius = max(1, int(radius))
        fy = torch.fft.fftfreq(height, device=x.device, dtype=x.dtype).view(height, 1)
        fx = torch.fft.fftfreq(width, device=x.device, dtype=x.dtype).view(1, width)
        phase = float(strength) * (
            torch.sin(2.0 * math.pi * radius * fx)
            + 0.80 * torch.sin(2.0 * math.pi * radius * fy)
            + 0.35 * torch.sin(2.0 * math.pi * radius * (fx + fy))
        )
        multiplier = torch.polar(torch.ones_like(phase), phase)
        spectrum = torch.fft.fft2(x, norm="ortho")
        return torch.fft.ifft2(spectrum * multiplier, norm="ortho").real.to(x.dtype)

    @staticmethod
    def _haar_2x2(x: torch.Tensor, inverse: bool) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if height % 2 or width % 2:
            raise ValueError(f"2x2 Haar gauge requires even H/W, got {(height, width)}")
        blocks = (
            x.reshape(batch, channels, height // 2, 2, width // 2, 2)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(batch, channels, height // 2, width // 2, 4)
        )
        matrix = x.new_tensor(
            [
                [0.5, 0.5, 0.5, 0.5],
                [0.5, -0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5, 0.5],
            ]
        )
        matrix = matrix.transpose(0, 1) if inverse else matrix
        mixed = torch.einsum("bchwk,lk->bchwl", blocks, matrix)
        return (
            mixed.reshape(batch, channels, height // 2, width // 2, 2, 2)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(batch, channels, height, width)
        )


def make_gauge(spec: GaugeSpec | Mapping[str, object]) -> OrthogonalGauge:
    return OrthogonalGauge(spec if isinstance(spec, GaugeSpec) else GaugeSpec(**dict(spec)))


def default_gauges() -> List[GaugeSpec]:
    return [
        GaugeSpec("identity"),
        GaugeSpec("roll_x2", kind="roll", shift_x=2),
        GaugeSpec("channel_0.5", kind="channel_givens", strength=0.5, seed=0),
        GaugeSpec("allpass_r1", kind="fourier_allpass", strength=0.65, radius=1),
        GaugeSpec("allpass_r3", kind="fourier_allpass", strength=0.65, radius=3),
        GaugeSpec("haar_2x2", kind="block_haar"),
    ]


def configure_reproducibility(seed: int = 0) -> None:
    configure_fp32()
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _encode_batches(adapter, images: torch.Tensor, batch_size: int) -> torch.Tensor:
    latents = []
    for start in range(0, len(images), batch_size):
        latents.append(adapter.encode(images[start : start + batch_size]).float().cpu())
    return torch.cat(latents, dim=0)


def prepare_codec_data(config: CodecDataConfig) -> CodecDataBundle:
    configure_reproducibility(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    train_dataset = load_named_dataset(
        config.dataset_name,
        config.data_root,
        split=config.train_split,
        dataset_path=config.dataset_path,
    )
    same_split = config.train_split == config.val_split
    if same_split:
        chosen = split_indices(len(train_dataset), config.train_count + config.val_count, config.seed)
        train_indices = chosen[: config.train_count]
        val_indices = chosen[config.train_count :]
        val_dataset = train_dataset
    else:
        val_dataset = load_named_dataset(
            config.dataset_name,
            config.data_root,
            split=config.val_split,
            dataset_path=config.dataset_path,
        )
        train_indices = split_indices(len(train_dataset), config.train_count, config.seed)
        val_indices = split_indices(len(val_dataset), config.val_count, config.seed + 10_000)
    train_images, _ = pick_dataset_images(train_dataset, indices=train_indices, image_size=config.image_size)
    val_images, _ = pick_dataset_images(val_dataset, indices=val_indices, image_size=config.image_size)
    rae_repo_path = Path(config.rae_repo_path).expanduser()
    if not rae_repo_path.is_absolute():
        rae_repo_path = ROOT / rae_repo_path
    adapter = load_baseline_adapter(
        config.model_key,
        device=device,
        rae_repo_path=rae_repo_path,
        posterior=config.posterior,
    )
    train_latents = _encode_batches(adapter, train_images, config.encode_batch_size)
    val_latents = _encode_batches(adapter, val_images, config.encode_batch_size)
    latent_scale = float(train_latents.square().mean().sqrt().clamp_min(1e-8))
    return CodecDataBundle(
        config=config,
        adapter=adapter,
        train_images=train_images,
        val_images=val_images,
        train_latents=train_latents,
        val_latents=val_latents,
        latent_scale=latent_scale,
        train_indices=train_indices,
        val_indices=val_indices,
    )


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = (prediction - target).flatten(1).square().sum(dim=1)
    denominator = target.flatten(1).square().sum(dim=1).clamp_min(EPS)
    return (numerator / denominator).sqrt()


def _channel_gram_eigenvalues(z: torch.Tensor) -> torch.Tensor:
    matrix = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1]).double()
    gram = matrix.transpose(0, 1) @ matrix / max(1, matrix.shape[0])
    return torch.linalg.eigvalsh(gram).clamp_min(0.0)


@torch.no_grad()
def exact_equivalence_table(
    z: torch.Tensor,
    specs: Sequence[GaugeSpec],
    *,
    device: str | torch.device = "cuda:0",
    seed: int = 0,
) -> pd.DataFrame:
    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    z = z.to(device=device, dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    epsilon = torch.randn(z.shape, generator=generator, dtype=torch.float32).to(device)
    t = torch.rand((len(z),), generator=generator, dtype=torch.float32).to(device)
    alpha = torch.cos(0.5 * math.pi * t).view(-1, 1, 1, 1)
    sigma = torch.sin(0.5 * math.pi * t).view(-1, 1, 1, 1)
    noisy = alpha * z + sigma * epsilon
    base_norm = z.flatten(1).norm(dim=1)
    base_dist = torch.pdist(z.flatten(1)) if len(z) > 1 else z.new_zeros(1)
    base_power = torch.fft.fft2(z, norm="ortho").abs().square().sum(dim=(0, 1))
    base_eigs = _channel_gram_eigenvalues(z)
    rows = []
    for spec in specs:
        gauge = make_gauge(spec)
        y = gauge.forward(z)
        z_roundtrip = gauge.inverse(y)
        transformed_noisy = gauge.forward(noisy)
        paired_noisy = alpha * y + sigma * gauge.forward(epsilon)
        power = torch.fft.fft2(y, norm="ortho").abs().square().sum(dim=(0, 1))
        eigs = _channel_gram_eigenvalues(y)
        distance = torch.pdist(y.flatten(1)) if len(y) > 1 else y.new_zeros(1)
        rows.append(
            {
                **asdict(spec),
                "inverse_rel_l2": float(relative_l2(z_roundtrip, z).mean()),
                "norm_rel_error": float(
                    ((y.flatten(1).norm(dim=1) - base_norm).abs() / base_norm.clamp_min(EPS)).mean()
                ),
                "pairwise_distance_rel_error": float(
                    (distance - base_dist).norm() / base_dist.norm().clamp_min(EPS)
                ),
                "paired_noise_rel_error": float(relative_l2(transformed_noisy, paired_noisy).mean()),
                "total_psd_rel_error": float((power - base_power).norm() / base_power.norm().clamp_min(EPS)),
                "channel_gram_spectrum_rel_error": float(
                    (eigs - base_eigs).norm() / base_eigs.norm().clamp_min(EPS)
                ),
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def reconstruction_equivalence_table(
    bundle: CodecDataBundle,
    specs: Sequence[GaugeSpec],
    *,
    count: int = 4,
) -> pd.DataFrame:
    z = bundle.val_latents[:count].to(bundle.device)
    reference = bundle.adapter.decode(z)
    rows = []
    for spec in specs:
        gauge = make_gauge(spec)
        roundtrip = bundle.adapter.decode(gauge.inverse(gauge.forward(z)))
        rows.append(
            {
                "gauge": spec.name,
                "image_max_abs_error": float((roundtrip - reference).abs().max()),
                "image_mean_abs_error": float((roundtrip - reference).abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def _to_image(x: torch.Tensor) -> np.ndarray:
    return ((x.detach().float().cpu().clamp(-1, 1) + 1) * 0.5).permute(1, 2, 0).numpy()


@torch.no_grad()
def visualize_gauge_roundtrip(
    bundle: CodecDataBundle,
    spec: GaugeSpec,
    *,
    count: int = 3,
    show_naive_decode: bool = True,
) -> plt.Figure:
    x = bundle.val_images[:count]
    z = bundle.val_latents[:count].to(bundle.device)
    gauge = make_gauge(spec)
    reconstruction = bundle.adapter.decode(z).cpu()
    equivalent = bundle.adapter.decode(gauge.inverse(gauge.forward(z))).cpu()
    naive = bundle.adapter.decode(gauge.forward(z)).cpu() if show_naive_decode else None
    columns = 4 if show_naive_decode else 3
    fig, axes = plt.subplots(count, columns, figsize=(3.5 * columns, 3.2 * count), squeeze=False)
    titles = ["input x", "D(z)", "D(A^-1 A z)"]
    if show_naive_decode:
        titles.append("D(A z), not the method")
    for row in range(count):
        images = [x[row], reconstruction[row], equivalent[row]]
        if show_naive_decode:
            images.append(naive[row])
        for column, image in enumerate(images):
            axes[row, column].imshow(_to_image(image))
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(titles[column], color=INK)
    fig.suptitle(f"Gauge reconstruction check: {spec.name}", color=INK)
    fig.tight_layout()
    return fig


@torch.no_grad()
def visualize_latent_pca(z: torch.Tensor, spec: GaugeSpec, *, count: int = 4) -> plt.Figure:
    z = z[:count].float()
    gauge = make_gauge(spec)
    y = gauge.forward(z)
    joined = torch.cat([z, y], dim=0)
    tokens = joined.permute(0, 2, 3, 1).reshape(-1, joined.shape[1])
    tokens = tokens - tokens.mean(dim=0, keepdim=True)
    _, _, basis = torch.pca_lowrank(tokens, q=min(3, tokens.shape[1]), center=False)
    rgb = (tokens @ basis[:, :3]).reshape(2 * count, z.shape[-2], z.shape[-1], -1)
    if rgb.shape[-1] < 3:
        rgb = F.pad(rgb, (0, 3 - rgb.shape[-1]))
    low = torch.quantile(rgb.reshape(-1, 3), 0.02, dim=0)
    high = torch.quantile(rgb.reshape(-1, 3), 0.98, dim=0)
    rgb = ((rgb - low) / (high - low).clamp_min(1e-6)).clamp(0, 1).cpu()
    fig, axes = plt.subplots(count, 2, figsize=(7.2, 3.1 * count), squeeze=False)
    for row in range(count):
        axes[row, 0].imshow(rgb[row].numpy())
        axes[row, 1].imshow(rgb[count + row].numpy())
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")
    axes[0, 0].set_title("PCA(z)", color=INK)
    axes[0, 1].set_title("PCA(Az), shared basis", color=INK)
    fig.suptitle(f"Latent coordinate view: {spec.name}", color=INK)
    fig.tight_layout()
    return fig


class TimeEmbedding(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.register_buffer("frequencies", 2.0 ** torch.arange(4, dtype=torch.float32), persistent=False)
        self.projection = nn.Sequential(nn.Linear(8, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * math.pi * t[:, None] * self.frequencies[None]
        return self.projection(torch.cat([angles.sin(), angles.cos()], dim=1))


class LocalBlock(nn.Module):
    def __init__(self, hidden: int, kernel_size: int):
        super().__init__()
        groups = math.gcd(hidden, 8)
        self.block = nn.Sequential(
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size, padding=kernel_size // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TinyLocalProbe(nn.Module):
    def __init__(self, channels: int, hidden: int, depth: int, kernel_size: int):
        super().__init__()
        self.input = nn.Conv2d(channels, hidden, 1)
        self.time = TimeEmbedding(hidden)
        self.blocks = nn.ModuleList(LocalBlock(hidden, kernel_size) for _ in range(depth))
        self.output = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        hidden = self.input(x) + self.time(t)[:, :, None, None]
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(F.silu(hidden))


class GlobalBlock(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        x = x + attended
        return x + self.mlp(self.norm2(x))


def _position_embedding(height: int, width: int, hidden: int, device, dtype) -> torch.Tensor:
    if hidden % 4:
        raise ValueError("global probe hidden size must be divisible by 4")
    quarter = hidden // 4
    frequencies = 2.0 ** torch.arange(quarter, device=device, dtype=dtype)
    y = torch.linspace(0, 1, height, device=device, dtype=dtype)[:, None] * frequencies[None]
    x = torch.linspace(0, 1, width, device=device, dtype=dtype)[:, None] * frequencies[None]
    y_features = torch.cat([torch.sin(2 * math.pi * y), torch.cos(2 * math.pi * y)], dim=1)
    x_features = torch.cat([torch.sin(2 * math.pi * x), torch.cos(2 * math.pi * x)], dim=1)
    return torch.cat(
        [
            y_features[:, None, :].expand(height, width, -1),
            x_features[None, :, :].expand(height, width, -1),
        ],
        dim=-1,
    ).reshape(1, height * width, hidden)


class TinyGlobalProbe(nn.Module):
    def __init__(self, channels: int, hidden: int, depth: int, heads: int):
        super().__init__()
        self.input = nn.Linear(channels, hidden)
        self.time = TimeEmbedding(hidden)
        self.blocks = nn.ModuleList(GlobalBlock(hidden, heads) for _ in range(depth))
        self.output = nn.Linear(hidden, channels)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        hidden = self.input(tokens)
        hidden = hidden + self.time(t)[:, None, :]
        hidden = hidden + _position_embedding(height, width, hidden.shape[-1], hidden.device, hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(F.silu(hidden)).reshape(batch, height, width, channels).permute(0, 3, 1, 2)


@dataclass(frozen=True)
class ProbeConfig:
    name: str
    kind: str = "local"
    hidden: int = 32
    depth: int = 2
    kernel_size: int = 3
    heads: int = 4

    @property
    def receptive_field(self) -> float:
        if self.kind == "global":
            return math.inf
        return float(1 + 2 * self.depth * (self.kernel_size - 1))


@dataclass(frozen=True)
class ProbeTrainingConfig:
    steps: int = 30
    eval_steps: Tuple[int, ...] = (0, 10, 30)
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    eval_batches: int = 4
    eval_full_dataset: bool = False
    time_bins: int = 8
    seed: int = 0


@dataclass
class ProbeRun:
    gauge: GaugeSpec
    probe: ProbeConfig
    seed: int
    channels: int
    state_dict: Dict[str, torch.Tensor]
    final_relative_mse: float
    parameter_count: int
    elapsed_seconds: float


def build_probe(channels: int, config: ProbeConfig) -> nn.Module:
    if config.kind == "local":
        return TinyLocalProbe(channels, config.hidden, config.depth, config.kernel_size)
    if config.kind == "global":
        return TinyGlobalProbe(channels, config.hidden, config.depth, config.heads)
    raise ValueError(f"unknown probe kind: {config.kind}")


def _paired_batch(
    latents: torch.Tensor,
    batch_size: int,
    seed: int,
    step: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 104_729 * int(step))
    indices = torch.randint(0, len(latents), (batch_size,), generator=generator)
    t = torch.rand((batch_size,), generator=generator).clamp(1e-4, 1 - 1e-4)
    epsilon = torch.randn((batch_size, *latents.shape[1:]), generator=generator)
    return (
        latents[indices].to(device=device, dtype=torch.float32),
        t.to(device=device, dtype=torch.float32),
        epsilon.to(device=device, dtype=torch.float32),
    )


def _flow_batch(
    latents: torch.Tensor,
    t: torch.Tensor,
    epsilon: torch.Tensor,
    gauge: OrthogonalGauge,
) -> Tuple[torch.Tensor, torch.Tensor]:
    alpha = torch.cos(0.5 * math.pi * t).view(-1, 1, 1, 1)
    sigma = torch.sin(0.5 * math.pi * t).view(-1, 1, 1, 1)
    noisy = alpha * latents + sigma * epsilon
    velocity = alpha * epsilon - sigma * latents
    return gauge.forward(noisy), gauge.forward(velocity)


@torch.no_grad()
def evaluate_probe(
    model: nn.Module,
    latents: torch.Tensor,
    gauge: OrthogonalGauge,
    training: ProbeTrainingConfig,
    device: torch.device,
    *,
    seed_offset: int = 1_000_000,
) -> Tuple[float, List[Dict[str, float]]]:
    model.eval()
    total_error = 0.0
    total_target = 0.0
    bin_error = np.zeros(training.time_bins, dtype=np.float64)
    bin_target = np.zeros(training.time_bins, dtype=np.float64)
    bin_count = np.zeros(training.time_bins, dtype=np.int64)
    if training.eval_full_dataset:
        batch_starts = range(0, len(latents), training.batch_size)
    else:
        batch_starts = range(training.eval_batches)
    for batch_index, batch_start in enumerate(batch_starts):
        if training.eval_full_dataset:
            batch_end = min(int(batch_start) + training.batch_size, len(latents))
            z = latents[int(batch_start) : batch_end].to(device=device, dtype=torch.float32)
            generator = torch.Generator(device="cpu").manual_seed(
                int(training.seed + seed_offset) + 104_729 * batch_index
            )
            t = torch.rand((len(z),), generator=generator).clamp(1e-4, 1 - 1e-4).to(device)
            epsilon = torch.randn(z.shape, generator=generator).to(device)
        else:
            z, t, epsilon = _paired_batch(
                latents,
                min(training.batch_size, len(latents)),
                training.seed + seed_offset,
                batch_index,
                device,
            )
        noisy, target = _flow_batch(z, t, epsilon, gauge)
        prediction = model(noisy, t)
        per_sample_error = (prediction - target).flatten(1).square().sum(dim=1)
        per_sample_target = target.flatten(1).square().sum(dim=1).clamp_min(EPS)
        total_error += float(per_sample_error.sum())
        total_target += float(per_sample_target.sum())
        bins = torch.clamp((t * training.time_bins).long(), max=training.time_bins - 1)
        for index in range(len(t)):
            bucket = int(bins[index])
            bin_error[bucket] += float(per_sample_error[index])
            bin_target[bucket] += float(per_sample_target[index])
            bin_count[bucket] += 1
    rows = []
    for bucket in range(training.time_bins):
        t_low = bucket / training.time_bins
        t_high = (bucket + 1) / training.time_bins
        t_center = 0.5 * (t_low + t_high)
        alpha = math.cos(0.5 * math.pi * t_center)
        sigma = math.sin(0.5 * math.pi * t_center)
        relative_mse = float(bin_error[bucket] / bin_target[bucket]) if bin_target[bucket] > 0 else float("nan")
        rows.append(
            {
                "time_bin": bucket,
                "t_low": t_low,
                "t_high": t_high,
                "t_center": t_center,
                "logsnr": math.log((alpha * alpha + EPS) / (sigma * sigma + EPS)),
                "relative_mse": relative_mse,
                "count": int(bin_count[bucket]),
            }
        )
    return float(total_error / max(total_target, EPS)), rows


def train_probe(
    train_latents: torch.Tensor,
    val_latents: torch.Tensor,
    gauge_spec: GaugeSpec,
    probe_config: ProbeConfig,
    training: ProbeTrainingConfig,
    *,
    device: str | torch.device = "cuda:0",
    run_seed: int = 0,
) -> Tuple[ProbeRun, List[Dict[str, object]], List[Dict[str, object]]]:
    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    torch.manual_seed(run_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(run_seed)
    model = build_probe(train_latents.shape[1], probe_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    gauge = make_gauge(gauge_spec)
    history: List[Dict[str, object]] = []
    time_rows: List[Dict[str, object]] = []
    eval_steps = sorted(set(training.eval_steps) | {0, training.steps})
    start_time = time.perf_counter()

    def record(step: int) -> float:
        relative_mse, rows = evaluate_probe(model, val_latents, gauge, training, device)
        common = {
            "gauge": gauge_spec.name,
            "gauge_kind": gauge_spec.kind,
            "gauge_strength": gauge_spec.strength,
            "gauge_radius": gauge_spec.radius,
            "probe": probe_config.name,
            "probe_kind": probe_config.kind,
            "receptive_field": probe_config.receptive_field,
            "seed": run_seed,
            "step": step,
            "relative_mse": relative_mse,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
        history.append(common)
        if step == training.steps:
            for row in rows:
                time_rows.append({**common, **row})
        return relative_mse

    final_loss = record(0)
    for step in range(1, training.steps + 1):
        model.train()
        z, t, epsilon = _paired_batch(
            train_latents,
            min(training.batch_size, len(train_latents)),
            training.seed + 10_000 * run_seed,
            step,
            device,
        )
        noisy, target = _flow_batch(z, t, epsilon, gauge)
        prediction = model(noisy, t)
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip)
        optimizer.step()
        if step in eval_steps:
            final_loss = record(step)
    elapsed = time.perf_counter() - start_time
    state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    run = ProbeRun(
        gauge=gauge_spec,
        probe=probe_config,
        seed=run_seed,
        channels=train_latents.shape[1],
        state_dict=state_dict,
        final_relative_mse=final_loss,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        elapsed_seconds=elapsed,
    )
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return run, history, time_rows


def run_probe_grid(
    train_latents: torch.Tensor,
    val_latents: torch.Tensor,
    gauges: Sequence[GaugeSpec],
    probes: Sequence[ProbeConfig],
    training: ProbeTrainingConfig,
    *,
    seeds: Sequence[int] = (0,),
    device: str | torch.device = "cuda:0",
    latent_scale: Optional[float] = None,
) -> Tuple[List[ProbeRun], pd.DataFrame, pd.DataFrame]:
    scale = float(latent_scale or train_latents.square().mean().sqrt().clamp_min(1e-8))
    train = train_latents.float().cpu() / scale
    val = val_latents.float().cpu() / scale
    runs: List[ProbeRun] = []
    history_rows: List[Dict[str, object]] = []
    time_rows: List[Dict[str, object]] = []
    total = len(gauges) * len(probes) * len(seeds)
    completed = 0
    for probe in probes:
        for gauge in gauges:
            for run_seed in seeds:
                run, history, bins = train_probe(
                    train,
                    val,
                    gauge,
                    probe,
                    training,
                    device=device,
                    run_seed=run_seed,
                )
                runs.append(run)
                history_rows.extend(history)
                time_rows.extend(bins)
                completed += 1
                print(
                    f"probe {completed}/{total}: {probe.name} | {gauge.name} | seed={run_seed} "
                    f"rel_mse={run.final_relative_mse:.4f}",
                    flush=True,
                )
    return runs, pd.DataFrame(history_rows), pd.DataFrame(time_rows)


def add_identity_ratios(history: pd.DataFrame) -> pd.DataFrame:
    keys = ["probe", "seed", "step"]
    identity = history[history["gauge"] == "identity"][keys + ["relative_mse"]].rename(
        columns={"relative_mse": "identity_relative_mse"}
    )
    merged = history.merge(identity, on=keys, how="left", validate="many_to_one")
    merged["loss_ratio_to_identity"] = merged["relative_mse"] / merged["identity_relative_mse"]
    return merged


def finite_difference_headroom(
    history: pd.DataFrame,
    *,
    plus_gauge: str,
    minus_gauge: str,
    delta: float,
) -> pd.DataFrame:
    final_step = int(history["step"].max())
    subset = history[history["step"] == final_step]
    wide = subset.pivot_table(
        index=["probe", "seed"],
        columns="gauge",
        values="relative_mse",
        aggfunc="first",
    ).reset_index()
    required = {"identity", plus_gauge, minus_gauge}
    missing = required - set(wide.columns)
    if missing:
        raise ValueError(f"finite difference gauges missing: {sorted(missing)}")
    wide["directional_gradient"] = (wide[plus_gauge] - wide[minus_gauge]) / (2 * delta)
    wide["directional_curvature"] = (
        wide[plus_gauge] + wide[minus_gauge] - 2 * wide["identity"]
    ) / (delta * delta)
    return wide


def _restore_probe(run: ProbeRun, device: torch.device) -> nn.Module:
    model = build_probe(run.channels, run.probe).to(device)
    model.load_state_dict(run.state_dict, strict=True)
    return model.eval()


@torch.no_grad()
def decoder_residual_table(
    bundle: CodecDataBundle,
    runs: Sequence[ProbeRun],
    training: ProbeTrainingConfig,
    *,
    count: int = 4,
    run_filter: Optional[Iterable[Tuple[str, str]]] = None,
) -> pd.DataFrame:
    allowed = None if run_filter is None else set(run_filter)
    device = bundle.device
    count = min(count, len(bundle.val_latents))
    z_raw = bundle.val_latents[:count].to(device)
    z = z_raw / bundle.latent_scale
    reference = bundle.adapter.decode(z_raw)
    rows = []
    for run_index, run in enumerate(runs):
        if allowed is not None and (run.probe.name, run.gauge.name) not in allowed:
            continue
        model = _restore_probe(run, device)
        gauge = make_gauge(run.gauge)
        generator = torch.Generator(device="cpu").manual_seed(
            training.seed + 2_000_000 + 104_729 * run_index + run.seed
        )
        t = torch.rand((count,), generator=generator).clamp(1e-4, 1 - 1e-4).to(device)
        epsilon = torch.randn(z.shape, generator=generator).to(device)
        noisy, _ = _flow_batch(z, t, epsilon, gauge)
        prediction = model(noisy, t)
        alpha = torch.cos(0.5 * math.pi * t).view(-1, 1, 1, 1)
        sigma = torch.sin(0.5 * math.pi * t).view(-1, 1, 1, 1)
        predicted_y0 = alpha * noisy - sigma * prediction
        predicted_z = gauge.inverse(predicted_y0) * bundle.latent_scale
        decoded = bundle.adapter.decode(predicted_z)
        latent_mse = (predicted_z - z_raw).square().flatten(1).mean(dim=1)
        image_mse = (decoded - reference).square().flatten(1).mean(dim=1)
        image_l1 = (decoded - reference).abs().flatten(1).mean(dim=1)
        rows.append(
            {
                "probe": run.probe.name,
                "probe_kind": run.probe.kind,
                "gauge": run.gauge.name,
                "seed": run.seed,
                "latent_mse": float(latent_mse.mean()),
                "latent_rmse": float(latent_mse.mean().sqrt()),
                "decoded_mse": float(image_mse.mean()),
                "decoded_l1": float(image_l1.mean()),
                "decoded_psnr": float(-10.0 * torch.log10(image_mse.mean().clamp_min(EPS))),
                "decoder_amplification": float(
                    image_mse.mean().sqrt() / latent_mse.mean().sqrt().clamp_min(EPS)
                ),
                "count": count,
            }
        )
        del model
    return pd.DataFrame(rows)


def probe_configs(hidden: int = 32) -> List[ProbeConfig]:
    return [
        ProbeConfig("local_rf5", kind="local", hidden=hidden, depth=1, kernel_size=3),
        ProbeConfig("local_rf9", kind="local", hidden=hidden, depth=2, kernel_size=3),
        ProbeConfig("global_attn", kind="global", hidden=hidden, depth=1, heads=4),
    ]


def _color_map(names: Sequence[str]) -> Dict[str, str]:
    return {name: PALETTE[index % len(PALETTE)] for index, name in enumerate(names)}


def plot_learning_curves(history: pd.DataFrame) -> plt.Figure:
    ratios = add_identity_ratios(history)
    probes = list(dict.fromkeys(ratios["probe"]))
    gauges = list(dict.fromkeys(ratios["gauge"]))
    colors = _color_map(gauges)
    fig, axes = plt.subplots(
        1,
        len(probes),
        figsize=(6.2 * len(probes), 4.5),
        squeeze=False,
        sharey=True,
    )
    for column, probe in enumerate(probes):
        axis = axes[0, column]
        subset = ratios[ratios["probe"] == probe]
        for gauge in gauges:
            rows = (
                subset[subset["gauge"] == gauge]
                .groupby("step")["loss_ratio_to_identity"]
                .agg(["mean", "std"])
                .reset_index()
            )
            axis.plot(
                rows["step"],
                rows["mean"],
                marker="o",
                linewidth=2,
                color=colors[gauge],
                label=gauge,
            )
            if rows["std"].notna().any():
                axis.fill_between(
                    rows["step"],
                    rows["mean"] - rows["std"].fillna(0),
                    rows["mean"] + rows["std"].fillna(0),
                    color=colors[gauge],
                    alpha=0.10,
                )
        axis.axhline(1.0, color=INK, linewidth=1, linestyle="--")
        axis.set_title(probe, color=INK)
        axis.set_xlabel("training step")
        axis.grid(axis="y", color=LIGHT_GREY, alpha=0.7)
    axes[0, 0].set_ylabel("held-out relative MSE / identity")
    axes[0, -1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Paired finite-horizon probe learning curves", color=INK)
    fig.tight_layout()
    return fig


def plot_locality_comparison(history: pd.DataFrame) -> plt.Figure:
    ratios = add_identity_ratios(history)
    final_step = ratios["step"].max()
    final = ratios[ratios["step"] == final_step]
    summary = (
        final.groupby(["probe", "gauge"])["loss_ratio_to_identity"]
        .agg(["mean", "std"])
        .reset_index()
    )
    probes = list(dict.fromkeys(summary["probe"]))
    gauges = list(dict.fromkeys(summary["gauge"]))
    colors = _color_map(probes)
    x = np.arange(len(gauges))
    fig, axis = plt.subplots(figsize=(max(10, 1.35 * len(gauges)), 5.2))
    offsets = np.linspace(-0.22, 0.22, len(probes)) if len(probes) > 1 else np.array([0.0])
    for offset, probe in zip(offsets, probes):
        rows = summary[summary["probe"] == probe].set_index("gauge").reindex(gauges)
        axis.errorbar(
            x + offset,
            rows["mean"],
            yerr=rows["std"].fillna(0),
            marker="o",
            linestyle="none",
            capsize=3,
            color=colors[probe],
            label=probe,
        )
    axis.axhline(1.0, color=INK, linewidth=1, linestyle="--")
    axis.set_xticks(x)
    axis.set_xticklabels(gauges, rotation=25, ha="right")
    axis.set_ylabel("held-out relative MSE / identity")
    axis.set_title(f"Locality comparison at step {int(final_step)}", color=INK)
    axis.grid(axis="y", color=LIGHT_GREY, alpha=0.7)
    axis.legend(frameon=False, ncol=min(3, len(probes)))
    fig.tight_layout()
    return fig


def plot_time_bin_heatmap(time_rows: pd.DataFrame, *, probe: str) -> plt.Figure:
    subset = time_rows[time_rows["probe"] == probe].copy()
    identity = subset[subset["gauge"] == "identity"]["time_bin seed relative_mse".split()].rename(
        columns={"relative_mse": "identity_mse"}
    )
    subset = subset.merge(identity, on=["time_bin", "seed"], how="left", validate="many_to_one")
    subset["ratio"] = subset["relative_mse"] / subset["identity_mse"]
    matrix = subset.pivot_table(index="gauge", columns="time_bin", values="ratio", aggfunc="mean")
    fig, axis = plt.subplots(figsize=(10, max(3.8, 0.55 * len(matrix) + 1.6)))
    values = matrix.to_numpy(dtype=float)
    masked_values = np.ma.masked_invalid(values)
    color_map = plt.colormaps["coolwarm"].copy()
    color_map.set_bad(LIGHT_GREY)
    image = axis.imshow(
        masked_values,
        aspect="auto",
        cmap=color_map,
        vmin=0.85,
        vmax=1.15,
    )
    axis.set_yticks(np.arange(len(matrix.index)))
    axis.set_yticklabels(matrix.index)
    axis.set_xticks(np.arange(len(matrix.columns)))
    axis.set_xticklabels([str(column) for column in matrix.columns])
    axis.set_xlabel("t bin: low noise to high noise")
    axis.set_title(f"Time-binned loss ratio to identity: {probe}", color=INK)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            if np.isfinite(value):
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", color=INK, fontsize=8)
            else:
                axis.text(column, row, "NA", ha="center", va="center", color=INK, fontsize=8)
    fig.colorbar(image, ax=axis, label="relative MSE / identity")
    fig.tight_layout()
    return fig


def plot_second_order_control(exact: pd.DataFrame, history: pd.DataFrame) -> plt.Figure:
    ratios = add_identity_ratios(history)
    final = ratios[ratios["step"] == ratios["step"].max()]
    loss = final.groupby("gauge", as_index=False)["loss_ratio_to_identity"].mean()
    merged = exact.merge(loss, left_on="name", right_on="gauge", how="inner")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    fields = ["total_psd_rel_error", "channel_gram_spectrum_rel_error"]
    labels = ["total PSD relative error", "channel Gram spectrum relative error"]
    for axis, field, label in zip(axes, fields, labels):
        axis.scatter(
            merged[field],
            merged["loss_ratio_to_identity"],
            s=55,
            color=BLUE,
            edgecolor=INK,
        )
        for _, row in merged.iterrows():
            axis.annotate(
                row["gauge"],
                (row[field], row["loss_ratio_to_identity"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axis.axhline(1.0, color=INK, linewidth=1, linestyle="--")
        axis.set_xlabel(label)
        axis.set_ylabel("held-out relative MSE / identity")
        axis.grid(color=LIGHT_GREY, alpha=0.7)
    fig.suptitle("Second-order controls versus probe sensitivity", color=INK)
    fig.tight_layout()
    return fig


def plot_decoder_residuals(decoder_rows: pd.DataFrame) -> plt.Figure:
    fig, axis = plt.subplots(figsize=(7.4, 5.2))
    probes = list(dict.fromkeys(decoder_rows["probe"]))
    colors = _color_map(probes)
    for probe in probes:
        rows = decoder_rows[decoder_rows["probe"] == probe]
        axis.scatter(
            rows["latent_rmse"],
            rows["decoded_l1"],
            s=65,
            color=colors[probe],
            edgecolor=INK,
            label=probe,
        )
        for _, row in rows.iterrows():
            axis.annotate(
                row["gauge"],
                (row["latent_rmse"], row["decoded_l1"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    axis.set_xlabel("predicted z0 latent RMSE")
    axis.set_ylabel("decoded image L1")
    axis.set_title("Decoder amplification of actual probe residuals", color=INK)
    axis.grid(color=LIGHT_GREY, alpha=0.7)
    axis.legend(frameon=False)
    fig.tight_layout()
    return fig


def mechanism_routing_table(
    history: pd.DataFrame,
    time_rows: pd.DataFrame,
    exact: pd.DataFrame,
    decoder_rows: Optional[pd.DataFrame] = None,
    *,
    improvement_threshold: float = 0.02,
    locality_threshold: float = 0.03,
) -> pd.DataFrame:
    ratios = add_identity_ratios(history)
    ordered_steps = sorted(ratios["step"].unique())
    early_step = ordered_steps[1] if len(ordered_steps) > 2 else ordered_steps[0]
    final_step = ratios["step"].max()
    early = ratios[ratios["step"] == early_step]
    final = ratios[ratios["step"] == final_step]
    final_summary = (
        final.groupby(["probe", "probe_kind", "gauge"], as_index=False)["loss_ratio_to_identity"].mean()
    )
    seeds = int(history["seed"].nunique())
    confidence = "confirmatory" if seeds >= 3 else "exploratory (<3 paired seeds)"
    rows: List[Dict[str, str]] = []

    exact_max = exact[["inverse_rel_l2", "norm_rel_error", "paired_noise_rel_error"]].to_numpy().max()
    rows.append(
        {
            "mechanism": "exact orthogonal wrapper",
            "status": "pass" if exact_max < 1e-5 else "fail",
            "evidence": f"max inverse/norm/noise error={exact_max:.2e}",
            "next_action": "continue" if exact_max < 1e-5 else "stop and fix gauge implementation",
            "confidence": "unit-test gate",
        }
    )

    non_identity = final_summary[final_summary["gauge"] != "identity"]
    best = non_identity.loc[non_identity["loss_ratio_to_identity"].idxmin()] if len(non_identity) else None
    has_headroom = best is not None and best["loss_ratio_to_identity"] <= 1.0 - improvement_threshold
    rows.append(
        {
            "mechanism": "identity headroom (H2)",
            "status": "candidate" if has_headroom else "not observed",
            "evidence": (
                "no non-identity gauge evaluated"
                if best is None
                else f"best={best['gauge']} on {best['probe']}, ratio={best['loss_ratio_to_identity']:.3f}"
            ),
            "next_action": (
                "repeat with >=3 paired seeds and +/- direction"
                if has_headroom
                else "widen structured directions before method training"
            ),
            "confidence": confidence,
        }
    )

    allpass = final_summary[final_summary["gauge"].str.startswith("allpass")]
    locality_gaps = []
    for gauge, group in allpass.groupby("gauge"):
        local = group[group["probe_kind"] == "local"]["loss_ratio_to_identity"]
        global_values = group[group["probe_kind"] == "global"]["loss_ratio_to_identity"]
        if len(local) and len(global_values):
            locality_gaps.append((gauge, float(local.max() - global_values.mean())))
    best_locality = max(locality_gaps, key=lambda item: item[1]) if locality_gaps else ("none", float("nan"))
    locality_supported = bool(locality_gaps) and best_locality[1] >= locality_threshold
    rows.append(
        {
            "mechanism": "higher-order locality mismatch",
            "status": "supported" if locality_supported else "not established",
            "evidence": (
                f"largest local-minus-global penalty={best_locality[1]:.3f} ({best_locality[0]})"
                if locality_gaps
                else "requires both local and global probes"
            ),
            "next_action": (
                "run RF sweep and cross-architecture gauge exchange"
                if locality_supported
                else "do not select LocalGauge yet"
            ),
            "confidence": confidence,
        }
    )

    early_mean = early.groupby("gauge")["loss_ratio_to_identity"].mean()
    late_mean = final.groupby("gauge")["loss_ratio_to_identity"].mean()
    shared = [name for name in early_mean.index if name != "identity" and name in late_mean.index]
    convergence = [(name, abs(early_mean[name] - 1) - abs(late_mean[name] - 1)) for name in shared]
    finite_only = max(convergence, key=lambda item: item[1]) if convergence else ("none", float("nan"))
    finite_candidate = bool(convergence) and finite_only[1] >= improvement_threshold
    rows.append(
        {
            "mechanism": "finite-horizon optimization",
            "status": "candidate" if finite_candidate else "not established",
            "evidence": (
                f"largest shrink toward identity={finite_only[1]:.3f} ({finite_only[0]})"
                if convergence
                else "needs at least two nonzero budgets"
            ),
            "next_action": (
                "report efficiency, not asymptotic quality"
                if finite_candidate
                else "retain longer-budget control"
            ),
            "confidence": confidence,
        }
    )

    identity_bins = time_rows[time_rows["gauge"] == "identity"][
        "probe seed time_bin relative_mse".split()
    ].rename(columns={"relative_mse": "identity_mse"})
    binned = time_rows.merge(
        identity_bins,
        on=["probe", "seed", "time_bin"],
        how="left",
        validate="many_to_one",
    )
    binned["delta"] = binned["relative_mse"] / binned["identity_mse"] - 1.0
    conflict = False
    conflict_name = "none"
    for gauge, group in binned[binned["gauge"] != "identity"].groupby("gauge"):
        means = group.groupby("time_bin")["delta"].mean()
        if (means < -improvement_threshold).any() and (means > improvement_threshold).any():
            conflict = True
            conflict_name = gauge
            break
    rows.append(
        {
            "mechanism": "time-dependent preference",
            "status": "candidate" if conflict else "not observed",
            "evidence": (
                f"sign-changing time-bin preference: {conflict_name}"
                if conflict
                else "no robust +/- time-bin reversal at current threshold"
            ),
            "next_action": "only now consider moving gauge" if conflict else "keep A static",
            "confidence": confidence,
        }
    )

    if decoder_rows is None or decoder_rows.empty:
        decoder_status = "not evaluated"
        decoder_evidence = "run decoder_residual_table on selected final probes"
        decoder_action = "collect actual probe residuals"
    else:
        amplification_mean = max(float(decoder_rows["decoder_amplification"].mean()), EPS)
        amplification_cv = float(
            decoder_rows["decoder_amplification"].std(ddof=0) / amplification_mean
        )
        decoder_status = "candidate" if amplification_cv > 0.10 else "not dominant"
        decoder_evidence = f"CV(image-RMSE/latent-RMSE)={amplification_cv:.3f}"
        decoder_action = (
            "separate decoder-aware project" if decoder_status == "candidate" else "keep decoder as a control"
        )
    rows.append(
        {
            "mechanism": "decoder directional amplification",
            "status": decoder_status,
            "evidence": decoder_evidence,
            "next_action": decoder_action,
            "confidence": confidence,
        }
    )
    return pd.DataFrame(rows)
