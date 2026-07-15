from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch


RAE_REPO_URL = "https://github.com/bytetriper/RAE.git"
RAE_HF_REPO = "nyu-visionx/RAE-collections"

RAE_SPECS: Dict[str, dict] = {
    "rae_dinov2": {
        "label": "rae_dinov2_b",
        "config": "configs/stage1/pretrained/DINOv2-B.yaml",
        "files": [
            "decoders/dinov2/wReg_base/ViTXL_n08/model.pt",
            "stats/dinov2/wReg_base/imagenet1k/stat.pt",
        ],
    },
    "rae_mae": {
        "label": "rae_mae_b",
        "config": "configs/stage1/pretrained/MAE.yaml",
        "files": [
            "decoders/mae/base_p16/ViTXL_n08/model.pt",
            "stats/mae/base_p16/ImageNet1k/stat.pt",
        ],
    },
    "rae_siglip2": {
        "label": "rae_siglip2_b",
        "config": "configs/stage1/pretrained/SigLIP2.yaml",
        "files": [
            "decoders/siglip2/base_p16_i256/ViTXL_n08/model.pt",
            "stats/siglip2/base_p16_i256/ImageNet1k/stat.pt",
        ],
    },
}


@dataclass
class RAEAdapter:
    model: torch.nn.Module
    device: torch.device
    dtype: torch.dtype
    key: str

    @torch.no_grad()
    def encode(self, x: torch.Tensor, posterior: Optional[str] = None) -> torch.Tensor:
        del posterior
        self.model.eval()
        x = x.to(device=self.device, dtype=self.dtype)
        x = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
        return self.model.encode(x)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        z = z.to(device=self.device, dtype=self.dtype).contiguous()
        x = self.model.decode(z).clamp(0.0, 1.0)
        return x * 2.0 - 1.0


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _run(cmd: list[str], cwd: Optional[Path] = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_rae_repo(repo_path: str | Path, auto_clone: bool = False) -> Path:
    repo_path = _as_path(repo_path)
    marker = repo_path / "src" / "stage1" / "rae.py"
    if marker.exists():
        return repo_path
    if auto_clone:
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--depth", "1", RAE_REPO_URL, str(repo_path)])
        return repo_path
    raise FileNotFoundError(
        "没有找到 RAE 官方仓库代码。请先运行：\n"
        f"  git clone --depth 1 {RAE_REPO_URL} {repo_path}\n"
        "或者在 notebook 里把 rae_auto_clone 改成 True。"
    )


def ensure_rae_files(
    repo_path: str | Path,
    files: Iterable[str],
    auto_download: bool = False,
) -> None:
    repo_path = _as_path(repo_path)
    missing = [name for name in files if not (repo_path / "models" / name).exists()]
    if not missing:
        return
    if auto_download:
        from huggingface_hub import hf_hub_download

        for name in missing:
            hf_hub_download(
                repo_id=RAE_HF_REPO,
                filename=name,
                local_dir=repo_path / "models",
            )
        return
    joined = "\n".join(f"  {name}" for name in missing)
    raise FileNotFoundError(
        "RAE 权重/统计文件还没有下载。缺少：\n"
        f"{joined}\n"
        "可以在 RAE 仓库目录运行：\n"
        f"  hf download {RAE_HF_REPO} --local-dir models\n"
        "或者在 notebook 里把 rae_auto_download 改成 True。"
    )


def get_rae_status(repo_path: str | Path) -> Dict[str, dict]:
    repo_path = _as_path(repo_path)
    repo_ready = (repo_path / "src" / "stage1" / "rae.py").exists()
    status = {}
    for key, spec in RAE_SPECS.items():
        missing = [name for name in spec["files"] if not (repo_path / "models" / name).exists()]
        config_ready = (repo_path / spec["config"]).exists()
        status[key] = {
            "repo": repo_ready,
            "config": config_ready,
            "weights": len(missing) == 0,
            "missing": missing,
        }
    return status


def _prepare_decoder_config_path(repo_path: Path, decoder_config_path: Path, decoder_patch_size: int) -> str:
    config_path = decoder_config_path / "config.json"
    if not config_path.exists():
        return str(decoder_config_path)

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if isinstance(config.get("patch_size"), int):
        return str(decoder_config_path)

    cache_dir = repo_path / ".adapter_cache" / decoder_config_path.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    config["patch_size"] = int(decoder_patch_size)
    with (cache_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return str(cache_dir)


def _resolve_cached_hf_model(model_id: str) -> str:
    path = Path(model_id).expanduser()
    if path.exists() or "/" not in model_id:
        return str(path.resolve()) if path.exists() else model_id
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=model_id, local_files_only=True)
    except Exception:
        return model_id


def _load_stage1_config(repo_path: Path, config_relpath: str) -> dict:
    import yaml

    config_path = repo_path / config_relpath
    if not config_path.exists():
        raise FileNotFoundError(f"RAE 配置不存在：{config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if "stage_1" not in config:
        raise ValueError(f"RAE 配置缺少 stage_1: {config_path}")
    params = dict(config["stage_1"]["params"])
    encoder_config_path = params.get("encoder_config_path")
    if isinstance(encoder_config_path, str):
        params["encoder_config_path"] = _resolve_cached_hf_model(encoder_config_path)
    encoder_params = dict(params.get("encoder_params", {}))
    for name in ("dinov2_path", "model_name"):
        value = encoder_params.get(name)
        if isinstance(value, str):
            encoder_params[name] = _resolve_cached_hf_model(value)
    params["encoder_params"] = encoder_params
    for name in ("decoder_config_path", "pretrained_decoder_path", "normalization_stat_path"):
        value = params.get(name)
        if value is not None:
            path = Path(value)
            params[name] = str(path if path.is_absolute() else repo_path / path)
    decoder_config_path = Path(params["decoder_config_path"])
    params["decoder_config_path"] = _prepare_decoder_config_path(
        repo_path,
        decoder_config_path,
        int(params.get("decoder_patch_size", 16)),
    )
    return params


def load_rae_adapter(
    key: str,
    repo_path: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    auto_clone: bool = False,
    auto_download: bool = False,
) -> RAEAdapter:
    if key not in RAE_SPECS:
        raise KeyError(f"未知 RAE baseline: {key}; 可选 {list(RAE_SPECS)}")

    repo_path = ensure_rae_repo(repo_path, auto_clone=auto_clone)
    spec = RAE_SPECS[key]
    ensure_rae_files(repo_path, spec["files"], auto_download=auto_download)

    src_path = str(repo_path / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from stage1 import RAE

    params = _load_stage1_config(repo_path, spec["config"])
    model = RAE(**params).to(device=device, dtype=dtype).eval()
    return RAEAdapter(model=model, device=device, dtype=dtype, key=key)
