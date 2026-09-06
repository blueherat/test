"""CPU-only extractor lineage spot-check; does not rewrite frozen features."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import sys
import time
import numpy as np
import torch
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.raev2_training_core import DeterministicImageNetPacked

torch.set_num_threads(4)
start = time.perf_counter()
extractor = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"], verbose=False).eval().requires_grad_(False)
dataset = DeterministicImageNetPacked(Path("/data/shared/imagenet-1k/random_access_v1"), split="train", image_size=256, horizontal_flip=False)
rows = []
for seed in [20260801, 20260802]:
    root = Path(f"/home/zhoushunyu/data/eqvae/experiments/raev2_ig_scale_response/n5000_seed{seed}_scales7_v1")
    with np.load(root / "sample_protocol.npz") as z:
        source_rows, labels = z["real_source_rows"], z["labels"]
    for condition in ["source", "real", "scale_s1p000000", "scale_s1p780000"]:
        if condition == "source":
            images = []
            for i in [0, 4]:
                image, actual_label, _ = dataset[int(source_rows[i])]
                assert actual_label == labels[i]
                images.append(image)
            x = torch.stack(images).clamp(0, 1).mul(255).to(torch.uint8)
        else:
            imgs = np.load(root / "decoded" / f"{condition}_rank00.npy", mmap_mode="r")[:2]
            x = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().mul(255).to(torch.uint8)
        with torch.inference_mode():
            current = extractor(x)[0].numpy()
        old = np.load(root / "inception" / f"{condition}_rank00.npy", mmap_mode="r")[:2]
        diff = current - old
        row = {"seed": seed, "condition": condition, "global_ids": [0, 4],
            "max_abs_feature_diff": float(np.abs(diff).max()), "feature_rmse": float(np.sqrt(np.square(diff).mean()))}
        rows.append(row)
        print(row, flush=True)
dataset.close()
weight = Path("/home/zhoushunyu/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth")
result = {"gpu_used": False, "torch": torch.__version__, "torch_fidelity": importlib.metadata.version("torch-fidelity"),
    "rows": rows, "current_weight_path": str(weight), "current_weight_sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
    "historical_weights_hash_archived": False,
    "historical_source_code_enables_tf32_matmul_and_cudnn": True,
    "note": "This is a numerical lineage spot-check, not a bitwise identity certificate. Historical decoded images are stored in float16; CPU extraction also differs from historical CUDA TF32 execution.",
    "elapsed_cpu_wall_seconds": time.perf_counter() - start}
Path(__file__).with_name("cpu_feature_spotcheck.json").write_text(json.dumps(result, indent=2) + "\n")
