#!/usr/bin/env python3
"""Freeze the official sequential CUDA noise draws for paired GPU workers."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202609068)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    path = out/"initial_noises.npy"
    bank = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(1000, 1024, 16, 16))
    raw_digest = hashlib.sha256()
    batches = []
    for start in range(0, 1000, 8):
        noise = torch.randn(8, 1024, 16, 16, generator=generator, device=device, dtype=torch.float32).cpu().numpy()
        bank[start:start+8] = noise
        raw = noise.tobytes()
        raw_digest.update(raw)
        batches.append({"start": start, "stop": start+8, "noise_sha256": hashlib.sha256(raw).hexdigest()})
    bank.flush()
    del bank
    labels = np.arange(1000, dtype=np.int64)
    label_path = out/"labels.npy"
    np.save(label_path, labels)
    manifest = {"protocol": "raev2_official_sequential_cuda_noise_bank_v1", "complete": True,
                "seed": args.seed, "shape": [1000, 1024, 16, 16], "dtype": "float32", "batch_size": 8,
                "path": str(path), "file_sha256": sha256(path), "noise_sha256": raw_digest.hexdigest(),
                "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(), "labels_file_sha256": sha256(label_path),
                "noise_schema": "one device torch.Generator(seed), sequential fixed-shape FP32 randn batches",
                "torch_version": str(torch.__version__), "device": torch.cuda.get_device_name(device),
                "source_sha256": sha256(Path(__file__)), "batches": batches}
    temporary = out/"manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2)+"\n")
    temporary.replace(out/"manifest.json")
    print(json.dumps({k: v for k, v in manifest.items() if k != "batches"}, indent=2))


if __name__ == "__main__":
    main()
