#!/usr/bin/env python3
"""Create a compact preview grid from numbered PNG samples."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a preview grid from a sample folder.")
    parser.add_argument("--sample-dir", required=True, help="Directory containing 000000.png style samples.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--num-images", type=int, default=64)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--thumb-size", type=int, default=128)
    parser.add_argument(
        "--resample",
        choices=("thumbnail", "nearest", "smooth"),
        default="thumbnail",
        help=(
            "thumbnail never enlarges; nearest/smooth also enlarge small images "
            "to the requested tile while preserving aspect ratio"
        ),
    )
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir).expanduser()
    output = Path(args.output).expanduser()
    files = sorted(sample_dir.glob("*.png"))[: args.num_images]
    if not files:
        raise FileNotFoundError(f"No PNG samples found under {sample_dir}")

    cols = max(1, int(args.cols))
    rows = (len(files) + cols - 1) // cols
    thumb = int(args.thumb_size)
    grid = Image.new("RGB", (cols * thumb, rows * thumb), "white")

    for idx, path in enumerate(files):
        image = Image.open(path).convert("RGB")
        if args.resample == "thumbnail":
            image.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
        else:
            scale = min(thumb / image.width, thumb / image.height)
            resized = (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            )
            method = (
                Image.Resampling.NEAREST
                if args.resample == "nearest"
                else Image.Resampling.LANCZOS
            )
            image = image.resize(resized, method)
        tile = Image.new("RGB", (thumb, thumb), "white")
        x = (thumb - image.width) // 2
        y = (thumb - image.height) // 2
        tile.paste(image, (x, y))
        draw = ImageDraw.Draw(tile)
        draw.text((4, 4), path.stem, fill=(0, 0, 0))
        grid.paste(tile, ((idx % cols) * thumb, (idx // cols) * thumb))

    output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output)
    print(output)


if __name__ == "__main__":
    main()
