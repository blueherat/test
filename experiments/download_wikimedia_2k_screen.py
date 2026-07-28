"""Download a small, deterministic, native-2K Wikimedia Commons screen set."""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


CATEGORIES = (
    "Featured pictures of landscapes",
    "Featured pictures of animals",
    "Featured pictures of architecture",
    "Featured pictures of people",
)
API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "eqvae-noise-responsibility-research/1.0"


def plain_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(value))).strip()


def category_candidates(category: str, limit: int = 50) -> list[dict]:
    response = requests.get(
        API_URL,
        params={
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": f"Category:{category}",
            "gcmtype": "file",
            "gcmlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata",
            "iiurlwidth": 2560,
            "format": "json",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    candidates = []
    for page in pages.values():
        info = page.get("imageinfo", [{}])[0]
        url = info.get("url", "")
        download_url = info.get("thumburl", url)
        if min(int(info.get("width", 0)), int(info.get("height", 0))) < 2048:
            continue
        if Path(urlparse(url).path).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        metadata = info.get("extmetadata", {})
        candidates.append(
            {
                "category": category,
                "title": page.get("title", ""),
                "width": int(info["width"]),
                "height": int(info["height"]),
                "source_url": url,
                "download_url": download_url,
                "description_url": info.get("descriptionurl", ""),
                "license": plain_text(metadata.get("LicenseShortName", {}).get("value")),
                "artist": plain_text(metadata.get("Artist", {}).get("value")),
                "description": plain_text(metadata.get("ImageDescription", {}).get("value")),
            }
        )
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/data/shared/Datasets/wikimedia_2k_screen"))
    parser.add_argument("--per-category", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    selected = []
    for category in CATEGORIES:
        candidates = category_candidates(category)
        rng.shuffle(candidates)
        if len(candidates) < args.per_category:
            raise RuntimeError(f"only {len(candidates)} qualifying files in {category}")
        selected.extend(candidates[: args.per_category])

    manifest = []
    manifest_path = args.output / "manifest.json"
    for index, item in enumerate(selected):
        suffix = Path(urlparse(item["source_url"]).path).suffix.lower()
        filename = f"{index:02d}_{item['category'].lower().replace(' ', '_')}{suffix}"
        destination = args.output / filename
        if not destination.is_file():
            for attempt in range(5):
                response = requests.get(
                    item["download_url"],
                    headers={"User-Agent": USER_AGENT},
                    timeout=180,
                    stream=True,
                )
                if response.status_code != 429:
                    break
                response.close()
                time.sleep(5 * (attempt + 1))
            response.raise_for_status()
            with response, destination.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            time.sleep(1)
        manifest.append(item | {"local_path": str(destination), "selection_seed": args.seed})
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        print(f"[{index + 1}/{len(selected)}] {destination.name} ({item['width']}x{item['height']})")

    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
