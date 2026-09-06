"""CPU-only source and note archive finalization; no model dependencies."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import time

wall_start = time.perf_counter()
cpu_start = time.process_time()
root = Path(__file__).resolve().parent
note = Path("/home/zhoushunyu/eqvae/docs/RAEV2_GUIDANCE_READING_ACTUAL_DISTRIBUTION_FEEDBACK_20260906_ZH.md")

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

bad = [(i, ord(c)) for i, c in enumerate(note.read_text()) if ord(c) < 32 and c not in "\n\r"]
if bad:
    raise ValueError(f"Unexpected note control characters: {bad}")

verified_sources = []
for src in json.loads((root / "sources.json").read_text()):
    path = root / src["file"]
    actual = sha256(path)
    if actual != src["sha256"]:
        raise ValueError(f"Source hash mismatch: {path}")
    verified_sources.append(src["file"])

shutil.copy2(note, root / "REVIEW_ZH.md")
exclusions = {"archive_manifest.json", "SHA256SUMS", "finalization_cost.json"}
files = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or ".git" in path.relative_to(root).parts:
        continue
    if path.parent == root and path.name in exclusions:
        continue
    files.append({
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    })
manifest = {
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "main_note": str(note),
    "main_note_sha256": sha256(note),
    "scope": "Two new papers; read-only literature, code inspection, and CPU archival. No model code execution, GPU, training, sampling, or new FID.",
    "original_source_hashes_verified": verified_sources,
    "git_internal_files_excluded": True,
    "full_reading_cpu_and_wall": "not measured",
    "files": files,
}
(root / "archive_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
cost = {
    "scope": "This finalization script only, through source checks, document copy, file hashing and manifest write; excludes later checksum and cost-file writes.",
    "wall_seconds": time.perf_counter() - wall_start,
    "cpu_process_seconds": time.process_time() - cpu_start,
    "gpu_calls": 0,
    "files_hashed": len(files),
    "note_control_character_check": "passed",
    "original_source_hashes_verified_count": len(verified_sources),
}
(root / "finalization_cost.json").write_text(json.dumps(cost, ensure_ascii=False, indent=2) + "\n")
checksums = [(item["sha256"], item["path"]) for item in files]
checksums.extend((sha256(root / name), name) for name in ("archive_manifest.json", "finalization_cost.json"))
(root / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for digest, name in checksums))
print(json.dumps({
    "main_note_sha256": sha256(note),
    "archive_manifest_sha256": sha256(root / "archive_manifest.json"),
    "sha256sums_sha256": sha256(root / "SHA256SUMS"),
    "finalization_cost": cost,
}, ensure_ascii=False, indent=2))
