#!/usr/bin/env python3
"""Prepare explicit, reviewable research archive inputs. Never stages or commits.

Source selection is frozen in sources_plan.json. Repeated execution verifies old
copies and refreshes only the Git proposal/link report, which are snapshots.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "docs/data/raev2_guidance_final_sources_20260906"
PORTABLE = REPO / "docs/data/raev2_guidance_final_20260906"
SOURCE = Path("/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906")
MAX_FILE = 100_000
MAX_TOTAL = 4_000_000


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def identity(path: Path) -> dict:
    data = path.read_bytes()
    return {"size_bytes": len(data), "sha256": sha(data)}


def verify_portable() -> dict:
    path = PORTABLE / "manifest.json"
    value = json.loads(path.read_text())
    rows = value["copied_files"]
    for row in rows:
        current = identity(PORTABLE / row["package_relative_path"])
        assert current == {key: row[key] for key in current}, row
    assert len(rows) == value["copied_file_count"]
    assert sum(row["size_bytes"] for row in rows) == value["copied_total_bytes"]
    exporter = value["export_script"]
    assert identity(Path(exporter["logical_path"])) == {key: exporter[key] for key in ("size_bytes", "sha256")}
    return {"manifest_sha256": identity(path)["sha256"], "revision": value["revision"],
            "copied_files_verified": len(rows), "copied_bytes_verified": value["copied_total_bytes"],
            "large_assets_reopened": False, "scope": "copied bytes and bound exporter, not scientific revalidation"}


def prepare_sources() -> dict:
    plan_path = PACKAGE / "sources_plan.json"
    plan = json.loads(plan_path.read_text())
    for row in plan["repository_identical_sources"]:
        assert identity(REPO / row["repository_path"]) == {key: row[key] for key in ("size_bytes", "sha256")}, row
    rows = []
    copied = {}
    for row in plan["sources"]:
        source = SOURCE / row["source_relative_path"]
        data = source.read_bytes()
        assert len(data) <= MAX_FILE
        assert identity(source) == {key: row[key] for key in ("size_bytes", "sha256")}, str(source)
        target = PACKAGE / row["package_relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_bytes() == data, str(target)
        else:
            target.write_bytes(data)
        copied[str(target.relative_to(PACKAGE))] = len(data)
        rows.append({**row, "source_logical_path": str(source), "source_resolved_path": str(source.resolve())})
    assert sum(copied.values()) <= MAX_TOTAL
    result = {"schema": "raev2_final_sources_v1", "source_plan": identity(plan_path),
              "export_script": identity(Path(__file__)), "source_root": str(SOURCE),
              "source_file_count": len(rows), "unique_copied_file_count": len(copied),
              "unique_copied_bytes": sum(copied.values()), "files": rows,
              "repository_identical_sources": plan["repository_identical_sources"],
              "excluded": plan["excluded"], "large_assets_reopened": False,
              "research_goal_complete": False,
              "scope": "frozen selected provenance JSON and locally authored auxiliary scripts; dependencies remain at recorded locations"}
    out = PACKAGE / "manifest.json"
    if out.exists():
        assert json.loads(out.read_text()) == result, "immutable source manifest changed"
    else:
        dump(out, result)
    return result


def allowed(path: str) -> bool:
    p = Path(path)
    if path == "docs/RESEARCH_STATUS.md":
        return True
    if p.parent == Path("docs"):
        return p.name.startswith("RAEV2_") and p.name.endswith("_20260906_ZH.md")
    if len(p.parts) >= 3 and p.parts[:2] == ("docs", "data"):
        return p.parts[2].startswith("raev2_") and "20260906" in p.parts[2]
    if p.parent == Path("experiments"):
        return "raev2" in p.name and p.suffix == ".py"
    if p.parent == Path("tests"):
        return p.name.startswith("test_") and "raev2" in p.name and p.suffix == ".py"
    return False


def link_report() -> dict:
    checked = Counter()
    missing = []
    documents = sorted((REPO / "docs").rglob("*.md"))
    pattern = re.compile(r"(?<!!)\[[^\]\n]+\]\((<[^>\n]+>|[^)\n]+)\)")
    for doc in documents:
        contents = doc.read_text(errors="replace")
        for match in pattern.finditer(contents):
            raw = match.group(1).strip().strip("<>")
            if raw.startswith("#") or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", raw):
                continue
            # Markdown optional title, anchors and rendered local :line notation.
            raw = re.split(r'\s+["\']', raw, maxsplit=1)[0]
            local = unquote(urlsplit(raw).path)
            local = re.sub(r":\d+(?::\d+)?$", "", local)
            if not local or "\n" in local:
                continue
            target = Path(local)
            if not target.is_absolute():
                target = doc.parent / target
            target = target.resolve()
            kind = "repository" if target.is_relative_to(REPO) else "external"
            checked[kind] += 1
            if not target.exists():
                missing.append({"document": str(doc.relative_to(REPO)),
                                "line": contents[:match.start()].count("\n") + 1,
                                "target": raw, "resolved_target": str(target), "kind": kind})
    return {"documents_checked": len(documents), "links_checked": dict(checked),
            "missing_counts": dict(Counter(x["kind"] for x in missing)), "missing": missing,
            "scope": "inline Markdown destinations only; no network, reference links, raw code paths, heading anchors or asset contents checked"}


def git_proposal(portable: dict, sources: dict) -> dict:
    # These generated inventory files cannot recursively contain their own hashes.
    generated = [PACKAGE / name for name in ("validation.json", "proposed_git_paths.txt", "proposed_git_inventory.json")]
    for p in generated:
        if not p.exists():
            p.write_text("" if p.suffix == ".txt" else "{}\n")
    links = link_report()
    dump(PACKAGE / "validation.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
          "portable_archive": portable, "source_package": {key: sources[key] for key in ("source_file_count", "unique_copied_file_count", "unique_copied_bytes")},
          "documentation_links": links, "staged_or_committed": False})
    changed = subprocess.check_output(["git", "diff", "--name-only", "HEAD", "-z"], cwd=REPO).split(b"\0")
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=REPO).split(b"\0")
    all_paths = sorted({p.decode() for p in changed + untracked if p})
    selected = [p for p in all_paths if allowed(p)]
    ignored = [p for p in all_paths if not allowed(p)]
    rows = []
    for relative in selected:
        path = REPO / relative
        assert path.is_file() and not path.is_symlink(), relative
        item = {"path": relative}
        if path in generated:
            item["identity_scope"] = "generated inventory/validation; excluded from recursive hash accounting"
        else:
            item.update(identity(path))
        rows.append(item)
    result = {"schema": "raev2_explicit_git_proposal_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "base_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
              "scope": "exact current changed paths within named September 6 RAEv2 docs/data/source/tests; proposal only",
              "file_count": len(rows), "identified_bytes_excluding_generated_inventories": sum(x.get("size_bytes", 0) for x in rows),
              "files": rows, "out_of_scope_changed_paths": ignored, "staged_or_committed": False}
    dump(PACKAGE / "proposed_git_inventory.json", result)
    (PACKAGE / "proposed_git_paths.txt").write_text("\n".join(selected) + "\n")
    return {"proposed_files": len(rows), "identified_bytes": result["identified_bytes_excluding_generated_inventories"],
            "out_of_scope": ignored, "link_counts": links["missing_counts"]}


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    portable = verify_portable()
    sources = prepare_sources()
    result = git_proposal(portable, sources)
    print(json.dumps({"portable": portable, "sources": {k: sources[k] for k in ("source_file_count", "unique_copied_file_count", "unique_copied_bytes")}, "proposal": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
