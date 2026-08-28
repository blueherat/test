#!/usr/bin/env python3
"""Audit third-pool endpoint-label reliability without opening any model metric.

Allowed inputs are deliberately restricted to:
  * the three immutable reviewer locks;
  * the immutable adjudication and consensus locks;
  * the endpoint-only blind-review/adjudication packs.

The script never opens trajectories, primary features, visual-track features,
endpoint embeddings, metric scores, thresholds, ranks, or alerts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/home/zhoushunyu/eqvae")
ANNOTATIONS = ROOT / "experiments" / "annotations"
REVIEW_PACK = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_bad_good_third_pool_v1_blind_review_pack"
)
ADJUDICATION_PACK = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_bad_good_third_pool_v1_adjudication_pack"
)
REVIEW_LOCKS = {
    r: ANNOTATIONS / f"dit_bad_good_third_pool_reviewer_{r}_lock_v1"
    for r in (1, 2, 3)
}
ADJUDICATION_LOCK = ANNOTATIONS / "dit_bad_good_third_pool_adjudication_lock_v1"
CONSENSUS_LOCK = ANNOTATIONS / "dit_bad_good_third_pool_consensus_lock_v1"

COMPONENTS = [
    "global_blur",
    "local_blur",
    "soft_fusion_or_melting",
    "discrete_duplication_or_extra_part",
    "detachment_or_floating_part",
    "topology_or_attachment_error",
    "limb_or_object_misalignment",
    "texture_break",
    "other",
]

# Independent endpoint-only re-audit performed after the immutable review and
# adjudication locks existed. These judgments are descriptive QA only: they do
# not alter the locks and must not be used as a substitute confirmatory label set.
# The standard is the user's intended one: clear defects materially below the
# frozen class band count as bad, including unmistakable blur/fusion/misalignment;
# ordinary model softness, crop, pose, distance, and ambiguity do not.
VISUAL_REAUDIT = {
    "a_9ad471876dc990d2d654": ("borderline_mild", "moderate softness; coherent face"),
    "a_336f97fcc1178bf2dbcd": ("borderline_mild", "soft/low-resolution portrait"),
    "a_cce9b548639bc67a200d": ("acceptable_for_class_band", "coherent profile; ordinary rendering"),
    "a_515a4b503c1fd57fd5ee": ("clear_bad_missed_by_adjudication", "strong global smoothing and plastic blur"),
    "a_ef44037d7ee862b96b89": ("clear_bad_missed_by_adjudication", "blurred/melted eyes and head contour"),
    "a_627316462cd5d7537fff": ("borderline_mild", "low-resolution distant pose"),
    "a_4858b4716a422c5eb990": ("clear_bad_missed_by_adjudication", "strong global blur with lost facial texture"),
    "a_720cc0c5f7c3e89c2a5a": ("clear_bad_missed_by_adjudication", "hind-limb/topology fusion and misalignment"),
    "a_3f63cf46c6130a63d31b": ("borderline_mild", "distance/occlusion makes softness ambiguous"),
    "a_e468ca547b1f131f7431": ("confirmed_clear_bad", "duplicated/incoherent gymnast limbs"),
    "a_4874ca86b95c84b66485": ("borderline_mild", "soft portrait but coherent subject"),
    "a_599eddbdd5f736978b89": ("borderline_mild", "soft puppy portrait"),
    "a_3e1309c9e3c749510d84": ("confirmed_clear_bad", "grossly fused and stretched gymnast/body"),
    "a_90977ccfce178905e8c9": ("borderline_mild", "moderate blur; recognizable coherent pose"),
    "a_5e6c3da3330b864cc75b": ("borderline_mild", "distant/cropped dog with ordinary softness"),
    "a_ed27d667ce92e5820e7f": ("clear_bad_missed_by_adjudication", "clear global blur below class band"),
    "a_a2416447e741903e7dfe": ("clear_bad_missed_by_adjudication", "face and eyes globally blurred below class band"),
    "a_f25b0da489235c5aa7cd": ("confirmed_clear_bad", "detached floating ski/object"),
    "a_666f48e492a03a24e107": ("borderline_mild", "soft puppy portrait; structure remains coherent"),
    "a_d584026c314976576004": ("acceptable_for_class_band", "coherent portrait with normal detail"),
    "a_5312507c0cf7c057bf40": ("confirmed_clear_bad", "gross fused/duplicated dog anatomy"),
    "a_b9cd71c896ba08733d0a": ("borderline_mild", "old-photo softness and background blur"),
    "a_d9f66b1812eebd828d92": ("clear_bad_missed_by_adjudication", "severe global blur and body/background fusion"),
    "a_e471e50a51ed243ba5bd": ("confirmed_clear_bad", "missing/fused gymnast arms"),
    "a_baba4b1e0d90f094f07f": ("borderline_mild", "sleeping pose and shallow-focus softness"),
    "a_97b9afd957ef1a159782": ("borderline_mild", "ordinary low-resolution portrait"),
    "a_5912cf551725a801481f": ("borderline_mild", "oversaturated/painterly but coherent portrait"),
    "a_fc5049c19a28864f43b7": ("confirmed_clear_bad", "fused/duplicated gymnast limbs and torso"),
    "a_f6f00ed9c164e9e19318": ("borderline_mild", "slightly unusual body proportions but plausible"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def safe_div(num: float, den: float) -> float | None:
    return num / den if den else None


def cohen_kappa(a: Sequence[int], b: Sequence[int], categories: Sequence[int]) -> float | None:
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in categories)
    return safe_div(po - pe, 1.0 - pe)


def weighted_kappa(
    a: Sequence[int], b: Sequence[int], categories: Sequence[int]
) -> float | None:
    """Quadratic-weighted Cohen kappa."""
    n = len(a)
    if n == 0 or len(b) != n:
        return None
    max_gap = max(categories) - min(categories)
    if max_gap == 0:
        return None
    ca, cb = Counter(a), Counter(b)
    obs = 0.0
    exp = 0.0
    for x in categories:
        for y in categories:
            weight = ((x - y) / max_gap) ** 2
            obs += weight * sum(u == x and v == y for u, v in zip(a, b)) / n
            exp += weight * (ca[x] / n) * (cb[y] / n)
    return 1.0 - obs / exp if exp else None


def fleiss_kappa(rows: Sequence[Sequence[int]], categories: Sequence[int]) -> float | None:
    """Fleiss kappa for a fixed number of raters per item."""
    if not rows:
        return None
    n_raters = len(rows[0])
    n_items = len(rows)
    if n_raters < 2 or any(len(row) != n_raters for row in rows):
        return None
    item_agreement = []
    totals = Counter()
    for row in rows:
        counts = Counter(row)
        totals.update(row)
        item_agreement.append(
            (sum(counts[c] ** 2 for c in categories) - n_raters)
            / (n_raters * (n_raters - 1))
        )
    p_bar = sum(item_agreement) / n_items
    denom = n_items * n_raters
    p_e = sum((totals[c] / denom) ** 2 for c in categories)
    return safe_div(p_bar - p_e, 1.0 - p_e)


def gwet_ac1_binary(a: Sequence[int], b: Sequence[int]) -> float | None:
    """Gwet AC1, included because clear-bad prevalence is extremely low."""
    n = len(a)
    if n == 0 or len(b) != n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    p = (sum(a) + sum(b)) / (2 * n)
    pe = 2 * p * (1 - p)
    return safe_div(po - pe, 1 - pe)


def pairwise_binary(a: Sequence[int], b: Sequence[int]) -> dict:
    both_pos = sum(x == 1 and y == 1 for x, y in zip(a, b))
    a_only = sum(x == 1 and y == 0 for x, y in zip(a, b))
    b_only = sum(x == 0 and y == 1 for x, y in zip(a, b))
    both_neg = sum(x == 0 and y == 0 for x, y in zip(a, b))
    n = len(a)
    return {
        "both_positive": both_pos,
        "a_only_positive": a_only,
        "b_only_positive": b_only,
        "both_negative": both_neg,
        "raw_agreement": (both_pos + both_neg) / n,
        "cohen_kappa": cohen_kappa(a, b, [0, 1]),
        "gwet_ac1": gwet_ac1_binary(a, b),
        "positive_agreement": safe_div(2 * both_pos, 2 * both_pos + a_only + b_only),
        "negative_agreement": safe_div(2 * both_neg, 2 * both_neg + a_only + b_only),
        "positive_jaccard": safe_div(both_pos, both_pos + a_only + b_only),
    }


def pixel_digest(path: Path) -> str:
    with Image.open(path) as image:
        normalized = image.convert("RGB")
        payload = (
            normalized.width.to_bytes(4, "big")
            + normalized.height.to_bytes(4, "big")
            + normalized.tobytes()
        )
    return hashlib.sha256(payload).hexdigest()


def load_reviewer(reviewer: int) -> dict[int, dict]:
    order = read_csv(REVIEW_PACK / f"reviewer_{reviewer}" / "review_order.csv")
    votes = read_csv(REVIEW_LOCKS[reviewer] / "review_rows.csv")
    order_by_id = {row["review_id"]: row for row in order}
    vote_by_id = {row["review_id"]: row for row in votes}
    if len(order) != 1800 or len(votes) != 1800:
        raise ValueError(f"reviewer {reviewer}: expected 1800 order/vote rows")
    if set(order_by_id) != set(vote_by_id):
        raise ValueError(f"reviewer {reviewer}: order/vote review_id mismatch")
    result: dict[int, dict] = {}
    for review_id, order_row in order_by_id.items():
        native_name = Path(order_row["native_image_relative_path"]).name
        sample_index = int(native_name.removeprefix("endpoint_").removesuffix(".png"))
        vote = vote_by_id[review_id]
        severity = int(vote["severity"])
        components = {name: int(vote[name]) for name in COMPONENTS}
        none = int(vote["none"])
        if severity not in (0, 1, 2, 3):
            raise ValueError(f"reviewer {reviewer}: invalid severity")
        if none != int(not any(components.values())):
            raise ValueError(f"reviewer {reviewer}: inconsistent none field")
        result[sample_index] = {
            "class_id": int(order_row["class_id"]),
            "review_id": review_id,
            "severity": severity,
            "components": components,
        }
    if set(result) != set(range(1800)):
        raise ValueError(f"reviewer {reviewer}: incomplete sample index axis")
    return result


def map_adjudication_cases() -> dict[str, int]:
    native_by_pixel = defaultdict(list)
    for sample_index in range(1800):
        path = REVIEW_PACK / "native" / f"endpoint_{sample_index:04d}.png"
        native_by_pixel[pixel_digest(path)].append(sample_index)
    mapping = {}
    for row in read_csv(ADJUDICATION_PACK / "adjudication_cases.csv"):
        path = ADJUDICATION_PACK / row["native_image_relative_path"]
        matches = native_by_pixel[pixel_digest(path)]
        if len(matches) != 1:
            raise ValueError(f"case {row['case_id']}: expected one pixel-identical endpoint")
        mapping[row["case_id"]] = matches[0]
    return mapping


def counts_by_class(values: dict[int, int], class_by_sample: dict[int, int]) -> dict:
    out = {}
    for class_id in sorted(set(class_by_sample.values())):
        sample_ids = [i for i, c in class_by_sample.items() if c == class_id]
        count = sum(values[i] for i in sample_ids)
        out[str(class_id)] = {"count": count, "rate": count / len(sample_ids), "n": len(sample_ids)}
    return out


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{100 * value:.2f}%"


def num(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def build_contact_sheet(
    case_rows: list[dict], decisions: dict[str, str], output: Path, columns: int = 5
) -> None:
    thumb_w, thumb_h, caption_h = 256, 256, 48
    rows = math.ceil(len(case_rows) / columns)
    canvas = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + caption_h)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, row in enumerate(case_rows):
        x = (idx % columns) * thumb_w
        y = (idx // columns) * (thumb_h + caption_h)
        path = ADJUDICATION_PACK / row["native_image_relative_path"]
        with Image.open(path) as image:
            canvas.paste(image.convert("RGB"), (x, y))
        action = decisions[row["case_id"]]
        color = (160, 0, 0) if action == "retain_clear_bad" else (0, 80, 0)
        draw.rectangle((x, y, x + thumb_w - 1, y + thumb_h - 1), outline=color, width=4)
        draw.text((x + 4, y + thumb_h + 3), row["case_id"], fill="black", font=font)
        draw.text((x + 4, y + thumb_h + 19), action, fill=color, font=font)
        draw.text((x + 4, y + thumb_h + 34), f"class {row['class_id']}", fill="black", font=font)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "audits" / "dit_bad_good_third_pool_label_reliability_v1",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reviewers = {r: load_reviewer(r) for r in (1, 2, 3)}
    class_by_sample = {i: reviewers[1][i]["class_id"] for i in range(1800)}
    for r in (2, 3):
        if {i: reviewers[r][i]["class_id"] for i in range(1800)} != class_by_sample:
            raise ValueError("class assignments differ across reviewer packs")

    severity = {r: [reviewers[r][i]["severity"] for i in range(1800)] for r in (1, 2, 3)}
    clear_bad = {r: [int(x >= 2) for x in severity[r]] for r in (1, 2, 3)}
    clean = {r: [int(x == 0) for x in severity[r]] for r in (1, 2, 3)}

    pairwise = {}
    for a, b in ((1, 2), (1, 3), (2, 3)):
        pairwise[f"reviewer_{a}_vs_{b}"] = {
            "four_level": {
                "raw_agreement": sum(x == y for x, y in zip(severity[a], severity[b])) / 1800,
                "cohen_kappa_unweighted": cohen_kappa(severity[a], severity[b], [0, 1, 2, 3]),
                "cohen_kappa_quadratic_weighted": weighted_kappa(
                    severity[a], severity[b], [0, 1, 2, 3]
                ),
            },
            "clear_bad_binary": pairwise_binary(clear_bad[a], clear_bad[b]),
            "clean_binary": pairwise_binary(clean[a], clean[b]),
        }

    per_class_agreement = {}
    for class_id in sorted(set(class_by_sample.values())):
        ids = [i for i in range(1800) if class_by_sample[i] == class_id]
        class_pairs = {}
        for a, b in ((1, 2), (1, 3), (2, 3)):
            sa = [severity[a][i] for i in ids]
            sb = [severity[b][i] for i in ids]
            ba = [clear_bad[a][i] for i in ids]
            bb = [clear_bad[b][i] for i in ids]
            class_pairs[f"reviewer_{a}_vs_{b}"] = {
                "four_level_kappa": cohen_kappa(sa, sb, [0, 1, 2, 3]),
                "binary_kappa": cohen_kappa(ba, bb, [0, 1]),
                "binary_positive_agreement": pairwise_binary(ba, bb)["positive_agreement"],
                "both_positive": sum(x and y for x, y in zip(ba, bb)),
            }
        per_class_agreement[str(class_id)] = class_pairs

    severity_rows = [[severity[r][i] for r in (1, 2, 3)] for i in range(1800)]
    clear_bad_rows = [[clear_bad[r][i] for r in (1, 2, 3)] for i in range(1800)]
    all_three_four_exact = sum(len(set(row)) == 1 for row in severity_rows) / 1800
    all_three_clear_bad_exact = sum(len(set(row)) == 1 for row in clear_bad_rows) / 1800

    reviewer_summary = {}
    for r in (1, 2, 3):
        per_class = {}
        for class_id in sorted(set(class_by_sample.values())):
            ids = [i for i in range(1800) if class_by_sample[i] == class_id]
            per_class[str(class_id)] = {
                "severity_counts": dict(sorted(Counter(severity[r][i] for i in ids).items())),
                "clear_bad_count": sum(clear_bad[r][i] for i in ids),
                "clear_bad_rate": sum(clear_bad[r][i] for i in ids) / len(ids),
            }
        component_counts = {
            component: sum(reviewers[r][i]["components"][component] for i in range(1800))
            for component in COMPONENTS
        }
        reviewer_summary[f"reviewer_{r}"] = {
            "severity_counts": dict(sorted(Counter(severity[r]).items())),
            "clear_bad_count": sum(clear_bad[r]),
            "clear_bad_rate": sum(clear_bad[r]) / 1800,
            "nonzero_severity_count": sum(x > 0 for x in severity[r]),
            "nonzero_severity_rate": sum(x > 0 for x in severity[r]) / 1800,
            "per_class": per_class,
            "component_counts": component_counts,
        }

    component_summary = {}
    for component in COMPONENTS:
        by_reviewer = {
            r: [reviewers[r][i]["components"][component] for i in range(1800)]
            for r in (1, 2, 3)
        }
        consensus = {
            i: int(sum(by_reviewer[r][i] for r in (1, 2, 3)) >= 2)
            for i in range(1800)
        }
        component_summary[component] = {
            "counts_by_reviewer": {f"reviewer_{r}": sum(by_reviewer[r]) for r in (1, 2, 3)},
            "majority_count": sum(consensus.values()),
            "majority_by_class": counts_by_class(consensus, class_by_sample),
            "pairwise_positive_intersections": {
                f"reviewer_{a}_vs_{b}": sum(
                    by_reviewer[a][i] and by_reviewer[b][i] for i in range(1800)
                )
                for a, b in ((1, 2), (1, 3), (2, 3))
            },
        }

    raw_majority = {
        i for i in range(1800) if sum(clear_bad[r][i] for r in (1, 2, 3)) >= 2
    }
    case_to_sample = map_adjudication_cases()
    if set(case_to_sample.values()) != raw_majority or len(raw_majority) != 29:
        raise ValueError("adjudication pack does not exactly match the 29 raw-majority cases")
    decisions = {
        row["case_id"]: row["action"]
        for row in read_csv(ADJUDICATION_LOCK / "adjudication_decisions.csv")
    }
    if set(decisions) != set(case_to_sample):
        raise ValueError("adjudication decision/case mismatch")
    if set(VISUAL_REAUDIT) != set(case_to_sample):
        raise ValueError("independent visual re-audit must cover exactly all 29 cases")

    sample_to_case = {sample: case for case, sample in case_to_sample.items()}
    raw_case_rows = []
    for sample_index in sorted(raw_majority):
        case_id = sample_to_case[sample_index]
        component_votes = {
            component: sum(
                reviewers[r][sample_index]["components"][component] for r in (1, 2, 3)
            )
            for component in COMPONENTS
        }
        raw_case_rows.append(
            {
                "sample_index": sample_index,
                "global_seed": 250 + sample_index // 3,
                "class_id": class_by_sample[sample_index],
                "case_id": case_id,
                "severities": [severity[r][sample_index] for r in (1, 2, 3)],
                "clear_bad_votes": sum(clear_bad[r][sample_index] for r in (1, 2, 3)),
                "majority_components": [k for k, v in component_votes.items() if v >= 2],
                "component_vote_counts": component_votes,
                "adjudication": decisions[case_id],
                "independent_visual_reaudit": VISUAL_REAUDIT[case_id][0],
                "independent_visual_reason": VISUAL_REAUDIT[case_id][1],
            }
        )

    final_retained = {
        row["sample_index"] for row in raw_case_rows if row["adjudication"] == "retain_clear_bad"
    }
    consensus_rows = read_csv(CONSENSUS_LOCK / "consensus_rows.csv")
    consensus_final = {
        int(row["sample_index"])
        for row in consensus_rows
        if row["final_severity"] == "clear_bad"
    }
    if consensus_final != final_retained:
        raise ValueError("final consensus clear-bad rows disagree with adjudication")
    visual_counts = Counter(status for status, _ in VISUAL_REAUDIT.values())
    visually_clear = {
        case_to_sample[case_id]
        for case_id, (status, _) in VISUAL_REAUDIT.items()
        if status in {"confirmed_clear_bad", "clear_bad_missed_by_adjudication"}
    }

    result = {
        "scope_guard": {
            "opened": [
                "three reviewer locks",
                "adjudication lock",
                "consensus lock",
                "endpoint-only blind review pack",
                "raw-majority-only adjudication pack",
            ],
            "explicitly_not_opened": [
                "trajectories/intermediate states",
                "primary metric features/scores",
                "visual-track features/scores",
                "endpoint embeddings/distances",
                "thresholds/ranks/alerts",
            ],
        },
        "grain": {"trajectory_count": 1800, "raters_per_trajectory": 3, "class_count": 3},
        "reviewer_summary": reviewer_summary,
        "agreement": {
            "all_three_four_level_exact": all_three_four_exact,
            "all_three_clear_bad_binary_exact": all_three_clear_bad_exact,
            "fleiss_kappa_four_level": fleiss_kappa(severity_rows, [0, 1, 2, 3]),
            "fleiss_kappa_clear_bad_binary": fleiss_kappa(clear_bad_rows, [0, 1]),
            "pairwise": pairwise,
            "per_class_pairwise": per_class_agreement,
        },
        "positive_sets": {
            "raw_majority_clear_bad_count": len(raw_majority),
            "unanimous_clear_bad_count": sum(sum(row) == 3 for row in clear_bad_rows),
            "exactly_two_clear_bad_votes_count": sum(sum(row) == 2 for row in clear_bad_rows),
            "at_least_one_clear_bad_vote_count": sum(any(row) for row in clear_bad_rows),
            "final_retained_count": len(final_retained),
            "downgraded_count": len(raw_majority - final_retained),
            "raw_majority_by_class": counts_by_class(
                {i: int(i in raw_majority) for i in range(1800)}, class_by_sample
            ),
            "final_retained_by_class": counts_by_class(
                {i: int(i in final_retained) for i in range(1800)}, class_by_sample
            ),
            "adjudication_retention_rate_overall": len(final_retained) / len(raw_majority),
            "adjudication_retention_rate_by_class": {
                str(class_id): safe_div(
                    sum(i in final_retained for i in raw_majority if class_by_sample[i] == class_id),
                    sum(class_by_sample[i] == class_id for i in raw_majority),
                )
                for class_id in sorted(set(class_by_sample.values()))
            },
        },
        "components": component_summary,
        "raw_majority_cases": raw_case_rows,
        "independent_visual_reaudit": {
            "counts": dict(sorted(visual_counts.items())),
            "visually_clear_count": len(visually_clear),
            "final_six_false_positive_count_in_reaudit": len(final_retained - visually_clear),
            "clear_bad_downgraded_count_in_reaudit": len(visually_clear - final_retained),
            "status": "DESCRIPTIVE_SINGLE_AUDITOR_QA_NOT_A_NEW_LABEL_LOCK",
        },
    }

    with (args.output_dir / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")

    pair_rows = []
    for key, values in pairwise.items():
        b = values["clear_bad_binary"]
        f = values["four_level"]
        pair_rows.append(
            f"| {key} | {pct(f['raw_agreement'])} | {num(f['cohen_kappa_unweighted'])} | "
            f"{num(f['cohen_kappa_quadratic_weighted'])} | {pct(b['raw_agreement'])} | "
            f"{num(b['cohen_kappa'])} | {pct(b['positive_agreement'])} | "
            f"{b['both_positive']} / {b['a_only_positive']} / {b['b_only_positive']} |"
        )

    reviewer_rows = []
    for r in (1, 2, 3):
        x = reviewer_summary[f"reviewer_{r}"]
        reviewer_rows.append(
            f"| reviewer {r} | {x['severity_counts']} | {x['clear_bad_count']} "
            f"({pct(x['clear_bad_rate'])}) | {x['nonzero_severity_count']} "
            f"({pct(x['nonzero_severity_rate'])}) |"
        )

    component_rows = []
    for component, x in component_summary.items():
        c = x["counts_by_reviewer"]
        component_rows.append(
            f"| {component} | {c['reviewer_1']} | {c['reviewer_2']} | "
            f"{c['reviewer_3']} | {x['majority_count']} |"
        )

    class_agreement_rows = []
    for class_id, pairs in per_class_agreement.items():
        for pair_name, values in pairs.items():
            class_agreement_rows.append(
                f"| {class_id} | {pair_name} | {num(values['four_level_kappa'])} | "
                f"{num(values['binary_kappa'])} | {pct(values['binary_positive_agreement'])} | "
                f"{values['both_positive']} |"
            )

    missed_rows = [
        row for row in raw_case_rows
        if row["independent_visual_reaudit"] == "clear_bad_missed_by_adjudication"
    ]
    missed_lines = [
        f"- `{row['case_id']}` (sample {row['sample_index']}, class {row['class_id']}): "
        f"{row['independent_visual_reason']}."
        for row in missed_rows
    ]

    report = f"""# Third-pool endpoint label reliability audit

This is an external-label quality audit, not a proposed detector. It opened only
the three endpoint-only review locks, adjudication/consensus locks, and endpoint
PNGs. It did not open any trajectory, model-derived feature, embedding, score,
threshold, rank, or alert.

## Dataset and grain

- 1,800 frozen-model endpoints: 600 each for ImageNet classes 207, 602, and 795.
- Three independent endpoint-only severity ratings per image on the ordinal scale
  0/1/2/3. For the binary audit, `clear bad = severity >= 2`.
- 29 images had at least two clear-bad votes. Conservative adjudication retained
  6 and downgraded 23; the frozen final consensus matches those decisions exactly.

## Rating prevalence

| Reviewer | Severity counts | Clear bad | Any imperfection (severity > 0) |
|---|---:|---:|---:|
{chr(10).join(reviewer_rows)}

The spread in positive prevalence is material: reviewer 3 calls far more clear
bad cases than reviewer 2. Agreement must therefore be interpreted with
positive agreement and kappa, not overall agreement alone.

## Agreement

| Pair | Four-level exact | Four-level κ | Quadratic κ | Binary exact | Binary κ | Positive agreement | Both / A-only / B-only |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(pair_rows)}

- All-three exact agreement: {pct(all_three_four_exact)} on four levels and
  {pct(all_three_clear_bad_exact)} after clear-bad binarization.
- Fleiss κ: {num(result['agreement']['fleiss_kappa_four_level'])} on four levels;
  {num(result['agreement']['fleiss_kappa_clear_bad_binary'])} for clear bad.
- Raw-majority clear bad: {len(raw_majority)}; unanimous clear bad:
  {result['positive_sets']['unanimous_clear_bad_count']}; exactly two votes:
  {result['positive_sets']['exactly_two_clear_bad_votes_count']}; at least one
  clear-bad vote: {result['positive_sets']['at_least_one_clear_bad_vote_count']}.

### Agreement by class

| Class | Pair | Four-level κ | Binary κ | Positive agreement | Both positive |
|---|---|---:|---:|---:|---:|
{chr(10).join(class_agreement_rows)}

## Component usage

| Component | Reviewer 1 | Reviewer 2 | Reviewer 3 | 2-of-3 majority |
|---|---:|---:|---:|---:|
{chr(10).join(component_rows)}

## Interpretation

- High overall agreement is dominated by the large negative class. The scarce
  positive labels have substantially weaker agreement and reviewer-specific
  thresholds.
- The 29-case majority set is suitable as a high-precision *screening set*, but
  adjudication removed 79.3% of it. All 23 downgrades are class 207; retention was
  1/24 (4.2%) for class 207 versus 4/4 and 1/1 for classes 602 and 795. This is a
  phenotype/class-dependent policy, not a neutral uniform tightening.
- Because adjudication was downgrade-only, false positives were reduced at the
  deliberate cost of unmeasured false negatives. The final six cannot establish
  recall, and a gate requiring 30 final clear-bad cases cannot be evaluated from
  this pool.

## Independent visual re-audit

All 29 raw-majority images were inspected at native 256x256 resolution, with
multiple frozen class-207 reference grids used to calibrate ordinary model
quality. This was a single-auditor QA pass, so it is evidence about label quality,
not a replacement immutable truth set.

- All final six remain clearly bad: no obvious adjudication false positive was
  found.
- Seven of the 23 downgrades still meet the user's intended clear-bad standard.
  This gives 13 visually clear cases in the 29-case screening set, 14 borderline
  cases, and 2 acceptable-for-class cases.
- Therefore the final six are **high precision but over-conservative** for the
  stated target. The missed clear cases are:

{chr(10).join(missed_lines)}

The strongest internal warning is `a_ed27d667ce92e5820e7f`: all three reviewers
assigned severity >=2, yet downgrade-only adjudication removed it. The image is
visibly globally blurred relative to the class band. This shows that adjudication
silently changed the estimand from "obvious below-model-average blur/fusion/
misalignment" toward "only catastrophic structural corruption."

## Recommended next-round rubric changes

1. Freeze 8-12 positive and 8-12 negative anchors per class before review. Include
   three boundary pairs for global blur: ordinary softness, clear below-band blur,
   and catastrophic melting.
2. Split the decision into two questions: `(a)` is there a visible defect, and
   `(b)` is it materially below the frozen class band? Severity 2 requires yes/yes.
3. For every severity 2/3, require a component and a one-line localization. For
   every downgrade of a unanimous bad vote, require two adjudicators or a written
   exception tied to a frozen anchor.
4. Run a 60-image calibration round before the next pool. Do not start production
   labeling until each pair reaches positive agreement >=60% and binary kappa
   >=0.50 on the calibration set; reconcile component definitions first.
5. Use stratified adjudication: expose an equal-sized blinded sample of majority
   negatives alongside majority positives. The current downgrade-only design can
   estimate precision but cannot reveal false negatives.
6. Keep endpoint labels strictly external. They evaluate whether an internal
   trajectory statistic works; FID/embeddings and these labels are not themselves
   candidate intervention signals.
"""
    with (args.output_dir / "AUDIT_REPORT.md").open("w", encoding="utf-8") as handle:
        handle.write(report)

    case_rows = read_csv(ADJUDICATION_PACK / "adjudication_cases.csv")
    build_contact_sheet(case_rows[:15], decisions, args.output_dir / "raw_majority_cases_01.png")
    build_contact_sheet(case_rows[15:], decisions, args.output_dir / "raw_majority_cases_02.png")
    retained_rows = [row for row in case_rows if decisions[row["case_id"]] == "retain_clear_bad"]
    build_contact_sheet(retained_rows, decisions, args.output_dir / "final_retained_cases.png", columns=3)

    print(json.dumps({
        "output_dir": str(args.output_dir),
        "raw_majority_count": len(raw_majority),
        "final_retained_count": len(final_retained),
        "fleiss_kappa_four_level": result["agreement"]["fleiss_kappa_four_level"],
        "fleiss_kappa_clear_bad_binary": result["agreement"]["fleiss_kappa_clear_bad_binary"],
    }, indent=2))


if __name__ == "__main__":
    main()
