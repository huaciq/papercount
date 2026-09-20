"""Compare shadow/pre-verifier and active final-verifier counts with TXT truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_txt_ground_truth import evaluate


def summarize(rows: list[dict], field: str) -> dict:
    scored = [row for row in rows if row["ground_truth"] is not None]
    errors = [row[field] - row["ground_truth"] for row in scored]
    return {
        "exact_count": sum(error == 0 for error in errors),
        "exact_accuracy": sum(error == 0 for error in errors) / max(1, len(errors)),
        "mean_absolute_error": sum(abs(error) for error in errors) / max(1, len(errors)),
        "total_signed_error": sum(errors),
    }


def compare(input_dir: Path, shadow: Path, active: Path) -> dict:
    before = evaluate(input_dir, shadow)
    after = evaluate(input_dir, active)
    before_by_image = {row["image"]: row for row in before["rows"]}
    rows = []
    for active_row in after["rows"]:
        old = before_by_image[active_row["image"]]
        debug_path = active / "images" / active_row["image"] / "debug_info.json"
        debug = json.loads(debug_path.read_text(encoding="utf-8"))
        verifier = debug.get("final_track_verifier", {})
        rows.append({
            "image": active_row["image"],
            "ground_truth": active_row["expected_count"],
            "count_before": old["predicted_count"],
            "count_after": active_row["predicted_count"],
            "change": active_row["predicted_count"] - old["predicted_count"],
            "proposed_reject_count": verifier.get("proposed_reject_count", 0),
            "applied_reject_count": verifier.get("applied_reject_count", 0),
            "applied_reject_track_ids": verifier.get("applied_reject_track_ids", []),
        })
    return {
        "shadow_batch": str(shadow), "active_batch": str(active),
        "before": summarize(rows, "count_before"),
        "after": summarize(rows, "count_after"), "rows": rows,
    }


def markdown(report: dict) -> str:
    before, after = report["before"], report["after"]
    lines = [
        "# Final verifier comparison", "",
        f"- Before: exact {before['exact_count']}/{len(report['rows'])} "
        f"({before['exact_accuracy']:.2%}), MAE {before['mean_absolute_error']:.3f}, "
        f"signed error {before['total_signed_error']:+d}",
        f"- After: exact {after['exact_count']}/{len(report['rows'])} "
        f"({after['exact_accuracy']:.2%}), MAE {after['mean_absolute_error']:.3f}, "
        f"signed error {after['total_signed_error']:+d}", "",
        "| Image | GT | Before | After | Change | Applied reject IDs |", 
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        ids = ", ".join(row["applied_reject_track_ids"]) or "—"
        lines.append(f"| {row['image']} | {row['ground_truth']} | {row['count_before']} | "
                     f"{row['count_after']} | {row['change']:+d} | {ids} |")
    lines += ["", "Count equality alone does not establish track correctness; known false and missed "
              "tracks can cancel each other. See false_track_audit.md and annotation_mapping.md."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--shadow", required=True, type=Path)
    parser.add_argument("--active", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = (args.output or args.active).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = compare(args.input.resolve(), args.shadow.resolve(), args.active.resolve())
    (output / "verifier_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "verifier_comparison.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"before": report["before"], "after": report["after"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
