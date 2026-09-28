"""Build one attribution table from completed front-end ablation batches."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def summarize(label: str, batch: Path) -> dict:
    comparison = json.loads((batch/"ground_truth_comparison.json").read_text(encoding="utf-8"))
    summary_csv = batch/"summary.csv"
    times = []
    if summary_csv.exists():
        with summary_csv.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                value = row.get("elapsed_seconds")
                if value:
                    times.append(float(value))
    else:
        summary_json = batch/"summary.json"
        if summary_json.exists():
            saved = json.loads(summary_json.read_text(encoding="utf-8"))
            times.extend(float(row["elapsed_seconds"]) for row in saved.get("results", [])
                         if row.get("elapsed_seconds") is not None)
    errors = [row for row in comparison["rows"] if row["error"]]
    return {"variant": label, "batch": str(batch),
            "exact_count": comparison["exact_count"],
            "truth_count": comparison["matched_truth_count"],
            "exact_accuracy": comparison["exact_accuracy"],
            "mae": comparison["mean_absolute_error"],
            "signed_error": comparison["total_signed_error"],
            "mean_elapsed_seconds": sum(times)/len(times) if times else None,
            "total_elapsed_seconds": sum(times) if times else None,
            "incorrect_images": [{"image": r["image"], "error": r["error"]} for r in errors]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", nargs=2, action="append", required=True, metavar=("LABEL", "BATCH"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = [summarize(label, Path(batch).resolve()) for label, batch in args.case]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"ablation_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Front-end ablation", "",
             "Backend and count truth set are fixed. The 16-image set was used during earlier tuning, so this is a regression/attribution result, not an independent generalization estimate.", "",
             "| variant | exact | accuracy | MAE | signed error | mean seconds/image | incorrect images |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        bad = ", ".join(f"{x['image']} ({x['error']:+d})" for x in row["incorrect_images"]) or "none"
        elapsed = "n/a" if row["mean_elapsed_seconds"] is None else f"{row['mean_elapsed_seconds']:.2f}"
        lines.append(f"| {row['variant']} | {row['exact_count']}/{row['truth_count']} | "
                     f"{row['exact_accuracy']:.2%} | {row['mae']:.3f} | {row['signed_error']:+d} | {elapsed} | {bad} |")
    (args.output/"ablation_summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
