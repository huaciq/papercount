"""Compare batch detector counts with per-image TXT ground truth files.

Example:
    python scripts/evaluate_txt_ground_truth.py \
        --input F:/纸张/数据/pictures \
        --batch output_batch_20260914_183122_595135

The script writes ``ground_truth_comparison.json`` and a human-readable
``ground_truth_comparison.md`` beside the batch result.  It never modifies
the source images or truth files.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_counter.ground_truth import load_count_text


IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# These aliases preserve the filenames supplied with the current dataset.  A
# report marks an alias match so a typo or a truth/image pairing mistake is
# visible instead of being silently treated as an exact filename match.
ALIASES = {
    "白底白板3.bmp": "白板白底3真值.txt",
    "金卡纸2.bmp": "金卡纸-2.txt",
    "灰底白板1.bmp": "灰底白板1 - 副本.txt",
}


def _normalized(value: str) -> str:
    return re.sub(r"[\s_\-—–]+", "", value).lower()


def find_truth(image: Path, truth_files: list[Path]) -> tuple[Path | None, str, str | None]:
    """Find a truth file and return (path, match_kind, warning)."""
    exact = image.with_suffix(".txt")
    candidates = [
        exact,
        image.with_name(image.stem + "真值.txt"),
    ]
    for candidate in candidates:
        if candidate in truth_files:
            return candidate, "exact", None
    alias_name = ALIASES.get(image.name)
    if alias_name:
        alias = image.with_name(alias_name)
        if alias in truth_files:
            return alias, "alias", f"truth filename differs from image filename: {alias.name}"

    # Last-resort normalized stem matching, useful for harmless spaces or
    # punctuation differences.  Keep it unique; ambiguous matches are left
    # unmatched and reported for manual correction.
    key = _normalized(image.stem)
    normalized = []
    for truth in truth_files:
        truth_key = _normalized(truth.stem.removesuffix("真值"))
        if truth_key == key:
            normalized.append(truth)
    if len(normalized) == 1:
        return normalized[0], "normalized", f"normalized filename match: {normalized[0].name}"
    return None, "missing", "no unique truth file matched"


def read_prediction(batch_dir: Path, image_name: str) -> int:
    debug = batch_dir / "images" / image_name / "debug_info.json"
    with debug.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if "count" in data:
        return int(data["count"])
    # Current debug_info stores the public result under this stable key.  The
    # fallback keeps evaluation compatible with older batch outputs.
    for key in ("predicted_count", "sheet_count"):
        if key in data:
            return int(data[key])
    raise KeyError(f"No count found in {debug}")


def evaluate(input_dir: Path, batch_dir: Path) -> dict:
    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    truth_files = sorted(input_dir.glob("*.txt"))
    rows = []
    for image in images:
        truth_path, match_kind, warning = find_truth(image, truth_files)
        predicted = read_prediction(batch_dir, image.name)
        expected = load_count_text(truth_path) if truth_path else None
        error = None if expected is None else predicted - expected
        rows.append({
            "image": image.name,
            "truth_file": truth_path.name if truth_path else None,
            "match_kind": match_kind,
            "mapping_warning": warning,
            "expected_count": expected,
            "predicted_count": predicted,
            "error": error,
            "absolute_error": abs(error) if error is not None else None,
            "exact": error == 0 if error is not None else None,
        })

    scored = [row for row in rows if row["error"] is not None]
    exact = sum(row["exact"] for row in scored)
    absolute_error = sum(row["absolute_error"] for row in scored)
    signed_error = sum(row["error"] for row in scored)
    summary = {
        "input_dir": str(input_dir),
        "batch_dir": str(batch_dir),
        "image_count": len(images),
        "matched_truth_count": len(scored),
        "missing_truth_count": len(images) - len(scored),
        "exact_count": exact,
        "exact_accuracy": exact / len(scored) if scored else None,
        "mean_absolute_error": absolute_error / len(scored) if scored else None,
        "total_signed_error": signed_error,
        "all_overcount": bool(scored) and all(row["error"] >= 0 for row in scored),
        "rows": rows,
    }
    return summary


def render_markdown(report: dict) -> str:
    pct = report["exact_accuracy"]
    pct_text = "n/a" if pct is None else f"{pct * 100:.2f}%"
    mae = report["mean_absolute_error"]
    mae_text = "n/a" if mae is None else f"{mae:.2f}"
    lines = [
        "# Ground-truth comparison",
        "",
        f"- Images: {report['image_count']}",
        f"- Matched TXT truth: {report['matched_truth_count']}",
        f"- Exact-count accuracy: {pct_text} ({report['exact_count']}/{report['matched_truth_count']})",
        f"- Mean absolute count error: {mae_text} sheets/image",
        f"- Total signed error (predicted - truth): {report['total_signed_error']}",
        f"- All scored errors are overcounts: {'yes' if report['all_overcount'] else 'no'}",
        "",
        "| Image | Truth file | Truth | Detected | Error | Match |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        truth = "—" if row["expected_count"] is None else str(row["expected_count"])
        detected = str(row["predicted_count"])
        error = "—" if row["error"] is None else f"{row['error']:+d}"
        match = row["match_kind"]
        if row["mapping_warning"]:
            match += " (verify)"
        lines.append(f"| {row['image']} | {row['truth_file'] or '—'} | {truth} | {detected} | {error} | {match} |")
    lines += [
        "",
        "The TXT files contain sheet counts only, so this report evaluates count accuracy; it does not compute edge-level precision/recall.",
        "Alias/normalized filename matches are marked `verify` and should be confirmed before using them as a benchmark.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="directory containing images and TXT truth files")
    parser.add_argument("--batch", required=True, type=Path, help="batch output directory")
    parser.add_argument("--output", type=Path, help="report directory; defaults to --batch")
    args = parser.parse_args()
    input_dir = args.input.resolve()
    batch_dir = args.batch.resolve()
    output_dir = (args.output or batch_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = evaluate(input_dir, batch_dir)
    (output_dir / "ground_truth_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "ground_truth_comparison.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in (
        "image_count", "matched_truth_count", "exact_count", "exact_accuracy",
        "mean_absolute_error", "total_signed_error", "all_overcount",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
