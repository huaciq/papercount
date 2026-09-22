"""Batch existing single-image CLI with isolated outputs and incremental summaries."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def run_one(image: Path, input_dir: Path, output: Path, config: Path, rotate_ccw: bool, ground_truth: Path | None = None) -> dict:
    """Keep processing other images even if this image fails."""
    relative = image.relative_to(input_dir)
    target = output / "images" / relative.parent / relative.name
    target.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    row = {"image": relative.as_posix(), "output": target.relative_to(output).as_posix(),
           "rotation_ccw_degrees": 90 if rotate_ccw else 0}
    try:
        with (target / "cli.log").open("w", encoding="utf-8") as stream:
            command = [sys.executable, "-X", "utf8=0", str(ROOT / "main.py"),
                "--image", str(image), "--config", str(config), "--output", str(target)]
            if rotate_ccw:
                command.append("--rotate-ccw")
            if ground_truth is not None:
                command.extend(["--ground-truth", str(ground_truth)])
            process = subprocess.run(command,
                stdout=stream, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if process.returncode != 0:
            raise RuntimeError(f"CLI exit code {process.returncode}; see cli.log")
        with (target / "debug_info.json").open(encoding="utf-8") as stream:
            debug = json.load(stream)
        row.update(status=debug["status"], count=debug["count"],
            detected=len(debug["detected_peaks"]), recovered=len(debug["repaired_peaks"]),
            rejected=len(debug.get("rejected_tracks", [])), period=debug["period"],
            local_period=debug.get("median_local_period"), confidence=debug["confidence"],
            active_roi=debug.get("active_stack_roi", debug["roi"]),
            polarity=debug["gradient_polarity"], warnings=debug.get("warnings", []),
            suspicious_gaps=len(debug.get("spacing_warnings", [])))
        row.update(sequence_rescued=len(debug.get("sequence_rescued_tracks", [])),
                   sequence_suppressed=len(debug.get("sequence_suppressed_tracks", [])))
        reg = debug.get("regularity", {})
        row.update(regular_region_ratio=reg.get("region_ratios", {}).get("REGULAR", 0.0),
                   mixed_region_ratio=reg.get("region_ratios", {}).get("MIXED", 0.0),
                   irregular_region_ratio=reg.get("region_ratios", {}).get("IRREGULAR", 0.0),
                   period_assisted_recoveries=len(debug.get("regular_period_assisted_recoveries", [])),
                   geometry_only_recoveries=len(debug.get("geometry_only_recoveries", [])),
                   strong_suppressions=sum(1 for t in debug.get("sequence_suppressed_tracks", []) if t.get("reason") == "sequence_suppressed_short_gap"),
                   unresolved_regular_gaps=len(debug.get("unresolved_regular_gaps", [])),
                   initial_stack_roi=debug.get("raw_stack_mask", {}).get("thresholds", {}).get("initial_top"),
                   refined_stack_roi=debug.get("mask_refinement", {}).get("refined_top"),
                   invalid_selected=debug.get("invalid_selected_track_count", 0),
                   margin_selected=len(debug.get("margin_selected_track_ids", [])))
        if "ground_truth" in debug:
            row["ground_truth"] = debug["ground_truth"]
    except Exception as error:
        row.update(status="error", count=None, error=str(error))
    row["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    return row


def save_summary(output: Path, rows: list[dict], total: int, rotate_ccw: bool = False) -> None:
    """Write both machine-readable results and a clickable report."""
    ordered = sorted(rows, key=lambda row: row["image"])
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump({"total_images": total, "completed": len(rows), "results": ordered},
                  stream, ensure_ascii=False, indent=2, allow_nan=False)
    lines = ["# 批量检测结果", "", f"已处理 {len(rows)} / {total} 张图像。", "",
        ("输入图像先逆时针旋转 90°。" if rotate_ccw else "输入图像未旋转。") + "统一使用本目录 default.yaml 配置检测，没有按单图调参。证据不足时的 0 不表示没有纸。",
        "Confidence 是算法一致性评分，不是计数准确率。", "",
        "| 图像 | 计数 | 直接检测 | 恢复 | Regular | Irregular | 周期 px | Confidence | 状态 | 输出 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for row in ordered:
        period = f"{row['period']:.2f}" if row.get("period") is not None else "—"
        confidence = f"{row['confidence']:.3f}" if "confidence" in row else "—"
        directory = row["output"]
        links = f"[标注图](<{directory}/result_overlay.png>) · [调试数据](<{directory}/debug_info.json>)" if row["status"] != "error" else f"[错误日志](<{directory}/cli.log>)"
        lines.append(f"| {row['image']} | {row.get('count', '—')} | {row.get('detected', '—')} | {row.get('recovered', '—')} | {row.get('regular_region_ratio', 0):.3f} | {row.get('irregular_region_ratio', 0):.3f} | {period} | {confidence} | {row['status']} | {links} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--rotate-ccw", action="store_true", help="Rotate every input counter-clockwise 90 degrees")
    parser.add_argument("--ground-truth", type=Path, default=None, help="Optional ground_truth.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output",
                        help="Parent directory for timestamped batch outputs")
    args = parser.parse_args()
    input_dir = args.input.resolve()
    if not input_dir.is_dir() or args.workers < 1:
        parser.error("Input must be a directory and workers must be positive")
    images = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not images:
        parser.error("No supported images found")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / ("batch_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(exist_ok=False)
    config = output / "default.yaml"
    shutil.copy2(args.config, config)
    rows = []
    print("OUTPUT:", output, flush=True)
    save_summary(output, rows, len(images), args.rotate_ccw)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_one, path, input_dir, output, config, args.rotate_ccw, args.ground_truth) for path in images]
        for job in as_completed(jobs):
            row = job.result()
            rows.append(row)
            save_summary(output, rows, len(images), args.rotate_ccw)
            print(json.dumps({"progress": f"{len(rows)}/{len(images)}", **row}, ensure_ascii=True), flush=True)
    print("DONE:", output, flush=True)


if __name__ == "__main__":
    main()
