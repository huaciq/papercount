"""Validate completed positive batch outputs and compare counts without accuracy claims."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative", type=Path, required=True)
    parser.add_argument("--positive", type=Path, required=True)
    args = parser.parse_args()
    neg = json.loads((args.negative / "summary.json").read_text(encoding="utf-8"))
    pos = json.loads((args.positive / "summary.json").read_text(encoding="utf-8"))
    key = lambda r: r["image"].replace("\\", "/")
    negative = {key(r): r for r in neg["results"]}
    positive = {key(r): r for r in pos["results"]}
    assert pos["completed"] == pos["total_images"], "Positive batch incomplete"
    assert set(negative) == set(positive), "Input image sets differ"
    required = ("result_overlay.png", "debug_info.json", "positive_gradient.png",
                "negative_gradient.png", "strip_profiles.png", "profiles.npz", "render_metadata.json")
    rows, failures = [], []
    groups = defaultdict(list)
    for image, p in sorted(positive.items()):
        n = negative[image]
        folder = args.positive / p["output"]
        if p["status"] != "ok":
            failures.append({"image": image, "reason": p.get("error", p["status"])})
            continue
        assert all((folder / name).is_file() for name in required), f"Missing output: {image}"
        debug = json.loads((folder / "debug_info.json").read_text(encoding="utf-8"))
        assert debug["gradient_polarity"] == "positive", image
        assert debug["primary_evidence_polarity"] == "positive", image
        assert debug["track_identity"]["physical_paper_side"] == "below", image
        assert p["rotation_ccw_degrees"] == 90, image
        row = {"image": image, "negative": n["count"], "positive": p["count"],
               "delta": p["count"] - n["count"], "confidence": p["confidence"],
               "period": p["period"], "local_period": p.get("local_period"),
               "suspicious_gaps": p["suspicious_gaps"],
               "verifier_removed": debug["final_track_verifier"]["applied_reject_count"],
               "roi_y1": p["active_roi"]["y1"], "roi_y2": p["active_roi"]["y2"],
               "invalid_selected": p.get("invalid_selected", 0),
               "elapsed_seconds": p["elapsed_seconds"]}
        rows.append(row)
        groups[image.split("/")[0]].append(row)
    with (args.positive / "polarity_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    overview = {"total": pos["total_images"], "successful": len(rows), "failed": failures,
                "same_counts": sum(r["delta"] == 0 for r in rows),
                "changed_counts": sum(r["delta"] != 0 for r in rows),
                "mean_processing_seconds": round(statistics.mean(r["elapsed_seconds"] for r in rows), 2) if rows else None,
                "negative_batch": str(args.negative), "positive_batch": str(args.positive),
                "accuracy": None, "accuracy_reason": "No complete ground truth; paired counts are diagnostics only"}
    (args.positive / "polarity_comparison.json").write_text(
        json.dumps({"overview": overview, "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 正梯度主检测与上一批负梯度结果对照", "",
             f"完成 {len(rows)}/{pos['total_images']} 张，失败 {len(failures)} 张。全部逆时针旋转90°。",
             "正梯度采用多尺度候选、原生正梯度周期锁定、正梯度恢复及按纸身在下方进行末端验证。",
             f"同数 {overview['same_counts']} 张；不同数 {overview['changed_counts']} 张。单图平均处理 {overview['mean_processing_seconds']} 秒。",
             "没有完整真值；同数、差值和 Confidence 都不等于准确率。计数线现在对应上边缘，旧 PNG 的局部标注不能直接搬用。", "",
             "## 各纸种对照", "", "| 纸种 | 图数 | 负梯度计数（按文件名排序） | 正梯度计数（同序） |", "|---|---:|---|---|"]
    for name, items in sorted(groups.items()):
        lines.append(f"| {name} | {len(items)} | {', '.join(str(r['negative']) for r in items)} | {', '.join(str(r['positive']) for r in items)} |")
    lines += ["", "## 单图对照", "", "| 图片 | 负 | 正 | 差值 | 正分支Confidence | 可疑间隙 | 末端删线 | 输出 |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        target = args.positive / positive[r["image"]]["output"] / "result_overlay.png"
        lines.append(f"| {r['image']} | {r['negative']} | {r['positive']} | {r['delta']:+d} | {r['confidence']:.3f} | {r['suspicious_gaps']} | {r['verifier_removed']} | [结果图](<{target.as_posix()}>) |")
    if failures:
        lines += ["", "## 失败图", "", json.dumps(failures, ensure_ascii=False, indent=2)]
    (args.positive / "polarity_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(overview, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
