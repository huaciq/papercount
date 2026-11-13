"""Create a count comparison report for a regions batch output."""
from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
from pathlib import Path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("evaluation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    root = Path(__file__).resolve().parents[1]
    evaluation = load_module(root / "scripts" / "evaluate_e1_5.py")

    inventory = json.loads((out / "audit" / "inventory.json").read_text(encoding="utf-8"))
    gt_dir = out / "audit" / "ground_truth"
    gt_dir.mkdir(exist_ok=True)
    frozen = {r["image"]: {"sha256": r["sha256"]} for r in inventory}
    gt, _ = evaluation.collect_gt(root.parent / "数据", {"inputs": frozen}, gt_dir)

    prior_path = root / "annotations" / "narrow_signal_gt_snapshot_20261016.json"
    if prior_path.exists():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        for key, entry in gt["images"].items():
            old = prior.get("images", {}).get(key, {})
            if old.get("status_reason", "").startswith("explicit_user_correction") and old.get("image_sha256") == entry["image_sha256"]:
                entry.update(status=old["status"], count=old["count"],
                             correction_source=old.get("correction_source"),
                             status_reason=old["status_reason"])

    regions_gt_path = root / "annotations" / "regions_count_gt_20261019.json"
    if regions_gt_path.exists():
        regions_gt = json.loads(regions_gt_path.read_text(encoding="utf-8"))
        for key, override in regions_gt["images"].items():
            if key not in gt["images"]:
                continue
            if gt["images"][key]["image_sha256"] != override["image_sha256"]:
                raise ValueError(f"hash mismatch for {key}")
            gt["images"][key].update(status="verified", count=override["count"],
                                      raw_annotation=f"{override['count']}张",
                                      status_reason="explicit_user_correction_20261019",
                                      correction_source=override.get("source"))

    evaluation.write_json(gt_dir / "gt_snapshot.json", gt)
    predictions = json.loads((out / "audit" / "results.json").read_text(encoding="utf-8"))
    rows = []
    for prediction in predictions:
        truth = gt["images"][prediction["image"]]
        scorable = prediction["status"] == "exported" and truth["status"] == "verified"
        delta = prediction["count"] - truth["count"] if scorable else None
        state = ("正确" if delta == 0 else "有误差") if scorable else ("运行失败" if prediction["status"] != "exported" else "无确定真值")
        rows.append({"image": prediction["image"], "count": prediction.get("count"),
                     "truth": truth.get("count"), "delta": delta, "state": state,
                     "result_image": prediction.get("result_image"),
                     "reason": truth.get("raw_annotation", "")})

    errors = [row for row in rows if row["state"] == "有误差"]
    unknown = [row for row in rows if row["state"] == "无确定真值"]
    correct = [row for row in rows if row["state"] == "正确"]
    summary = {"total": len(rows), "exported": sum(p["status"] == "exported" for p in predictions),
               "scorable": len(correct) + len(errors), "correct": len(correct),
               "errors": len(errors), "unknown": len(unknown)}
    evaluation.write_json(out / "audit" / "count_comparison.json", {"summary": summary, "rows": rows})

    with (out / "计数对照.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["图片", "检测张数", "人工真值", "误差（检测减真值）", "结论"])
        writer.writerows([row["image"], row["count"], row["truth"], row["delta"], row["state"]] for row in rows)

    lines = ["# regions 计数误差报告", "", f"共{summary['total']}张，成功导出{summary['exported']}张；可核对{summary['scorable']}张，其中计数正确{summary['correct']}张、有误差{summary['errors']}张；另{summary['unknown']}张无确定真值。", "", "误差为检测张数减人工真值；灰色淋膜纸、长条淋膜纸仍按用户要求保留在报告中，但不作为本轮重点。", "", "## 有计数误差的图片", "", "| 图片 | 检测 | 真值 | 误差 |", "| --- | ---: | ---: | ---: |"]
    for row in errors:
        lines.append(f"| [{row['image']}]({row['result_image']}) | {row['count']} | {row['truth']} | {row['delta']:+d} |")
    lines += ["", "## 无确定真值，待人工核对", "", "| 图片 | 检测 | 标注情况 |", "| --- | ---: | --- |"]
    for row in unknown:
        lines.append(f"| [{row['image']}]({row['result_image']}) | {row['count']} | {row['reason'] or '未提供真值'} |")
    (out / "误差报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    html_rows = []
    for row in sorted(rows, key=lambda item: (item["state"] != "有误差", item["image"])):
        link = html.escape(row["result_image"] or "", quote=True)
        name = html.escape(row["image"])
        delta = f"{row['delta']:+d}" if row["delta"] is not None else "—"
        truth = row["truth"] if row["truth"] is not None else "—"
        html_rows.append(f"<tr><td><a href='{link}'>{name}</a></td><td>{row['count']}</td><td>{truth}</td><td>{delta}</td><td>{row['state']}</td></tr>")
    page = "<!doctype html><meta charset='utf-8'><title>regions 检测结果</title><style>body{font:16px sans-serif;margin:30px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}a{color:#1762ad}</style>"
    page += f"<h1>regions 检测结果</h1><p>共{summary['total']}张；可核对{summary['scorable']}张，正确{summary['correct']}张，误差{summary['errors']}张；无确定真值{summary['unknown']}张。</p>"
    (out / "index.html").write_text(page + "<table><tr><th>图片</th><th>检测</th><th>真值</th><th>误差</th><th>结论</th></tr>" + "".join(html_rows) + "</table>", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    for row in errors:
        print(f"{row['image']}\t{row['count']}\t{row['truth']}\t{row['delta']:+d}")


if __name__ == "__main__":
    main()
