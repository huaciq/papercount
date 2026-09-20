"""Run real sequence regressions in a new directory and retain exact decisions."""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_counter.config import CounterConfig


def main():
    output = ROOT / ("output_vertical_sequence_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(exist_ok=False)
    print("OUTPUT", output, flush=True)
    rows = []
    for name in ("2.bmp", "4.bmp", "3.png"):
        cfg = CounterConfig.from_yaml(ROOT / "configs" / "default.yaml")
        if name == "4.bmp":
            cfg = replace(cfg, max_period=80)
        target = output / Path(name).stem
        target.mkdir()
        (target / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
        with (target / "cli.log").open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-X", "utf8=0", str(ROOT / "main.py"), "--image",
                str(ROOT.parent / "数据" / name), "--config", str(target / "config.yaml"), "--output", str(target)],
                check=True, stdout=stream, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        new = json.loads((target / "debug_info.json").read_text(encoding="utf-8"))
        old = json.loads((ROOT / "output_tracks_20260914_100847_868173" / Path(name).stem / "debug_info.json").read_text(encoding="utf-8"))
        assert new["raw_track_count"] == len(new["paper_edge_tracks"]) + len(new["rejected_tracks"])
        all_tracks = new["paper_edge_tracks"] + new["rejected_tracks"]
        assert len({t["track_id"] for t in all_tracks}) == len(all_tracks), "Duplicate track IDs"
        assert len(new["detected_peaks"]) + len(new["repaired_peaks"]) == new["count"]
        decisions = new["vertical_sequence"]["track_decisions"]
        ys = (1480.5, 1680, 1834) if name == "2.bmp" else (1970, 1986, 2027) if name == "4.bmp" else ()
        targets = [r for r in decisions if r["center_y"] in ys]
        pool = {t["track_id"]: t for t in new["track_candidate_pool"]}
        for row in targets:
            row["support_count"] = pool[row["track_id"]]["support_count"]
            row["strip_positions"] = pool[row["track_id"]]["strip_positions"]
        row = {"image": name, "old_count": old["count"], "new_count": new["count"],
            "strong_pool": len(new["strong_tracks"]), "marginal_pool": len(new["marginal_tracks"]),
            "sequence_suppressed": len(new["sequence_suppressed_tracks"]),
            "sequence_rescued": len(new["sequence_rescued_tracks"]),
            "weak_image_selected": sum(t["source"] == "recovered" for t in new["paper_edge_tracks"]),
            "remaining_suspicious_gaps": new["remaining_suspicious_gaps"],
            "target_decisions": targets, "confidence": new["confidence"], "output": str(target)}
        rows.append(row)
        (output / "regression.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=True), flush=True)
    lines = ["# 纵向序列真实图像回归", "", "旧结果来自 output_tracks_20260914_100847_868173。原图不旋转；4.bmp 沿用已指定的 max_period=80。", "",
             "| 图像 | 旧计数 | 新计数 | 序列抑制 | 序列恢复 | 弱图像新恢复 | 残留异常间距 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        stem = Path(row["image"]).stem
        lines.append(f"| [{row['image']}]({stem}/result_overlay.png) | {row['old_count']} | {row['new_count']} | {row['sequence_suppressed']} | {row['sequence_rescued']} | {row['weak_image_selected']} | {len(row['remaining_suspicious_gaps'])} |")
    lines += ["", "目标轨迹的 unary、进入/离开间距、pair 分数、竞争邻居、决策原因和完整条带位置见 [regression.json](regression.json)。",
              "新计数是算法结果；没有全图人工真值，不能据此声明整图准确率。"]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
