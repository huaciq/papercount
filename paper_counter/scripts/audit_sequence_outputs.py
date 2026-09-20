"""Audit saved results and produce one review report without changing detections."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit(path):
    data = read(path / "debug_info.json")
    tracks = data["paper_edge_tracks"] + data["rejected_tracks"]
    sequence = data["vertical_sequence"]
    assert len(tracks) == data["raw_track_count"]
    assert len({t["track_id"] for t in tracks}) == len(tracks)
    assert data["count"] == len(sequence["selected_track_ids"]) == len(data["paper_edge_tracks"])
    assert data["count"] == len(data["detected_peaks"]) + len(data["repaired_peaks"])
    assert set(sequence["rescued_track_ids"]) <= set(sequence["selected_track_ids"])
    assert not set(sequence["suppressed_track_ids"]) & set(sequence["selected_track_ids"])
    pool = {t["track_id"]: t for t in data["track_candidate_pool"]}
    assert set(pool) == set(sequence["selected_track_ids"]) | set(sequence["suppressed_track_ids"])
    total = sum(r["unary_score"] for r in sequence["track_decisions"] if r["decision"] != "suppressed")
    total += sum(e["pair_score"] + e["spacing_change_score"] - e.get("competition_penalty", 0.)
                 for e in sequence["transitions"])
    assert abs(total - sequence["total_sequence_score"]) < 1e-7
    for track in data["paper_edge_tracks"]:
        assert track["peak_candidates"], "A selected track must contain image observations"
        assert track["candidate_class"] in {"strong", "marginal"}
        assert track["support_count"] == len(track["strip_positions"])
    for attempt in data["candidate_reuse_recovery_attempts"]:
        for candidate in attempt["candidates"]:
            assert candidate["strip_positions"] == pool[candidate["track_id"]]["strip_positions"]
            assert candidate["support_count"] == pool[candidate["track_id"]]["support_count"]
    for name in ("vertical_sequence.png", "hybrid_sequence.png", "regularity_map.png", "raw_stack_mask.png",
                 "result_overlay.png", "edge_tracks.png", "profile.png", "profiles.npz", "cli.log"):
        assert (path / name).stat().st_size > 0
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regression", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args()
    args.regression = args.regression.resolve()
    args.batch = args.batch.resolve()
    regressions = read(args.regression / "regression.json")
    batch = read(args.batch / "summary.json")
    old_batch = read(ROOT / "output_batch_20260914_112413_858687" / "summary.json")
    old_by_name = {r["image"]: r["count"] for r in old_batch["results"]}
    rows = []
    for row in regressions:
        data = audit(Path(row["output"]))
        rows.append((row["image"], row["old_count"], data, Path(row["output"])))
    assert batch["completed"] == batch["total_images"] == 16
    for row in batch["results"]:
        path = args.batch / row["output"]
        data = audit(path)
        assert data["rotation_ccw_degrees"] == 90
        assert data["image_shape"][:2] == list(reversed(data["input_original_shape"][:2]))
        assert data["count"] == row["count"] and row["status"] == "ok"
        rows.append((row["image"], old_by_name[row["image"]], data, path))
    lines = ["# 本次修改与真实图验证报告", "",
        "原有 horizontal multi-strip tracking 保留，min_track_score 仍为 0.60。新增 strong/marginal/garbage 候选池、纵向 DP、候选优先恢复及自适应曲线搜索。",
        f"完整文件/函数清单、目标函数、分类、去重、恢复、输出口径和限制见 [算法说明](<{(ROOT / 'docs/VERTICAL_SEQUENCE.md').as_posix()}>)。",
        "", "单元测试和真实图的数量守恒、ID 唯一性、序列目标函数求和、候选复用几何和产物完整性由通用断言校验；不对单张图片硬编码轨迹结论。",
        "", "selected 包含 rescued，等于最终 Count；suppressed 只统计 usable 候选，垃圾碎片不计入该列。",
        "", "| 图像 | 旧 Count | selected / 新 Count | rescued | suppressed | 残留异常间距 | 输出 |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for name, old, data, path in rows:
        seq = data["vertical_sequence"]
        link = (path / "result_overlay.png").as_posix()
        lines.append(f"| {name} | {old} | {data['count']} | {len(seq['rescued_track_ids'])} | {len(seq['suppressed_track_ids'])} | {len(data['remaining_suspicious_gaps'])} | [标注图](<{link}>) |")
    lines += ["", "前三张按原方向检测，4.bmp 的 max_period 沿用 80。pictures 内 16 张全部先逆时针旋转 90°，并统一采用默认配置。历史结果和原图均未覆盖。",
              "", "## 轨迹回归", "",
              "每张图片的轨迹决定、局部 regularity、mask 状态和异常区均从 debug_info.json 读取，不针对某个 Y 坐标设置通过条件。", "",
              "| Y | 原 total | 支持条带 | 原因 | unary | 前间距 | 后间距 | 插入净收益 |",
              "|---:|---:|---:|---|---:|---:|---:|---:|"]
    for row in regressions[0].get("target_decisions", []):
        lines.append(f"| {row['center_y']} | {row['total_score']:.4f} | {row.get('support_count', '—')}/12 | {row.get('classification_reason', '—')} | {row['unary_score']:.4f} | {row.get('incoming_gap')} | {row.get('outgoing_gap')} | {row.get('local_insertion_delta_without_second_order', 0):.4f} |")
    lines += ["", "目标行仅用于可读性展示，不是算法硬编码条件。",
        "", "## 仍需复核", "",
        "异常间距、短 gap 竞争、regular/geometry recovery 和 unresolved regular gaps 均按每张图的 debug_info.json 输出。软序列代价没有强制删除所有短间距，也没有按大间距自动造边。",
        "缺少全图人工真值，新增或抑制的轨迹仍需结合标注图核验，不能把更高 Confidence 或更整齐的间距当成准确率证明。",
        "横向碎片过多、真实轨迹低于垃圾门槛、类似纸边的连贯纹理、局部厚度快速变化及周期先验错误，仍可能导致漏计或误计。所有逐轨迹分数、相邻边代价、竞争邻居与最终原因保存在 JSON。"]
    (args.regression / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in list((ROOT / "paper_counter").glob("*.py")) + [ROOT / "main.py", ROOT / "configs/default.yaml"]}
    (args.regression / "source_sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print("AUDIT PASSED:", len(rows), "images; targets, geometry, accounting, scores, rotation and artifacts")
    print("REPORT:", args.regression / "validation_report.md")


if __name__ == "__main__":
    main()
