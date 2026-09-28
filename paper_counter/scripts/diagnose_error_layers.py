"""Layer reviewer-labelled errors using saved detector evidence.

This tool is deliberately conservative: image visibility is never inferred from
count stability.  Cases without enough machine evidence remain pending human
review and every decision keeps the nearby candidates/tracks and image crops.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_counter.coordinates import reconstruct_render_metadata, transform_from_metadata
from paper_counter.track_annotations import load_track_annotations, resolve_annotation_image


KINDS = {
    "false_tracks": "false_positive",
    "true_tracks": "true_selected",
    "missed_true_tracks": "missed_true",
    "correctly_rejected_tracks": "correct_rejection",
}
ERROR_LAYERS = ("原图不可辨", "梯度弱", "候选漏检", "关联失败", "序列误选", "重复边界", "待人工核验")


def read_color(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"cannot encode {path}")
    encoded.tofile(path)


def center_y(track: dict) -> float:
    return float(track.get("center_y", track.get("y_median", -1e9)))


def flatten_candidates(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        out.extend(item if isinstance(item, list) else [item])
    return [x for x in out if isinstance(x, dict)]


def nearest(rows: list[dict], y: float, radius: float, getter=center_y) -> list[dict]:
    found = []
    for row in rows:
        delta = abs(getter(row) - y)
        if delta <= radius:
            found.append((delta, row))
    return [row for _, row in sorted(found, key=lambda pair: pair[0])]


def summarize_track(row: dict) -> dict:
    return {key: row.get(key) for key in (
        "track_id", "center_y", "support_count", "support_ratio", "candidate_class",
        "classification_reason", "sequence_decision", "reason", "source",
        "total_score", "confidence", "y_range", "y_mad")}


def classify(kind: str, y: float, debug: dict, radius: float) -> tuple[str, str, dict]:
    candidates = flatten_candidates(debug.get("strip_peak_candidates", []))
    candidate_near = nearest(candidates, y, radius, lambda row: float(row.get("y", -1e9)))
    pool_near = nearest(list(debug.get("track_candidate_pool", [])), y, radius)
    selected_near = nearest(list(debug.get("paper_edge_tracks", [])), y, radius)
    rejected_near = nearest(list(debug.get("rejected_tracks", [])), y, radius)
    evidence = {
        "strip_candidate_count": len(candidate_near),
        "strip_indices": sorted({int(x.get("strip_index", -1)) for x in candidate_near}),
        "candidate_strengths": Counter(str(x.get("strength_type", "unknown")) for x in candidate_near),
        "candidate_prominence_max": max((float(x.get("prominence", 0)) for x in candidate_near), default=0.0),
        "candidate_raw_support_max": max((float(x.get("raw_support", 0)) for x in candidate_near), default=0.0),
        "pool_tracks": [summarize_track(x) for x in pool_near[:6]],
        "selected_tracks": [summarize_track(x) for x in selected_near[:6]],
        "rejected_tracks": [summarize_track(x) for x in rejected_near[:6]],
    }
    if kind == "false_positive":
        selected = sorted(center_y(x) for x in debug.get("paper_edge_tracks", []))
        matched = min(selected, key=lambda value: abs(value-y), default=None)
        neighbors = [abs(value-matched) for value in selected if matched is not None and value != matched]
        period = float(debug.get("median_local_period") or debug.get("period") or 20.0)
        if matched is not None and abs(matched-y) <= radius and neighbors and min(neighbors) < .55*period:
            return "重复边界", "误计轨迹与另一条最终轨迹间距小于局部周期的55%", evidence
        return "序列误选", "人工标为误计，但轨迹进入最终序列", evidence
    if kind == "true_selected":
        return ("正确选中", "人工确认真边且最终序列附近有轨迹", evidence) if selected_near else (
            "序列误选", "人工确认真边，但最终序列附近没有轨迹", evidence)
    if kind == "correct_rejection":
        return "正确拒绝", "人工确认该响应不应计数", evidence
    if selected_near:
        return "待人工核验", "人工标为漏检，但容差内存在最终轨迹；需核对是否属于相邻纸边", evidence
    if rejected_near or pool_near:
        return "序列误选", "候选已经形成轨迹，但在分类或最终序列阶段被排除", evidence
    strips = len(evidence["strip_indices"])
    if strips >= 2:
        return "关联失败", "多个strip有局部候选，但未形成候选轨迹", evidence
    if candidate_near:
        weak_only = all(str(x.get("strength_type")) == "weak" for x in candidate_near)
        if weak_only or evidence["candidate_raw_support_max"] < .25:
            return "梯度弱", "仅有弱候选或原始梯度支持不足", evidence
        return "关联失败", "孤立strip候选未获得横向关联支持", evidence
    return "候选漏检", "容差内没有strip峰候选；原图是否可辨仍需人工看局部图确认", evidence


def crop_evidence(source: Path, destination: Path, center_y: float,
                  half_height: int = 90) -> None:
    image = read_color(source)
    center = int(round(center_y))
    top, bottom = max(0, center-half_height), min(image.shape[0], center+half_height+1)
    crop = image[top:bottom].copy()
    local_y = center-top
    cv2.line(crop, (0, local_y), (crop.shape[1]-1, local_y), (0, 255, 255), 1)
    write_png(destination, crop)


def plot_local_profiles(folder: Path, destination: Path, detection_y: float,
                        half_height: int = 40) -> None:
    with np.load(folder/"profiles.npz", allow_pickle=False) as archive:
        y = archive["original_y"]
        profiles = archive["negative_profiles"]
    mask = np.abs(y-detection_y) <= half_height
    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    for index, profile in enumerate(profiles):
        ax.plot(y[mask], profile[mask], lw=.65, alpha=.45, label=f"strip {index}")
    ax.plot(y[mask], np.median(profiles[:, mask], axis=0), color="black", lw=1.8,
            label="strip median")
    ax.axvline(detection_y, color="red", ls=":", lw=1.2, label="review Y")
    ax.set(xlabel="detection Y", ylabel="normalized negative response")
    ax.grid(alpha=.2)
    ax.legend(ncol=4, fontsize=6)
    fig.savefig(destination, dpi=145)
    plt.close(fig)


def run(batch: Path, annotations_path: Path, output: Path) -> dict:
    annotations = load_track_annotations(annotations_path)
    available = sorted(path.name for path in (batch / "images").iterdir() if path.is_dir())
    tolerance = float(annotations.get("overlay_y_tolerance_px", 8.0))
    rows, warnings = [], []
    crop_root = output / "evidence"
    crop_root.mkdir(parents=True, exist_ok=True)
    for annotation_name, entry in annotations["images"].items():
        image_name, warning = resolve_annotation_image(annotation_name, available)
        if warning:
            warnings.append(warning)
        if image_name is None:
            continue
        folder = batch / "images" / image_name
        debug = json.loads((folder / "debug_info.json").read_text(encoding="utf-8"))
        overlay = read_color(folder / "result_overlay.png")
        metadata_path = folder / "render_metadata.json"
        metadata = (json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists()
                    else reconstruct_render_metadata(debug, overlay.shape[1], overlay.shape[0]))
        if metadata.get("mapping_status") == "ambiguous":
            warnings.append(f"{image_name}: ambiguous coordinate mapping")
            continue
        transform = transform_from_metadata(metadata)
        detection_radius = tolerance / transform.scale_y
        for source_key, kind in KINDS.items():
            for overlay_y in entry.get(source_key, []):
                _, detection_y = transform.overlay_to_detection(0.0, float(overlay_y))
                layer, reason, evidence = classify(kind, detection_y, debug, detection_radius)
                tag = f"{Path(image_name).stem}_{kind}_{float(overlay_y):g}".replace(" ", "_")
                evidence_files = []
                for artifact in ("result_overlay.png", "gray.png", "negative_gradient.png"):
                    source = folder / artifact
                    if source.exists():
                        target = crop_root / f"{tag}_{source.stem}.png"
                        center = (float(overlay_y) if artifact == "result_overlay.png" else
                                  detection_y-float(debug["roi"]["y1"]))
                        crop_evidence(source, target, center)
                        evidence_files.append(str(target.relative_to(output)))
                profile_target = crop_root/f"{tag}_local_profiles.png"
                plot_local_profiles(folder, profile_target, detection_y)
                evidence_files.append(str(profile_target.relative_to(output)))
                rows.append({
                    "image": image_name, "kind": kind, "overlay_y": float(overlay_y),
                    "detection_y": detection_y, "layer": layer, "reason": reason,
                    "note": entry.get("notes", {}).get(str(int(float(overlay_y)))),
                    "evidence_files": evidence_files, "evidence": evidence,
                })
    counts = Counter({name: 0 for name in ERROR_LAYERS})
    counts.update(row["layer"] for row in rows)
    true_rows = [r for r in rows if r["kind"] in {"true_selected", "missed_true"}]
    false_rows = [r for r in rows if r["kind"] == "false_positive"]
    candidate_hits = sum(r["evidence"]["strip_candidate_count"] > 0 for r in true_rows)
    selected_true = sum(bool(r["evidence"]["selected_tracks"]) for r in true_rows)
    selected_false = sum(bool(r["evidence"]["selected_tracks"]) for r in false_rows)
    metrics = {
        "scope": "annotated_subset_only_not_global_edge_metrics",
        "annotated_true_edges": len(true_rows), "annotated_false_edges": len(false_rows),
        "candidate_recall": candidate_hits / len(true_rows) if true_rows else None,
        "final_recall": selected_true / len(true_rows) if true_rows else None,
        "final_precision_on_labelled_selected_subset": (
            selected_true / (selected_true + selected_false) if selected_true + selected_false else None),
    }
    return {"batch": str(batch), "annotations": str(annotations_path),
            "layer_counts": dict(counts), "annotated_subset_metrics": metrics,
            "warnings": warnings, "rows": rows}


def markdown(report: dict) -> str:
    lines = ["# Error layer diagnostics", "",
             "Metrics below cover only explicitly labelled coordinates; the annotation set is partial, so they are not global edge Precision/Recall.", "",
             "## Layer counts", ""]
    lines += [f"- {key}: {value}" for key, value in report["layer_counts"].items()]
    lines += ["", "## Annotated subset metrics", ""]
    for key, value in report["annotated_subset_metrics"].items():
        lines.append(f"- {key}: {value:.4f}" if isinstance(value, float) else f"- {key}: {value}")
    lines += ["", "## Cases", "",
              "| image | kind | overlay Y | detection Y | layer | reason | evidence |",
              "|---|---|---:|---:|---|---|---|"]
    for row in report["rows"]:
        links = ", ".join(f"[{Path(p).stem}]({p.replace(chr(92), '/')})" for p in row["evidence_files"])
        lines.append(f"| {row['image']} | {row['kind']} | {row['overlay_y']:.1f} | "
                     f"{row['detection_y']:.1f} | {row['layer']} | {row['reason']} | {links} |")
    lines += ["", "`原图不可辨`不会由稳定性或极性一致性自动推断。没有足够图像证据的案例保留为`待人工核验`。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--annotations", type=Path, default=ROOT/"annotations"/"track_ground_truth.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = run(args.batch.resolve(), args.annotations.resolve(), args.output.resolve())
    (args.output/"error_layers.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=dict), encoding="utf-8")
    (args.output/"error_layers.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"layer_counts": report["layer_counts"],
                      "metrics": report["annotated_subset_metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
