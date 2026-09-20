"""Audit partial false-track annotations against saved batch detections."""
from __future__ import annotations

import argparse
import csv
import json
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
from paper_counter.track_annotations import (
    load_track_annotations, match_false_track_annotations, resolve_annotation_image,
    track_reference_y,
)


def image_dimensions(path: Path) -> tuple[int, int]:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read overlay: {path}")
    return int(image.shape[1]), int(image.shape[0])


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def render_mapping_overlay(output: Path, debug: dict, transform, false_matches,
                           true_matches=(), missed_matches=(), rejected_matches=()) -> None:
    """Show annotation mapping without altering the saved detector overlay."""
    canvas = read_image(output / "result_overlay.png")
    all_tracks = (annotation_candidate_tracks(debug) + list(debug.get("rejected_tracks", [])))
    tracks = {str(row.get("track_id", "")): row for row in all_tracks}
    strips = debug.get("strip_bounds_roi_x", [])
    roi_x = float(debug.get("roi", {}).get("x1", 0))

    def draw_track(track_id: str, color: tuple[int, int, int], thickness: int = 2) -> None:
        track = tracks.get(track_id)
        if not track:
            return
        points = []
        for key, y in sorted(track.get("strip_positions", {}).items(), key=lambda item: int(item[0])):
            strip = int(key)
            if strip >= len(strips):
                continue
            x = roi_x + .5 * (strips[strip][0] + strips[strip][1])
            ox, oy = transform.detection_to_overlay(x, float(y))
            points.append((int(round(ox)), int(round(oy))))
        if len(points) > 1:
            cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, thickness)

    for track_id in debug.get("track_identity", {}).get("reference_track_ids", []):
        draw_track(track_id, (255, 255, 0), 1)  # cyan: provisional reference
    for match in false_matches:
        y = int(round(match.annotation_overlay_y))
        if match.status in {"matched", "matched_clear_nearest", "matched_human_override"}:
            color = (255, 0, 255)  # magenta: mapped known false
            draw_track(match.matched_track_id or "", color, 2)
        elif match.status == "ambiguous_annotation_match":
            color = (0, 165, 255)  # orange: ambiguous candidates
            for candidate in match.candidates:
                draw_track(candidate["track_id"], color, 2)
        else:
            color = (0, 0, 255)  # red: no selected track within tolerance
        if 0 <= y < canvas.shape[0]:
            cv2.line(canvas, (0, y), (canvas.shape[1] - 1, y), color, 1)
            label = f"ann {match.annotation_overlay_y:g}: {match.status}"
            cv2.putText(canvas, label, (8, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        .42, color, 1, cv2.LINE_AA)
    for match in true_matches:
        y = int(round(match.annotation_overlay_y))
        color = (255, 255, 0)
        if match.matched_track_id:
            draw_track(match.matched_track_id, color, 2)
        if 0 <= y < canvas.shape[0]:
            cv2.line(canvas, (0, y), (canvas.shape[1] - 1, y), color, 1)
            cv2.putText(canvas, f"true ann {match.annotation_overlay_y:g}",
                        (8, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        .42, color, 1, cv2.LINE_AA)
    for match in missed_matches:
        y = int(round(match.annotation_overlay_y))
        color = (0, 255, 255)  # yellow: human-confirmed missed edge
        if match.matched_track_id:
            draw_track(match.matched_track_id, color, 2)
        if 0 <= y < canvas.shape[0]:
            cv2.line(canvas, (0, y), (canvas.shape[1] - 1, y), color, 1)
            cv2.putText(canvas, f"missed true {match.annotation_overlay_y:g}",
                        (8, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        .42, color, 1, cv2.LINE_AA)
    for match in rejected_matches:
        y = int(round(match.annotation_overlay_y))
        color = (255, 128, 0)  # blue: human-confirmed correct rejection
        if match.matched_track_id:
            draw_track(match.matched_track_id, color, 2)
        if 0 <= y < canvas.shape[0]:
            cv2.line(canvas, (0, y), (canvas.shape[1] - 1, y), color, 1)
            cv2.putText(canvas, f"correct reject {match.annotation_overlay_y:g}",
                        (8, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        .42, color, 1, cv2.LINE_AA)
    write_image(output / "false_track_overlay.png", canvas)


def annotation_candidate_tracks(debug: dict) -> list[dict]:
    """Return the tracks that entered the verifier in shadow or active mode.

    In active mode verifier-rejected tracks have moved from paper_edge_tracks to
    rejected_tracks.  Bringing back only the IDs applied by this terminal stage
    keeps annotation mapping comparable before and after activation.
    """
    selected = list(debug.get("paper_edge_tracks", []))
    verifier_ids = set(debug.get("final_track_verifier", {}).get("applied_reject_track_ids", []))
    if not verifier_ids:
        return selected
    by_id = {str(row.get("track_id", "")): row for row in selected}
    for row in debug.get("rejected_tracks", []):
        track_id = str(row.get("track_id", ""))
        if track_id in verifier_ids:
            by_id[track_id] = row
    return list(by_id.values())


def nearby_tracks(tracks: list[dict], transform, annotation_y: float,
                  tolerance: float) -> list[dict]:
    nearby = []
    for track in tracks:
        track_y = track_reference_y(track)
        track_overlay_y = transform.detection_to_overlay(0.0, track_y)[1]
        delta = abs(track_overlay_y - annotation_y)
        if delta <= tolerance:
            nearby.append({"track_id": str(track.get("track_id", "")),
                           "track_y": track_y, "overlay_y": track_overlay_y,
                           "delta_overlay_y": delta, "reason": track.get("reason"),
                           "sequence_decision": track.get("sequence_decision")})
    return sorted(nearby, key=lambda item: item["delta_overlay_y"])


def _numeric_feature_names(rows: list[dict]) -> list[str]:
    excluded = {"track_id", "identity_score", "identity_z_score", "template_outlier_z"}
    if not rows:
        return []
    return [key for key, value in rows[0].items()
            if key not in excluded and isinstance(value, (int, float)) and not isinstance(value, bool)]


def _robust_summary(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    median = float(np.median(array))
    return median, float(np.median(np.abs(array - median)))


def feature_distribution(false_features: list[dict], reference_features: list[dict]) -> list[dict]:
    rows = []
    for name in _numeric_feature_names(false_features or reference_features):
        false_values = [float(row[name]) for row in false_features if np.isfinite(row.get(name, np.nan))]
        reference_values = [float(row[name]) for row in reference_features if np.isfinite(row.get(name, np.nan))]
        if not false_values or not reference_values:
            continue
        fm, fd = _robust_summary(false_values)
        rm, rd = _robust_summary(reference_values)
        scale = max(1.4826 * np.sqrt(fd * fd + rd * rd), .05)
        effect = (fm - rm) / scale
        false_range, reference_range = (fm - fd, fm + fd), (rm - rd, rm + rd)
        overlap_width = max(0.0, min(false_range[1], reference_range[1]) -
                            max(false_range[0], reference_range[0]))
        union_width = max(false_range[1], reference_range[1]) - min(false_range[0], reference_range[0])
        overlap = overlap_width / max(union_width, 1e-9)
        rows.append({"feature": name, "false_median": fm, "false_mad": fd,
                     "reference_median": rm, "reference_mad": rd,
                     "difference": fm - rm, "robust_effect_size": effect,
                     "overlap_estimate": overlap})
    return sorted(rows, key=lambda row: abs(row["robust_effect_size"]), reverse=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def render_false_audit_markdown(report: dict) -> str:
    summary = report["shadow_evaluation"]
    lines = ["# False track audit", "",
             f"- known false annotations: {summary['known_false_count']}",
             f"- known false proposed rejects: {summary['known_false_proposed_rejects']}",
             f"- known false rejection rate: {summary['known_false_rejection_rate']:.2%}",
             f"- known false keeps: {summary['known_false_keeps']}",
             f"- unannotated proposed rejections: {summary['unannotated_proposed_rejections']}",
             "", "| image | overlay_y | track_id | identity_score | z | family | proposed decision |",
             "|---|---:|---|---:|---:|---|---|"]
    for row in report["false_tracks"]:
        feature = row["features"]
        lines.append(f"| {row['image']} | {row['annotation_overlay_y']:.1f} | {row['matched_track_id']} | "
                     f"{feature['identity_score']:.3f} | {feature['identity_z_score']:.2f} | "
                     f"{feature['false_family']} | {feature['proposed_decision']} |")
    return "\n".join(lines) + "\n"


def render_distribution_markdown(rows: list[dict]) -> str:
    lines = ["# False vs provisional-reference feature distribution", "",
             "Medians and MAD are robust per-track summaries. Effect size is signed; larger absolute values separate the groups more.",
             "", "| feature | false median | false MAD | reference median | reference MAD | difference | effect | overlap |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['feature']} | {row['false_median']:.4g} | {row['false_mad']:.4g} | "
                     f"{row['reference_median']:.4g} | {row['reference_mad']:.4g} | "
                     f"{row['difference']:.4g} | {row['robust_effect_size']:.3f} | "
                     f"{row['overlap_estimate']:.3f} |")
    return "\n".join(lines) + "\n"


def plot_false_profiles(output: Path, false_track_ids: list[str]) -> None:
    archive_path = output / "profiles.npz"
    if not archive_path.exists() or not false_track_ids:
        return
    with np.load(archive_path, allow_pickle=False) as archive:
        if "identity_track_ids" not in archive:
            return
        ids = [str(value) for value in archive["identity_track_ids"]]
        indices = [(track_id, ids.index(track_id)) for track_id in false_track_ids if track_id in ids]
        if not indices:
            return
        relative = archive["identity_relative_y"]
        names = ("raw", "negative", "positive", "normalized")
        fig, axes = plt.subplots(len(names), len(indices), figsize=(4.2 * len(indices), 10),
                                 sharex=True, squeeze=False, constrained_layout=True)
        for column, (track_id, index) in enumerate(indices):
            for row, name in enumerate(names):
                axes[row, column].plot(relative, archive[f"identity_template_{name}"],
                                       color="black", lw=1.8, label="template")
                axes[row, column].plot(relative, archive[f"identity_{name}_profiles"][index],
                                       color="magenta", lw=1.0, label="known false")
                axes[row, column].axvline(0, color="steelblue", ls=":", lw=.7)
                axes[row, column].grid(alpha=.15)
                if column == 0:
                    axes[row, column].set_ylabel(name)
            axes[0, column].set_title(track_id)
        axes[0, 0].legend(fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("Relative Y / spacing")
        fig.savefig(output / "track_identity_profiles.png", dpi=145)
        plt.close(fig)


def render_mapping_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# Annotation mapping",
        "",
        f"- annotations: {summary['annotation_count']}",
        f"- known false annotations: {summary['false_annotation_count']}",
        f"- confirmed selected-edge annotations: {summary['true_annotation_count']}",
        f"- confirmed missed-edge annotations: {summary['missed_true_annotation_count']}",
        f"- confirmed correct-rejection annotations: {summary['correct_rejection_annotation_count']}",
        f"- unique matches: {summary['matched_count']}",
        f"- ambiguous: {summary['ambiguous_count']}",
        f"- unmatched: {summary['unmatched_count']}",
        f"- ambiguous render mappings: {summary['ambiguous_mapping_count']}",
        "",
        "| image | kind | annotation image | overlay_y | detection_y | matched_track_id | track_y | delta_overlay | status | nearby non-final |",
        "|---|---|---|---:|---:|---|---:|---:|---|---|",
    ]
    for row in report["rows"]:
        def value(key, digits=2):
            item = row.get(key)
            return "—" if item is None else f"{item:.{digits}f}"
        lines.append(
            f"| {row.get('image') or '—'} | {row.get('annotation_kind', 'known_false')} | {row['annotation_image']} | "
            f"{value('annotation_overlay_y', 1)} | {value('mapped_detection_y')} | "
            f"{row.get('matched_track_id') or '—'} | {value('matched_track_y')} | "
            f"{value('delta_overlay_y')} | {row['status']} | "
            f"{', '.join(x['track_id'] for x in row.get('near_nonfinal_candidates', [])) or '—'} |"
        )
    warnings = report.get("warnings", [])
    if warnings:
        lines += ["", "## Warnings", ""] + [f"- {warning}" for warning in warnings]
    lines += [
        "",
        "Mapping equation: overlay_y = overlay_offset_y + "
        "(detection_y - detection_offset_y) * preview_scale_y.",
    ]
    return "\n".join(lines) + "\n"


def audit(batch: Path, annotations_path: Path) -> dict:
    annotations = load_track_annotations(annotations_path)
    image_root = batch / "images"
    available = sorted(path.name for path in image_root.iterdir() if path.is_dir())
    tolerance = float(annotations.get("overlay_y_tolerance_px", 8.0))
    ambiguity_margin = float(annotations.get("ambiguity_margin_px", 2.0))
    rows, warnings = [], []
    false_audit, reference_audit = [], []
    proposed_pairs: set[tuple[str, str]] = set()
    matched_false_pairs: set[tuple[str, str]] = set()
    ambiguous_mapping_count = 0
    for annotation_name, entry in annotations["images"].items():
        image_name, filename_warning = resolve_annotation_image(annotation_name, available)
        if filename_warning:
            warnings.append(filename_warning)
        if image_name is None:
            for kind in ("false_tracks", "true_tracks", "missed_true_tracks",
                         "correctly_rejected_tracks"):
                for overlay_y in entry[kind]:
                    kind_names = {"false_tracks": "known_false", "true_tracks": "explicit_true",
                                  "missed_true_tracks": "missed_true",
                                  "correctly_rejected_tracks": "correct_rejection"}
                    rows.append({"image": None, "annotation_image": annotation_name,
                                 "annotation_kind": kind_names[kind],
                                 "annotation_overlay_y": overlay_y, "status": "image_unmatched"})
            continue
        output = image_root / image_name
        debug = json.loads((output / "debug_info.json").read_text(encoding="utf-8"))
        metadata_path = output / "render_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            overlay_width, overlay_height = image_dimensions(output / "result_overlay.png")
            metadata = reconstruct_render_metadata(debug, overlay_width, overlay_height)
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        if metadata.get("mapping_status") == "ambiguous":
            ambiguous_mapping_count += 1
            for kind in ("false_tracks", "true_tracks", "missed_true_tracks",
                         "correctly_rejected_tracks"):
                for overlay_y in entry[kind]:
                    kind_names = {"false_tracks": "known_false", "true_tracks": "explicit_true",
                                  "missed_true_tracks": "missed_true",
                                  "correctly_rejected_tracks": "correct_rejection"}
                    rows.append({"image": image_name, "annotation_image": annotation_name,
                                 "annotation_kind": kind_names[kind],
                                 "annotation_overlay_y": overlay_y, "status": "ambiguous_mapping",
                                 "mapping_reason": metadata.get("reason")})
            continue
        transform = transform_from_metadata(metadata)
        identity_debug = debug.get("track_identity", {})
        feature_by_id = {row["track_id"]: row for row in identity_debug.get("features", [])}
        proposed_pairs.update((image_name, track_id) for track_id in
                              debug.get("final_track_verifier", {}).get("proposed_reject_track_ids", []))
        audit_tracks = annotation_candidate_tracks(debug)
        matches = match_false_track_annotations(
            entry["false_tracks"], audit_tracks, transform,
            tolerance_px=tolerance, ambiguity_margin_px=ambiguity_margin,
            match_overrides=entry.get("track_match_overrides", {}),
        )
        true_matches = match_false_track_annotations(
            entry["true_tracks"], audit_tracks, transform,
            tolerance_px=tolerance, ambiguity_margin_px=ambiguity_margin,
        )
        final_tracks = list(debug.get("paper_edge_tracks", []))
        final_ids = {str(track.get("track_id", "")) for track in final_tracks}
        verifier_rejected_ids = set(
            debug.get("final_track_verifier", {}).get("applied_reject_track_ids", []))
        missed_matches = match_false_track_annotations(
            entry["missed_true_tracks"], final_tracks, transform,
            tolerance_px=tolerance, ambiguity_margin_px=ambiguity_margin,
        )
        rejected_tracks = list(debug.get("rejected_tracks", []))
        correct_rejection_matches = match_false_track_annotations(
            entry["correctly_rejected_tracks"], rejected_tracks, transform,
            tolerance_px=tolerance, ambiguity_margin_px=ambiguity_margin,
            match_overrides=entry.get("track_match_overrides", {}),
        )
        render_mapping_overlay(output, debug, transform, matches, true_matches,
                               missed_matches, correct_rejection_matches)
        matched_image_false_ids = []
        for match in matches:
            row = {"image": image_name, "annotation_image": annotation_name,
                   "annotation_kind": "known_false",
                   "filename_warning": filename_warning, **match.to_dict()}
            if match.status == "unmatched":
                row["near_nonfinal_candidates"] = nearby_tracks(
                    rejected_tracks, transform, match.annotation_overlay_y, tolerance)
            rows.append(row)
            if match.matched_track_id and match.matched_track_id in feature_by_id:
                matched_false_pairs.add((image_name, match.matched_track_id))
                matched_image_false_ids.append(match.matched_track_id)
                false_audit.append({"image": image_name, "annotation_image": annotation_name,
                                    "annotation_overlay_y": match.annotation_overlay_y,
                                    "mapped_detection_y": match.mapped_detection_y,
                                    "matched_track_id": match.matched_track_id,
                                    "matched_track_y": match.matched_track_y,
                                    "delta_overlay_y": match.delta_overlay_y,
                                    "features": feature_by_id[match.matched_track_id]})
        for match in true_matches:
            row = {"image": image_name, "annotation_image": annotation_name,
                   "annotation_kind": "explicit_true",
                   "filename_warning": filename_warning, **match.to_dict()}
            if match.status == "unmatched":
                row["near_nonfinal_candidates"] = nearby_tracks(
                    rejected_tracks, transform, match.annotation_overlay_y, tolerance)
                row["status"] = "explicit_true_not_selected"
            elif match.matched_track_id in verifier_rejected_ids:
                row["status"] = "explicit_true_verifier_rejected"
            elif match.matched_track_id in final_ids:
                row["status"] = "explicit_true_selected"
            else:
                row["status"] = "explicit_true_nonfinal"
            rows.append(row)
        for match in missed_matches:
            row = {"image": image_name, "annotation_image": annotation_name,
                   "annotation_kind": "missed_true", "filename_warning": filename_warning,
                   **match.to_dict(), "annotation_note": entry.get("notes", {}).get(
                       str(int(match.annotation_overlay_y)))}
            if match.status == "unmatched":
                row["status"] = "confirmed_missed_true"
                row["near_nonfinal_candidates"] = nearby_tracks(
                    rejected_tracks, transform, match.annotation_overlay_y, tolerance)
            else:
                # The reviewer explicitly says no line represents this edge. A nearby
                # selected line is therefore context, not proof that the miss is fixed.
                row["status"] = "confirmed_missed_true_with_nearby_selected"
            rows.append(row)
        for match in correct_rejection_matches:
            row = {"image": image_name, "annotation_image": annotation_name,
                   "annotation_kind": "correct_rejection", "filename_warning": filename_warning,
                   **match.to_dict()}
            row["status"] = ("confirmed_correct_rejection" if match.matched_track_id
                             else "correct_rejection_candidate_unmatched")
            rows.append(row)
        for track_id in identity_debug.get("reference_track_ids", []):
            if track_id in feature_by_id:
                reference_audit.append({"image": image_name, "track_id": track_id,
                                        "features": feature_by_id[track_id]})
        plot_false_profiles(output, matched_image_false_ids)
    false_rows = [row for row in rows if row.get("annotation_kind") == "known_false"]
    true_rows = [row for row in rows if row.get("annotation_kind") == "explicit_true"]
    missed_rows = [row for row in rows if row.get("annotation_kind") == "missed_true"]
    correct_rejection_rows = [row for row in rows
                              if row.get("annotation_kind") == "correct_rejection"]
    matched = sum(row["status"] in {"matched", "matched_clear_nearest", "matched_human_override"}
                  for row in false_rows)
    ambiguous = sum(row["status"] == "ambiguous_annotation_match" for row in false_rows)
    unmatched = len(false_rows) - matched - ambiguous
    false_features = [row["features"] for row in false_audit]
    reference_features = [row["features"] for row in reference_audit]
    distributions = feature_distribution(false_features, reference_features)
    proposed_false = sum(row["features"].get("proposed_decision", "").startswith("REJECT")
                         for row in false_audit)
    known_false_count = len(false_audit)
    shadow = {"known_false_count": known_false_count,
              "known_false_proposed_rejects": proposed_false,
              "known_false_rejection_rate": proposed_false / max(1, known_false_count),
              "known_false_keeps": known_false_count - proposed_false,
              "unannotated_proposed_rejections": len(proposed_pairs - matched_false_pairs),
              "unannotated_proposed_rejection_tracks": [
                  {"image": image, "track_id": track_id}
                  for image, track_id in sorted(proposed_pairs - matched_false_pairs)]}
    return {
        "batch": str(batch), "annotations": str(annotations_path),
        "coordinate_space": annotations.get("coordinate_space"),
        "overlay_y_tolerance_px": tolerance,
        "ambiguity_margin_px": ambiguity_margin,
        "summary": {"annotation_count": len(rows), "false_annotation_count": len(false_rows),
                    "true_annotation_count": len(true_rows),
                    "missed_true_annotation_count": len(missed_rows),
                    "correct_rejection_annotation_count": len(correct_rejection_rows),
                    "matched_count": matched,
                    "ambiguous_count": ambiguous, "unmatched_count": unmatched,
                    "ambiguous_mapping_count": ambiguous_mapping_count},
        "warnings": sorted(set(warnings)), "rows": rows,
        "false_tracks": false_audit, "reference_tracks": reference_audit,
        "feature_distribution": distributions, "shadow_evaluation": shadow,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--annotations", type=Path, default=ROOT / "annotations" / "track_ground_truth.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = (args.output or args.batch).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = audit(args.batch.resolve(), args.annotations.resolve())
    (output / "annotation_mapping.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "annotation_mapping.md").write_text(render_mapping_markdown(report), encoding="utf-8")
    audit_report = {key: report[key] for key in
                    ("false_tracks", "reference_tracks", "shadow_evaluation", "warnings")}
    (output / "false_track_audit.json").write_text(
        json.dumps(audit_report, ensure_ascii=False, indent=2), encoding="utf-8")
    flat_false = [{k: v for k, v in row.items() if k != "features"} | row["features"]
                  for row in report["false_tracks"]]
    write_csv(output / "false_track_audit.csv", flat_false)
    (output / "false_track_audit.md").write_text(render_false_audit_markdown(report), encoding="utf-8")
    write_csv(output / "false_vs_reference_features.csv", report["feature_distribution"])
    (output / "false_vs_reference_summary.md").write_text(
        render_distribution_markdown(report["feature_distribution"]), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
