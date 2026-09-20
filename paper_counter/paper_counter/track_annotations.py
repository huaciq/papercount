"""Partial track-level annotation loading and coordinate-safe matching."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .coordinates import OverlayTransform


DEFAULT_FILENAME_ALIASES = {"金卡纸-2.bmp": "金卡纸2.bmp"}


def _merge_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Preserve duplicate ``false_tracks`` keys instead of dropping earlier values."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key == "false_tracks" and key in result:
            previous = result[key] if isinstance(result[key], list) else [result[key]]
            incoming = value if isinstance(value, list) else [value]
            result[key] = previous + incoming
        else:
            result[key] = value
    return result


def normalize_track_values(values: Any) -> list[float]:
    if values is None:
        return []
    if isinstance(values, (int, float)) and not isinstance(values, bool):
        return [float(values)]
    if not isinstance(values, list):
        raise ValueError("Track annotations must be numbers or a list of numbers")
    flattened: list[float] = []
    for value in values:
        flattened.extend(normalize_track_values(value))
    return flattened


def load_track_annotations(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8-sig")
    data = json.loads(text, object_pairs_hook=_merge_duplicate_keys)
    if not isinstance(data, dict) or not isinstance(data.get("images"), dict):
        raise ValueError("Track annotation JSON must contain an images object")
    annotation_fields = (
        "false_tracks", "true_tracks", "missed_true_tracks",
        "correctly_rejected_tracks",
    )
    for entry in data["images"].values():
        for field in annotation_fields:
            entry[field] = normalize_track_values(entry.get(field, []))
    return data


def resolve_annotation_image(annotation_name: str, available_names: Iterable[str],
                             aliases: dict[str, str] | None = None) -> tuple[str | None, str | None]:
    available = set(available_names)
    if annotation_name in available:
        return annotation_name, None
    alias = (aliases or DEFAULT_FILENAME_ALIASES).get(annotation_name)
    if alias in available:
        return alias, f"filename alias: annotation {annotation_name} -> dataset {alias}"
    return None, f"image not found for annotation name: {annotation_name}"


def track_reference_y(track: Any) -> float:
    positions = track.get("strip_positions", {}) if isinstance(track, dict) else track.strip_positions
    values = [float(value) for value in positions.values()]
    if values:
        return float(np.median(values))
    return float(track.get("center_y") if isinstance(track, dict) else track.center_y)


@dataclass
class AnnotationMatch:
    annotation_overlay_y: float
    mapped_detection_y: float
    status: str
    matched_track_id: str | None
    matched_track_y: float | None
    matched_track_overlay_y: float | None
    delta_overlay_y: float | None
    candidates: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def match_false_track_annotations(overlay_ys: Iterable[float], tracks: Iterable[Any],
                                  transform: OverlayTransform, tolerance_px: float = 8.0,
                                  ambiguity_margin_px: float = 2.0,
                                  match_overrides: dict[str, Any] | None = None) -> list[AnnotationMatch]:
    """Map approximate overlay Y labels to final selected track curves."""
    track_rows = []
    for track in tracks:
        y = track_reference_y(track)
        overlay_y = transform.detection_to_overlay(0.0, y)[1]
        get = track.get if isinstance(track, dict) else lambda key, default=None: getattr(track, key, default)
        track_rows.append({
            "track_id": str(get("track_id", "")), "track_y": y,
            "overlay_y": overlay_y, "score": float(get("total_score", 0.0)),
            "source": str(get("source", "unknown")),
        })
    matches = []
    for annotation_y in overlay_ys:
        detection_y = transform.overlay_to_detection(transform.overlay_offset_x, float(annotation_y))[1]
        candidates = sorted(({
            **row, "delta_overlay_y": abs(row["overlay_y"] - float(annotation_y))
        } for row in track_rows if abs(row["overlay_y"] - float(annotation_y)) <= tolerance_px),
            key=lambda row: (row["delta_overlay_y"], -row["score"], row["track_id"]))
        chosen = None
        status = "unmatched"
        override = (match_overrides or {}).get(str(int(annotation_y)))
        if override is not None:
            target_y = float(override["detection_y"])
            choices = sorted(track_rows, key=lambda row: abs(row["track_y"] - target_y))
            if choices and abs(choices[0]["track_y"] - target_y) <= 1.0:
                chosen = {**choices[0],
                          "delta_overlay_y": abs(choices[0]["overlay_y"] - float(annotation_y))}
                status = "matched_human_override"
                if all(row["track_id"] != chosen["track_id"] for row in candidates):
                    candidates.insert(0, {**chosen,
                        "delta_overlay_y": abs(chosen["overlay_y"] - float(annotation_y))})
        elif len(candidates) == 1:
            chosen, status = candidates[0], "matched"
        elif len(candidates) > 1:
            if candidates[1]["delta_overlay_y"] - candidates[0]["delta_overlay_y"] > ambiguity_margin_px:
                chosen, status = candidates[0], "matched_clear_nearest"
            else:
                status = "ambiguous_annotation_match"
        matches.append(AnnotationMatch(
            float(annotation_y), float(detection_y), status,
            chosen["track_id"] if chosen else None,
            chosen["track_y"] if chosen else None,
            chosen["overlay_y"] if chosen else None,
            chosen["delta_overlay_y"] if chosen else None,
            candidates,
        ))
    return matches


__all__ = [
    "DEFAULT_FILENAME_ALIASES", "AnnotationMatch", "load_track_annotations",
    "normalize_track_values", "resolve_annotation_image", "track_reference_y",
    "match_false_track_annotations",
]
