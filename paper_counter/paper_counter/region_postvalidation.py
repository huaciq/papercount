# -*- coding: utf-8 -*-
"""Evidence-bounded validation for region boxes.

Thickness outliers are review signals, not proof that a box is false. A box
is removed only when it violates a physical constraint; merge and split
operations require an actual boundary observation and are audited.
"""
from __future__ import annotations

from typing import List, Tuple
import numpy as np


def _height(box: dict) -> float:
    return float(box["detection_xyxy"][3] - box["detection_xyxy"][1])


def _profile(observation: dict) -> np.ndarray | None:
    values = observation.get("raw_signals") if observation else None
    if values is None:
        return None
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not values.shape[0]:
        return None
    return np.median(values, axis=0)


def _boundary_strength(profile: np.ndarray | None, y: float) -> float:
    if profile is None or len(profile) < 5:
        return 0.0
    index = int(np.clip(round(y), 1, len(profile) - 2))
    gradient = np.gradient(profile.astype(np.float32))
    noise = 1.4826 * float(np.median(np.abs(gradient - np.median(gradient))))
    return float(abs(gradient[index]) / max(noise, 0.35))


def _robust_reference(heights: np.ndarray) -> tuple[float, float]:
    median = float(np.median(heights))
    mad = 1.4826 * float(np.median(np.abs(heights - median)))
    return median, max(mad, 0.12 * max(median, 1.0), 1.0)


def identify_suspicious_boxes(boxes: List[dict], debug: dict) -> List[dict]:
    """Mark adaptive thickness outliers without deleting them."""
    if len(boxes) < 3:
        return []
    heights = np.asarray([_height(box) for box in boxes], dtype=np.float32)
    reference, scale = _robust_reference(heights)
    region = debug.get("region_detection") or {}
    profile = _profile(region.get("three_strip_observation") or {})
    suspicious = []
    for index, box in enumerate(boxes):
        height = float(heights[index])
        ratio = height / max(reference, 1.0)
        robust_z = abs(height - reference) / scale
        issues = []
        if ratio <= 0.55:
            issues.append(dict(type="small_thickness_outlier", severity="review",
                               detail=f"height={height:.2f}, reference={reference:.2f}, ratio={ratio:.3f}"))
        elif ratio >= 1.8:
            issues.append(dict(type="large_thickness_outlier", severity="review",
                               detail=f"height={height:.2f}, reference={reference:.2f}, ratio={ratio:.3f}"))
        elif robust_z >= 3.5:
            issues.append(dict(type="robust_thickness_outlier", severity="review",
                               detail=f"height={height:.2f}, reference={reference:.2f}, robust_z={robust_z:.2f}"))
        if not box.get("band_ids") and box.get("thickness_inferred"):
            issues.append(dict(type="inferred_without_band", severity="review",
                               detail="no directly observed band identity"))
        if index in (0, len(boxes) - 1) and box.get("thickness_inferred"):
            body = (box.get("evidence") or {}).get("joint_resegmentation") or {}
            top_support = float(body.get("top_gray_support", 0.0))
            bottom_support = float(body.get("bottom_gray_support", 0.0))
            source_ids = body.get("source_unit_ids") or []
            paper_coverage = float(body.get("paper_coverage", 0.0))
            weak_edge = min(top_support, bottom_support) < 0.15
            short_edge = ratio < 0.60
            low_score = float(box.get("score", 0.0)) < 1.0
            orphan = not source_ids or not box.get("band_ids")
            # A conservative mask can leave a real, faint outer sheet with
            # no direct band identity. Keep it when most of its interval is
            # still supported as foreground; remove only an edge candidate
            # that is both weak at the faces and mostly background.
            if weak_edge and paper_coverage < 0.75 and (short_edge or low_score or orphan):
                issues.append(dict(type="edge_background_candidate", severity="constraint",
                                   detail=(f"height_ratio={ratio:.3f}, "
                                           f"gray_support={top_support:.3f}/{bottom_support:.3f}, "
                                           f"source_units={len(source_ids)}")))
        if index and boxes[index - 1]["detection_xyxy"][3] > box["detection_xyxy"][1]:
            issues.append(dict(type="overlap_with_previous", severity="constraint",
                               detail="box intervals overlap"))
        # Boundary diagnostics are useful for outliers, but attaching a weak
        # boundary warning to every normal textured row makes the review list
        # unreadable.  Keep it as supporting evidence only for an outlier.
        if issues and _boundary_strength(profile, box["detection_xyxy"][1]) < 1.0:
            issues.append(dict(type="weak_top_boundary", severity="review",
                               detail="top boundary has weak raw-gray edge support"))
        if issues and _boundary_strength(profile, box["detection_xyxy"][3]) < 1.0:
            issues.append(dict(type="weak_bottom_boundary", severity="review",
                               detail="bottom boundary has weak raw-gray edge support"))
        if issues:
            suspicious.append(dict(box_number=box["number"], box_index=index,
                                   issues=issues, box=box, height=height,
                                   reference_height=reference, ratio=ratio, robust_z=robust_z))
    return suspicious


def correct_suspicious_boxes(boxes: List[dict], suspicious: List[dict],
                             bands: List[dict], observation: dict,
                             content_verification: dict | None = None) -> dict:
    """Apply only hard non-overlap repairs; keep uncertain boxes for review."""
    corrected = [dict(box) for box in boxes]
    corrections = []
    edge_indices = {s["box_index"] for s in suspicious
                    if any(i["type"] == "edge_background_candidate" for i in s["issues"])}
    for index in range(1, len(corrected)):
        previous, current = corrected[index - 1], corrected[index]
        previous_bottom = float(previous["detection_xyxy"][3])
        current_top = float(current["detection_xyxy"][1])
        if previous_bottom <= current_top:
            continue
        cut = (previous_bottom + current_top) / 2.0
        previous["detection_xyxy"][3] = cut
        previous["aligned_xyxy"][3] = min(float(previous["aligned_xyxy"][3]), cut)
        current["detection_xyxy"][1] = cut
        current["aligned_xyxy"][1] = max(float(current["aligned_xyxy"][1]), cut)
        corrections.append(dict(type="clip_overlap", box_numbers=[previous["number"], current["number"]],
                                 reason="hard ordered non-overlap constraint; shared midpoint cut",
                                 cut=float(cut)))
    for index in sorted(edge_indices, reverse=True):
        if 0 <= index < len(corrected):
            number = corrected[index]["number"]
            corrected.pop(index)
            corrections.append(dict(type="remove_edge_background", box_numbers=[number],
                                     reason="edge box has no body boundary support and is an adaptive thickness outlier"))
    for number, box in enumerate(corrected, 1):
        box["number"] = number
    return dict(boxes=corrected, corrections=corrections,
                review_boxes=[s["box_number"] for s in suspicious if s["box_index"] not in edge_indices],
                removed_boxes=[s["box_number"] for s in suspicious if s["box_index"] in edge_indices],
                split_boxes=[], merged_boxes=[])


def validate_box_sequence(boxes: List[dict], bands: List[dict],
                          observation: dict, debug: dict) -> Tuple[List[dict], dict]:
    if not boxes:
        return boxes, dict(enabled=False, reason="no_boxes", corrections_applied=[])
    suspicious = identify_suspicious_boxes(boxes, debug)
    corrected = correct_suspicious_boxes(boxes, suspicious, bands, observation)
    result = dict(enabled=True, original_count=len(boxes), final_count=len(corrected["boxes"]),
                  suspicious_boxes=suspicious, corrections_applied=corrected["corrections"],
                  review_boxes=corrected["review_boxes"], removed_boxes=corrected["removed_boxes"],
                  merged_boxes=[], split_boxes=[], policy="retain_uncertain_boxes")
    return corrected["boxes"], result


def check_physical_constraints(boxes: List[dict]) -> dict:
    violations = []
    for index, box in enumerate(boxes):
        top, bottom = box["detection_xyxy"][1], box["detection_xyxy"][3]
        if not bottom > top:
            violations.append(dict(index=index, type="non_positive_height"))
        if index and top < boxes[index - 1]["detection_xyxy"][3] - 1e-8:
            violations.append(dict(index=index, type="overlap"))
    return dict(passed=not violations, violations=violations,
                hard_constraints=["positive_height", "ordered", "nonoverlapping"])
