from __future__ import annotations

"""Orchestration of the full-resolution counting pipeline."""

from typing import Any

import numpy as np

from .config import CounterConfig
from .confidence import compute_confidence
from .models import PaperCountResult
from .peaks import detect_candidates, remove_duplicates, strip_peak_indices, validate_candidates
from .period import score_polarity
from .preprocessing import crop_roi, display_preview, normalize_illumination, signed_gradient, to_gray
from .profiles import make_strip_profiles, normalize_and_fuse, robust_scale
from .repair import repair_missing_peaks


class PaperCounter:
    """Count one selected-polarity edge per visible paper layer."""

    def __init__(self, config: CounterConfig | dict[str, Any] | None = None):
        self.config = config if isinstance(config, CounterConfig) else CounterConfig(**(config or {}))

    def process(self, image: np.ndarray) -> PaperCountResult:
        """Process one image; return original-image coordinates and debug artifacts."""
        cfg = self.config
        roi, bounds = crop_roi(image, cfg.roi)
        gray = to_gray(roi)
        normalized = normalize_illumination(gray, cfg.illumination_sigma, cfg.epsilon)
        gy = signed_gradient(normalized)
        preview = lambda a: display_preview(a, cfg.display_max_height, cfg.display_max_width)
        previews = {"gray": preview(gray), "normalized": preview(normalized)}
        # Render each sign in chunks; do not retain two full-sized response matrices.
        for name, sign in (("positive", 1.0), ("negative", -1.0)):
            rendered = np.empty(gy.shape, dtype=np.uint8)
            scale = max(float(np.max(gy) if sign > 0 else -np.min(gy)), cfg.epsilon)
            for start in range(0, len(gy), cfg.debug_chunk_rows):
                chunk = gy[start:start + cfg.debug_chunk_rows] * sign
                rendered[start:start + len(chunk)] = np.clip(chunk * (255.0 / scale), 0, 255).astype(np.uint8)
            previews[name + "_gradient"] = preview(rendered)
            del rendered
        del normalized
        data = {}
        for polarity in ("positive", "negative"):
            raw, strips = make_strip_profiles(gy, polarity, cfg)
            profiles, fused, stats = normalize_and_fuse(raw, cfg.epsilon)
            data[polarity] = {"raw_profiles": raw, "profiles": profiles, "global": fused,
                              "strip_stats": stats, "assessment": score_polarity(fused, cfg)}
        del gy
        chosen = cfg.gradient_polarity
        if chosen == "auto":
            chosen = max(data, key=lambda p: (data[p]["assessment"]["score"], data[p]["assessment"]["acf_score"]))
        selected = data[chosen]
        assessment = selected["assessment"]
        period = assessment["period"]
        raw_candidates, accepted, repaired, rejected, repair_attempts = [], [], [], [], []
        threshold = 0.0
        weak_candidates = []
        warnings = []
        if period is not None:
            raw_candidates, threshold = detect_candidates(selected["global"], period, cfg)
            accepted, rejected = validate_candidates(raw_candidates, strip_peak_indices(selected["profiles"], cfg), period, cfg)
            accepted, duplicates = remove_duplicates(accepted, period, cfg)
            rejected.extend(duplicates)
            weak_candidates, _ = detect_candidates(selected["global"], period, cfg, weak=True)
            repaired, repair_attempts = repair_missing_peaks(accepted, weak_candidates,
                strip_peak_indices(selected["profiles"], cfg, weak=True), period, cfg, {p.y for p in raw_candidates})
        else:
            warnings.append("No reliable period: count=0 means no validated edges, not proof that no paper exists.")
        final = sorted(accepted + repaired, key=lambda p: p.y)
        confidence, confidence_parts = compute_confidence(final, period, assessment["acf_score"], cfg)
        if len(final) < 3:
            warnings.append("Too few edges for a stable count.")
        flat_rows = float(np.mean(np.ptp(gray, axis=1) == 0))
        if flat_rows > 0.5:
            warnings.append("More than half the ROI rows are horizontally constant; inspect the ROI and image content.")
        if any(s["scale_method"] == "nonzero_deviation_median" for s in selected["strip_stats"]):
            warnings.append("MAD was zero in sparse strips; used median nonzero deviation fallback.")
        # Internal arrays use ROI rows. All exported candidate records use original Y.
        all_records = {id(p): p for p in raw_candidates + repaired + weak_candidates}
        for p in all_records.values():
            p.y += bounds["y1"]
        for attempt in repair_attempts:
            for key in ("left", "right", "predicted_y", "selected_y"):
                if attempt.get(key) is not None:
                    attempt[key] += bounds["y1"]
        gap_mean = float(np.mean(np.diff([p.y for p in final]))) if len(final) > 1 else None
        debug = {"config": cfg.to_dict(), "image_shape": list(image.shape), "image_dtype": str(image.dtype),
            "roi": bounds, "coordinate_system": "original image; ROI x2/y2 exclusive",
            "gradient_polarity": chosen, "period": period, "mean_peak_spacing": gap_mean,
            "polarity_scores": {p: {k: v for k, v in d["assessment"].items() if k != "acf"} for p, d in data.items()},
            "strip_bounds_roi_x": strips, "strip_stats": {p: d["strip_stats"] for p, d in data.items()},
            "global_scale": dict(zip(("median", "scale", "method"), robust_scale(selected["global"], cfg.epsilon))),
            "prominence_threshold": threshold, "raw_candidate_count": len(raw_candidates),
            "raw_candidates": [p.to_dict() for p in raw_candidates],
            "detected_peaks": [p.to_dict() for p in accepted], "repaired_peaks": [p.to_dict() for p in repaired],
            "rejected_peaks": [p.to_dict() for p in rejected], "weak_candidates": [p.to_dict() for p in weak_candidates],
            "repair_attempts": repair_attempts, "count": len(final), "peak_positions": [p.y for p in final],
            "confidence": confidence, "confidence_components": confidence_parts, "warnings": warnings,
            "status": "ok" if period is not None and len(final) >= 3 else "insufficient_evidence",
            "count_semantics": "One selected-polarity visible edge per sheet; boundary/occluded sheets need ground truth validation",
            "full_resolution_detection": True}
        return PaperCountResult(len(final), [p.y for p in final], accepted, repaired, rejected,
                                period, chosen, confidence, debug,
                                {"roi": roi, "previews": previews, "polarities": data})

