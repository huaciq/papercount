from __future__ import annotations

"""Interpretable heuristic confidence, not a calibrated accuracy probability."""

from typing import Any

import numpy as np

from .config import CounterConfig
from .models import PeakCandidate


def compute_geometry_confidence(tracks: list, sequence: object, config: CounterConfig,
                                regularity=None) -> tuple[float, dict]:
    """Image/ordering quality and ambiguity with a bounded regularity term."""
    if len(tracks) < 3:
        return 0., {"reason": "insufficient_tracks"}
    edges = sequence.transitions
    strong_suppressed = sum(t.candidate_class == "strong" for t in sequence.suppressed)
    near = sum(t.reason == "sequence_suppressed_short_gap" for t in sequence.suppressed)
    conflict = sum(t.reason == "sequence_crossing_conflict" for t in sequence.suppressed)
    ambiguity_ratio = (near + strong_suppressed + conflict) / max(1, len(tracks) + len(sequence.suppressed))
    raw_ratio = float(np.mean([t.raw_valid_ratio for t in tracks]))
    parts = {"track": float(np.mean([t.total_score for t in tracks])),
        "image_support": float(np.mean([t.support_ratio * t.continuity_ratio for t in tracks])),
        "raw_validity": raw_ratio,
        "non_crossing": float(np.mean([e.get("non_crossing_ratio", 0.) for e in edges])) if edges else 0.,
        "geometry": float(np.mean([e.get("shared_geometry_confidence", 0.) * e.get("gap_profile_smoothness", 0.) for e in edges])) if edges else 0.,
        "ambiguity": float(1 / (1 + ambiguity_ratio))}
    if regularity is not None and len(getattr(regularity, "final_regularity", [])):
        parts["regularity"] = float(np.mean(regularity.final_regularity))
        parts["regularity_anomaly_ratio"] = float(np.mean(regularity.final_regularity < config.regularity_irregular_threshold))
    weights = config.geometry_confidence_weights
    score = sum(parts[k] * weights[k] for k in weights) / sum(weights.values())
    rescued_ratio = len(sequence.rescued) / len(tracks)
    score *= (1 - config.repair_confidence_penalty * rescued_ratio)
    score *= 1 / (1 + config.geometry_confidence_ambiguity_penalty * ambiguity_ratio)
    score *= parts["ambiguity"] ** config.geometry_confidence_ambiguity_exponent
    score *= min(1., len(tracks) / config.confidence_min_edges)
    # A mostly regular stack receives a small bounded consistency lift; local
    # anomalies only affect their own diagnostics rather than collapsing the
    # whole-image confidence.
    if "regularity" in parts:
        score = float(np.clip(score * (0.90 + 0.10 * parts["regularity"]), 0, 1))
    return float(np.clip(score, 0, 1)), {**parts, "near_track_competitions": near,
        "suppressed_strong_tracks": strong_suppressed, "crossing_conflicts": conflict,
        "sequence_ambiguity_ratio": ambiguity_ratio, "rescued_ratio": rescued_ratio,
        "spacing_consistency_weight": 0., "note": "Image and geometry confidence, not a calibrated count accuracy"}


def compute_track_confidence(tracks: list, local: object, config: CounterConfig) -> tuple[float, dict]:
    """Track quality/consistency, explicitly not a correctness probability."""
    if len(tracks) < 3 or local is None:
        return 0.0, {"reason": "insufficient_tracks"}
    centers = np.array([t.center_y for t in tracks])
    gaps = np.diff(centers)
    expected = np.array([local.at(float(y)) for y in (centers[:-1] + centers[1:]) / 2])
    errors = np.abs(gaps - expected) / expected
    parts = {"track": float(np.mean([t.total_score for t in tracks])),
             "support": float(np.mean([t.support_ratio for t in tracks])),
             "continuity": float(np.mean([t.continuity_ratio for t in tracks])),
             "smoothness": float(np.mean([t.smoothness_score for t in tracks])),
             "spacing": float(1 / (1 + np.mean(errors)))}
    suspicious = float(np.mean(errors > config.missing_gap_tolerance))
    recovered = sum(t.source == "recovered" or t.sequence_decision == "rescued" for t in tracks) / len(tracks)
    sample_factor = min(1.0, len(tracks) / config.confidence_min_edges)
    score = sum(parts[k] * v for k, v in config.track_confidence_weights.items()) / sum(config.track_confidence_weights.values())
    score *= (1 - config.repair_confidence_penalty * recovered) * (1 - config.suspicious_gap_penalty * suspicious) * sample_factor
    return float(np.clip(score, 0, 1)), {**parts, "recovered_ratio": recovered,
        "suspicious_gap_ratio": suspicious, "sample_factor": sample_factor,
        "note": "Algorithm quality/consistency score, not count-correctness probability"}


def compute_confidence(peaks: list[PeakCandidate], period: float | None, acf_score: float,
                       config: CounterConfig) -> tuple[float, dict[str, Any]]:
    """Penalize irregular gaps, weak support, repairs and small sample counts."""
    if len(peaks) < 3 or period is None:
        return 0.0, {"reason": "insufficient_edges_or_no_period"}
    gaps = np.diff([p.y for p in peaks])
    cv = float(np.std(gaps) / np.mean(gaps))
    mismatch = float(np.mean(np.abs(gaps - period)) / period)
    spacing = 1.0 / (1.0 + cv + mismatch)
    vote = float(np.mean([p.vote_ratio for p in peaks]))
    acf = float(np.clip(acf_score, 0, 1))
    repaired_fraction = sum(p.source == "repaired" for p in peaks) / len(peaks)
    sample_factor = min(1.0, len(peaks) / config.confidence_min_edges)
    parts = {"spacing": spacing, "vote": vote, "acf": acf}
    weighted = sum(parts[k] * w for k, w in config.confidence_weights.items()) / sum(config.confidence_weights.values())
    result = weighted * (1 - config.repair_confidence_penalty * repaired_fraction) * sample_factor
    return float(np.clip(result, 0, 1)), {**parts, "gap_cv": cv, "period_mismatch": mismatch,
        "repaired_fraction": repaired_fraction, "sample_factor": sample_factor,
        "note": "Heuristic score, not a calibrated probability of correct count"}

