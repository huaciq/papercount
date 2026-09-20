"""Reuse full horizontal proposals first; search weak image evidence only second."""
from __future__ import annotations
import numpy as np
from .config import CounterConfig
from .local_period import LocalPeriod
from .models import PaperEdgeTrack, StripPeakCandidate
from .tracking import associate_strip_peaks, validate_tracks
from .track_recovery import positive_aux_score
from .vertical_sequence import period_mad_at


def spacing_anomalies(tracks: list[PaperEdgeTrack], local: LocalPeriod, cfg: CounterConfig, regularity=None) -> list[dict]:
    """Flag deviations from one local period, including noninteger large gaps."""
    if not cfg.sequence_period_multiple_mode:
        from .geometry_recovery import geometry_anomalies
        return geometry_anomalies(tracks, local, cfg, regularity)
    result = []
    ordered = sorted(tracks, key=lambda t: t.center_y)
    for left, right in zip(ordered, ordered[1:]):
        midpoint = (left.center_y + right.center_y) / 2
        period, mad = local.at(midpoint), period_mad_at(local, midpoint)
        gap = right.center_y - left.center_y
        threshold = max(1.0, cfg.sequence_spacing_scale_ratio * period, cfg.spacing_anomaly_mad_k * 1.4826 * mad)
        if abs(gap - period) <= threshold:
            continue
        result.append({"left": left.center_y, "right": right.center_y,
            "left_track_id": left.track_id, "right_track_id": right.track_id,
            "gap": gap, "local_period": period, "period_mad": mad,
            "spacing_error": gap - period, "anomaly_threshold": threshold,
            "reason": "large_spacing_anomaly" if gap > period else "short_spacing_anomaly",
            "search": period < gap <= cfg.max_sequence_gap_periods * period,
            "action": "search_observed_evidence" if gap > period else "review_only"})
    return result


def adaptive_search_radius(period: float, period_mad: float, neighbor_uncertainty: float,
                           cfg: CounterConfig) -> float:
    raw = (cfg.recovery_base_search_ratio * period + cfg.recovery_period_mad_weight * period_mad
           + cfg.recovery_track_geometry_weight * neighbor_uncertainty)
    return float(np.clip(raw, cfg.recovery_min_search_ratio * period, cfg.recovery_max_search_ratio * period))


def recovery_hypotheses(tracks: list[PaperEdgeTrack], local: LocalPeriod, cfg: CounterConfig, regularity=None) -> list[dict]:
    """Interpolate neighbor geometry at each strip, even for noninteger gaps."""
    if not cfg.sequence_period_multiple_mode:
        from .geometry_recovery import geometry_recovery_regions
        return geometry_recovery_regions(tracks, local, cfg, regularity)
    by_id = {t.track_id: t for t in tracks}
    hypotheses = []
    for anomaly in spacing_anomalies(tracks, local, cfg):
        if not anomaly["search"]:
            hypotheses.append(anomaly)
            continue
        left, right = by_id[anomaly["left_track_id"]], by_id[anomaly["right_track_id"]]
        repeats = max(2, int(round(anomaly["gap"] / anomaly["local_period"])))
        uncertainty = (left.y_mad + right.y_mad) / 2
        radius = adaptive_search_radius(anomaly["local_period"], anomaly["period_mad"], uncertainty, cfg)
        def curve(t):
            keys = sorted(t.strip_positions)
            return np.interp(np.arange(cfg.num_strips), keys, [t.strip_positions[i] for i in keys])
        a, b = curve(left), curve(right)
        for k in range(1, repeats):
            fraction = k / repeats
            predicted = a + fraction * (b - a)
            hypotheses.append({**anomaly, "predicted_y": left.center_y + fraction * anomaly["gap"],
                "predicted_strip_positions": {i: float(y) for i, y in enumerate(predicted)},
                "left_strip_positions": {i: float(y) for i, y in enumerate(a)},
                "right_strip_positions": {i: float(y) for i, y in enumerate(b)},
                "neighbor_geometry_uncertainty": uncertainty, "search_radius": radius,
                "suspected_missing_count": repeats - 1})
    return hypotheses


def recover_sequence_candidates(hypotheses: list[dict], candidates: list[list[StripPeakCandidate]],
                                existing: list[PaperEdgeTrack], pool: list[PaperEdgeTrack],
                                valid_count: int, cfg: CounterConfig,
                                auxiliary: list[list[StripPeakCandidate]] | None = None,
                                raw_mask=None, negative_evidence: bool = True,
                                regularity=None
                                ) -> tuple[list[PaperEdgeTrack], list[PaperEdgeTrack], list[dict], list[dict]]:
    """Return only new observed paths; pool reuse never truncates or duplicates them.

    Reused paths already participate in the global DP. A rejected pool path
    does not authorize another search for the same evidence. The final DP also
    adjudicates new weak paths; recovery is not a direct append to the count.
    """
    if not cfg.sequence_period_multiple_mode:
        from .geometry_recovery import recover_geometry_candidates
        return recover_geometry_candidates(hypotheses, candidates, existing, pool, valid_count,
                                           cfg, auxiliary, raw_mask, negative_evidence, regularity)
    added, rejected, reuse_attempts, weak_attempts = [], [], [], []
    occupied = {(p.strip_index, p.y) for t in existing + pool for p in t.peak_candidates}
    selected_ids = {t.track_id for t in existing}
    for hypothesis in hypotheses:
        if not hypothesis["search"]:
            continue
        period = hypothesis["local_period"]
        reusable = [t for t in pool if t.candidate_class in {"strong", "marginal"}
                    and hypothesis["left"] < t.center_y < hypothesis["right"]
                    and abs(t.center_y - hypothesis["predicted_y"]) <= cfg.recovery_reuse_candidate_radius_ratio * period]
        reuse = {"predicted_y": hypothesis["predicted_y"], "local_period": period,
                 "raw_track_count": 0, "candidate_track_ids": [t.track_id for t in reusable],
                 "reason": "recovery_reused_marginal_track" if reusable else "recovery_no_existing_candidate",
                 "candidates": [{"track_id": t.track_id, "center_y": t.center_y,
                     "support_count": t.support_count, "strip_positions": dict(t.strip_positions),
                     "already_sequence_selected": t.track_id in selected_ids} for t in reusable]}
        reuse_attempts.append(reuse)
        if reusable:
            continue
        radius = hypothesis["search_radius"]
        def eligible(p):
            return (not p.reason and (p.strip_index, p.y) not in occupied and
                    hypothesis["left_strip_positions"][p.strip_index] < p.y < hypothesis["right_strip_positions"][p.strip_index])
        gap_nodes = [[p for p in strip if eligible(p)] for strip in candidates]
        window = [[p for p in strip if abs(p.y - hypothesis["predicted_strip_positions"][p.strip_index]) <= radius]
                  for strip in gap_nodes]
        raw = associate_strip_peaks(window, period, cfg, valid_count, recovery=True)
        for track in raw:
            track.predicted_y = hypothesis["predicted_y"]
            if auxiliary is not None:
                track.positive_aux_score = positive_aux_score(track, auxiliary, period, cfg)
                track.total_score = min(1.0, track.total_score + cfg.positive_aux_weight * track.positive_aux_score)
                track.confidence = track.total_score
        for i, track in enumerate(raw):
            track.track_id = f"weak_{len(weak_attempts):04d}_{i:05d}"
        accepted, failures = validate_tracks(raw, cfg, recovery=True)
        for track in accepted:
            track.candidate_class = "marginal"
            track.classification_reason = "weak_image_evidence"
        for track in failures:
            track.candidate_class = "garbage"
            track.classification_reason = track.reason
            if track.reason == "low_support":
                track.reason = "recovery_low_support"
        added.extend(accepted)
        rejected.extend(failures)
        occupied.update((p.strip_index, p.y) for t in raw for p in t.peak_candidates)
        reason = "recovery_observed_candidate" if accepted else "recovery_no_image_evidence"
        if not accepted and any(gap_nodes) and not any(window):
            reason = "recovery_search_window_miss"
        elif not accepted and failures:
            reason = "recovery_low_support" if all(t.reason == "recovery_low_support" for t in failures) else failures[0].reason
        weak_attempts.append({"predicted_y": hypothesis["predicted_y"], "local_period": period,
            "search_radius": radius, "predicted_strip_positions": hypothesis["predicted_strip_positions"],
            "period_mad": hypothesis["period_mad"], "neighbor_geometry_uncertainty": hypothesis["neighbor_geometry_uncertainty"],
            "candidate_count": sum(map(len, window)), "gap_candidate_count": sum(map(len, gap_nodes)),
            "raw_track_count": len(raw), "accepted_candidate_ids": [t.track_id for t in accepted],
            "reason": reason})
    return added, rejected, reuse_attempts, weak_attempts
