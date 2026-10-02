"""Inspect anomalous observed layer gaps without predicting missing layers."""
from __future__ import annotations
import numpy as np
from dataclasses import replace
from .track_geometry import geometry_pair_score, fit_track_curve
from .raw_mask import StackState, track_mask_permission
from .tracking import associate_strip_peaks, validate_tracks
from .track_recovery import positive_aux_score


def geometry_anomalies(tracks, local, cfg, regularity=None):
    result = []
    ordered = sorted(tracks, key=lambda t: t.center_y)
    for left, right in zip(ordered, ordered[1:]):
        profile = geometry_pair_score(left, right, local, cfg)
        if not (profile["large_gap_anomaly"] or profile["short_gap_warning"] or profile["hard_reject"]):
            continue
        large = profile["large_gap_anomaly"]
        midpoint = (left.center_y + right.center_y) / 2
        r = float(regularity.at(midpoint)) if regularity is not None else 0.5
        # A large gap launches a missing-edge search only in a regular/mixed
        # zone.  Irregular zones remain geometry-first and are review-only.
        period_hint = bool(cfg.adaptive_period_reasoning and profile["large_gap_anomaly"] and
                           ((r >= cfg.regularity_regular_threshold and cfg.regular_large_gap_search_enabled) or
                            (cfg.regularity_irregular_threshold < r < cfg.regularity_regular_threshold) or
                            (r <= cfg.regularity_irregular_threshold and cfg.irregular_gap_search_enabled)))
        result.append({**profile, "left": left.center_y, "right": right.center_y,
            "left_track_id": left.track_id, "right_track_id": right.track_id,
            "gap": profile["gap_median"], "local_period": profile["local_spacing_scale"],
            "period_mad": float(np.interp((left.center_y + right.center_y) / 2, local.y, local.mad)) if len(local.y) else 0.0,
            "reason": "large_gap_anomaly_not_missing" if large else "short_gap_geometry_anomaly",
            "regularity": r,
            "regular_period_hint": period_hint,
            "search": bool(large and not profile["hard_reject"] and (period_hint or regularity is None)),
            "action": "regular_period_assisted_search" if period_hint else ("inspect_image_evidence_only" if large else "review_geometry")})
    return result


def geometry_recovery_regions(tracks, local, cfg, regularity=None):
    """One inspection region per real gap; never n-1 predicted paper locations."""
    by_id = {t.track_id: t for t in tracks}
    result = []
    for anomaly in geometry_anomalies(tracks, local, cfg, regularity):
        if not anomaly["search"]:
            result.append(anomaly)
            continue
        a = fit_track_curve(by_id[anomaly["left_track_id"]], cfg)
        b = fit_track_curve(by_id[anomaly["right_track_id"]], cfg)
        valid = a.valid & b.valid
        strips = np.flatnonzero(valid)
        center = {int(s): float((a.positions[s] + b.positions[s]) / 2) for s in strips}
        # This curve centers an inspection corridor, not a predicted paper.
        radius = {int(s): max(0., float((b.positions[s] - a.positions[s]) / 2) - cfg.sequence_min_geometric_gap) for s in strips}
        if anomaly.get("regular_period_hint"):
            widened = cfg.regular_recovery_search_ratio * max(anomaly["local_period"], cfg.min_period)
            radius = {s: max(v, widened) for s, v in radius.items()}
        result.append({**anomaly, "predicted_y": (anomaly["left"] + anomaly["right"]) / 2,
            "prediction_semantics": "search_corridor_only_not_a_paper_prediction",
            "predicted_strip_positions": center, "search_radius_by_strip": radius,
            "left_strip_positions": {int(s): float(a.positions[s]) for s in strips},
            "right_strip_positions": {int(s): float(b.positions[s]) for s in strips},
            "search_radius": float(np.median(list(radius.values()))) if radius else 0.,
            "neighbor_geometry_uncertainty": float(np.median(a.uncertainty[valid] + b.uncertainty[valid])) if len(strips) else 0.})
    return result


def recover_geometry_candidates(hypotheses, candidates, existing, pool, valid_count, cfg,
                                auxiliary=None, raw_mask=None, negative_evidence=True,
                                regularity=None, primary_evidence=None):
    """Reuse complete tracks, otherwise associate observed primary-polarity nodes."""
    added, rejected, reuse_attempts, weak_attempts = [], [], [], []
    occupied = {(p.strip_index, p.y) for t in existing + pool for p in t.peak_candidates}
    selected_ids = {t.track_id for t in existing}
    for h in hypotheses:
        if not h["search"]:
            continue
        period = h["local_period"]
        r = float(regularity.at(h.get("predicted_y", h["left"])) if regularity is not None else 0.5)
        reusable = [t for t in pool if t.candidate_class in {"strong", "marginal"}
            and t.raw_mask_state != "INVALID" and h["left"] < t.center_y < h["right"]
            and (t.raw_mask_state != "MARGIN" or t.candidate_class == "strong" or cfg.margin_allow_marginal)]
        reuse_attempts.append({"predicted_y": h["predicted_y"], "local_period": period,
            "raw_track_count": 0, "candidate_track_ids": [t.track_id for t in reusable],
            "reason": "recovery_reused_marginal_track" if reusable else "recovery_no_existing_candidate",
            "candidates": [{"track_id": t.track_id, "center_y": t.center_y,
                "support_count": t.support_count, "strip_positions": dict(t.strip_positions),
                "already_sequence_selected": t.track_id in selected_ids} for t in reusable]})
        if reusable:
            continue
        attempt = {"left": h["left"], "right": h["right"], "predicted_y": h["predicted_y"],
            "prediction_semantics": "inspection_only_not_missing_paper", "local_period": period,
            "raw_track_count": 0, "candidate_count": 0, "accepted_candidate_ids": [],
            "reason": "recovery_no_image_evidence", "regularity": r,
            "recovery_mode": "period_assisted" if r >= cfg.regularity_regular_threshold else "geometry_only"}
        weak_attempts.append(attempt)
        evidence_required = (cfg.recovery_require_primary_evidence if primary_evidence is not None
                             else cfg.recovery_require_negative_evidence)
        has_evidence = primary_evidence if primary_evidence is not None else negative_evidence
        if evidence_required and not has_evidence:
            attempt["reason"] = ("recovery_primary_evidence_required" if primary_evidence is not None
                                 else "recovery_negative_evidence_required")
            continue
        mask_rejected = 0
        def eligible(p):
            nonlocal mask_rejected
            s = p.strip_index
            if p.reason or (s, p.y) in occupied or s not in h["left_strip_positions"]:
                return False
            if not h["left_strip_positions"][s] < p.y < h["right_strip_positions"][s]:
                return False
            if raw_mask is not None:
                state = raw_mask.state_at(p.y)
                if state == StackState.INVALID or (state == StackState.MARGIN and not cfg.margin_allow_weak_recovery):
                    mask_rejected += 1
                    return False
            return abs(p.y - h["predicted_strip_positions"][s]) <= h["search_radius_by_strip"][s]
        window = [[p for p in strip if eligible(p)] for strip in candidates]
        attempt.update(candidate_count=sum(map(len, window)), raw_mask_rejected_nodes=mask_rejected,
            search_radius_by_strip=h["search_radius_by_strip"], predicted_strip_positions=h["predicted_strip_positions"])
        if not any(window):
            if mask_rejected:
                attempt["reason"] = "recovery_raw_mask_forbidden"
            continue
        raw = associate_strip_peaks(window, period, cfg, valid_count, recovery=True)
        for i, track in enumerate(raw):
            track.track_id = f"weak_{len(weak_attempts)-1:04d}_{i:05d}"
            if auxiliary is not None:
                track.positive_aux_score = positive_aux_score(track, auxiliary, period, cfg)
                track.total_score = min(1., track.total_score + cfg.positive_aux_weight * track.positive_aux_score)
        validation_cfg = cfg
        if r >= cfg.regularity_regular_threshold:
            validation_cfg = replace(cfg,
                min_recovery_support_ratio=max(.01, cfg.min_recovery_support_ratio * cfg.regular_recovery_support_relaxation),
                min_recovery_gradient_score=max(0.0, cfg.min_recovery_gradient_score * cfg.regular_recovery_weak_threshold_scale))
        accepted, failures = validate_tracks(raw, validation_cfg, recovery=True)
        eligible_tracks = []
        for track in accepted:
            track.candidate_class = "marginal"
            track.classification_reason = ("weak_primary_image_evidence" if primary_evidence is not None
                                           else "weak_negative_image_evidence")
            if raw_mask is None or track_mask_permission(track, raw_mask, cfg, recovery=True):
                eligible_tracks.append(track)
            else:
                failures.append(track)
        for track in failures:
            track.candidate_class = "garbage"
            track.classification_reason = track.reason
            if track.reason == "low_support":
                track.reason = "recovery_low_support"
        added.extend(eligible_tracks)
        rejected.extend(failures)
        occupied.update((p.strip_index, p.y) for t in raw for p in t.peak_candidates)
        attempt.update(raw_track_count=len(raw), accepted_candidate_ids=[t.track_id for t in eligible_tracks],
            reason="recovery_observed_candidate" if eligible_tracks else
                (failures[0].reason if failures else "recovery_no_image_evidence"))
    return added, rejected, reuse_attempts, weak_attempts
