"""Multi-strip-first counting; global profile is only a scale prior/debug signal."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from .config import CounterConfig
from .models import PaperCountResult, PeakCandidate
from .preprocessing import crop_roi, to_gray, normalize_illumination, signed_gradient_levels, display_preview
from .profiles import make_strip_profiles_diagnostic, normalize_and_fuse
from .period import score_polarity, estimate_period
from .local_detection import detect_strip_peaks
from .tracking import associate_strip_peaks, validate_tracks, deduplicate_tracks, classify_tracks, deduplicate_obvious_tracks
from .local_period import estimate_local_period, missing_edge_hypotheses
from .track_recovery import recover_weak_tracks
from .active_roi import detect_active_stack_roi
from .confidence import compute_track_confidence
from .vertical_sequence import optimize_vertical_sequence, SequenceResult, transition_score
from .sequence_recovery import recovery_hypotheses, recover_sequence_candidates, spacing_anomalies
from .raw_mask import detect_raw_stack_mask, filter_mask_peaks, track_mask_permission, StackState, refine_stack_mask_with_strong_tracks
from .track_geometry import estimate_local_spacing, geometry_pair_score
from .regularity import estimate_local_regularity, evaluate_middle_track_competition
from .track_identity import (analyze_track_identity, apply_final_track_verifier,
                             make_intensity_strip_profiles)


def process_tracks(image: np.ndarray, cfg: CounterConfig) -> PaperCountResult:
    """Detect at full resolution and export observed track centers."""
    roi, bounds = crop_roi(image, cfg.roi)
    requested_bounds = dict(bounds)
    gray = to_gray(roi)
    raw_preview = display_preview(gray, cfg.display_max_height, cfg.display_max_width)
    raw_mask = detect_raw_stack_mask(gray, cfg, offset=bounds["y1"])
    # The raw three-state mask decides whether individual candidates are
    # permitted; it is intentionally conservative and cannot locate the stack
    # when a scratch splits an otherwise empty black border.  Locate the main
    # sustained texture interval independently, then retain the raw mask for
    # candidate/track permission inside that interval.
    if cfg.stack_roi_mode == "foreground_envelope":
        top, bottom, active_info = detect_active_stack_roi(gray, None, cfg, original=roi)
    else:
        top, bottom, active_info = detect_active_stack_roi(gray, None, cfg)
    roi = roi[top:bottom]
    gray = gray[top:bottom]
    bounds = {**bounds, "y1": bounds["y1"] + top, "y2": bounds["y1"] + bottom}
    normalized = normalize_illumination(gray, cfg.illumination_sigma, cfg.epsilon)
    gradient_levels, native_gy = signed_gradient_levels(normalized, cfg)
    invalid_rows = raw_mask.states[top:bottom] == StackState.INVALID
    native_gy[invalid_rows] = 0
    for _, gradient, _ in gradient_levels:
        gradient[invalid_rows] = 0
    preview = lambda a: display_preview(a, cfg.display_max_height, cfg.display_max_width)
    previews = {"gray": preview(gray), "normalized": preview(normalized)}
    preview_gradient = gradient_levels[-1][1]
    for name, sign in (("positive", 1.0), ("negative", -1.0)):
        rendered = np.empty(preview_gradient.shape, dtype=np.uint8)
        scale = max(float(np.max(preview_gradient) if sign > 0 else -np.min(preview_gradient)), cfg.epsilon)
        for start in range(0, len(preview_gradient), cfg.debug_chunk_rows):
            chunk = preview_gradient[start:start + cfg.debug_chunk_rows] * sign
            rendered[start:start + len(chunk)] = np.clip(chunk * (255 / scale), 0, 255).astype(np.uint8)
        previews[name + "_gradient"] = preview(rendered)
    del rendered
    data = {}
    for polarity in ("positive", "negative"):
        level_profiles, level_details = [], []
        for scale_value, gradient, source in gradient_levels:
            raw_level, strips, aggregation = make_strip_profiles_diagnostic(gradient, polarity, cfg)
            raw_level[:, invalid_rows] = 0
            level_profiles.append(raw_level)
            level_details.append({"scale": scale_value, "source": source,
                                  "aggregation": aggregation})
        native_raw, _, native_aggregation = make_strip_profiles_diagnostic(native_gy, polarity, cfg)
        native_raw[:, invalid_rows] = 0
        stack = np.stack(level_profiles)
        winners = np.argmax(stack, axis=0)
        raw = np.take_along_axis(stack, winners[None, ...], axis=0)[0]
        scale_values = np.asarray([item[0] for item in gradient_levels], dtype=np.float32)
        scale_sources = scale_values[winners]
        profiles, fused, stats = normalize_and_fuse(raw, cfg.epsilon)
        native_profiles, native_fused, _ = normalize_and_fuse(native_raw, cfg.epsilon)
        data[polarity] = {"raw_profiles": raw, "profiles": profiles, "global": fused,
                          "native_raw_profiles": native_raw, "native_profiles": native_profiles,
                          "native_global": native_fused,
                          "scale_sources": scale_sources,
                          "strip_stats": stats, "assessment": score_polarity(fused, cfg),
                          "native_assessment": score_polarity(native_fused, cfg),
                          "gradient_levels": level_details,
                          "native_aggregation": native_aggregation}
    raw_intensity_profiles = make_intensity_strip_profiles(gray, strips, cfg.debug_chunk_rows)
    normalized_intensity_profiles = make_intensity_strip_profiles(normalized, strips, cfg.debug_chunk_rows)
    del normalized, native_gy, gradient_levels, preview_gradient
    chosen = cfg.gradient_polarity
    if chosen == "auto":
        chosen = max(data, key=lambda p: (data[p]["assessment"]["score"], data[p]["assessment"]["acf_score"]))
    selected = data[chosen]
    opposite = "positive" if chosen == "negative" else "negative"
    # Resolve auto before the identity verifier interprets boundary direction.
    identity_cfg = replace(cfg, gradient_polarity=chosen)
    use_native_period = cfg.enhanced_period_from_native and (
        cfg.directional_smoothing_enabled or cfg.multiscale_gradient_enabled)
    period_assessment = selected["native_assessment"] if use_native_period else selected["assessment"]
    period = period_assessment["period"]
    period_source = ("native_gradient_frontend_lock" if period is not None and use_native_period
                     else "global_profile" if period is not None else "unavailable")
    strip_periods = []
    # Large curvature can wash out the fused signal. Measured strip ACF consensus
    # supplies the scale in that case, without inferring any paper locations.
    if period is None:
        for signal in selected["profiles"]:
            measured, strength, _ = estimate_period(signal, cfg.min_period, cfg.max_period,
                cfg.min_acf_score, cfg.acf_harmonic_preference, cfg.epsilon)
            if measured is not None:
                strip_periods.append(measured)
        if len(strip_periods) >= np.ceil(cfg.min_track_support_ratio * cfg.num_strips):
            consensus = float(np.median(strip_periods))
            agreement = sum(abs(p - consensus) <= cfg.missing_gap_tolerance * consensus for p in strip_periods)
            if agreement >= np.ceil(cfg.min_track_support_ratio * cfg.num_strips):
                period = consensus
                period_source = "strip_acf_consensus"
    if period is None and not cfg.sequence_period_multiple_mode and np.any(raw_mask.evidence) and any(s["active"] for s in selected["strip_stats"]):
        period = (cfg.min_period + cfg.max_period) / 2
        period_source = "bootstrap_spacing_scale_not_periodicity"
    tracks, rejected, raw_tracks, candidates, statistics = [], [], [], [], []
    local = None
    hypotheses, attempts, recovered = [], [], []
    strong, marginal, garbage = [], [], []
    sequence = SequenceResult()
    reuse_attempts, weak_attempts, pool, remaining_gaps = [], [], [], []
    invalid_peak_count = 0
    if period is not None:
        candidates, statistics = detect_strip_peaks(
            selected["profiles"], period, cfg, bounds["y1"],
            selected["native_profiles"], selected["scale_sources"])
        invalid_peak_count = sum(raw_mask.state_at(p.y) == StackState.INVALID for strip in candidates for p in strip)
        candidates = filter_mask_peaks(candidates, raw_mask)
        valid = sum(s["active"] for s in selected["strip_stats"])
        raw_tracks = associate_strip_peaks(candidates, period, cfg, valid)
        strong, marginal, garbage = classify_tracks(raw_tracks, cfg)
        mask_refinement = refine_stack_mask_with_strong_tracks(raw_mask, strong, cfg, period)
        for track in strong + marginal:
            if not track_mask_permission(track, raw_mask, cfg):
                garbage.append(track)
        strong = [t for t in strong if t.candidate_class == "strong"]
        marginal = [t for t in marginal if t.candidate_class == "marginal"]
        if cfg.vertical_sequence_enabled:
            pool, duplicates = deduplicate_obvious_tracks(strong + marginal)
            estimate_spacing = estimate_local_period if cfg.sequence_period_multiple_mode else estimate_local_spacing
            local = estimate_spacing([t for t in pool if t.candidate_class == "strong"], period, cfg)
            regularity = estimate_local_regularity([t for t in pool if t.candidate_class == "strong"],
                                                   selected["global"], period, cfg) if cfg.regularity_enabled else None
            sequence = optimize_vertical_sequence(pool, local, cfg, regularity)
            # Gaps between selected strong anchors may already contain a rescued
            # pool track. Log that reuse before allowing any weak-image search.
            anchors = [t for t in sequence.selected if t.candidate_class == "strong"]
            hypotheses = recovery_hypotheses(anchors, local, cfg, regularity)
            auxiliary = None
            if cfg.positive_aux_weight > 0:
                auxiliary, _ = detect_strip_peaks(
                    data[opposite]["profiles"], period, cfg, bounds["y1"],
                    data[opposite]["native_profiles"], data[opposite]["scale_sources"])
                auxiliary = filter_mask_peaks(auxiliary, raw_mask)
            added, failures, reuse_attempts, weak_attempts = recover_sequence_candidates(
                hypotheses, candidates, sequence.selected, pool, valid, cfg, auxiliary,
                raw_mask=raw_mask, negative_evidence=chosen == "negative", regularity=regularity,
                primary_evidence=True)
            pool, more_duplicates = deduplicate_obvious_tracks(pool + added)
            sequence = optimize_vertical_sequence(pool, local, cfg, regularity)
            tracks, recovered = sequence.selected, sequence.rescued
            rejected = garbage + duplicates + more_duplicates + failures + sequence.suppressed
            attempts = reuse_attempts + weak_attempts
            remaining_gaps = spacing_anomalies(tracks, local, cfg, regularity)
            for attempt in reuse_attempts:
                attempt["final_selected_track_ids"] = [t.track_id for t in tracks if t.track_id in attempt["candidate_track_ids"]]
            for attempt in weak_attempts:
                attempt["final_selected_track_ids"] = [t.track_id for t in tracks if t.track_id in attempt["accepted_candidate_ids"]]
        else:
            tracks, rejected = list(strong), list(garbage) + list(marginal)
            tracks, duplicates = deduplicate_tracks(tracks, lambda y: period, cfg)
            rejected.extend(duplicates)
            local = estimate_local_period(tracks, period, cfg)
            tracks, duplicates = deduplicate_tracks(tracks, local.at, cfg)
            rejected.extend(duplicates)
            local = estimate_local_period(tracks, period, cfg)
            hypotheses = missing_edge_hypotheses(tracks, local, cfg)
            auxiliary = None
            if cfg.positive_aux_weight > 0:
                auxiliary, _ = detect_strip_peaks(data[opposite]["profiles"], period, cfg, bounds["y1"],
                    data[opposite]["native_profiles"], data[opposite]["scale_sources"])
            recovery_candidates = [[p for p in strip if raw_mask.state_at(p.y) == StackState.CORE or
                (cfg.margin_allow_weak_recovery and raw_mask.state_at(p.y) == StackState.MARGIN)] for strip in candidates]
            recovered, failures, attempts = recover_weak_tracks(hypotheses, recovery_candidates, tracks, valid, cfg, auxiliary)
            rejected.extend(failures)
            tracks = sorted(tracks + recovered, key=lambda t: t.center_y)
            local = estimate_local_period(tracks, period, cfg)
    positions = [int(round(t.center_y)) for t in tracks]
    def compatibility_peaks(items: list) -> list[PeakCandidate]:
        return [PeakCandidate(int(round(t.center_y)), float(np.median(list(t.strip_peak_strengths.values()))),
                float(np.median([p.height for p in t.peak_candidates])),
                float(np.median([p.width for p in t.peak_candidates])), t.support_ratio,
                "repaired" if t.source == "recovered" or t.sequence_decision == "rescued" else "detected", t.reason) for t in items]
    detected = compatibility_peaks([t for t in tracks if t.source == "detected" and t.sequence_decision != "rescued"])
    repaired = compatibility_peaks(recovered)
    rejected_peaks = compatibility_peaks(rejected)
    regularity = locals().get("regularity", None)
    confidence, confidence_parts = compute_track_confidence(tracks, local, cfg)
    if cfg.vertical_sequence_enabled and not cfg.sequence_period_multiple_mode:
        from .confidence import compute_geometry_confidence
        confidence, confidence_parts = compute_geometry_confidence(tracks, sequence, cfg, regularity)
    selected_pairs = [(transition_score(a, b, local, cfg, regularity)
                      or geometry_pair_score(a, b, local, cfg)) for a, b in zip(tracks, tracks[1:])] if local else []
    competitions = []
    if local and len(tracks) >= 3:
        for a, b, c in zip(tracks, tracks[1:], tracks[2:]):
            competitions.append(evaluate_middle_track_competition(
                a, b, c, local.at(b.center_y), cfg,
                regularity.at(b.center_y) if regularity is not None else 0.5))
    layer_gaps = [p["gap_median"] for p in selected_pairs]
    spacing_warnings = []
    if local is not None:
        for left, right in zip(tracks[:-1], tracks[1:]):
            expected = local.at((left.center_y + right.center_y) / 2)
            gap = right.center_y - left.center_y
            if abs(gap - expected) / expected > cfg.missing_gap_tolerance:
                spacing_warnings.append({"left": left.center_y, "right": right.center_y,
                    "gap": gap, "local_period": expected, "reason": "unexplained_spacing",
                    "action": "review_only_no_period_based_deletion"})
    if cfg.vertical_sequence_enabled:
        spacing_warnings = remaining_gaps
    pre_verifier_tracks = list(tracks)
    identity_features = {}
    identity_bundle = {"debug": {"template": {"status": "disabled"}, "features": []},
                       "artifacts": {}}
    verifier_debug = {"enabled": False, "shadow_mode": cfg.final_track_verifier_shadow_mode,
                      "count_before": len(tracks), "count_after": len(tracks),
                      "proposed_reject_count": 0, "applied_reject_count": 0,
                      "proposed_reject_track_ids": [], "applied_reject_track_ids": [],
                      "track_decisions": [], "recovery_after_verifier": False}
    verifier_rejected = []
    if cfg.final_track_verifier_enabled:
        identity_features, identity_bundle = analyze_track_identity(
            tracks, raw_intensity_profiles, normalized_intensity_profiles,
            data["negative"]["profiles"], data["positive"]["profiles"],
            bounds["y1"], local, period or (cfg.min_period + cfg.max_period) / 2,
            regularity, raw_mask, competitions, remaining_gaps, identity_cfg,
        )
        tracks, verifier_rejected, verifier_debug = apply_final_track_verifier(
            tracks, identity_features, cfg)
        rejected.extend(verifier_rejected)
    # Verifier is the terminal stage. Rebuild exported count records only;
    # recovery is deliberately never called after a verifier rejection.
    positions = [int(round(t.center_y)) for t in tracks]
    recovered = [t for t in tracks if t.source == "recovered" or t.sequence_decision == "rescued"]
    detected = compatibility_peaks([t for t in tracks if t.source == "detected" and t.sequence_decision != "rescued"])
    repaired = compatibility_peaks(recovered)
    rejected_peaks = compatibility_peaks(rejected)
    confidence, confidence_parts = compute_track_confidence(tracks, local, cfg)
    if cfg.vertical_sequence_enabled and not cfg.sequence_period_multiple_mode:
        from .confidence import compute_geometry_confidence
        confidence, confidence_parts = compute_geometry_confidence(tracks, sequence, cfg, regularity)
    debug = {"algorithm": "multi_strip_tracks", "config": cfg.to_dict(), "image_shape": list(image.shape),
        "image_dtype": str(image.dtype), "roi": bounds, "coordinate_system": "original image; x2/y2 exclusive",
        "requested_roi": requested_bounds, "active_stack_roi": bounds,
        "active_roi_diagnostics": {k: v for k, v in active_info.items() if k != "energy"},
        "raw_stack_mask": raw_mask.to_dict(), "mask_refinement": locals().get("mask_refinement", {}),
        "raw_mask_invalid_peaks_removed": invalid_peak_count,
        "invalid_selected_track_count": sum(any(raw_mask.state_at(y) == StackState.INVALID for y in t.strip_positions.values()) for t in tracks),
        "margin_selected_track_ids": [t.track_id for t in tracks if t.raw_mask_state == "MARGIN"],
        "sequence_model": "period_multiple_comparison" if cfg.sequence_period_multiple_mode else "geometry_ordered_layers",
        "hybrid_mode": "regular_first_geometry_exception" if cfg.regularity_enabled else "geometry_only",
        "regularity": regularity.to_dict() if regularity is not None else {},
        "initial_spacing_scale": period, "local_spacing_scale": local.to_samples() if local else [],
        "selected_neighbor_gap_profiles": selected_pairs,
        "secondary_texture_competitions": competitions,
        "track_identity": identity_bundle["debug"],
        "pre_verifier_sequence": {"count": len(pre_verifier_tracks),
                                  "track_ids": [t.track_id for t in pre_verifier_tracks]},
        "post_verifier_sequence": {"count": len(tracks),
                                   "track_ids": [t.track_id for t in tracks]},
        "final_track_verifier": verifier_debug,
        "selected_neighbor_gap_distribution": {"median": float(np.median(layer_gaps)),
            "mad": float(np.median(np.abs(np.array(layer_gaps) - np.median(layer_gaps)))),
            "q10": float(np.percentile(layer_gaps, 10)), "q90": float(np.percentile(layer_gaps, 90))} if layer_gaps else {},
        "short_gap_suppression_count": sum(t.reason == "sequence_suppressed_short_gap" for t in sequence.suppressed),
        "large_gap_anomaly_count": sum(p["large_gap_anomaly"] for p in selected_pairs),
        "evidence_based_recovery_count": len(sequence.rescued),
        "gradient_polarity": chosen, "primary_evidence_polarity": chosen,
        "auxiliary_evidence_polarity": opposite,
        "counted_boundary": "paper_top" if chosen == "positive" else "paper_bottom",
        "recovery_evidence_policy": "native_selected_polarity_required",
        "period": period, "initial_global_period": period,
        "initial_period_source": period_source, "strip_period_estimates": strip_periods,
        "median_local_period": float(np.median(local.values)) if local is not None and len(local.values) else None,
        "local_period_samples": local.to_samples() if local is not None else [],
        "missing_edge_hypotheses": hypotheses, "recovery_attempts": attempts,
        "exception_regions": hypotheses if cfg.pass2_exception_only else [],
        "regular_missing_hypotheses": [h for h in hypotheses if h.get("regular_period_hint")],
        "regular_period_assisted_recoveries": [a for a in attempts if a.get("recovery_mode") == "period_assisted" or a.get("reason") == "recovery_observed_candidate" and a.get("regularity", 0) >= cfg.regularity_regular_threshold],
        "geometry_only_recoveries": [a for a in attempts if a.get("recovery_mode") == "geometry_only"],
        "unresolved_regular_gaps": ([g for g in remaining_gaps if g.get("regularity", 0) >= cfg.regularity_regular_threshold and not g.get("search", False)]
            + [a for a in attempts if a.get("regularity", 0) >= cfg.regularity_regular_threshold
               and a.get("reason") in {"recovery_no_image_evidence", "recovery_low_support", "recovery_search_window_miss"}]),
        "spacing_warnings": spacing_warnings,
        "track_candidate_pool": [t.to_dict() for t in (pool if cfg.vertical_sequence_enabled else strong + marginal)],
        "strong_tracks": [t.to_dict() for t in strong],
        "marginal_tracks": [t.to_dict() for t in marginal],
        "garbage_tracks": [t.to_dict() for t in garbage],
        "vertical_sequence": sequence.to_dict(),
        "sequence_suppressed_tracks": [t.to_dict() for t in sequence.suppressed],
        "sequence_rescued_tracks": [t.to_dict() for t in sequence.rescued],
        "remaining_suspicious_gaps": remaining_gaps,
        "candidate_reuse_recovery_attempts": reuse_attempts,
        "weak_image_recovery_attempts": weak_attempts,
        "mean_peak_spacing": float(np.mean(np.diff(positions))) if len(positions) > 1 else None,
        "polarity_scores": {p: {k: v for k, v in d["assessment"].items() if k != "acf"} for p, d in data.items()},
        "strip_bounds_roi_x": strips, "strip_stats": {p: d["strip_stats"] for p, d in data.items()},
        "front_end": {"strip_aggregation_mode": cfg.strip_aggregation_mode,
            "strip_width_ratio": cfg.strip_width_ratio,
            "directional_smoothing_enabled": cfg.directional_smoothing_enabled,
            "directional_sigma_x": cfg.directional_sigma_x,
            "directional_sigma_y": cfg.directional_sigma_y,
            "multiscale_gradient_enabled": cfg.multiscale_gradient_enabled,
            "enhanced_period_from_native": cfg.enhanced_period_from_native,
            "period_evidence": period_source,
            "gradient_scales": cfg.gradient_scales,
            "scale_source": "fixed_config_not_period",
            "cross_scale_deduplication": "pointwise_max_before_peak_detection",
            "enhanced_only_observations_count_as_tracks": False,
            "gradient_levels": {p: [{"scale": level["scale"], "source": level["source"],
                                      "aggregation": level["aggregation"]}
                                     for level in d["gradient_levels"]] for p, d in data.items()}},
        "prominence_threshold": None, "threshold_mode": "per_strip_local_adaptive",
        "strip_peak_candidates": [[p.to_dict() for p in strip] for strip in candidates],
        "strong_strip_candidate_count": sum(p.strength_type == "strong" and not p.reason for strip in candidates for p in strip),
        "weak_strip_candidate_count": sum(p.strength_type == "weak" and not p.reason for strip in candidates for p in strip),
        "enhanced_only_proposal_count": sum(p.reason == "enhanced_only_proposal" for strip in candidates for p in strip),
        "raw_seed_track_count": len(raw_tracks),
        "raw_recovery_track_count": sum(a["raw_track_count"] for a in attempts),
        "raw_track_count": len(raw_tracks) + sum(a["raw_track_count"] for a in attempts),
        "raw_candidate_count": len(raw_tracks) + sum(a["raw_track_count"] for a in attempts),
        "rejected_tracks": [t.to_dict() for t in rejected], "paper_edge_tracks": [t.to_dict() for t in tracks],
        "count": len(tracks), "peak_positions": positions, "edge_positions": [t.center_y for t in tracks],
        "detected_peaks": [p.to_dict() for p in detected], "repaired_peaks": [p.to_dict() for p in repaired],
        "rejected_peaks": [p.to_dict() for p in rejected_peaks],
        "confidence": confidence, "confidence_components": confidence_parts, "full_resolution_detection": True,
        "status": "ok" if len(tracks) >= 3 else "insufficient_evidence",
        "warnings": [] if period else ["No reliable period; no validated tracks. Count zero is not proof of no paper."]}
    return PaperCountResult(len(tracks), positions, detected, repaired, rejected_peaks, period, chosen, confidence,
        debug, {"roi": roi, "previews": previews, "polarities": data,
        "raw_stack_mask": raw_mask, "raw_gray_preview": raw_preview,
                "track_identity": identity_bundle["artifacts"],
                "strip_candidates": candidates, "local_statistics": statistics, "local_period": local,
                "active_energy": active_info.get("energy"), "active_energy_y_offset": requested_bounds["y1"]}, tracks, rejected)
