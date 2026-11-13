"""Small, auditable postprocessing pass for horizontally prepared tracks."""
from __future__ import annotations

from statistics import median


def _quality(track) -> float:
    """Compare duplicate candidates without treating a score as ground truth."""
    raw = getattr(track, "raw_evidence_ratio", 0.0)
    return (float(getattr(track, "total_score", 0.0))
            + .12 * float(getattr(track, "support_ratio", 0.0))
            + .08 * float(getattr(track, "smoothness_score", 0.0))
            + .08 * float(raw))


def apply_horizontal_gap_postprocessing(tracks, period, config):
    """Remove only near-identical, weak duplicate tracks.

    Sequence selection and identity verification remain the primary decisions.
    This terminal pass is intentionally narrow: it only acts when two final
    tracks are closer than a small fraction of the measured spacing *and* share
    several real strip observations.  It never inserts a periodic or synthetic
    track, and it leaves close physical layers alone when their observations do
    not overlap.
    """
    debug = {"enabled": bool(getattr(config, "tracks_postprocessing_enabled", False)),
             "decisions": [], "input_count": len(tracks), "output_count": len(tracks)}
    if not debug["enabled"] or len(tracks) < 2:
        return list(tracks), [], debug

    ordered = sorted(tracks, key=lambda t: (float(t.center_y), str(t.track_id)))
    gaps = [float(b.center_y - a.center_y) for a, b in zip(ordered, ordered[1:])
            if float(b.center_y - a.center_y) > 0]
    expected = float(period) if period is not None and period > 0 else (median(gaps) if gaps else 0.0)
    if expected <= 0:
        debug["reason"] = "no_spacing"
        return ordered, [], debug
    threshold = expected * float(config.tracks_post_min_gap_ratio)
    removed = []
    removed_ids = set()
    for left, right in zip(ordered, ordered[1:]):
        if left.track_id in removed_ids or right.track_id in removed_ids:
            continue
        gap = float(right.center_y - left.center_y)
        if gap >= threshold:
            continue
        common = sorted(set(left.strip_positions) & set(right.strip_positions))
        offsets = [abs(float(left.strip_positions[s] - right.strip_positions[s])) for s in common]
        # A short gap without shared observations is allowed: it may be a real
        # compressed layer visible in disjoint portions of the stack.
        if len(common) < max(3, int(getattr(config, "min_absolute_track_support", 3))) or not offsets:
            debug["decisions"].append({"left": left.track_id, "right": right.track_id,
                "gap": gap, "action": "retain_disjoint_observations"})
            continue
        if median(offsets) > max(2.0, threshold):
            debug["decisions"].append({"left": left.track_id, "right": right.track_id,
                "gap": gap, "common_strips": len(common), "action": "retain_separate_geometry"})
            continue
        weaker = left if _quality(left) <= _quality(right) else right
        stronger = right if weaker is left else left
        margin = abs(_quality(left) - _quality(right))
        if margin < .04:
            debug["decisions"].append({"left": left.track_id, "right": right.track_id,
                "gap": gap, "common_strips": len(common), "action": "retain_ambiguous"})
            continue
        weaker.reason = "horizontal_gap_duplicate"
        weaker.sequence_decision = "suppressed"
        removed.append(weaker)
        removed_ids.add(weaker.track_id)
        debug["decisions"].append({"left": left.track_id, "right": right.track_id,
            "gap": gap, "common_strips": len(common), "removed": weaker.track_id,
            "kept": stronger.track_id, "quality_margin": margin,
            "action": "suppress_near_duplicate"})

    kept = [track for track in ordered if track.track_id not in removed_ids]
    debug["output_count"] = len(kept)
    debug["expected_spacing"] = expected
    debug["threshold"] = threshold
    return kept, removed, debug

