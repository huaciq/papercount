"""Second-order dynamic-programming linking and structural track validation."""
from __future__ import annotations
from typing import Callable
import numpy as np
from .config import CounterConfig
from .models import StripPeakCandidate, PaperEdgeTrack


def make_track(peaks: list[StripPeakCandidate], period: float, valid_strip_count: int,
               config: CounterConfig, source: str = "detected") -> PaperEdgeTrack:
    """Score observed nodes; skipped strips are never counted as evidence."""
    ordered = sorted(peaks, key=lambda p: p.strip_index)
    xs = np.array([p.strip_index for p in ordered])
    ys = np.array([p.y for p in ordered], dtype=np.float32)
    med = float(np.median(ys))
    run = longest = 1
    for gap in np.diff(xs):
        run = run + 1 if gap == 1 else 1
        longest = max(longest, run)
    coverage = len(ordered) / max(1, valid_strip_count)
    continuity = longest / config.num_strips
    slopes = np.diff(ys) / np.maximum(1, np.diff(xs))
    curves = np.diff(slopes)
    shift_scale = max(config.max_neighbor_shift_ratio * period, config.epsilon)
    slope_penalty = float(np.mean(np.abs(slopes)) / shift_scale) if len(slopes) else 1.0
    curve_penalty = float(np.mean(np.abs(curves)) / shift_scale) if len(curves) else 0.0
    smoothness = float(np.exp(-config.track_shift_penalty * slope_penalty
                              - config.track_curvature_penalty * curve_penalty))
    # Alternating direction changes are different from a stable tilt or a
    # smooth bow.  Peak localization is integer-valued, however, so a path
    # oscillating by one pixel must not be treated like random-jump texture.
    # Count a reversal only when both adjacent slopes exceed the localization
    # tolerance expressed as a fraction of the measured paper spacing.
    if len(slopes) > 1:
        reversal_threshold = config.track_reversal_amplitude_ratio * period
        reversal_mask = ((slopes[:-1] * slopes[1:] < 0)
                         & (np.minimum(np.abs(slopes[:-1]), np.abs(slopes[1:]))
                            > reversal_threshold))
        reversal = float(np.mean(reversal_mask))
        if config.track_continuity_geometry_enabled or (reversal > 0.5 and np.ptp(ys) <= 0.30 * period):
            smoothness *= float(np.exp(-config.track_reversal_weight * reversal))
    continuity_geometry = {}
    if config.track_continuity_geometry_enabled:
        allowance = max(config.track_max_neighbor_jump_ratio * period, config.epsilon)
        median_derivative = float(np.median(np.abs(slopes))) if len(slopes) else 0.0
        derivative_mad = float(np.median(np.abs(slopes - np.median(slopes)))) if len(slopes) else 0.0
        variation = float(np.median(np.abs(curves))) / allowance if len(curves) else 0.0
        extreme = float(np.mean(np.abs(slopes) > allowance)) if len(slopes) else 0.0
        reversal = float(np.mean((slopes[:-1] * slopes[1:] < 0) &
            (np.minimum(np.abs(slopes[:-1]), np.abs(slopes[1:])) > config.track_reversal_amplitude_ratio * period))) if len(slopes) > 1 else 0.0
        smoothness = float(np.exp(-config.track_derivative_variation_weight * variation
            -config.track_extreme_jump_weight * extreme -config.track_reversal_weight * reversal))
        continuity_geometry = {"median_abs_dy": median_derivative, "mad_dy": derivative_mad,
            "extreme_jump_ratio": extreme, "large_reversal_ratio": reversal,
            "derivative_variation": variation, "neighbor_jump_allowance": allowance}
    # Ratio is local contrast significance, so brightness drift does not dominate.
    prominence_significance = float(np.median([p.prominence / max(p.local_prominence_reference,
        p.local_threshold, config.epsilon) for p in ordered]))
    gradient = prominence_significance / (1.0 + prominence_significance)
    weights = (config.track_gradient_weight, config.track_coverage_weight,
               config.track_continuity_weight, config.track_smoothness_weight)
    score = float(np.dot(weights, [gradient, coverage, continuity, smoothness]) / sum(weights))
    strong_count = sum(p.strength_type == "strong" for p in ordered)
    return PaperEdgeTrack(med, {p.strip_index: p.y for p in ordered},
        {p.strip_index: p.prominence for p in ordered}, len(ordered), coverage, longest,
        continuity, med, float(np.median(np.abs(ys - med))), float(np.ptp(ys)),
        smoothness, gradient, coverage, continuity, score, score, source,
        strong_count, len(ordered) - strong_count, peak_candidates=ordered,
        continuity_geometry=continuity_geometry)


def associate_strip_peaks(candidates: list[list[StripPeakCandidate]], period: float,
                          config: CounterConfig, valid_strip_count: int | None = None,
                          recovery: bool = False) -> list[PaperEdgeTrack]:
    """Find second-order optimal paths through strip candidates.

    DP states keep the last two nodes and whether a strong node was seen.
    Linking may cross at most max_skip_strips missing strips. Curvature uses
    slopes per strip, so skipped strips do not create artificial large slopes.
    Paths are selected by score and node-exclusive; no peak supports two tracks.
    """
    nodes = sorted([p for strip in candidates for p in strip if p.reason is None],
                   key=lambda p: (p.strip_index, p.y))
    if not nodes:
        return []
    valid = valid_strip_count if valid_strip_count is not None else len(candidates)
    by_strip: dict[int, list[int]] = {}
    states: list[dict[tuple[int, bool], tuple[float, tuple[int, ...]]]] = []
    shift = (config.track_max_neighbor_jump_ratio if config.track_continuity_geometry_enabled else config.max_neighbor_shift_ratio) * period
    for j, node in enumerate(nodes):
        strong = node.strength_type == "strong"
        significance = node.prominence / max(node.local_prominence_reference, node.local_threshold, config.epsilon)
        reward = 1.0 + significance / (1.0 + significance)
        current = {(-1, strong): (reward, (j,))}
        for strip in range(max(0, node.strip_index - config.max_skip_strips - 1), node.strip_index):
            for i in by_strip.get(strip, []):
                previous = nodes[i]
                dx = node.strip_index - previous.strip_index
                slope = (node.y - previous.y) / dx
                if abs(slope) > shift:
                    continue
                for (before, has_strong), (value, path) in states[i].items():
                    curvature = 0.0
                    if before >= 0:
                        earlier = nodes[before]
                        old_slope = (previous.y - earlier.y) / (previous.strip_index - earlier.strip_index)
                        curvature = abs(slope - old_slope)
                    shift_weight = config.track_geometry_shift_penalty if config.track_continuity_geometry_enabled else config.track_shift_penalty
                    curve_weight = config.track_geometry_curvature_penalty if config.track_continuity_geometry_enabled else config.track_curvature_penalty
                    cost = (shift_weight * abs(slope) / max(shift, config.epsilon)
                            + curve_weight * curvature / max(shift, config.epsilon)
                            + config.track_skip_penalty * (dx - 1))
                    if config.track_continuity_geometry_enabled:
                        # Stable tilt has zero excess acceleration; local small
                        # undulations are free of this jump cost. Random joins
                        # cannot win merely by adding one noisy strip node.
                        change_scale = max(config.track_max_neighbor_change_ratio * period, config.epsilon)
                        cost += config.track_unreasonable_change_penalty * max(0., curvature / change_scale - 1)
                    key = (i, has_strong or strong)
                    score = value + reward - cost
                    if key not in current or score > current[key][0]:
                        current[key] = (score, path + (j,))
        states.append(current)
        by_strip.setdefault(node.strip_index, []).append(j)
    # DP best paths from every endpoint, followed by conflict-free extraction.
    paths = sorted((entry for state in states for (_, strong), entry in state.items()
                    if recovery or strong), key=lambda entry: entry[0], reverse=True)
    used: set[int] = set()
    tracks = []
    for _, path in paths:
        if any(i in used for i in path):
            continue
        used.update(path)
        tracks.append(make_track([nodes[i] for i in path], period, valid, config,
                                 "recovered" if recovery else "detected"))
    return sorted(tracks, key=lambda t: t.center_y)


def validate_tracks(tracks: list[PaperEdgeTrack], config: CounterConfig,
                    recovery: bool = False) -> tuple[list[PaperEdgeTrack], list[PaperEdgeTrack]]:
    """Apply independent coverage, continuity, curvature and score gates."""
    accepted, rejected = [], []
    support = config.min_recovery_support_ratio if recovery else config.min_track_support_ratio
    continuity = config.min_recovery_continuity_ratio if recovery else config.min_continuity_ratio
    score = config.min_recovery_track_score if recovery else config.min_track_score
    for track in tracks:
        reason = None
        if track.support_count < config.min_absolute_track_support or track.support_ratio < support:
            reason = "low_support"
        elif track.continuity_ratio < continuity:
            reason = "low_continuity"
        elif track.smoothness_score < config.min_track_smoothness:
            reason = "low_smoothness"
        elif not recovery and track.strong_support_count < config.min_strong_support:
            reason = "low_gradient_score"
        elif track.gradient_score < (config.min_recovery_gradient_score if recovery else config.min_track_gradient_score):
            reason = "low_gradient_score"
        elif track.total_score < score:
            reason = "low_total_score"
        track.reason = reason
        (rejected if reason else accepted).append(track)
    return accepted, rejected


def classify_tracks(tracks: list[PaperEdgeTrack], config: CounterConfig
                    ) -> tuple[list[PaperEdgeTrack], list[PaperEdgeTrack], list[PaperEdgeTrack]]:
    """Keep plausible proposals without changing the strong acceptance gates.

    Half the support/continuity gate is the proposal boundary, with at least
    three observed strips. Gradient and smoothness remain evidence gates: a
    periodic location cannot rescue a tiny/noisy/strongly irregular path.
    """
    strong, marginal, garbage = [], [], []
    for i, track in enumerate(tracks):
        if not track.track_id:
            track.track_id = f"seed_{i:05d}"
        accepted, _ = validate_tracks([track], config, recovery=track.source == "recovered")
        reason = track.reason
        if accepted:
            track.candidate_class = "strong"
            strong.append(track)
        else:
            hard = None
            if track.smoothness_score < config.min_track_smoothness:
                hard = "hard_geometry_failure"
            elif (track.support_count < config.min_absolute_track_support or
                  track.support_ratio < 0.5 * config.min_track_support_ratio):
                hard = "low_support"
            elif track.continuity_ratio < 0.5 * config.min_continuity_ratio:
                hard = "low_continuity"
            elif track.gradient_score < (config.min_recovery_gradient_score if track.source == "recovered"
                                         else config.min_track_gradient_score):
                hard = "low_gradient_score"
            elif track.source != "recovered" and track.strong_support_count < config.min_strong_support:
                hard = "low_gradient_score"
            elif track.total_score < config.track_proposal_floor:
                hard = "low_total_score"
            track.candidate_class = "garbage" if hard else "marginal"
            track.reason = hard or reason
            (garbage if hard else marginal).append(track)
        track.classification_reason = track.reason
    return strong, marginal, garbage


def deduplicate_tracks(tracks: list[PaperEdgeTrack], period_at: Callable[[float], float],
                       config: CounterConfig) -> tuple[list[PaperEdgeTrack], list[PaperEdgeTrack]]:
    """Suppress overlapping nearby paths by track quality and gradient evidence."""
    kept, rejected = [], []
    for track in sorted(tracks, key=lambda t: (t.total_score, t.gradient_score), reverse=True):
        duplicate = False
        for other in kept:
            period = period_at((track.center_y + other.center_y) / 2)
            if abs(track.center_y - other.center_y) >= config.duplicate_track_ratio * period:
                continue
            common = set(track.strip_positions) & set(other.strip_positions)
            close = sum(abs(track.strip_positions[i] - other.strip_positions[i]) <=
                        config.duplicate_position_ratio * period for i in common)
            if common and close / min(track.support_count, other.support_count) >= config.duplicate_overlap_ratio:
                duplicate = True
                break
        if duplicate:
            track.reason = "duplicate_track"
            rejected.append(track)
        else:
            kept.append(track)
    return sorted(kept, key=lambda t: t.center_y), rejected


def deduplicate_obvious_tracks(tracks: list[PaperEdgeTrack]) -> tuple[list[PaperEdgeTrack], list[PaperEdgeTrack]]:
    """Only collapse near-identical observed paths before sequence selection.

    At least 80% of the shorter track's observed strips must coincide within
    one pixel. Center distance alone is never a duplicate criterion here.
    """
    kept, rejected = [], []
    for track in sorted(tracks, key=lambda t: t.total_score, reverse=True):
        duplicate = next((other for other in kept if sum(
            abs(y - other.strip_positions[i]) <= 1.0 for i, y in track.strip_positions.items()
            if i in other.strip_positions) >= .8 * min(track.support_count, other.support_count)), None)
        if duplicate is None:
            kept.append(track)
        else:
            track.reason = "duplicate_track"
            track.sequence_decision = "duplicate"
            rejected.append(track)
    return sorted(kept, key=lambda t: t.center_y), rejected
