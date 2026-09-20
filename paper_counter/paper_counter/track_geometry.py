"""Shape-preserving curves and ordered layer gaps; no physical thickness claim."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import CounterConfig
from .local_period import LocalPeriod
from .models import PaperEdgeTrack


@dataclass
class TrackCurve:
    positions: np.ndarray
    observed: np.ndarray
    valid: np.ndarray
    uncertainty: np.ndarray

    def to_dict(self) -> dict:
        return {"positions": [float(y) if v else None for y, v in zip(self.positions, self.valid)],
            "support_mask": self.observed.tolist(), "valid_mask": self.valid.tolist(),
            "uncertainty": [float(u) if v else None for u, v in zip(self.uncertainty, self.valid)]}


def fit_track_curve(track: PaperEdgeTrack, config: CounterConfig) -> TrackCurve:
    """Interpolate only short internal holes; never extrapolate evidence."""
    key = (tuple(sorted(track.strip_positions.items())), config.num_strips,
           config.track_curve_max_interpolation_gap, config.track_curve_observation_uncertainty)
    if getattr(track, "_curve_key", None) == key:
        return track._curve_cache
    xs = np.array(sorted(track.strip_positions), dtype=int)
    ys = np.array([track.strip_positions[int(x)] for x in xs], dtype=float)
    positions = np.zeros(config.num_strips)
    observed = np.zeros(config.num_strips, dtype=bool)
    observed[xs] = True
    positions[xs] = ys
    valid = observed.copy()
    uncertainty = np.full(config.num_strips, config.track_curve_observation_uncertainty)
    slopes = np.diff(ys) / np.maximum(1, np.diff(xs))
    slope_mad = float(np.median(np.abs(slopes - np.median(slopes)))) if len(slopes) else 0.0
    for a, b in zip(xs, xs[1:]):
        if b - a <= config.track_curve_max_interpolation_gap:
            for s in range(a + 1, b):
                positions[s] = positions[a] + (positions[b] - positions[a]) * (s - a) / (b - a)
                valid[s] = True
                uncertainty[s] += slope_mad * min(s - a, b - s)
    curve = TrackCurve(positions, observed, valid, uncertainty)
    track.curve_geometry = {**curve.to_dict(), "method": config.track_curve_method,
                            "observed_y_mad": track.y_mad, "slope_mad": slope_mad}
    track._curve_key, track._curve_cache = key, curve
    return curve


def gap_profile(upper: PaperEdgeTrack, lower: PaperEdgeTrack, config: CounterConfig,
                spacing_scale: float) -> dict:
    """Compare complete overlapping curves and record measurement provenance."""
    a, b = fit_track_curve(upper, config), fit_track_curve(lower, config)
    valid = a.valid & b.valid
    observed = a.observed & b.observed
    xs = np.flatnonzero(valid)
    gaps = b.positions[valid] - a.positions[valid]
    center = lower.center_y - upper.center_y
    if len(gaps):
        med = float(np.median(gaps))
        mad = float(np.median(np.abs(gaps - med)))
        positive = float(np.mean(gaps > config.sequence_min_geometric_gap))
        small = float(np.mean(gaps < config.sequence_short_gap_ratio * spacing_scale))
        derivative = np.diff(gaps) / np.maximum(1, np.diff(xs))
        allowance = max(config.sequence_min_geometric_gap, config.sequence_gap_derivative_ratio * max(abs(med), 1.0))
        variation = float(np.mean(np.maximum(0, np.abs(derivative) / allowance - 1))) if len(derivative) else 0.0
        quality = 1 / (1 + variation)
        uncertainty = float(np.median(a.uncertainty[valid] + b.uncertainty[valid]))
        shared = (int(observed.sum()) + int((valid & ~observed).sum()) / 2) / config.num_strips
        shared /= 1 + uncertainty / max(abs(med), config.epsilon)
        minimum, maximum = float(np.min(gaps)), float(np.max(gaps))
    else:
        med, mad, positive, small, quality, shared, uncertainty = center, 0., 0., 0., 0., 0., 0.
        minimum = maximum = center
    measured = b.positions[observed] - a.positions[observed]
    crossing = float(np.mean(measured <= config.sequence_min_geometric_gap)) if len(measured) else 0.0
    hard = len(measured) >= config.sequence_min_shared_strips and crossing >= config.sequence_crossing_hard_reject_ratio
    return {"upper_track_id": upper.track_id, "lower_track_id": lower.track_id,
        "center_gap": center, "gap_median": med, "gap_mad": mad, "gap_min": minimum, "gap_max": maximum,
        "valid_shared_strip_count": int(valid.sum()), "observed_shared_strip_count": int(observed.sum()),
        "positive_gap_ratio": positive, "non_crossing_ratio": positive, "small_gap_ratio": small,
        "gap_profile_smoothness": quality, "shared_geometry_confidence": shared,
        "track_uncertainty": uncertainty, "crossing_ratio": crossing, "crossing_hard_reject": bool(hard),
        "strip_indices": xs.tolist(), "gap_values": gaps.tolist(),
        "gap_source": "overlapping_curves" if len(gaps) else "center_fallback_no_shared_geometry"}


def geometry_pair_score(upper: PaperEdgeTrack, lower: PaperEdgeTrack, local: LocalPeriod,
                        config: CounterConfig) -> dict:
    scale = local.at((upper.center_y + lower.center_y) / 2)
    result = gap_profile(upper, lower, config, scale)
    values = np.array(result["gap_values"] or [result["center_gap"]])
    # Persistently narrow across the width is stronger evidence than center gap.
    narrow = float(np.mean(np.maximum(0, 1 - values / (config.sequence_short_gap_ratio * scale))))
    short = config.sequence_short_spacing_weight * narrow
    ratio = max(result["gap_median"], config.epsilon) / scale
    long = config.sequence_long_spacing_weight * (max(0., ratio - 1) / max(1., ratio))
    prior = config.sequence_spacing_prior_weight * min(1., abs(np.log(max(ratio, config.epsilon))))
    shared = result["shared_geometry_confidence"]
    # Best geometry is neutral, not a bonus for inserting additional layers.
    # Only image unary provides positive evidence for an extra paper.
    geometry = -shared * (config.sequence_non_crossing_weight * (1 - result["non_crossing_ratio"])
                         + config.sequence_gap_geometry_weight * (1 - result["gap_profile_smoothness"]))
    ambiguity = config.sequence_geometry_ambiguity_weight * (1 - shared)
    crossing_cost = config.sequence_non_crossing_weight * result["crossing_ratio"]
    result.update(local_spacing_scale=scale, short_gap_penalty=short, long_gap_penalty=long,
        spacing_prior_penalty=prior, geometry_score=geometry, neighbor_geometry_score=geometry - ambiguity - crossing_cost,
        geometry_ambiguity_penalty=ambiguity,
        pair_score=geometry - ambiguity - crossing_cost - short - long - prior,
        large_gap_anomaly=ratio > config.sequence_large_gap_anomaly_ratio,
        short_gap_warning=result["small_gap_ratio"] > 0,
        hard_reject=result["crossing_hard_reject"], hard_reject_reason="track_crossing" if result["crossing_hard_reject"] else None)
    return result


def estimate_local_spacing(tracks: list[PaperEdgeTrack], initial: float, config: CounterConfig) -> LocalPeriod:
    """Robust neighbor gap distribution without n*d filtering or decomposition."""
    ordered = sorted(tracks, key=lambda t: t.center_y)
    ys, gaps = [], []
    for a, b in zip(ordered, ordered[1:]):
        profile = gap_profile(a, b, config, initial)
        if profile["valid_shared_strip_count"] >= config.sequence_min_shared_strips and not profile["crossing_hard_reject"] and profile["gap_median"] > 0:
            ys.append((a.center_y + b.center_y) / 2)
            gaps.append(profile["gap_median"])
    if not gaps:
        return LocalPeriod(np.array([]), np.array([]), np.array([]), initial)
    values, deviations = [], []
    half = config.local_period_neighbors // 2
    for i in range(len(gaps)):
        group = np.array(gaps[max(0, i - half):i + half + 1])
        median = float(np.median(group))
        values.append(median)
        deviations.append(float(np.median(np.abs(group - median))))
    return LocalPeriod(np.array(ys), np.array(values), np.array(deviations), initial)
