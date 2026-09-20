"""Local regularity estimates used by the regular-first hybrid detector.

The score is deliberately a *soft* diagnostic/prior.  It never creates a
paper edge by itself.  Observed high quality tracks provide the spacing
anchor; the profile and track quality only modulate confidence around it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import CounterConfig
from .models import PaperEdgeTrack


def _clip01(value: float) -> float:
    return float(np.clip(value if np.isfinite(value) else 0.0, 0.0, 1.0))


@dataclass
class RegularityMap:
    y: np.ndarray
    local_spacing: np.ndarray
    spacing_mad: np.ndarray
    spacing_regularity: np.ndarray
    acf_period: np.ndarray
    acf_score: np.ndarray
    track_regularity: np.ndarray
    gap_geometry_regularity: np.ndarray
    final_regularity: np.ndarray
    zone: list[str]
    regular_threshold: float
    irregular_threshold: float

    def at(self, y: float) -> float:
        if len(self.y) == 0:
            return 0.5
        return float(np.interp(y, self.y, self.final_regularity))

    def sample_at(self, y: float) -> dict:
        if len(self.y) == 0:
            return {"y": float(y), "local_spacing": 0.0, "spacing_mad": 0.0,
                    "spacing_regularity": 0.0, "acf_period": 0.0, "acf_score": 0.0,
                    "track_regularity": 0.0, "gap_geometry_regularity": 0.0,
                    "final_regularity": 0.5, "zone": "MIXED"}
        i = int(np.clip(np.searchsorted(self.y, y), 0, len(self.y) - 1))
        return {"y": float(y), "local_spacing": float(self.local_spacing[i]),
                "spacing_mad": float(self.spacing_mad[i]),
                "spacing_regularity": float(self.spacing_regularity[i]),
                "acf_period": float(self.acf_period[i]), "acf_score": float(self.acf_score[i]),
                "track_regularity": float(self.track_regularity[i]),
                "gap_geometry_regularity": float(self.gap_geometry_regularity[i]),
                "final_regularity": float(self.final_regularity[i]), "zone": self.zone[i]}

    def to_dict(self) -> dict:
        return {"samples": [self.sample_at(float(y)) for y in self.y],
                "regular_threshold": float(self.regular_threshold),
                "irregular_threshold": float(self.irregular_threshold),
                "regular_regions": self.regions("REGULAR"),
                "mixed_regions": self.regions("MIXED"),
                "irregular_regions": self.regions("IRREGULAR"),
                "region_ratios": self.region_ratios()}

    def regions(self, wanted: str) -> list[list[float]]:
        if len(self.y) == 0:
            return []
        out: list[list[float]] = []
        start = None
        for i, z in enumerate(self.zone):
            if z == wanted and start is None:
                start = float(self.y[i])
            if start is not None and (z != wanted or i == len(self.zone) - 1):
                end = float(self.y[i] if z == wanted else self.y[max(0, i - 1)])
                out.append([start, end])
                start = None
        return out

    def region_ratios(self) -> dict[str, float]:
        if len(self.y) == 0:
            return {"REGULAR": 0.0, "MIXED": 0.0, "IRREGULAR": 0.0}
        values = {z: sum(1 for x in self.zone if x == z) for z in ("REGULAR", "MIXED", "IRREGULAR")}
        return {z: float(n / len(self.zone)) for z, n in values.items()}


def _robust_gap_stats(gaps: np.ndarray, initial: float, config: CounterConfig) -> tuple[float, float, float]:
    """Return (local spacing, MAD, inlier density), rejecting texture/missing gaps."""
    gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    if not len(gaps):
        return float(initial), 0.0, 0.0
    anchor = float(initial) if np.isfinite(initial) and initial > 0 else float(np.median(gaps))
    # One robust pass around the physical spacing avoids a 0.5d texture and a
    # 2d missing edge from changing the prior.  If no values survive, retain
    # the local median rather than inventing a period.
    lo, hi = config.regularity_gap_min_ratio * anchor, config.regularity_gap_max_ratio * anchor
    inliers = gaps[(gaps >= lo) & (gaps <= hi)]
    if len(inliers) < 2:
        inliers = gaps
    med = float(np.median(inliers))
    mad = float(np.median(np.abs(inliers - med)))
    density = float(len(inliers) / max(1, len(gaps)))
    return med, mad, density


def _local_acf(profile: np.ndarray | None, y: float, spacing: float,
               window_periods: float, config: CounterConfig) -> tuple[float, float]:
    if profile is None or len(profile) < 4 or not np.isfinite(spacing) or spacing <= 0:
        return 0.0, float(spacing)
    center = int(round(y))
    half = max(int(round(window_periods * spacing)), int(round(2 * spacing)))
    a, b = max(0, center - half), min(len(profile), center + half + 1)
    signal = np.asarray(profile[a:b], dtype=float)
    if len(signal) < max(8, int(2.5 * spacing)):
        return 0.0, float(spacing)
    signal -= np.median(signal)
    norm = float(np.dot(signal, signal))
    if norm <= 1e-9:
        return 0.0, float(spacing)
    # Evaluate the normalized correlation on the plausible spacing interval,
    # then compare the winning lag with the observed local spacing.
    min_lag = max(2, int(round(config.regularity_acf_min_ratio * spacing)))
    max_lag = min(len(signal) - 2, int(round(config.regularity_acf_max_ratio * spacing)))
    if max_lag < min_lag:
        return 0.0, float(spacing)
    scores = []
    for lag in range(min_lag, max_lag + 1):
        scores.append(float(np.dot(signal[:-lag], signal[lag:]) /
                           max(np.sqrt(np.dot(signal[:-lag], signal[:-lag]) * np.dot(signal[lag:], signal[lag:])), 1e-9)))
    j = int(np.argmax(scores))
    acf = _clip01((scores[j] + 1.0) * 0.5)
    lag = float(min_lag + j)
    agreement = np.exp(-abs(lag - spacing) / max(spacing, 1.0))
    return _clip01(acf * agreement), lag


def estimate_local_regularity(tracks: Iterable[PaperEdgeTrack], profile: np.ndarray | None,
                              initial_spacing: float, config: CounterConfig) -> RegularityMap:
    """Estimate continuous local regularity from reliable observed tracks.

    The function accepts an empty pool and is safe for direct use in diagnostics
    and tests.  Samples are placed at track centers, which is sufficient for
    interpolation by the sequence objective and keeps the map tied to evidence.
    """
    seed = sorted((t for t in tracks if t.candidate_class in {"strong", "unclassified"}
                   and np.isfinite(t.center_y)), key=lambda t: t.center_y)
    if not seed:
        return RegularityMap(np.array([], dtype=float), *(np.array([], dtype=float) for _ in range(8)),
                             [], config.regularity_regular_threshold, config.regularity_irregular_threshold)
    ys = np.array([t.center_y for t in seed], dtype=float)
    scores = {k: [] for k in ("spacing", "mad", "acf_period", "acf", "track", "geometry", "spacing_value")}
    half = max(1, int(config.regularity_window_tracks // 2))
    for i, track in enumerate(seed):
        a, b = max(0, i - half), min(len(seed), i + half + 1)
        neighborhood = seed[a:b]
        local_gaps = np.diff([t.center_y for t in neighborhood])
        spacing, mad, density = _robust_gap_stats(local_gaps, initial_spacing, config)
        mad_ratio = 1.4826 * mad / max(spacing, config.epsilon)
        spacing_score = _clip01(np.exp(-mad_ratio / config.regularity_spacing_mad_scale) * (0.55 + 0.45 * density))
        acf_score, acf_period = _local_acf(profile, track.center_y, spacing,
                                            config.regularity_acf_window_periods, config)
        qualities = np.array([float(np.clip(t.total_score, 0, 1)) for t in neighborhood])
        supports = np.array([float(np.clip(t.support_ratio, 0, 1)) for t in neighborhood])
        qmed = float(np.median(qualities)) if len(qualities) else 0.0
        qmad = float(np.median(np.abs(qualities - qmed))) if len(qualities) else 1.0
        smed = float(np.median(supports)) if len(supports) else 0.0
        track_score = _clip01(qmed * np.exp(-qmad / config.regularity_track_mad_scale) * (0.5 + 0.5 * smed))
        # The inlier density is a cheap geometry-consistency proxy.  A local
        # gap profile can refine this later without changing the API.
        geometry_score = _clip01(0.5 * density + 0.5 * np.exp(-mad_ratio / config.regularity_geometry_mad_scale))
        scores["spacing"].append(spacing_score); scores["mad"].append(mad)
        scores["acf_period"].append(acf_period); scores["acf"].append(acf_score)
        scores["track"].append(track_score); scores["geometry"].append(geometry_score)
        scores["spacing_value"].append(spacing)
    weights = np.array([config.regularity_spacing_weight, config.regularity_acf_weight,
                        config.regularity_track_weight, config.regularity_geometry_weight], dtype=float)
    weights /= max(weights.sum(), config.epsilon)
    final = np.clip(weights[0] * np.asarray(scores["spacing"])
                    + weights[1] * np.asarray(scores["acf"])
                    + weights[2] * np.asarray(scores["track"])
                    + weights[3] * np.asarray(scores["geometry"]), 0, 1)
    zones = ["REGULAR" if v >= config.regularity_regular_threshold else
             "IRREGULAR" if v <= config.regularity_irregular_threshold else "MIXED" for v in final]
    return RegularityMap(ys, np.asarray(scores["spacing_value"], dtype=float), np.asarray(scores["mad"], dtype=float),
                         np.asarray(scores["spacing"], dtype=float), np.asarray(scores["acf_period"], dtype=float),
                         np.asarray(scores["acf"], dtype=float), np.asarray(scores["track"], dtype=float),
                         np.asarray(scores["geometry"], dtype=float), final, zones,
                         config.regularity_regular_threshold, config.regularity_irregular_threshold)


def evaluate_middle_track_competition(upper: PaperEdgeTrack, middle: PaperEdgeTrack,
                                      lower: PaperEdgeTrack, local_spacing: float | None = None,
                                      config: CounterConfig | None = None, regularity: float = 0.5) -> dict:
    """Score a possible secondary texture without hard deleting it."""
    config = config or CounterConfig()
    local_spacing = float(local_spacing or abs(lower.center_y - upper.center_y) or config.min_period)
    ab = abs(middle.center_y - upper.center_y)
    bc = abs(lower.center_y - middle.center_y)
    ac = abs(lower.center_y - upper.center_y)
    short = max(0.0, 1.0 - ab / max(config.sequence_short_gap_ratio * local_spacing, config.epsilon))
    short += max(0.0, 1.0 - bc / max(config.sequence_short_gap_ratio * local_spacing, config.epsilon))
    direct = np.exp(-abs(ac - local_spacing) / max(local_spacing, 1.0))
    remainder = np.exp(-abs((ab + bc) - ac) / max(local_spacing, 1.0))
    unary = float(np.clip(middle.total_score, 0, 1))
    score = float(np.clip(0.45 * short * direct * (0.55 + 0.45 * regularity)
                          + 0.25 * remainder * regularity
                          - 0.30 * unary, -1, 1))
    return {"upper_track_id": upper.track_id, "middle_track_id": middle.track_id,
            "lower_track_id": lower.track_id, "ab_gap": ab, "bc_gap": bc, "ac_gap": ac,
            "local_spacing": float(local_spacing), "regularity": float(regularity),
            "short_gap_evidence": float(short), "direct_gap_consistency": float(direct),
            "remainder_consistency": float(remainder), "middle_unary": unary,
            "competition_score": score, "secondary_texture_likely": bool(score > 0.15)}


__all__ = ["RegularityMap", "estimate_local_regularity", "evaluate_middle_track_competition"]
