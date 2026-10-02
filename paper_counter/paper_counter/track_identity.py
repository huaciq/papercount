"""Conservative image-boundary identity features for final selected tracks.

The main detector answers whether a coherent horizontal structure exists. This
module runs after recovery and asks whether a small set of suspicious selected
tracks looks like the image's typical paper-bottom boundary.  It uses no image
names, annotation coordinates, learned model, or global material thresholds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.signal import find_peaks, peak_widths

from .config import CounterConfig
from .models import PaperEdgeTrack
from .raw_mask import RawStackMask, StackState


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return 0.0
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def _robust_normalize(profile: np.ndarray, epsilon: float) -> np.ndarray:
    values = np.asarray(profile, dtype=float)
    med = float(np.median(values))
    scale = 1.4826 * _mad(values)
    if scale <= epsilon:
        scale = float(np.percentile(values, 90) - np.percentile(values, 10))
    return (values - med) / max(scale, epsilon)


def _correlation(a: np.ndarray, b: np.ndarray, epsilon: float) -> float:
    aa = _robust_normalize(a, epsilon)
    bb = _robust_normalize(b, epsilon)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / max(denom, epsilon))


def make_intensity_strip_profiles(image: np.ndarray, strips: list[tuple[int, int]],
                                  chunk_rows: int = 1024) -> np.ndarray:
    """Median vertical intensity profile for every detector strip."""
    profiles = np.empty((len(strips), image.shape[0]), dtype=np.float32)
    for strip, (x1, x2) in enumerate(strips):
        for start in range(0, len(image), chunk_rows):
            end = min(len(image), start + chunk_rows)
            profiles[strip, start:end] = np.median(image[start:end, x1:x2], axis=1)
    return profiles


@dataclass
class AlignedTrackProfiles:
    relative_y: np.ndarray
    raw: np.ndarray
    normalized: np.ndarray
    negative: np.ndarray
    positive: np.ndarray
    raw_by_strip: np.ndarray
    normalized_by_strip: np.ndarray
    negative_by_strip: np.ndarray
    positive_by_strip: np.ndarray
    profile_across_strip_mad: float


def extract_track_aligned_profiles(track: PaperEdgeTrack, local_spacing: float,
                                   raw_profiles: np.ndarray,
                                   normalized_profiles: np.ndarray,
                                   negative_profiles: np.ndarray,
                                   positive_profiles: np.ndarray,
                                   y_offset: float, config: CounterConfig) -> AlignedTrackProfiles:
    """Sample around observed Y in a canonical paper-to-dark direction.

    For positive boundaries the local Y axis is reversed and polarity channels
    are exchanged. Historical negative/above fields then mean primary/paper-side;
    global track coordinates remain untouched.
    """
    relative = np.linspace(-config.identity_profile_extent_ratio,
                           config.identity_profile_extent_ratio,
                           config.identity_profile_samples)
    positive_primary = config.gradient_polarity == "positive"
    direction = -1.0 if positive_primary else 1.0
    arrays = (raw_profiles, normalized_profiles,
              positive_profiles if positive_primary else negative_profiles,
              negative_profiles if positive_primary else positive_profiles)
    sampled: list[list[np.ndarray]] = [[] for _ in arrays]
    for strip, y in sorted(track.strip_positions.items()):
        if not 0 <= int(strip) < raw_profiles.shape[0]:
            continue
        coordinates = float(y) - float(y_offset) + direction * relative * max(local_spacing, config.epsilon)
        if np.min(coordinates) < 0 or np.max(coordinates) > raw_profiles.shape[1] - 1:
            continue
        for target, source in zip(sampled, arrays):
            target.append(np.interp(coordinates, np.arange(source.shape[1]), source[int(strip)]))
    if not sampled[0]:
        empty = np.empty((0, len(relative)), dtype=float)
        zeros = np.zeros(len(relative), dtype=float)
        return AlignedTrackProfiles(relative, zeros, zeros, zeros, zeros,
                                    empty, empty, empty, empty, 1.0)
    matrices = [np.asarray(values, dtype=float) for values in sampled]
    aggregates = [np.median(matrix, axis=0) for matrix in matrices]
    normalized_strips = np.asarray([_robust_normalize(row, config.epsilon) for row in matrices[2]])
    strip_mad = float(np.median(np.abs(normalized_strips - np.median(normalized_strips, axis=0))))
    return AlignedTrackProfiles(relative, *aggregates, *matrices, strip_mad)


@dataclass
class EdgeTemplate:
    status: str
    reference_track_ids: list[str]
    raw: np.ndarray
    normalized: np.ndarray
    negative: np.ndarray
    positive: np.ndarray
    raw_mad: np.ndarray
    normalized_mad: np.ndarray
    negative_mad: np.ndarray
    positive_mad: np.ndarray

    def to_debug_dict(self) -> dict[str, Any]:
        return {"status": self.status, "reference_track_ids": self.reference_track_ids,
                "reference_track_count": len(self.reference_track_ids)}


def build_edge_template(profiles: dict[str, AlignedTrackProfiles], reference_ids: list[str],
                        config: CounterConfig) -> EdgeTemplate:
    """Build an iterative median/MAD image template; never use the mean."""
    usable = [track_id for track_id in reference_ids if track_id in profiles]
    names = ("raw", "normalized", "negative", "positive")
    size = config.identity_profile_samples
    if not usable:
        zeros = np.zeros(size, dtype=float)
        return EdgeTemplate("insufficient", [], zeros, zeros, zeros, zeros,
                            np.ones(size), np.ones(size), np.ones(size), np.ones(size))
    # A median is already robust. One conservative trim removes only extreme
    # whole-profile outliers when enough references remain.
    negative = np.stack([_robust_normalize(profiles[track_id].negative, config.epsilon)
                         for track_id in usable])
    provisional = np.median(negative, axis=0)
    distances = np.median(np.abs(negative - provisional), axis=1)
    center, spread = float(np.median(distances)), 1.4826 * _mad(distances)
    if len(usable) >= 2 * config.identity_template_min_tracks and spread > config.epsilon:
        keep = distances <= center + config.identity_template_trim_mad_k * spread
        if int(np.sum(keep)) >= config.identity_template_min_tracks:
            usable = [track_id for track_id, accepted in zip(usable, keep) if accepted]
    medians, mads = {}, {}
    for name in names:
        matrix = np.stack([getattr(profiles[track_id], name) for track_id in usable])
        medians[name] = np.median(matrix, axis=0)
        mads[name] = np.median(np.abs(matrix - medians[name]), axis=0)
    status = "ok" if len(usable) >= config.identity_template_min_tracks else "insufficient"
    return EdgeTemplate(status, usable, *(medians[name] for name in names),
                        *(mads[name] for name in names))


@dataclass
class TrackIdentityFeatures:
    track_id: str
    negative_template_similarity: float
    raw_template_similarity: float
    positive_template_similarity: float
    normalized_template_similarity: float
    negative_profile_corr: float
    negative_profile_robust_distance: float
    raw_profile_corr: float
    raw_profile_robust_distance: float
    positive_profile_corr: float
    normalized_profile_distance: float
    negative_prominence: float
    negative_width: float
    negative_asymmetry: float
    negative_isolation: float
    local_second_strongest_peak: float
    peak_neighbor_ratio: float
    above_raw_level: float
    below_raw_level: float
    above_raw_p10: float
    above_raw_p90: float
    below_raw_p10: float
    below_raw_p90: float
    above_texture: float
    below_texture: float
    above_normalized_level: float
    below_normalized_level: float
    paper_support_above: float
    raw_signed_contrast: float
    background_above_score: float
    background_below_score: float
    distance_to_stack_boundary: float
    above_mask_state: str
    below_mask_state: str
    background_connected_above: bool
    background_connected_below: bool
    positive_pair_score: float
    positive_pair_support: float
    positive_pair_distance_ratio: float
    positive_pair_distance_mad: float
    positive_pair_geometry_similarity: float
    band_evidence_score: float
    dark_run_below_ratio: float
    rebound_distance_ratio: float
    below_transition_score: float
    profile_across_strip_mad: float
    pair_offset_mad: float
    contrast_mad: float
    identity_strip_consistency: float
    regularity: float
    candidate_class: str
    source: str
    raw_mask_state: str
    sequence_decision: str
    local_spacing: float
    template_status: str
    template_distance: float = 0.0
    template_distance_outlier_z: float = 0.0
    raw_similarity_outlier_z: float = 0.0
    positive_corr_outlier_z: float = 0.0
    template_outlier_z: float = 0.0
    identity_score: float = 0.0
    identity_z_score: float = 0.0
    needs_final_verification: bool = False
    suspicious_reasons: list[str] | None = None
    evidence_families: list[str] | None = None
    false_family: str = "NONE"
    proposed_decision: str = "TRUSTED"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _profile_comparison(profile: np.ndarray, template: np.ndarray, envelope: np.ndarray,
                        epsilon: float) -> tuple[float, float, float]:
    corr = _correlation(profile, template, epsilon)
    p = _robust_normalize(profile, epsilon)
    t = _robust_normalize(template, epsilon)
    # Envelope prevents nearly zero template MAD from exploding distances.
    floor = max(float(np.median(envelope)), .15)
    distance = float(np.median(np.abs(p - t) / np.maximum(envelope, floor)))
    similarity = _clip01(.5 * (corr + 1.0) * np.exp(-.35 * distance))
    return corr, distance, similarity


def _negative_shape(profile: np.ndarray, config: CounterConfig) -> dict[str, float]:
    values = np.asarray(profile, dtype=float)
    center = len(values) // 2
    baseline = float(np.median(np.r_[values[:max(1, center // 2)], values[-max(1, center // 2):]]))
    prominence = max(0.0, float(values[center] - baseline))
    peaks, properties = find_peaks(values, prominence=(0, None), width=(0, None))
    if len(peaks):
        nearest = int(np.argmin(abs(peaks - center)))
        width = float(properties["widths"][nearest]) / max(1, len(values) - 1)
        peak_prominence = float(properties["prominences"][nearest])
    else:
        width, peak_prominence = 0.0, prominence
    exclusion = max(2, int(round(.08 * len(values))))
    mask = np.ones(len(values), dtype=bool)
    mask[max(0, center - exclusion):min(len(values), center + exclusion + 1)] = False
    second = float(np.max(values[mask])) if np.any(mask) else 0.0
    isolation = _clip01((values[center] - second) / max(abs(values[center]), config.epsilon) * .5 + .5)
    ratio = float(max(0.0, values[center]) / max(max(0.0, second), config.epsilon))
    half = max(1, int(round(.15 * len(values))))
    left = values[max(0, center - half):center]
    right = values[center + 1:min(len(values), center + half + 1)]
    asymmetry = float(abs(np.sum(left) - np.sum(right)) / max(abs(np.sum(left)) + abs(np.sum(right)), config.epsilon))
    return {"prominence": max(prominence, peak_prominence), "width": width,
            "asymmetry": asymmetry, "isolation": isolation,
            "second": second, "ratio": ratio}


def _region(values: np.ndarray, relative: np.ndarray, low: float, high: float) -> dict[str, float]:
    chosen = np.asarray(values)[(relative >= low) & (relative <= high)]
    if not len(chosen):
        return {"median": 0.0, "p10": 0.0, "p90": 0.0, "texture": 0.0}
    return {"median": float(np.median(chosen)), "p10": float(np.percentile(chosen, 10)),
            "p90": float(np.percentile(chosen, 90)),
            "texture": float(np.median(np.abs(np.diff(chosen)))) if len(chosen) > 1 else 0.0}


def _paper_score(level: float, mask: RawStackMask, config: CounterConfig) -> float:
    rows = np.asarray(mask.statistics["row_median"], dtype=float)
    core = rows[mask.states == StackState.CORE]
    invalid = rows[mask.states == StackState.INVALID]
    if not len(core):
        return .5
    paper = float(np.median(core))
    background = float(np.median(invalid)) if len(invalid) else float(np.percentile(core, 10))
    if abs(paper - background) <= config.epsilon:
        return .5
    return _clip01((level - background) / (paper - background))


def _mask_context(y: float, spacing: float, mask: RawStackMask, direction: float = 1.0) -> dict[str, Any]:
    above_y, below_y = y - direction * .25 * spacing, y + direction * .25 * spacing
    above, below = mask.state_at(above_y), mask.state_at(below_y)
    transitions = np.flatnonzero(np.diff(mask.states.astype(int)) != 0) + mask.offset
    distance = float(np.min(np.abs(transitions - y))) if len(transitions) else float(len(mask.states))
    return {"above": StackState(above).name, "below": StackState(below).name,
            "distance": distance,
            "connected_above": bool(above == StackState.INVALID),
            "connected_below": bool(below == StackState.INVALID)}


def background_false_evidence(paper_support_above: float, background_connected_above: bool,
                              above_mask_state: str, distance_to_boundary: float,
                              local_spacing: float, config: CounterConfig) -> bool:
    near_or_background = (background_connected_above or above_mask_state == "INVALID"
                          or distance_to_boundary < .5 * local_spacing)
    return bool(near_or_background and paper_support_above < config.identity_low_paper_support)


def conservative_identity_decision(families: list[str], needs_verification: bool,
                                   identity_z_score: float, config: CounterConfig
                                   ) -> tuple[str, str]:
    """Require two independent evidence families; uncertainty always keeps."""
    unique = set(families)
    if not needs_verification:
        return "NONE", "TRUSTED"
    enough = len(unique) >= config.identity_min_reject_evidence_families
    if enough and "background" in unique and ("template" in unique or "paper_support" in unique):
        return "BACKGROUND_FALSE_EDGE", "REJECT_BACKGROUND"
    if enough and "competition" in unique and "template" in unique:
        return "SECONDARY_TEXTURE", "REJECT_SECONDARY_TEXTURE"
    if enough and "template" in unique and unique & {"negative_shape", "paper_support"}:
        return "SECONDARY_TEXTURE", "REJECT_SECONDARY_TEXTURE"
    if identity_z_score < -config.identity_ambiguous_z:
        return "UNKNOWN_AMBIGUOUS", "AMBIGUOUS_KEEP"
    return "NONE", "SUSPICIOUS_KEEP"


def _positive_pair(track: PaperEdgeTrack, spacing: float, positive_profiles: np.ndarray,
                   y_offset: float, config: CounterConfig) -> dict[str, float]:
    offsets, strengths = [], []
    direction = -1.0 if config.gradient_polarity == "positive" else 1.0
    for strip, y in track.strip_positions.items():
        signal = positive_profiles[int(strip)]
        center = float(y) - y_offset
        endpoints = (center - direction * 1.20 * spacing, center - direction * .10 * spacing)
        a = max(0, int(np.floor(min(endpoints))))
        b = min(len(signal), int(np.ceil(max(endpoints))) + 1)
        if b <= a:
            continue
        local = signal[a:b]
        index = int(np.argmax(local))
        strength = float(local[index])
        if strength >= config.identity_positive_pair_min_response:
            offsets.append(direction * (center - (a + index)))
            strengths.append(strength)
    support = len(offsets) / max(1, len(track.strip_positions))
    median = float(np.median(offsets)) if offsets else 0.0
    mad = _mad(np.asarray(offsets)) if offsets else spacing
    geometry = float(np.exp(-mad / max(.15 * spacing, config.epsilon)))
    strength_score = _clip01(float(np.median(strengths)) / max(config.identity_positive_pair_reference, config.epsilon)) if strengths else 0.0
    score = _clip01(support * geometry * (.5 + .5 * strength_score))
    return {"support": support, "distance": median / max(spacing, config.epsilon),
            "mad": mad / max(spacing, config.epsilon), "geometry": geometry, "score": score}


def _transition_features(profile: np.ndarray, relative: np.ndarray) -> tuple[float, float, float]:
    below = np.asarray(profile)[relative >= .05]
    if not len(below):
        return 0.0, 0.0, 0.0
    low = float(np.percentile(profile, 25))
    dark = below <= low
    run = 0
    for value in dark:
        if not value:
            break
        run += 1
    dark_ratio = run / max(1, len(below))
    rebound = next((i for i, value in enumerate(below) if value >= np.percentile(profile, 60)), len(below))
    rebound_ratio = rebound / max(1, len(below))
    score = _clip01(.5 * dark_ratio + .5 * (1 - rebound_ratio))
    return dark_ratio, rebound_ratio, score


def select_reference_track_ids(tracks: list[PaperEdgeTrack], regularity: Any,
                               competitions: list[dict], config: CounterConfig) -> list[str]:
    competition_ids = {row["middle_track_id"] for row in competitions
                       if row.get("secondary_texture_likely")}
    selected = []
    for track in tracks:
        r = float(regularity.at(track.center_y)) if regularity is not None else .5
        if (track.candidate_class == "strong" and track.source == "detected"
                and track.sequence_decision != "rescued" and track.raw_mask_state == "CORE"
                and track.support_ratio >= config.identity_reference_min_support
                and track.total_score >= config.identity_reference_min_score
                and r >= config.regularity_regular_threshold
                and track.track_id not in competition_ids):
            selected.append(track.track_id)
    # If a material has too few regular-zone tracks, retain the same strong
    # image-evidence requirements and relax only the regularity context.
    if len(selected) < config.identity_template_min_tracks:
        selected = [track.track_id for track in tracks
                    if track.candidate_class == "strong" and track.source == "detected"
                    and track.sequence_decision != "rescued" and track.raw_mask_state == "CORE"
                    and track.support_ratio >= config.identity_reference_min_support
                    and track.total_score >= config.identity_reference_min_score
                    and track.track_id not in competition_ids]
    return selected


def analyze_track_identity(tracks: list[PaperEdgeTrack], raw_profiles: np.ndarray,
                           normalized_profiles: np.ndarray, negative_profiles: np.ndarray,
                           positive_profiles: np.ndarray, y_offset: float, local: Any,
                           period: float, regularity: Any, raw_mask: RawStackMask,
                           competitions: list[dict], spacing_anomalies: list[dict],
                           config: CounterConfig) -> tuple[dict[str, TrackIdentityFeatures], dict[str, Any]]:
    """Extract aligned profiles, robust template, and explainable decisions."""
    aligned: dict[str, AlignedTrackProfiles] = {}
    spacing_by_id = {}
    for track in tracks:
        spacing = float(local.at(track.center_y)) if local is not None else float(period)
        spacing_by_id[track.track_id] = spacing
        aligned[track.track_id] = extract_track_aligned_profiles(
            track, spacing, raw_profiles, normalized_profiles,
            negative_profiles, positive_profiles, y_offset, config,
        )
    reference_ids = select_reference_track_ids(tracks, regularity, competitions, config)
    template = build_edge_template(aligned, reference_ids, config)
    competition_by_id = {row["middle_track_id"]: row for row in competitions
                         if row.get("secondary_texture_likely")}
    anomaly_ids = {item.get("left_track_id") for item in spacing_anomalies} | {
        item.get("right_track_id") for item in spacing_anomalies}
    features: dict[str, TrackIdentityFeatures] = {}
    for track in tracks:
        profile = aligned[track.track_id]
        spacing = spacing_by_id[track.track_id]
        neg_corr, neg_dist, neg_sim = _profile_comparison(
            profile.negative, template.negative, template.negative_mad, config.epsilon)
        raw_corr, raw_dist, raw_sim = _profile_comparison(
            profile.raw, template.raw, template.raw_mad, config.epsilon)
        pos_corr, _, pos_sim = _profile_comparison(
            profile.positive, template.positive, template.positive_mad, config.epsilon)
        _, norm_dist, norm_sim = _profile_comparison(
            profile.normalized, template.normalized, template.normalized_mad, config.epsilon)
        shape = _negative_shape(profile.negative, config)
        above = _region(profile.raw, profile.relative_y, -.55, -.10)
        below = _region(profile.raw, profile.relative_y, .05, .45)
        above_norm = _region(profile.normalized, profile.relative_y, -.55, -.10)
        below_norm = _region(profile.normalized, profile.relative_y, .05, .45)
        paper_above = _paper_score(above["median"], raw_mask, config)
        paper_below = _paper_score(below["median"], raw_mask, config)
        direction = -1.0 if config.gradient_polarity == "positive" else 1.0
        mask_context = _mask_context(track.center_y, spacing, raw_mask, direction)
        pair_profiles = negative_profiles if config.gradient_polarity == "positive" else positive_profiles
        pair = _positive_pair(track, spacing, pair_profiles, y_offset, config)
        dark, rebound, transition = _transition_features(profile.raw, profile.relative_y)
        contrast_by_strip = []
        for row in profile.raw_by_strip:
            a = _region(row, profile.relative_y, -.55, -.10)["median"]
            b = _region(row, profile.relative_y, .05, .45)["median"]
            contrast_by_strip.append(a - b)
        contrast_mad = _mad(np.asarray(contrast_by_strip))
        consistency = _clip01(np.exp(-profile.profile_across_strip_mad)
                              * np.exp(-contrast_mad / max(abs(above["median"] - below["median"]), 1.0)))
        r = float(regularity.at(track.center_y)) if regularity is not None else .5
        features[track.track_id] = TrackIdentityFeatures(
            track.track_id, neg_sim, raw_sim, pos_sim, norm_sim,
            neg_corr, neg_dist, raw_corr, raw_dist, pos_corr, norm_dist,
            shape["prominence"], shape["width"], shape["asymmetry"], shape["isolation"],
            shape["second"], shape["ratio"],
            above["median"], below["median"], above["p10"], above["p90"],
            below["p10"], below["p90"], above["texture"], below["texture"],
            above_norm["median"], below_norm["median"], paper_above,
            above["median"] - below["median"], 1 - paper_above, 1 - paper_below,
            mask_context["distance"], mask_context["above"], mask_context["below"],
            mask_context["connected_above"], mask_context["connected_below"],
            pair["score"], pair["support"], pair["distance"], pair["mad"], pair["geometry"],
            _clip01(pair["score"] * (.5 + .5 * paper_above)), dark, rebound, transition,
            profile.profile_across_strip_mad, pair["mad"], contrast_mad, consistency,
            r, track.candidate_class, track.source, track.raw_mask_state,
            track.sequence_decision or "selected", spacing, template.status,
            template_distance=float(np.median([neg_dist, raw_dist, norm_dist])),
        )

    reference_distances = np.asarray([features[i].template_distance for i in template.reference_track_ids])
    distance_center = float(np.median(reference_distances)) if len(reference_distances) else 0.0
    distance_scale = max(1.4826 * _mad(reference_distances), config.identity_template_distance_floor)
    reference_raw_similarity = np.asarray([features[i].raw_template_similarity for i in template.reference_track_ids])
    reference_positive_corr = np.asarray([features[i].positive_profile_corr for i in template.reference_track_ids])
    raw_center = float(np.median(reference_raw_similarity)) if len(reference_raw_similarity) else .5
    positive_center = float(np.median(reference_positive_corr)) if len(reference_positive_corr) else 0.0
    raw_scale = max(1.4826 * _mad(reference_raw_similarity), config.identity_similarity_mad_floor)
    positive_scale = max(1.4826 * _mad(reference_positive_corr), config.identity_similarity_mad_floor)
    weights = config.identity_score_weights
    ordered_tracks = sorted(tracks, key=lambda item: item.center_y)
    nearest_gap_ratio: dict[str, float] = {}
    for index, track in enumerate(ordered_tracks):
        spacing = max(spacing_by_id[track.track_id], config.epsilon)
        neighbor_gaps = [abs(track.center_y - ordered_tracks[j].center_y) / spacing
                         for j in (index - 1, index + 1)
                         if 0 <= j < len(ordered_tracks)]
        nearest_gap_ratio[track.track_id] = min(neighbor_gaps, default=float("inf"))
    for track in tracks:
        feature = features[track.track_id]
        template_score = float(np.median([feature.negative_template_similarity,
                                          feature.raw_template_similarity,
                                          feature.normalized_template_similarity]))
        negative_score = _clip01(.5 * feature.negative_isolation
                                 + .5 * min(1.0, feature.peak_neighbor_ratio / 2.0))
        context = _clip01(.5 * track.total_score + .5 * feature.regularity)
        values = {"template": template_score, "negative": negative_score,
                  "above": feature.paper_support_above, "positive": feature.positive_pair_score,
                  "consistency": feature.identity_strip_consistency, "context": context}
        effective = dict(weights)
        if template.status != "ok":
            effective["template"] *= config.identity_insufficient_template_weight_scale
        total_weight = max(sum(effective.values()), config.epsilon)
        feature.identity_score = float(sum(effective[key] * values[key] for key in values) / total_weight)
        feature.template_distance_outlier_z = max(0.0, (feature.template_distance - distance_center) / distance_scale)
        feature.raw_similarity_outlier_z = max(0.0, (raw_center - feature.raw_template_similarity) / raw_scale)
        feature.positive_corr_outlier_z = max(0.0, (positive_center - feature.positive_profile_corr) / positive_scale)
        # Both RAW boundary shape and the paired positive-band morphology must
        # be abnormal. This protects a true neighbor that differs on only one
        # profile family, as seen in close gold-card tracks.
        feature.template_outlier_z = min(feature.raw_similarity_outlier_z,
                                         feature.positive_corr_outlier_z)
        reasons = []
        if track.raw_mask_state == "MARGIN": reasons.append("margin_track")
        if feature.distance_to_stack_boundary <= config.identity_near_boundary_ratio * feature.local_spacing:
            reasons.append("near_raw_boundary")
        if track.source == "recovered" or track.sequence_decision == "rescued": reasons.append("rescued_track")
        if track.candidate_class == "marginal": reasons.append("marginal_track")
        if track.track_id in competition_by_id: reasons.append("short_gap_competition")
        if track.track_id in anomaly_ids: reasons.append("local_gap_anomaly")
        if feature.template_outlier_z >= config.identity_suspicious_outlier_z: reasons.append("template_identity_anomaly")
        if feature.paper_support_above < config.identity_low_paper_support: reasons.append("low_paper_support_above")
        if feature.positive_pair_score < config.identity_low_positive_pair_score: reasons.append("positive_pair_abnormal")
        rescued_short_gap = ((track.source == "recovered" or track.sequence_decision == "rescued"
                              or track.candidate_class == "marginal")
                             and nearest_gap_ratio[track.track_id]
                             <= config.identity_close_competition_gap_ratio
                             and feature.template_outlier_z >= config.identity_reject_outlier_z)
        if rescued_short_gap:
            reasons.append("rescued_short_gap_competition")
        feature.needs_final_verification = bool(reasons)
        feature.suspicious_reasons = reasons

    reference_scores = np.asarray([features[i].identity_score for i in template.reference_track_ids])
    score_center = float(np.median(reference_scores)) if len(reference_scores) else .5
    score_scale = max(1.4826 * _mad(reference_scores), config.identity_score_mad_floor)
    for track in tracks:
        feature = features[track.track_id]
        feature.identity_z_score = (feature.identity_score - score_center) / score_scale
        families = []
        if background_false_evidence(feature.paper_support_above, feature.background_connected_above,
                                     feature.above_mask_state, feature.distance_to_stack_boundary,
                                     feature.local_spacing, config):
            families.append("background")
        if feature.paper_support_above < config.identity_low_paper_support:
            families.append("paper_support")
        if (feature.template_status == "ok"
                and feature.template_outlier_z >= config.identity_reject_outlier_z):
            families.append("template")
        if track.track_id in competition_by_id:
            families.append("competition")
        if "rescued_short_gap_competition" in (feature.suspicious_reasons or []):
            families.append("competition")
        if feature.identity_strip_consistency < config.identity_low_strip_consistency:
            families.append("strip_consistency")
        if feature.negative_isolation < config.identity_low_negative_isolation:
            families.append("negative_shape")
        feature.evidence_families = families
        feature.false_family, feature.proposed_decision = conservative_identity_decision(
            families, feature.needs_final_verification, feature.identity_z_score, config)

    debug = {
        "primary_polarity": config.gradient_polarity,
        "profile_coordinate_system": "canonical paper-to-dark; global track Y unchanged",
        "physical_paper_side": "below" if config.gradient_polarity == "positive" else "above",
        "legacy_field_semantics": {"negative": "primary gradient", "positive": "opposite gradient",
                                   "above": "paper side", "below": "gap side"},
        "template": template.to_debug_dict(),
        "reference_track_ids": template.reference_track_ids,
        "template_distance_distribution": {"median": distance_center, "mad_scale": distance_scale},
        "identity_score_distribution": {"median": score_center, "mad_scale": score_scale},
        "features": [features[track.track_id].to_dict() for track in tracks],
    }
    artifacts = {"relative_y": next(iter(aligned.values())).relative_y if aligned else np.linspace(
                     -config.identity_profile_extent_ratio, config.identity_profile_extent_ratio,
                     config.identity_profile_samples),
                 "profiles": aligned, "template": template}
    return features, {"debug": debug, "artifacts": artifacts}


REJECT_DECISIONS = {"REJECT_BACKGROUND", "REJECT_SECONDARY_TEXTURE", "REJECT_DUPLICATE"}


def _boundary_morphology_score(feature: TrackIdentityFeatures) -> float:
    """Score only aligned boundary shape, excluding material brightness context."""
    positive_corr = _clip01(.5 * (feature.positive_profile_corr + 1.0))
    return float(.30 * feature.raw_template_similarity
                 + .25 * feature.normalized_template_similarity
                 + .25 * positive_corr
                 + .20 * feature.negative_template_similarity)


def apply_final_track_verifier(tracks: list[PaperEdgeTrack], features: dict[str, TrackIdentityFeatures],
                               config: CounterConfig) -> tuple[list[PaperEdgeTrack], list[PaperEdgeTrack], dict]:
    """Apply only final decisions; no recovery is called after this function."""
    proposed_ids = {track.track_id for track in tracks
                    if features[track.track_id].proposed_decision in REJECT_DECISIONS}
    ordered = sorted(tracks, key=lambda item: item.center_y)
    pair_adjudications = []
    # A low-paper-context rule can choose the wrong member of a close pair.
    # When both are direct, well-supported detections and both are template
    # outliers, keep the member whose aligned boundary morphology is clearly
    # closer to the image template and reject its close competitor instead.
    for index, track in enumerate(ordered):
        if track.track_id not in proposed_ids:
            continue
        feature = features[track.track_id]
        if (track.source != "detected" or track.candidate_class != "strong"
                or track.support_ratio < config.identity_reference_min_support
                or "paper_support" not in (getattr(feature, "evidence_families", None) or [])):
            continue
        current_score = _boundary_morphology_score(feature)
        choices = []
        for neighbor_index in (index - 1, index + 1):
            if not 0 <= neighbor_index < len(ordered):
                continue
            neighbor = ordered[neighbor_index]
            if neighbor.track_id in proposed_ids or neighbor.source != "detected" or neighbor.candidate_class != "strong":
                continue
            other = features[neighbor.track_id]
            gap_ratio = abs(track.center_y - neighbor.center_y) / max(feature.local_spacing, config.epsilon)
            if (gap_ratio > config.identity_close_competition_gap_ratio
                    or neighbor.support_ratio < config.identity_reference_min_support
                    or other.template_outlier_z < config.identity_reject_outlier_z
                    or other.identity_z_score >= -config.identity_ambiguous_z
                    or "template" not in (other.evidence_families or [])
                    or "paper_support" in (other.evidence_families or [])):
                continue
            advantage = current_score - _boundary_morphology_score(other)
            if advantage >= config.identity_pair_morphology_margin:
                choices.append((advantage, neighbor, other, gap_ratio))
        if not choices:
            continue
        advantage, neighbor, other, gap_ratio = max(choices, key=lambda item: item[0])
        proposed_ids.remove(track.track_id)
        proposed_ids.add(neighbor.track_id)
        pair_adjudications.append({
            "kept_track_id": track.track_id,
            "rejected_competitor_track_id": neighbor.track_id,
            "gap_ratio": gap_ratio,
            "kept_morphology_score": current_score,
            "competitor_morphology_score": _boundary_morphology_score(other),
            "morphology_advantage": advantage,
            "reason": "close_pair_boundary_morphology_swap",
        })
    raw_conflicts = []
    if config.strip_sequence_raw_body_safeguard_enabled:
        for track in ordered:
            feature = features[track.track_id]
            proof = track.local_evidence.get('raw_body_confirmation', {})
            if (track.track_id in proposed_ids and proof.get('supported') and
                    feature.false_family == 'SECONDARY_TEXTURE' and track.raw_mask_state != 'INVALID'):
                proposed_ids.remove(track.track_id)
                raw_conflicts.append(track.track_id)
    proposed = [track for track in tracks if track.track_id in proposed_ids]
    rejected = [] if config.final_track_verifier_shadow_mode else proposed
    rejected_ids = {track.track_id for track in rejected}
    kept = [track for track in tracks if track.track_id not in rejected_ids]
    records = []
    for index, track in enumerate(ordered):
        feature = features[track.track_id]
        records.append({"track_id": track.track_id, "center_y": track.center_y,
                        "identity_score": feature.identity_score,
                        "identity_z_score": feature.identity_z_score,
                        "false_family": feature.false_family,
                        "proposed_decision": feature.proposed_decision,
                        "terminal_decision": ("KEEP_RAW_BODY_CONFLICT_REVIEW" if track.track_id in raw_conflicts else
                            "REJECT_PAIR_COMPETITOR" if any(
                            row["rejected_competitor_track_id"] == track.track_id
                            for row in pair_adjudications) else
                            "KEEP_PAIR_MORPHOLOGY" if any(
                                row["kept_track_id"] == track.track_id
                                for row in pair_adjudications) else feature.proposed_decision),
                        "applied": track.track_id in rejected_ids,
                        "neighbor_track_ids": [item.track_id for item in ordered[max(0, index - 1):index]
                                               + ordered[index + 1:index + 2]],
                        "features": feature.to_dict()})
    return kept, rejected, {
        "enabled": bool(config.final_track_verifier_enabled),
        "shadow_mode": bool(config.final_track_verifier_shadow_mode),
        "count_before": len(tracks), "count_after": len(kept),
        "proposed_reject_count": len(proposed), "applied_reject_count": len(rejected),
        "proposed_reject_track_ids": [track.track_id for track in proposed],
        "applied_reject_track_ids": [track.track_id for track in rejected],
        "pair_adjudications": pair_adjudications,
        "raw_body_conflicts_retained": raw_conflicts,
        "track_decisions": records,
        "recovery_after_verifier": False,
    }


__all__ = [
    "AlignedTrackProfiles", "EdgeTemplate", "TrackIdentityFeatures",
    "make_intensity_strip_profiles", "extract_track_aligned_profiles",
    "build_edge_template", "select_reference_track_ids", "analyze_track_identity",
    "background_false_evidence", "conservative_identity_decision",
    "apply_final_track_verifier", "REJECT_DECISIONS",
]
