from __future__ import annotations

"""Validated, serializable algorithm configuration."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class CounterConfig:
    roi: Optional[dict[str, int]] = None
    illumination_sigma: float = 15.0
    epsilon: float = 1e-6
    gradient_polarity: str = "negative"
    num_strips: int = 12
    strip_aggregation_mode: str = "median"
    strip_width_ratio: float = 1.0
    strip_alignment_bins: int = 5
    strip_alignment_window_rows: int = 192
    strip_alignment_max_shift: int = 8
    directional_smoothing_enabled: bool = False
    directional_sigma_x: float = 2.0
    directional_sigma_y: float = 0.35
    multiscale_gradient_enabled: bool = False
    gradient_scales: list[float] = field(default_factory=lambda: [0.0, 0.7, 1.4])
    multiscale_raw_support_ratio: float = 0.10
    multiscale_raw_support_radius: int = 2
    algorithm: str = "tracks"
    local_window_periods: float = 8.0
    strong_prominence_k: float = 2.0
    weak_prominence_k: float = 0.8
    mad_floor_ratio: float = 0.1
    local_statistics_chunk_rows: int = 512
    max_neighbor_shift_ratio: float = 0.50
    track_shift_penalty: float = 0.25
    track_curvature_penalty: float = 0.15
    track_skip_penalty: float = 0.4
    max_skip_strips: int = 2
    min_strong_support: int = 2
    min_absolute_track_support: int = 3
    min_track_support_ratio: float = 0.60
    min_continuity_ratio: float = 0.30
    min_track_score: float = 0.60
    vertical_sequence_enabled: bool = True
    sequence_period_multiple_mode: bool = False
    # Regular-first hybrid controls.  The legacy period-multiple mode remains
    # available for comparisons, while the default geometry sequence uses this
    # continuous local regularity score as a soft prior.
    regularity_enabled: bool = True
    regularity_window_tracks: int = 7
    regularity_acf_window_periods: float = 8.0
    regularity_spacing_weight: float = 0.45
    regularity_acf_weight: float = 0.25
    regularity_track_weight: float = 0.30
    regularity_geometry_weight: float = 0.10
    regularity_spacing_mad_scale: float = 0.12
    regularity_gap_min_ratio: float = 0.55
    regularity_gap_max_ratio: float = 1.55
    regularity_track_mad_scale: float = 0.18
    regularity_geometry_mad_scale: float = 0.18
    regularity_acf_min_ratio: float = 0.60
    regularity_acf_max_ratio: float = 1.45
    regularity_regular_threshold: float = 0.70
    regularity_irregular_threshold: float = 0.40
    adaptive_period_reasoning: bool = True
    regular_spacing_prior_weight: float = 0.30
    mixed_spacing_prior_weight: float = 0.12
    irregular_spacing_prior_weight: float = 0.02
    regular_large_gap_search_enabled: bool = True
    regular_gap_multiple_tolerance: float = 0.25
    regular_recovery_weak_threshold_scale: float = 0.85
    regular_recovery_support_relaxation: float = 0.85
    regular_recovery_search_ratio: float = 0.35
    irregular_gap_search_enabled: bool = False
    raw_stack_mask_enabled: bool = True
    raw_mask_brightness_k: float = 4.0
    raw_mask_texture_k: float = 4.0
    raw_mask_background_percentile: float = 20.0
    raw_mask_relative_floor: float = 0.005
    raw_mask_close_ratio: float = 2.0
    raw_mask_margin_spacing_ratio: float = 1.0
    raw_mask_min_active_ratio: float = 0.005
    raw_mask_boundary_padding_ratio: float = 0.10
    raw_mask_min_track_evidence_ratio: float = 0.60
    margin_allow_marginal: bool = True
    margin_allow_weak_recovery: bool = True
    margin_min_gradient_score: float = 0.40
    margin_min_support_ratio: float = 0.80
    margin_min_track_score: float = 0.60
    margin_recovery_negative_support: float = 0.75
    mask_edge_growth_enabled: bool = True
    mask_high_confidence_black_k: float = 4.0
    mask_min_black_run: int = 3
    mask_refine_with_strong_tracks: bool = True
    mask_refine_max_spacing_ratio: float = 1.5
    pass2_exception_only: bool = True
    track_curve_method: str = "piecewise_linear"
    track_curve_max_interpolation_gap: int = 3
    track_curve_observation_uncertainty: float = 0.5
    track_continuity_geometry_enabled: bool = False
    track_max_neighbor_jump_ratio: float = 0.50
    track_geometry_shift_penalty: float = 0.05
    track_geometry_curvature_penalty: float = 0.08
    track_max_neighbor_change_ratio: float = 0.25
    track_unreasonable_change_penalty: float = 3.0
    track_derivative_variation_weight: float = 0.15
    track_extreme_jump_weight: float = 3.0
    track_reversal_weight: float = 1.2
    track_reversal_amplitude_ratio: float = 0.12
    sequence_non_crossing_weight: float = 0.10
    sequence_gap_geometry_weight: float = 0.10
    sequence_geometry_ambiguity_weight: float = 0.30
    sequence_short_spacing_weight: float = 3.0
    sequence_long_spacing_weight: float = 0.05
    sequence_short_gap_ratio: float = 0.55
    sequence_spacing_prior_weight: float = 0.03
    sequence_large_gap_anomaly_ratio: float = 1.50
    sequence_crossing_hard_reject_ratio: float = 0.50
    sequence_min_shared_strips: int = 3
    sequence_min_geometric_gap: float = 0.5
    sequence_gap_derivative_ratio: float = 0.50
    recovery_require_negative_evidence: bool = True
    geometry_confidence_ambiguity_penalty: float = 0.60
    geometry_confidence_ambiguity_exponent: float = 2.0
    geometry_confidence_weights: dict[str, float] = field(default_factory=lambda: {
        "track": 0.20, "image_support": 0.25, "raw_validity": 0.10,
        "non_crossing": 0.15, "geometry": 0.15, "ambiguity": 0.15})
    track_proposal_floor: float = 0.45
    sequence_unary_weight: float = 1.0
    sequence_spacing_weight: float = 0.45
    sequence_spacing_change_weight: float = 0.0
    sequence_spacing_scale_ratio: float = 0.20
    sequence_min_spacing_ratio: float = 0.55
    sequence_short_gap_penalty: float = 2.0
    sequence_missing_period_penalty: float = 0.45
    max_sequence_gap_periods: int = 4
    sequence_rescue_bonus: float = 0.05
    sequence_suppress_margin: float = 0.02
    spacing_anomaly_mad_k: float = 2.5
    recovery_reuse_candidate_radius_ratio: float = 0.65
    recovery_base_search_ratio: float = 0.20
    recovery_period_mad_weight: float = 1.5
    recovery_track_geometry_weight: float = 0.5
    recovery_min_search_ratio: float = 0.20
    recovery_max_search_ratio: float = 0.65
    min_track_smoothness: float = 0.45
    local_peak_reference_percentile: float = 75.0
    min_track_gradient_score: float = 0.20
    min_recovery_gradient_score: float = 0.10
    track_gradient_weight: float = 0.35
    track_coverage_weight: float = 0.25
    track_continuity_weight: float = 0.15
    track_smoothness_weight: float = 0.25
    duplicate_track_ratio: float = 0.50
    duplicate_position_ratio: float = 0.25
    duplicate_overlap_ratio: float = 0.50
    local_period_neighbors: int = 7
    local_period_smooth_window: int = 5
    local_period_gap_min_ratio: float = 0.60
    local_period_gap_max_ratio: float = 1.50
    missing_gap_tolerance: float = 0.25
    recovery_search_ratio: float = 0.25
    min_recovery_support_ratio: float = 0.60
    min_recovery_continuity_ratio: float = 0.30
    min_recovery_track_score: float = 0.55
    positive_aux_weight: float = 0.05
    positive_aux_search_ratio: float = 0.50
    auto_stack_roi: bool = True
    stack_energy_smooth_periods: float = 1.0
    stack_roi_margin_ratio: float = 1.0
    stack_energy_threshold_ratio: float = 0.08
    stack_energy_percentile: float = 75.0
    stack_min_extent_periods: float = 2.0
    stack_gap_close_periods: float = 2.0
    track_confidence_weights: dict[str, float] = field(default_factory=lambda: {
        "track": 0.30, "support": 0.20, "continuity": 0.15, "smoothness": 0.15, "spacing": 0.20})
    suspicious_gap_penalty: float = 0.5
    # Final boundary-identity verifier. Shadow mode computes and reports its
    # decisions after all recovery, without changing the count.
    final_track_verifier_enabled: bool = True
    final_track_verifier_shadow_mode: bool = True
    identity_profile_samples: int = 81
    identity_profile_extent_ratio: float = 0.60
    identity_template_min_tracks: int = 5
    identity_template_trim_mad_k: float = 4.0
    identity_template_distance_floor: float = 0.15
    identity_similarity_mad_floor: float = 0.04
    identity_reference_min_support: float = 0.80
    identity_reference_min_score: float = 0.70
    identity_insufficient_template_weight_scale: float = 0.20
    identity_positive_pair_min_response: float = 1.0
    identity_positive_pair_reference: float = 4.0
    identity_near_boundary_ratio: float = 1.0
    identity_low_paper_support: float = 0.35
    identity_low_positive_pair_score: float = 0.10
    identity_suspicious_outlier_z: float = 3.0
    identity_reject_outlier_z: float = 4.0
    identity_low_template_similarity: float = 0.45
    identity_low_strip_consistency: float = 0.25
    identity_low_negative_isolation: float = 0.25
    identity_min_reject_evidence_families: int = 2
    identity_score_mad_floor: float = 0.04
    identity_ambiguous_z: float = 3.0
    identity_close_competition_gap_ratio: float = 0.55
    identity_pair_morphology_margin: float = 0.05
    identity_score_weights: dict[str, float] = field(default_factory=lambda: {
        "template": 0.30, "negative": 0.15, "above": 0.20,
        "positive": 0.10, "consistency": 0.15, "context": 0.10})
    horizontal_margin_ratio: float = 0.05
    profile_smooth_sigma: float = 1.0
    min_period: int = 5
    max_period: int = 50
    min_acf_score: float = 0.10
    acf_harmonic_preference: float = 0.85
    prominence_k: float = 2.0
    distance_ratio: float = 0.55
    vote_tolerance_ratio: float = 0.20
    min_vote_ratio: float = 0.5
    strip_prominence_ratio: float = 0.6
    duplicate_ratio: float = 0.50
    max_peak_width_ratio: float = 0.65
    repair_tolerance_ratio: float = 0.25
    weak_prominence_ratio: float = 0.5
    repair_min_vote_ratio: float = 0.5
    repair_gap_error_ratio: float = 0.30
    max_repair_gap_periods: int = 4
    confidence_weights: dict[str, float] = field(default_factory=lambda: {
        "spacing": 0.35, "vote": 0.35, "acf": 0.30})
    repair_confidence_penalty: float = 0.7
    confidence_min_edges: int = 8
    display_max_height: int = 4000
    display_max_width: int = 1600
    debug_chunk_rows: int = 1024

    def __post_init__(self) -> None:
        """Reject invalid parameters before any image processing."""
        import math
        from typing import get_type_hints
        for name, kind in get_type_hints(type(self)).items():
            value = getattr(self, name)
            if kind is bool and type(value) is not bool:
                raise ValueError(f"{name} must be boolean")
            if kind is int and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer")
            if kind is float and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.algorithm not in {"legacy", "tracks"}:
            raise ValueError("algorithm must be legacy or tracks")
        if self.track_curve_method != "piecewise_linear":
            raise ValueError("Only shape-preserving piecewise_linear curves are supported")
        if not 0 < self.raw_mask_background_percentile < 50:
            raise ValueError("RAW background percentile must be in (0,50)")
        for name in ("raw_mask_relative_floor", "raw_mask_min_active_ratio", "raw_mask_min_track_evidence_ratio",
                     "margin_min_gradient_score", "margin_min_support_ratio", "sequence_short_gap_ratio",
                     "sequence_crossing_hard_reject_ratio", "regularity_regular_threshold",
                     "regularity_irregular_threshold", "regular_recovery_support_relaxation",
                     "margin_recovery_negative_support"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        if self.regularity_irregular_threshold >= self.regularity_regular_threshold:
            raise ValueError("regularity_irregular_threshold must be below regularity_regular_threshold")
        if self.regularity_window_tracks < 3 or self.mask_min_black_run < 1 or self.mask_refine_max_spacing_ratio <= 0:
            raise ValueError("regularity_window_tracks and mask_min_black_run must be positive")
        for name in ("regularity_spacing_weight", "regularity_acf_weight", "regularity_track_weight",
                     "regularity_geometry_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if sum(getattr(self, name) for name in ("regularity_spacing_weight", "regularity_acf_weight",
                                                 "regularity_track_weight", "regularity_geometry_weight")) <= 0:
            raise ValueError("Regularity weights must have positive sum")
        if not 0 < self.regularity_acf_window_periods or not 0 < self.regular_gap_multiple_tolerance:
            raise ValueError("Regularity windows and gap tolerance must be positive")
        for name in ("regularity_spacing_mad_scale", "regularity_gap_min_ratio", "regularity_gap_max_ratio",
                     "regularity_track_mad_scale", "regularity_geometry_mad_scale", "regularity_acf_min_ratio",
                     "regularity_acf_max_ratio"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.regularity_gap_min_ratio >= 1 or self.regularity_gap_max_ratio <= 1:
            raise ValueError("Regularity gap ratios must straddle one")
        for name in ("track_curve_max_interpolation_gap", "track_max_neighbor_jump_ratio",
                     "sequence_min_shared_strips", "sequence_gap_derivative_ratio"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.sequence_short_spacing_weight <= self.sequence_long_spacing_weight:
            raise ValueError("Short spacing weight must exceed long spacing weight")
        if self.sequence_large_gap_anomaly_ratio <= 1:
            raise ValueError("Large gap anomaly ratio must exceed one")
        if not 0 <= self.track_proposal_floor < self.min_track_score <= 1:
            raise ValueError("Require proposal floor < strong track score <= 1")
        if not 0 < self.sequence_spacing_scale_ratio <= 1 or not 0 < self.sequence_min_spacing_ratio < 1:
            raise ValueError("Invalid sequence spacing ratios")
        if not 0 < self.recovery_min_search_ratio <= self.recovery_base_search_ratio <= self.recovery_max_search_ratio <= 1:
            raise ValueError("Require 0 < min search <= base search <= max search <= 1")
        if not 0 < self.recovery_reuse_candidate_radius_ratio <= 1 or self.max_sequence_gap_periods < 2:
            raise ValueError("Invalid recovery radius or sequence gap bound")
        if not 0 < self.weak_prominence_k < self.strong_prominence_k:
            raise ValueError("Require 0 < weak_prominence_k < strong_prominence_k")
        for name in ("local_window_periods", "max_neighbor_shift_ratio", "local_statistics_chunk_rows",
                     "min_strong_support", "min_absolute_track_support", "local_period_neighbors", "local_period_smooth_window",
                     "stack_energy_smooth_periods", "stack_min_extent_periods"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("mad_floor_ratio", "min_track_support_ratio", "min_continuity_ratio",
                     "min_track_score", "min_track_smoothness", "duplicate_track_ratio",
                     "duplicate_position_ratio", "duplicate_overlap_ratio", "missing_gap_tolerance",
                     "recovery_search_ratio", "min_recovery_support_ratio", "min_recovery_continuity_ratio",
                     "min_recovery_track_score", "positive_aux_weight", "positive_aux_search_ratio",
                     "stack_energy_threshold_ratio", "local_period_gap_min_ratio"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0 < self.stack_energy_percentile <= 100:
            raise ValueError("stack_energy_percentile must be in (0, 100]")
        if not 0 < self.local_peak_reference_percentile <= 100:
            raise ValueError("local_peak_reference_percentile must be in (0, 100]")
        if not 0 <= self.min_recovery_gradient_score <= self.min_track_gradient_score <= 1:
            raise ValueError("Require 0 <= recovery gradient gate <= track gradient gate <= 1")
        if self.local_period_gap_max_ratio <= 1 or self.local_period_gap_min_ratio <= 0:
            raise ValueError("Local gap bounds must straddle 1")
        if sum((self.track_gradient_weight, self.track_coverage_weight,
                self.track_continuity_weight, self.track_smoothness_weight)) <= 0:
            raise ValueError("Track weights must have positive sum")
        if set(self.track_confidence_weights) != {"track", "support", "continuity", "smoothness", "spacing"}:
            raise ValueError("Invalid track_confidence_weights keys")
        if set(self.geometry_confidence_weights) != {"track", "image_support", "raw_validity", "non_crossing", "geometry", "ambiguity"}:
            raise ValueError("Invalid geometry_confidence_weights keys")
        if not 0 <= self.geometry_confidence_ambiguity_penalty <= 1:
            raise ValueError("Geometry confidence penalty must be in [0,1]")
        for values in (self.track_confidence_weights, self.confidence_weights, self.geometry_confidence_weights):
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values.values()) or sum(values.values()) <= 0:
                raise ValueError("Confidence weights must be finite nonnegative numbers with positive sum")
        if self.suspicious_gap_penalty > 1:
            raise ValueError("suspicious_gap_penalty must be in [0, 1]")
        if self.identity_profile_samples < 9 or self.identity_profile_samples % 2 == 0:
            raise ValueError("identity_profile_samples must be an odd integer >= 9")
        if self.identity_template_min_tracks < 1 or self.identity_min_reject_evidence_families < 2:
            raise ValueError("Identity template/reject counts are invalid")
        for name in ("identity_reference_min_support", "identity_reference_min_score",
                     "identity_insufficient_template_weight_scale", "identity_low_paper_support",
                     "identity_low_positive_pair_score", "identity_low_template_similarity",
                     "identity_low_strip_consistency", "identity_low_negative_isolation",
                     "identity_close_competition_gap_ratio", "identity_pair_morphology_margin"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if set(self.identity_score_weights) != {"template", "negative", "above", "positive", "consistency", "context"}:
            raise ValueError("Invalid identity_score_weights keys")
        if any(not 0 <= value < float("inf") for value in self.identity_score_weights.values()) or sum(self.identity_score_weights.values()) <= 0:
            raise ValueError("Identity score weights must be finite nonnegative with positive sum")
        if self.gradient_polarity not in {"auto", "positive", "negative"}:
            raise ValueError("gradient_polarity must be auto, positive or negative")
        if self.strip_aggregation_mode not in {"median", "narrow_median", "slope_aligned"}:
            raise ValueError("strip_aggregation_mode must be median, narrow_median or slope_aligned")
        if not 0 < self.strip_width_ratio <= 1:
            raise ValueError("strip_width_ratio must be in (0,1]")
        if (self.directional_sigma_x < 0 or self.directional_sigma_y < 0 or
                not self.gradient_scales or any(type(v) not in (int, float) or v < 0 for v in self.gradient_scales)):
            raise ValueError("Directional sigmas and gradient_scales must be finite nonnegative values")
        if sorted(set(float(v) for v in self.gradient_scales)) != [float(v) for v in self.gradient_scales]:
            raise ValueError("gradient_scales must be unique and increasing")
        if not 0 <= self.multiscale_raw_support_ratio <= 1:
            raise ValueError("multiscale_raw_support_ratio must be in [0,1]")
        for name in ("num_strips", "min_period", "max_period", "confidence_min_edges",
                     "display_max_height", "display_max_width", "debug_chunk_rows",
                     "identity_profile_samples", "identity_template_min_tracks",
                     "identity_min_reject_evidence_families",
                     "strip_alignment_bins", "strip_alignment_window_rows",
                     "strip_alignment_max_shift", "multiscale_raw_support_radius",
                     "max_repair_gap_periods"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_period < 2 or self.max_period < self.min_period:
            raise ValueError("Require 2 <= min_period <= max_period")
        for name in ("illumination_sigma", "epsilon", "prominence_k", "distance_ratio",
                     "duplicate_ratio", "max_peak_width_ratio"):
            if not 0 < getattr(self, name) < float("inf"):
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.horizontal_margin_ratio < 0.5:
            raise ValueError("horizontal_margin_ratio must be in [0, 0.5)")
        if not 0 <= self.profile_smooth_sigma < float("inf"):
            raise ValueError("profile_smooth_sigma must be finite and nonnegative")
        for name in ("min_acf_score", "acf_harmonic_preference", "vote_tolerance_ratio",
                     "min_vote_ratio", "strip_prominence_ratio", "repair_tolerance_ratio",
                     "weak_prominence_ratio", "repair_min_vote_ratio", "repair_gap_error_ratio",
                     "repair_confidence_penalty"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if set(self.confidence_weights) != {"spacing", "vote", "acf"}:
            raise ValueError("confidence_weights requires spacing, vote and acf")
        if any(not 0 <= v < float("inf") for v in self.confidence_weights.values()) or sum(self.confidence_weights.values()) <= 0:
            raise ValueError("confidence weights must be finite, nonnegative and have positive sum")
        if self.roi is not None:
            if set(self.roi) != {"x1", "x2", "y1", "y2"}:
                raise ValueError("roi must contain exactly x1, x2, y1, y2")
            if any(not isinstance(v, int) or isinstance(v, bool) for v in self.roi.values()):
                raise ValueError("ROI coordinates must be integers")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CounterConfig":
        """Load YAML, rejecting misspelled keys."""
        with Path(path).open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a mapping")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Return the complete effective configuration."""
        return asdict(self)

