from __future__ import annotations

"""Result records; all public Y coordinates use the original image."""

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class PeakCandidate:
    y: int
    prominence: float
    height: float
    width: float
    vote_ratio: float = 0.0
    source: str = "detected"
    reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PaperCountResult:
    count: int
    peak_positions: list[int]
    detected_peaks: list[PeakCandidate]
    repaired_peaks: list[PeakCandidate]
    rejected_peaks: list[PeakCandidate]
    period: Optional[float]
    gradient_polarity: str
    confidence: float
    debug_info: dict[str, Any]
    # Small 1-D arrays plus a reference to the ROI; full-size gradients are released.
    artifacts: dict[str, Any] = field(default_factory=dict, repr=False)
    paper_edge_tracks: list[PaperEdgeTrack] = field(default_factory=list)
    rejected_tracks: list[PaperEdgeTrack] = field(default_factory=list)

    @property
    def edge_positions(self) -> list[float]:
        """Subpixel median centers in original image coordinates."""
        return [t.center_y for t in self.paper_edge_tracks]


@dataclass
class StripPeakCandidate:
    strip_index: int
    y: float
    height: float
    prominence: float
    width: float
    local_threshold: float
    strength_type: str
    weak_threshold: float = 0.0
    reason: Optional[str] = None
    local_prominence_reference: float = 0.0
    response_scale: float = 0.0
    raw_support: float = 1.0
    observation_source: str = "native"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PaperEdgeTrack:
    center_y: float
    strip_positions: dict[int, float]
    strip_peak_strengths: dict[int, float]
    support_count: int
    support_ratio: float
    longest_contiguous_run: int
    continuity_ratio: float
    y_median: float
    y_mad: float
    y_range: float
    smoothness_score: float
    gradient_score: float
    coverage_score: float
    continuity_score: float
    total_score: float
    confidence: float
    source: str
    strong_support_count: int
    weak_support_count: int
    reason: Optional[str] = None
    predicted_y: Optional[float] = None
    positive_aux_score: float = 0.0
    peak_candidates: list[StripPeakCandidate] = field(default_factory=list, repr=False)
    track_id: str = ""
    candidate_class: str = "unclassified"
    classification_reason: Optional[str] = None
    sequence_decision: Optional[str] = None
    raw_mask_state: str = "CORE"
    raw_evidence_ratio: float = 1.0
    raw_valid_ratio: float = 1.0
    curve_geometry: dict[str, Any] = field(default_factory=dict)
    continuity_geometry: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

