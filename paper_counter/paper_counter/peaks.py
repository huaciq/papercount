from __future__ import annotations

"""Adaptive peak extraction, strip voting and quality-based suppression."""

import numpy as np
from scipy.signal import find_peaks

from .config import CounterConfig
from .models import PeakCandidate
from .profiles import robust_scale


def detect_candidates(profile: np.ndarray, period: float, config: CounterConfig,
                      weak: bool = False) -> tuple[list[PeakCandidate], float]:
    """Find local maxima with MAD-based prominence and period-based distance."""
    _, scale, _ = robust_scale(profile, config.epsilon)
    threshold = config.prominence_k * scale
    if scale <= config.epsilon:
        return [], threshold
    if weak:
        threshold *= config.weak_prominence_ratio
    indices, props = find_peaks(profile, prominence=threshold, width=(None, None),
                                distance=1 if weak else max(1, int(np.ceil(config.distance_ratio * period))))
    return [PeakCandidate(int(y), float(props["prominences"][i]), float(profile[y]),
                          float(props["widths"][i]), source="repaired" if weak else "detected")
            for i, y in enumerate(indices)], threshold


def strip_peak_indices(profiles: np.ndarray, config: CounterConfig, weak: bool = False) -> list[np.ndarray]:
    """Build per-strip local-peak evidence; flat strips never vote."""
    support = []
    for profile in profiles:
        _, scale, _ = robust_scale(profile, config.epsilon)
        threshold = config.prominence_k * config.strip_prominence_ratio * scale
        if weak:
            threshold *= config.weak_prominence_ratio
        support.append(find_peaks(profile, prominence=threshold)[0] if scale > config.epsilon else np.array([], dtype=int))
    return support


def vote_ratio(y: int, strip_peaks: list[np.ndarray], tolerance: float) -> float:
    """Fraction of all configured strips containing a nearby supported maximum."""
    votes = 0
    for peaks in strip_peaks:
        idx = np.searchsorted(peaks, y - tolerance)
        votes += bool(idx < len(peaks) and peaks[idx] <= y + tolerance)
    return votes / len(strip_peaks) if strip_peaks else 0.0


def validate_candidates(candidates: list[PeakCandidate], strip_peaks: list[np.ndarray],
                        period: float, config: CounterConfig) -> tuple[list[PeakCandidate], list[PeakCandidate]]:
    """Retain vote failures as explicit rejected records."""
    accepted, rejected = [], []
    for peak in candidates:
        peak.vote_ratio = vote_ratio(peak.y, strip_peaks, config.vote_tolerance_ratio * period)
        if peak.width > config.max_peak_width_ratio * period:
            peak.reason = "peak_too_wide"
            rejected.append(peak)
        elif peak.vote_ratio < config.min_vote_ratio:
            peak.reason = "insufficient_votes"
            rejected.append(peak)
        else:
            accepted.append(peak)
    return accepted, rejected


def candidate_quality(peak: PeakCandidate) -> float:
    """Reward prominence, positive response and cross-strip agreement equally."""
    return max(0.0, peak.height) + peak.prominence + peak.vote_ratio * peak.prominence


def remove_duplicates(candidates: list[PeakCandidate], period: float,
                      config: CounterConfig) -> tuple[list[PeakCandidate], list[PeakCandidate]]:
    """Greedy quality-ordered suppression avoids unstable adjacent-pair choices."""
    kept, rejected = [], []
    for candidate in sorted(candidates, key=candidate_quality, reverse=True):
        if any(abs(candidate.y - other.y) < config.duplicate_ratio * period for other in kept):
            candidate.reason = "duplicate"
            rejected.append(candidate)
        else:
            kept.append(candidate)
    return sorted(kept, key=lambda p: p.y), sorted(rejected, key=lambda p: p.y)

