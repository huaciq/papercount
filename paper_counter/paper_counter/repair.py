from __future__ import annotations

"""Gap repair only at observed weak maxima with strip evidence."""

from typing import Any

from .config import CounterConfig
from .models import PeakCandidate
from .peaks import candidate_quality, vote_ratio


def repair_missing_peaks(accepted: list[PeakCandidate], weak_candidates: list[PeakCandidate],
                         weak_strip_peaks: list, period: float, config: CounterConfig,
                         forbidden_positions: set[int]) -> tuple[list[PeakCandidate], list[dict[str, Any]]]:
    """Inspect internal near-integer gaps; never extrapolate into blank margins."""
    repaired, attempts = [], []
    for left, right in zip(accepted[:-1], accepted[1:]):
        gap = right.y - left.y
        repeats = int(round(gap / period))
        if repeats < 2:
            continue
        if repeats > config.max_repair_gap_periods or abs(gap / repeats - period) > config.repair_gap_error_ratio * period:
            attempts.append({"left": left.y, "right": right.y, "reason": "gap_inconsistent_or_too_large"})
            continue
        for k in range(1, repeats):
            predicted = left.y + k * gap / repeats
            evidence = []
            for peak in weak_candidates:
                if peak.width > config.max_peak_width_ratio * period:
                    continue
                if peak.y in forbidden_positions or abs(peak.y - predicted) > config.repair_tolerance_ratio * period:
                    continue
                if any(abs(peak.y - p.y) < config.duplicate_ratio * period for p in accepted + repaired):
                    continue
                peak.vote_ratio = vote_ratio(peak.y, weak_strip_peaks, config.vote_tolerance_ratio * period)
                if peak.vote_ratio >= config.repair_min_vote_ratio:
                    evidence.append(peak)
            chosen = max(evidence, key=candidate_quality) if evidence else None
            attempts.append({"predicted_y": predicted, "selected_y": chosen.y if chosen else None,
                             "reason": "weak_peak_and_strip_evidence" if chosen else "no_image_evidence"})
            if chosen is not None:
                repaired.append(chosen)
    return sorted(repaired, key=lambda p: p.y), attempts

