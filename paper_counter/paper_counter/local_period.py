"""Robust, slowly varying spacing prior estimated from observed tracks."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import median_filter
from .models import PaperEdgeTrack
from .config import CounterConfig


@dataclass
class LocalPeriod:
    y: np.ndarray
    values: np.ndarray
    mad: np.ndarray
    initial: float

    def at(self, y: float) -> float:
        """Interpolate between robust samples, holding endpoints constant."""
        return float(np.interp(y, self.y, self.values)) if len(self.y) else self.initial

    def to_samples(self) -> list[dict[str, float]]:
        return [{"y": float(y), "period": float(p), "mad": float(m)}
                for y, p, m in zip(self.y, self.values, self.mad)]


def estimate_local_period(tracks: list[PaperEdgeTrack], initial: float,
                          config: CounterConfig) -> LocalPeriod:
    """Median nearby non-multiple gaps, then median-smooth the estimates.

    The initial prior only rejects obvious merged/doubled gaps. Empty local
    neighborhoods keep the prior instead of treating a missing edge as thickness.
    """
    ys = np.array(sorted(t.center_y for t in tracks), dtype=np.float32)
    gaps = np.diff(ys)
    samples = np.full(len(ys), initial, dtype=np.float32)
    deviations = np.zeros_like(samples)
    half = config.local_period_neighbors // 2
    for i in range(len(ys)):
        start = max(0, min(i - half, len(gaps) - config.local_period_neighbors))
        local = gaps[start:start + config.local_period_neighbors]
        local = local[(local >= config.local_period_gap_min_ratio * initial) &
                      (local <= config.local_period_gap_max_ratio * initial)]
        if len(local):
            samples[i] = np.median(local)
            deviations[i] = np.median(np.abs(local - samples[i]))
    if len(samples):
        samples = median_filter(samples, size=config.local_period_smooth_window | 1, mode="nearest")
    return LocalPeriod(ys, samples, deviations, initial)


def missing_edge_hypotheses(tracks: list[PaperEdgeTrack], local: LocalPeriod,
                            config: CounterConfig) -> list[dict]:
    """Propose bounded internal searches; a hypothesis contributes no count."""
    hypotheses = []
    ordered = sorted(tracks, key=lambda t: t.center_y)
    for left, right in zip(ordered[:-1], ordered[1:]):
        midpoint = (left.center_y + right.center_y) / 2
        period = local.at(midpoint)
        gap = right.center_y - left.center_y
        repeats = int(round(gap / period))
        if repeats < 2:
            continue
        if repeats > config.max_repair_gap_periods or abs(gap / period - repeats) > config.missing_gap_tolerance:
            hypotheses.append({"left": left.center_y, "right": right.center_y, "gap": gap,
                               "local_period": period, "reason": "inconsistent_or_large_gap", "search": False})
            continue
        for k in range(1, repeats):
            fraction = k / repeats
            # Follow the neighboring curves, rather than a horizontal center-only band.
            predicted_positions = {}
            for strip in range(config.num_strips):
                a = np.interp(strip, sorted(left.strip_positions), [left.strip_positions[x] for x in sorted(left.strip_positions)])
                b = np.interp(strip, sorted(right.strip_positions), [right.strip_positions[x] for x in sorted(right.strip_positions)])
                predicted_positions[strip] = float(a + fraction * (b - a))
            hypotheses.append({"left": left.center_y, "right": right.center_y, "gap": gap,
                "local_period": period, "predicted_y": left.center_y + fraction * gap,
                "predicted_strip_positions": predicted_positions,
                "suspected_missing_count": repeats - 1, "reason": "near_integer_gap", "search": True})
    return hypotheses
