"""Find the main sustained texture interval; no sheet positions are inferred."""
from __future__ import annotations
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, label
from .config import CounterConfig


def detect_active_stack_roi(gray: np.ndarray, period: float | None,
                            config: CounterConfig) -> tuple[int, int, dict]:
    """Use raw gradient energy to avoid normalization halos in empty regions.

    The bootstrap scale is the midpoint of configured period bounds. It only
    sizes ROI smoothing/margins, never becomes the reported measured period.
    """
    scale = period if period is not None else (config.min_period + config.max_period) / 2
    if not config.auto_stack_roi:
        return 0, len(gray), {"reason": "disabled", "scale": scale}
    gradient = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    energy = np.empty(len(gray), dtype=np.float32)
    for start in range(0, len(gray), config.debug_chunk_rows):
        chunk = np.abs(gradient[start:start + config.debug_chunk_rows])
        energy[start:start + len(chunk)] = np.percentile(chunk, config.stack_energy_percentile, axis=1)
    del gradient
    smoothed = gaussian_filter1d(energy, max(config.epsilon, scale * config.stack_energy_smooth_periods))
    positive = smoothed[smoothed > config.epsilon]
    if not len(positive):
        return 0, len(gray), {"reason": "no_texture", "scale": scale, "energy": smoothed}
    reference = float(np.percentile(positive, config.stack_energy_percentile))
    threshold = max(config.epsilon, reference * config.stack_energy_threshold_ratio)
    mask = smoothed > threshold
    # Fill only short INTERNAL gaps; do not erode partially visible first/last edges.
    starts = np.flatnonzero(np.diff(np.r_[False, mask].astype(int)) == 1)
    ends = np.flatnonzero(np.diff(np.r_[mask, False].astype(int)) == -1) + 1
    for end, start in zip(ends[:-1], starts[1:]):
        if start - end <= config.stack_gap_close_periods * scale:
            mask[end:start] = True
    regions, count = label(mask)
    intervals = []
    for region in range(1, count + 1):
        rows = np.flatnonzero(regions == region)
        if len(rows) >= config.stack_min_extent_periods * scale:
            intervals.append((float(smoothed[rows].sum()), int(rows[0]), int(rows[-1] + 1)))
    if not intervals:
        return 0, len(gray), {"reason": "no_sustained_region", "scale": scale, "energy": smoothed,
                             "threshold": threshold}
    _, top, bottom = max(intervals)
    margin = int(np.ceil(config.stack_roi_margin_ratio * scale))
    return max(0, top - margin), min(len(gray), bottom + margin), {
        "reason": "sustained_gradient_energy", "scale": scale, "threshold": threshold,
        "energy": smoothed, "texture_top": top, "texture_bottom": bottom, "margin": margin}
