"""Find the main sustained texture interval; no sheet positions are inferred."""
from __future__ import annotations
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, label, median_filter
from .config import CounterConfig


def detect_active_stack_roi(gray: np.ndarray, period: float | None,
                            config: CounterConfig, original: np.ndarray | None = None) -> tuple[int, int, dict]:
    """Use raw gradient energy to avoid normalization halos in empty regions.

    The bootstrap scale is the midpoint of configured period bounds. It only
    sizes ROI smoothing/margins, never becomes the reported measured period.
    """
    scale = period if period is not None else (config.min_period + config.max_period) / 2
    if not config.auto_stack_roi:
        return 0, len(gray), {"reason": "disabled", "scale": scale}
    if config.stack_roi_mode == "foreground_envelope":
        return foreground_envelope_roi(gray, config, original)
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


def foreground_envelope_roi(gray: np.ndarray, config: CounterConfig,
                            original: np.ndarray | None = None) -> tuple[int, int, dict]:
    """Trim only exterior dark background; retain low-texture paper and gaps.

    The envelope includes every sustained, sufficiently wide RAW foreground
    interval. It deliberately does not choose the most textured component.
    Color input uses its brightest channel so colored paper is not discarded
    through grayscale conversion. Ambiguous contrast keeps the requested ROI.
    """
    raw = gray if original is None or original.ndim == 2 else np.max(original[:, :, :3], axis=2)
    h, w = raw.shape
    margin_x = int(config.horizontal_margin_ratio * w)
    raw = raw[:, margin_x:w-margin_x or None]
    sample = raw[::max(1, h // 1000), ::max(1, w // 256)].astype(np.float32)
    background, bright = (float(v) for v in np.percentile(sample, [2, 95]))
    depth = float(np.iinfo(gray.dtype).max)
    contrast = bright - background
    detail = {"mode": "foreground_envelope", "background_level": background,
              "bright_level": bright, "uses_color": original is not None and original.ndim == 3,
              "policy": "all_supported_intervals_envelope_not_largest_texture_component"}
    if contrast < depth * 0.02:
        return 0, h, {**detail, "reason": "ambiguous_foreground_keep_requested_roi"}
    threshold = background + max(depth * 0.015, config.foreground_roi_contrast_ratio * contrast)
    fractions = np.mean(raw > threshold, axis=1)
    active = median_filter(fractions, size=3, mode="nearest") >= config.foreground_roi_min_width_ratio
    regions, count = label(active)
    intervals = []
    for i in range(1, count + 1):
        rows = np.flatnonzero(regions == i)
        if len(rows) >= 3:
            intervals.append([int(rows[0]), int(rows[-1] + 1)])
    if not intervals:
        return 0, h, {**detail, "reason": "no_sustained_foreground_keep_requested_roi",
                      "threshold": threshold}
    top, bottom = intervals[0][0], intervals[-1][1]
    margin = config.foreground_roi_margin_rows
    return max(0, top-margin), min(h, bottom+margin), {
        **detail, "reason": "raw_foreground_envelope", "threshold": threshold,
        "foreground_intervals": intervals, "foreground_top": top,
        "foreground_bottom": bottom, "margin": margin,
        "energy": fractions.astype(np.float32), "energy_semantics": "raw_foreground_width_fraction"}
