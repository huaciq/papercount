from __future__ import annotations

"""Strip median projections and robust profile fusion."""

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .config import CounterConfig


def robust_scale(values: np.ndarray, epsilon: float) -> tuple[float, float, str]:
    """Use MAD, falling back to median nonzero deviation for sparse signals."""
    median = float(np.median(values))
    deviations = np.abs(values - median)
    scale = float(np.median(deviations))
    method = "mad"
    if scale <= epsilon:
        nonzero = deviations[deviations > epsilon]
        scale = float(np.median(nonzero)) if nonzero.size else 0.0
        method = "nonzero_deviation_median" if nonzero.size else "flat"
    return median, scale, method


def make_strip_profiles(gy: np.ndarray, polarity: str, config: CounterConfig) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Project signed responses using row medians in bounded-size chunks."""
    h, w = gy.shape
    margin = int(w * config.horizontal_margin_ratio)
    if w - 2 * margin < config.num_strips:
        raise ValueError("ROI width after margins must be >= num_strips")
    edges = np.linspace(margin, w - margin, config.num_strips + 1, dtype=int)
    bounds = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]
    profiles = np.empty((config.num_strips, h), dtype=np.float32)
    sign = 1.0 if polarity == "positive" else -1.0
    for index, (x1, x2) in enumerate(bounds):
        for start in range(0, h, config.debug_chunk_rows):
            end = min(h, start + config.debug_chunk_rows)
            response = gy[start:end, x1:x2] * sign
            np.maximum(response, 0, out=response)
            profiles[index, start:end] = np.median(response, axis=1)
        if config.profile_smooth_sigma > 0:
            profiles[index] = gaussian_filter1d(profiles[index], config.profile_smooth_sigma)
    return profiles, bounds


def normalize_and_fuse(raw_profiles: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """MAD-normalize each strip and take the median across non-flat strips."""
    normalized = np.zeros_like(raw_profiles)
    stats = []
    active = []
    for i, profile in enumerate(raw_profiles):
        med, scale, method = robust_scale(profile, epsilon)
        valid = scale > epsilon
        if valid:
            normalized[i] = (profile - med) / (scale + epsilon)
            active.append(i)
        stats.append({"strip": i, "median": med, "scale": scale, "scale_method": method, "active": valid})
    fused = np.median(normalized[active], axis=0) if active else np.zeros(raw_profiles.shape[1], dtype=np.float32)
    return normalized, fused, stats

