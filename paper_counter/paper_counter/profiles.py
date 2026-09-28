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


def _aligned_profile(response: np.ndarray, config: CounterConfig) -> tuple[np.ndarray, dict[str, Any]]:
    """Locally align sub-strip profiles by measured integer row lags."""
    h, w = response.shape
    bins = min(config.strip_alignment_bins, w)
    edges = np.linspace(0, w, bins + 1, dtype=int)
    columns = np.stack([np.median(response[:, a:b], axis=1) for a, b in zip(edges[:-1], edges[1:])])
    output = np.empty(h, dtype=np.float32)
    records = []
    window = config.strip_alignment_window_rows
    max_shift = config.strip_alignment_max_shift
    coordinates = np.arange(h)
    for start in range(0, h, window):
        end = min(h, start + window)
        target = np.median(columns[:, start:end], axis=0)
        aligned = []
        lags = []
        target_centered = target - np.median(target)
        target_norm = np.linalg.norm(target_centered)
        for profile in columns:
            best = (float("-inf"), 0)
            for lag in range(-max_shift, max_shift + 1):
                values = np.interp(np.arange(start, end) + lag, coordinates, profile)
                centered = values - np.median(values)
                score = float(np.dot(target_centered, centered) /
                              max(config.epsilon, target_norm * np.linalg.norm(centered)))
                if score > best[0]:
                    best = (score, lag)
            aligned.append(np.interp(np.arange(start, end) + best[1], coordinates, profile))
            lags.append(best[1])
        output[start:end] = np.median(np.stack(aligned), axis=0)
        records.append({"start": start, "end": end, "lags": lags,
                        "lag_range": int(max(lags) - min(lags))})
    return output, {"mode": "slope_aligned", "windows": records,
                    "max_observed_lag_range": max((r["lag_range"] for r in records), default=0)}


def make_strip_profiles_diagnostic(gy: np.ndarray, polarity: str, config: CounterConfig
                                  ) -> tuple[np.ndarray, list[tuple[int, int]], list[dict[str, Any]]]:
    """Project signed responses using row medians in bounded-size chunks."""
    h, w = gy.shape
    margin = int(w * config.horizontal_margin_ratio)
    if w - 2 * margin < config.num_strips:
        raise ValueError("ROI width after margins must be >= num_strips")
    edges = np.linspace(margin, w - margin, config.num_strips + 1, dtype=int)
    nominal = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]
    bounds = []
    for a, b in nominal:
        if config.strip_aggregation_mode == "narrow_median":
            width = max(1, int(round((b - a) * config.strip_width_ratio)))
            center = (a + b) // 2
            a, b = max(a, center - width // 2), min(b, center - width // 2 + width)
        bounds.append((a, b))
    profiles = np.empty((config.num_strips, h), dtype=np.float32)
    diagnostics = []
    sign = 1.0 if polarity == "positive" else -1.0
    for index, (x1, x2) in enumerate(bounds):
        response = gy[:, x1:x2] * sign
        np.maximum(response, 0, out=response)
        if config.strip_aggregation_mode == "slope_aligned":
            profiles[index], detail = _aligned_profile(response, config)
        else:
            for start in range(0, h, config.debug_chunk_rows):
                end = min(h, start + config.debug_chunk_rows)
                profiles[index, start:end] = np.median(response[start:end], axis=1)
            detail = {"mode": config.strip_aggregation_mode, "windows": []}
        detail.update(strip=index, x1=x1, x2=x2, width=x2-x1)
        diagnostics.append(detail)
        if config.profile_smooth_sigma > 0:
            profiles[index] = gaussian_filter1d(profiles[index], config.profile_smooth_sigma)
    return profiles, bounds, diagnostics


def make_strip_profiles(gy: np.ndarray, polarity: str, config: CounterConfig) -> tuple[np.ndarray, list[tuple[int, int]]]:
    profiles, bounds, _ = make_strip_profiles_diagnostic(gy, polarity, config)
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

