"""Per-strip local statistics and independent strong/weak peak extraction."""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from numpy.lib.stride_tricks import sliding_window_view

from .config import CounterConfig
from .models import StripPeakCandidate
from .profiles import robust_scale


def local_statistics(profile: np.ndarray, period: float, config: CounterConfig) -> dict[str, np.ndarray]:
    """Exact centered-window median/MAD using bounded float32 chunks.

    Each window uses its own median, not a second median-filter approximation.
    Python loops cover chunks, never individual pixels.
    """
    size = max(3, int(round(period * config.local_window_periods)) | 1)
    signal = np.asarray(profile, dtype=np.float32)
    padded = np.pad(signal, size // 2, mode="reflect")
    windows = sliding_window_view(padded, size)
    baseline = np.empty_like(signal)
    mad = np.empty_like(signal)
    for start in range(0, len(signal), config.local_statistics_chunk_rows):
        block = windows[start:start + config.local_statistics_chunk_rows]
        med = np.median(block, axis=1)
        baseline[start:start + len(block)] = med
        mad[start:start + len(block)] = np.median(np.abs(block - med[:, None]), axis=1)
    _, scale, _ = robust_scale(signal, config.epsilon)
    # MAD(y)=median(|window-median(window)|)，默认以 2.0/0.8 倍区分强弱峰。
    # 稀疏窗口可能 MAD=0，故用整条带尺度的一个比例托底，避免零阈值收噪声。
    effective = np.maximum(mad, max(scale * config.mad_floor_ratio, config.epsilon))
    return {"baseline": baseline, "mad": mad, "effective_mad": effective,
            "strong_threshold": effective * config.strong_prominence_k,
            "weak_threshold": effective * config.weak_prominence_k}


def detect_strip_peaks(profiles: np.ndarray, period: float, config: CounterConfig,
                       y_offset: int = 0, raw_profiles: np.ndarray | None = None,
                       scale_sources: np.ndarray | None = None
                       ) -> tuple[list[list[StripPeakCandidate]], list[dict[str, np.ndarray]]]:
    """Detect local maxima before fusion; weak maxima cannot seed ordinary tracks.

    Profile endpoints are padded with their local baseline so visible partial
    edge maxima are eligible, rather than unconditionally deleting end rows.
    """
    result, statistics = [], []
    for strip, profile in enumerate(profiles):
        stats = local_statistics(profile, period, config)
        statistics.append(stats)
        extended = np.r_[stats["baseline"][0], profile, stats["baseline"][-1]]
        # prominence 衡量峰相对周围基底的突出程度，并不等于峰的绝对高度。
        ys, properties = find_peaks(extended, prominence=(None, None), width=(None, None))
        candidates = []
        for i, padded_y in enumerate(ys):
            y = int(padded_y - 1)
            if not 0 <= y < len(profile):
                continue
            prominence = float(properties["prominences"][i])
            if prominence < float(stats["weak_threshold"][y]) or prominence <= config.epsilon:
                continue
            strong_threshold = float(stats["strong_threshold"][y])
            width = float(properties["widths"][i])
            radius = config.multiscale_raw_support_radius
            native = profiles[strip] if raw_profiles is None else raw_profiles[strip]
            raw_height = float(np.max(native[max(0, y-radius):min(len(native), y+radius+1)]))
            raw_support = raw_height / max(float(profile[y]), config.epsilon)
            scale = float(scale_sources[strip, y]) if scale_sources is not None else 0.0
            enhanced_only = (config.multiscale_gradient_enabled and scale > config.epsilon and
                             raw_support < config.multiscale_raw_support_ratio)
            candidates.append(StripPeakCandidate(strip, float(y + y_offset), float(profile[y]),
                prominence, width, strong_threshold,
                "strong" if prominence >= strong_threshold else "weak",
                float(stats["weak_threshold"][y]),
                "peak_too_wide" if width > config.max_peak_width_ratio * period else
                "enhanced_only_proposal" if enhanced_only else None,
                response_scale=scale, raw_support=raw_support,
                observation_source="native_supported_multiscale" if scale > config.epsilon else "native"))
        # A local foreground reference is used only for track gradient scoring.
        # It never replaces the requested k * local-MAD strong/weak thresholds.
        usable = [p for p in candidates if p.reason is None]
        candidate_y = np.array([p.y for p in usable])
        candidate_prominence = np.array([p.prominence for p in usable], dtype=np.float32)
        half_window = period * config.local_window_periods / 2
        for peak in usable:
            left = np.searchsorted(candidate_y, peak.y - half_window)
            right = np.searchsorted(candidate_y, peak.y + half_window, side="right")
            reference = float(np.percentile(candidate_prominence[left:right], config.local_peak_reference_percentile))
            peak.local_prominence_reference = max(reference, peak.local_threshold)
        result.append(candidates)
    return result, statistics
