from __future__ import annotations

"""Bounded autocorrelation period estimation and polarity comparison."""

from typing import Any, Optional

import numpy as np
from scipy.signal import correlate, find_peaks

from .config import CounterConfig
from .profiles import robust_scale


def estimate_period(profile: np.ndarray, min_period: int, max_period: int,
                    min_acf_score: float = 0.1, harmonic_preference: float = 0.85,
                    epsilon: float = 1e-6) -> tuple[Optional[float], float, np.ndarray]:
    """Select the shortest strong local ACF maximum within the lag bounds.

    ACF is divided by lag-zero energy (no large-lag unbiased amplification).
    Similar-strength harmonics prefer the shorter repeat; no reliable maximum
    yields None rather than an invented period.
    """
    x = np.asarray(profile, dtype=np.float64) - np.mean(profile)
    energy = float(np.dot(x, x))
    limit = min(max_period, len(x) - 2)
    if energy <= epsilon or limit < min_period:
        return None, 0.0, np.zeros(min(len(x), max_period + 2))
    # ACF(k) = sum_y x(y)x(y+k) / sum_y x(y)^2；重复纸边会在其间距处相关。
    # 这里只给搜索/关联提供尺度，不能据此按“纸堆高度 / 周期”生成张数。
    corr = correlate(x, x, mode="full", method="fft")[len(x) - 1:len(x) + limit + 1] / energy
    indices, _ = find_peaks(corr)
    indices = indices[(indices >= min_period) & (indices <= limit)]
    if not indices.size:
        return None, 0.0, corr
    strongest = float(np.max(corr[indices]))
    if strongest < min_acf_score:
        return None, max(0.0, strongest), corr
    eligible = indices[corr[indices] >= max(min_acf_score, strongest * harmonic_preference)]
    lag = int(eligible[0])
    return float(lag), float(corr[lag]), corr


def score_polarity(profile: np.ndarray, config: CounterConfig) -> dict[str, Any]:
    """Combine ACF strength with preliminary peak-spacing regularity."""
    period, strength, corr = estimate_period(profile, config.min_period, config.max_period,
                                            config.min_acf_score, config.acf_harmonic_preference, config.epsilon)
    _, scale, method = robust_scale(profile, config.epsilon)
    peaks = np.array([], dtype=int)
    cv = None
    score = 0.0
    if period is not None and scale > config.epsilon:
        peaks, _ = find_peaks(profile, prominence=config.prominence_k * scale,
                              width=(None, config.max_peak_width_ratio * period),
                              distance=max(1, int(np.ceil(config.distance_ratio * period))))
        if len(peaks) >= 3:
            gaps = np.diff(peaks)
            cv = float(np.std(gaps) / np.mean(gaps))
            score = strength / (1 + cv)
    return {"period": period, "acf_score": strength, "gap_cv": cv, "score": score,
            "preliminary_peak_count": len(peaks), "scale_method": method, "acf": corr}

