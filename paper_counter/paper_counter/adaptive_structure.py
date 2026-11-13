from __future__ import annotations

"""Automatic paper structure detection for adaptive algorithm parameter adjustment.

This module analyzes the intrinsic structure of stacked papers to detect:
1. Paper structure type: 'double_bright', 'single_bright', 'narrow_gap'
2. Characteristic spacing and width ratios
3. Local vs global regularity metrics

The goal is to enable a single algorithm to handle diverse paper types without manual tuning.
"""

import numpy as np
import cv2
from typing import Optional
from dataclasses import dataclass


@dataclass
class PaperStructureAnalysis:
    """Adaptive structure analysis result."""
    structure_type: str  # 'double_bright' | 'single_bright' | 'narrow_gap' | 'unknown'
    confidence: float  # 0-1 confidence in the structure type determination
    period: Optional[float]  # Detected period, if available
    bright_width_ratio: float  # Ratio of bright region width to period
    dark_width_ratio: float  # Ratio of dark region width to period
    has_narrow_gaps: bool  # Whether narrow dark gaps detected between papers
    gap_width_pixels: Optional[float]  # Average gap width if narrow_gap type
    tilt_angle_degrees: float  # Estimated tilt angle in degrees
    tilt_confidence: float  # Confidence in tilt estimation
    local_period_variance: float  # Coefficient of variation of local periods
    edge_contrast_ratio: float  # Ratio of max to median edge responses
    recommended_adjustments: dict  # Suggested parameter modifications


def analyze_paper_structure(gray: np.ndarray, initial_period: Optional[float],
                            gradient_response: np.ndarray, config) -> PaperStructureAnalysis:
    """Analyze paper structure to enable adaptive algorithm behavior.

    Args:
        gray: Grayscale image of paper stack
        initial_period: Initial period estimate from ACF or other method
        gradient_response: Signed gradient response (positive or negative polarity)
        config: Algorithm configuration

    Returns:
        PaperStructureAnalysis with detected characteristics and recommendations
    """
    h, w = gray.shape

    # If no initial period, try to estimate it first
    if initial_period is None or initial_period < config.min_period:
        initial_period = _estimate_period_robust(gray, config)

    # Analyze tilt
    tilt_angle, tilt_conf = _estimate_tilt_angle(gradient_response, initial_period)

    # Analyze structure pattern using autocorrelation
    structure_type, structure_conf, bright_ratio, dark_ratio = _classify_structure_type(
        gray, initial_period, config)

    # Detect narrow gaps (critical for compressed papers)
    has_narrow_gaps, gap_width = _detect_narrow_gaps(gradient_response, initial_period, config)

    # Analyze local period variation
    local_var = _compute_local_period_variance(gradient_response, initial_period, config)

    # Compute edge contrast characteristics
    edge_contrast = _compute_edge_contrast_ratio(gradient_response, config)

    # Generate adaptive parameter recommendations
    adjustments = _generate_adaptive_adjustments(
        structure_type, has_narrow_gaps, tilt_angle, local_var, edge_contrast, config)

    return PaperStructureAnalysis(
        structure_type=structure_type,
        confidence=structure_conf,
        period=initial_period,
        bright_width_ratio=bright_ratio,
        dark_width_ratio=dark_ratio,
        has_narrow_gaps=has_narrow_gaps,
        gap_width_pixels=gap_width,
        tilt_angle_degrees=tilt_angle,
        tilt_confidence=tilt_conf,
        local_period_variance=local_var,
        edge_contrast_ratio=edge_contrast,
        recommended_adjustments=adjustments
    )


def _estimate_period_robust(gray: np.ndarray, config) -> float:
    """Robust period estimation with multiple fallback strategies."""
    # Strategy 1: Global vertical profile ACF
    profile = np.median(gray, axis=1).astype(np.float32)
    profile = (profile - np.mean(profile)) / (np.std(profile) + config.epsilon)

    acf = np.correlate(profile, profile, mode='full')
    acf = acf[len(acf)//2:]
    acf = acf / acf[0]

    # Find first significant peak after lag > min_period
    peaks = []
    for lag in range(config.min_period, min(len(acf), config.max_period)):
        if lag > 0 and lag < len(acf) - 1:
            if acf[lag] > acf[lag-1] and acf[lag] > acf[lag+1] and acf[lag] > 0.1:
                peaks.append((lag, acf[lag]))

    if peaks:
        return float(max(peaks, key=lambda x: x[1])[0])

    # Fallback: Use gradient response spectrum
    gy = cv2.Scharr(gray.astype(np.float32), cv2.CV_32F, 0, 1)
    profile_grad = np.median(np.abs(gy), axis=1)
    profile_grad = (profile_grad - np.mean(profile_grad)) / (np.std(profile_grad) + config.epsilon)

    acf_grad = np.correlate(profile_grad, profile_grad, mode='full')
    acf_grad = acf_grad[len(acf_grad)//2:]
    acf_grad = acf_grad / (acf_grad[0] + config.epsilon)

    for lag in range(config.min_period, min(len(acf_grad), config.max_period)):
        if lag > 0 and lag < len(acf_grad) - 1:
            if acf_grad[lag] > acf_grad[lag-1] and acf_grad[lag] > acf_grad[lag+1] and acf_grad[lag] > 0.1:
                return float(lag)

    # Last resort: return a default reasonable value
    return float(config.min_period + (config.max_period - config.min_period) / 2)


def _estimate_tilt_angle(gradient_response: np.ndarray, period: float) -> tuple[float, float]:
    """Estimate tilt angle using Hough transform and gradient orientation.

    Returns:
        (angle_degrees, confidence)
    """
    h, w = gradient_response.shape

    # Strategy 1: Hough line detection on strong edges
    edge_mag = np.abs(gradient_response)
    threshold = np.percentile(edge_mag, 95)
    edges = (edge_mag > threshold).astype(np.uint8) * 255

    # Apply Hough transform
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=int(w * 0.3))

    angles = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle_rad = theta - np.pi/2  # Convert to angle from horizontal
            angle_deg = np.degrees(angle_rad)
            # Filter to reasonable tilt range
            if -45 < angle_deg < 45:
                angles.append(angle_deg)

    if len(angles) >= 3:
        # Use median angle
        median_angle = float(np.median(angles))
        # Confidence based on consistency
        mad = np.median(np.abs(np.array(angles) - median_angle))
        confidence = np.clip(1.0 - mad / 10.0, 0.0, 1.0)
        return median_angle, confidence

    # Strategy 2: Local gradient orientation analysis
    # Sample horizontal strips and track peak positions
    num_strips = min(12, int(h / period))
    if num_strips < 3:
        return 0.0, 0.5  # Not enough data

    strip_height = h // num_strips
    peak_positions = []

    for i in range(num_strips):
        y_start = i * strip_height
        y_end = min((i + 1) * strip_height, h)
        strip_profile = np.median(edge_mag[y_start:y_end, :], axis=0)

        # Find strongest peak
        if len(strip_profile) > 10:
            peak_idx = np.argmax(strip_profile)
            peak_positions.append((y_start + strip_height/2, peak_idx))

    if len(peak_positions) >= 3:
        # Linear fit to estimate slope
        y_coords = np.array([p[0] for p in peak_positions])
        x_coords = np.array([p[1] for p in peak_positions])

        # Robust line fit
        coeffs = np.polyfit(y_coords, x_coords, 1)
        slope = coeffs[0]
        angle_deg = np.degrees(np.arctan(slope))

        # Estimate confidence from fit residuals
        x_pred = np.polyval(coeffs, y_coords)
        residuals = np.abs(x_coords - x_pred)
        rmse = np.sqrt(np.mean(residuals**2))
        confidence = np.clip(1.0 - rmse / (w * 0.05), 0.0, 1.0)

        return float(angle_deg), float(confidence)

    return 0.0, 0.3  # Low confidence default


def _classify_structure_type(gray: np.ndarray, period: float, config) -> tuple[str, float, float, float]:
    """Classify paper structure type using profile analysis.

    Returns:
        (structure_type, confidence, bright_width_ratio, dark_width_ratio)
    """
    h, w = gray.shape

    # Extract a representative vertical profile from the center region
    center_x1 = w // 4
    center_x2 = 3 * w // 4
    center_region = gray[:, center_x1:center_x2]
    profile = np.median(center_region, axis=1).astype(np.float32)

    # Normalize
    profile = (profile - np.min(profile)) / (np.max(profile) - np.min(profile) + config.epsilon)

    # Sample multiple complete periods
    num_periods = int(h / period) - 1
    if num_periods < 2:
        return 'unknown', 0.3, 0.5, 0.5

    # Extract aligned periods
    period_samples = []
    sample_length = int(period)

    for i in range(1, min(num_periods, 10)):
        start_idx = int(i * period)
        end_idx = start_idx + sample_length
        if end_idx < len(profile):
            period_samples.append(profile[start_idx:end_idx])

    if len(period_samples) < 2:
        return 'unknown', 0.3, 0.5, 0.5

    # Average the periods to get characteristic shape
    period_samples = np.array(period_samples)
    avg_period = np.median(period_samples, axis=0)

    # Analyze the shape characteristics
    # Compute autocorrelation of the average period
    acf_period = np.correlate(avg_period - np.mean(avg_period),
                              avg_period - np.mean(avg_period), mode='full')
    acf_period = acf_period[len(acf_period)//2:]
    acf_period = acf_period / (acf_period[0] + config.epsilon)

    # Check for half-period peak (indicates double-bright structure)
    half_period_idx = len(acf_period) // 2
    search_window = max(3, int(len(acf_period) * 0.1))
    half_period_region = acf_period[max(0, half_period_idx - search_window):
                                     min(len(acf_period), half_period_idx + search_window)]

    has_half_period_peak = len(half_period_region) > 0 and np.max(half_period_region) > 0.3

    # Analyze bright and dark region widths
    threshold = np.median(avg_period)
    bright_mask = avg_period > threshold
    dark_mask = ~bright_mask

    # Find contiguous regions
    bright_runs = _find_runs(bright_mask)
    dark_runs = _find_runs(dark_mask)

    if not bright_runs or not dark_runs:
        return 'unknown', 0.3, 0.5, 0.5

    avg_bright_width = np.mean([r[1] - r[0] for r in bright_runs])
    avg_dark_width = np.mean([r[1] - r[0] for r in dark_runs])
    min_dark_width = np.min([r[1] - r[0] for r in dark_runs])

    bright_ratio = avg_bright_width / sample_length
    dark_ratio = avg_dark_width / sample_length

    # Classification logic
    confidence = 0.6

    # Type 1: Narrow gaps (dark region much smaller than bright)
    if min_dark_width < sample_length * 0.15 and dark_ratio < 0.2:
        structure_type = 'narrow_gap'
        confidence = 0.8
    # Type 2: Double bright (two bright edges per paper, half-period peak)
    elif has_half_period_peak and bright_ratio > 0.3:
        structure_type = 'double_bright'
        confidence = 0.7
    # Type 3: Single bright edge (standard case)
    elif bright_ratio > 0.2 and dark_ratio > 0.2:
        structure_type = 'single_bright'
        confidence = 0.75
    else:
        structure_type = 'unknown'
        confidence = 0.4

    return structure_type, confidence, float(bright_ratio), float(dark_ratio)


def _detect_narrow_gaps(gradient_response: np.ndarray, period: float, config) -> tuple[bool, Optional[float]]:
    """Detect narrow dark gaps between papers (critical for compressed stacks).

    Returns:
        (has_narrow_gaps, average_gap_width_pixels)
    """
    h, w = gradient_response.shape

    # Use second derivative (Laplacian) to detect narrow gaps
    gray_approx = np.cumsum(gradient_response, axis=0)  # Approximate integration
    laplacian = cv2.Laplacian(gray_approx.astype(np.float32), cv2.CV_32F)

    # Vertical profile of Laplacian response
    lap_profile = np.median(np.abs(laplacian), axis=1)

    # Find narrow peaks in Laplacian (indicates sharp transitions)
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(lap_profile, height=np.percentile(lap_profile, 70),
                                    distance=max(3, period * 0.3))

    if len(peaks) < 3:
        return False, None

    # Analyze peak widths
    widths = properties.get('widths', [])
    if len(widths) == 0:
        # Estimate widths manually
        widths = []
        threshold = np.percentile(lap_profile, 70)
        for peak in peaks:
            # Find width at half maximum
            left = peak
            while left > 0 and lap_profile[left] > threshold * 0.5:
                left -= 1
            right = peak
            while right < len(lap_profile) - 1 and lap_profile[right] > threshold * 0.5:
                right += 1
            widths.append(right - left)

    if len(widths) > 0:
        avg_width = np.median(widths)
        # Narrow gap threshold: less than 20% of period
        if avg_width < period * 0.2:
            return True, float(avg_width)

    return False, None


def _compute_local_period_variance(gradient_response: np.ndarray, global_period: float, config) -> float:
    """Compute coefficient of variation of local periods to detect irregularity."""
    h, w = gradient_response.shape

    # Divide into horizontal bands
    num_bands = max(3, int(h / (global_period * 5)))
    band_height = h // num_bands

    local_periods = []

    for i in range(num_bands):
        y_start = i * band_height
        y_end = min((i + 1) * band_height, h)
        band = gradient_response[y_start:y_end, :]

        # Compute ACF for this band
        profile = np.median(np.abs(band), axis=1)
        if len(profile) < config.min_period * 2:
            continue

        profile = (profile - np.mean(profile)) / (np.std(profile) + config.epsilon)
        acf = np.correlate(profile, profile, mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / (acf[0] + config.epsilon)

        # Find first peak
        for lag in range(config.min_period, min(len(acf), int(global_period * 2))):
            if lag > 0 and lag < len(acf) - 1:
                if acf[lag] > acf[lag-1] and acf[lag] > acf[lag+1] and acf[lag] > 0.1:
                    local_periods.append(float(lag))
                    break

    if len(local_periods) < 2:
        return 0.5  # Default moderate variance

    # Coefficient of variation
    cv = np.std(local_periods) / (np.mean(local_periods) + config.epsilon)
    return float(cv)


def _compute_edge_contrast_ratio(gradient_response: np.ndarray, config) -> float:
    """Compute ratio of maximum to median edge responses."""
    edge_mag = np.abs(gradient_response)

    # Use local maxima to avoid outliers
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(edge_mag, size=5)
    peaks = edge_mag == local_max

    peak_values = edge_mag[peaks]
    if len(peak_values) < 10:
        return 1.0

    max_val = np.percentile(peak_values, 95)
    median_val = np.median(peak_values)

    ratio = max_val / (median_val + config.epsilon)
    return float(ratio)


def _find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous True regions in a boolean mask."""
    # Pad to handle edges
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def _generate_adaptive_adjustments(structure_type: str, has_narrow_gaps: bool,
                                  tilt_angle: float, local_variance: float,
                                  edge_contrast: float, config) -> dict:
    """Generate adaptive parameter adjustments based on detected characteristics."""
    adjustments = {}

    # Adjust for structure type
    if structure_type == 'narrow_gap' or has_narrow_gaps:
        # Enable Laplacian-based detection, reduce minimum gap threshold
        adjustments['use_second_derivative'] = True
        adjustments['sequence_min_spacing_ratio'] = 0.40  # More sensitive to small gaps
        adjustments['regularity_gap_min_ratio'] = 0.45
        adjustments['sequence_short_gap_ratio'] = 0.45
        adjustments['track_min_continuity_ratio'] = 0.25  # Relax for compressed papers

    elif structure_type == 'double_bright':
        # Need to distinguish paired edges
        adjustments['use_second_derivative'] = False
        adjustments['sequence_min_spacing_ratio'] = 0.55  # Standard
        adjustments['pair_edge_detection'] = True  # Flag for paired edge logic

    else:  # single_bright or unknown
        # Standard parameters
        adjustments['use_second_derivative'] = False
        adjustments['sequence_min_spacing_ratio'] = 0.55

    # Adjust for tilt
    if abs(tilt_angle) > 5.0:
        # Increase strip alignment tolerance
        tilt_factor = min(abs(tilt_angle) / 30.0, 1.0)
        adjustments['max_neighbor_shift_ratio'] = 0.50 + 0.25 * tilt_factor
        adjustments['track_max_neighbor_jump_ratio'] = 0.50 + 0.25 * tilt_factor
        # Increase geometry weight to compensate for weak local evidence
        adjustments['regularity_geometry_weight'] = 0.10 + 0.15 * tilt_factor
        adjustments['regular_spacing_prior_weight'] = 0.30 + 0.20 * tilt_factor

    # Adjust for local period variance (irregularity)
    if local_variance > 0.15:
        # High variance: reduce regularity assumptions
        adjustments['regular_spacing_prior_weight'] = max(0.10, 0.30 - local_variance)
        adjustments['adaptive_local_period'] = True
    else:
        # Low variance: strong regularity, increase prior weight
        adjustments['regular_spacing_prior_weight'] = 0.30 + 0.15 * (1.0 - local_variance / 0.15)

    # Adjust for edge contrast variation
    if edge_contrast > 3.0:
        # High variation: some edges much weaker than others
        adjustments['margin_min_gradient_score'] = 0.30  # Relax threshold
        adjustments['min_recovery_gradient_score'] = 0.05
        adjustments['weak_edge_recovery_enabled'] = True

    return adjustments
