from __future__ import annotations

"""Enhanced edge detection combining first and second derivatives for compressed papers.

For paper structures with narrow dark gaps (bright-narrow_gap-bright), standard gradient
detection may miss the gap. Second derivatives (Laplacian) detect the two transitions
at the gap boundaries, helping identify papers that would otherwise be counted as one.
"""

import numpy as np
import cv2
from typing import Optional


def enhanced_gradient_response(normalized: np.ndarray, structure_analysis, config) -> tuple[np.ndarray, dict]:
    """Compute enhanced gradient response combining first and second derivatives.

    Args:
        normalized: Illumination-normalized float32 image
        structure_analysis: PaperStructureAnalysis from adaptive_structure module
        config: Algorithm configuration

    Returns:
        (enhanced_response, debug_info)
    """
    # Standard first derivative (Scharr gradient)
    first_deriv = cv2.Scharr(normalized, cv2.CV_32F, 0, 1)

    debug_info = {
        'used_second_derivative': False,
        'fusion_mode': 'first_only',
        'structure_type': structure_analysis.structure_type if structure_analysis else 'unknown'
    }

    # Check if second derivative enhancement is needed
    if structure_analysis is None:
        return first_deriv, debug_info

    use_second_deriv = structure_analysis.recommended_adjustments.get('use_second_derivative', False)

    if not use_second_deriv:
        return first_deriv, debug_info

    # Compute second derivative (Laplacian)
    # For narrow gaps: Laplacian responds to both edges of the gap
    second_deriv = cv2.Laplacian(normalized, cv2.CV_32F)

    debug_info['used_second_derivative'] = True

    # Fusion strategy based on structure type
    if structure_analysis.structure_type == 'narrow_gap':
        # For narrow gaps: emphasize Laplacian in potential gap regions
        enhanced = _fuse_for_narrow_gaps(first_deriv, second_deriv, structure_analysis, config)
        debug_info['fusion_mode'] = 'narrow_gap_adaptive'

    else:
        # Conservative fusion: use Laplacian only where first derivative is weak
        enhanced = _conservative_fusion(first_deriv, second_deriv, config)
        debug_info['fusion_mode'] = 'conservative'

    return enhanced, debug_info


def _fuse_for_narrow_gaps(first_deriv: np.ndarray, second_deriv: np.ndarray,
                          structure_analysis, config) -> np.ndarray:
    """Adaptive fusion emphasizing Laplacian in narrow gap regions."""

    # Estimate expected gap width
    gap_width = structure_analysis.gap_width_pixels
    if gap_width is None or gap_width < 2:
        gap_width = structure_analysis.period * 0.1 if structure_analysis.period else 2.0

    # Create a gap probability map based on image characteristics
    # High Laplacian response + low first derivative magnitude = likely gap
    first_mag = np.abs(first_deriv)
    second_mag = np.abs(second_deriv)

    # Normalize responses
    first_norm = first_mag / (np.percentile(first_mag, 95) + config.epsilon)
    second_norm = second_mag / (np.percentile(second_mag, 95) + config.epsilon)

    # Gap probability: high Laplacian, low-to-moderate first derivative
    # Use a smooth weighting function instead of hard threshold
    gap_prob = second_norm * (1.0 - np.minimum(first_norm, 1.0) * 0.5)

    # Smooth gap probability map
    kernel_size = max(3, int(gap_width * 2) | 1)  # Ensure odd
    gap_prob_smooth = cv2.GaussianBlur(gap_prob, (kernel_size, kernel_size),
                                       sigmaX=gap_width * 0.5, sigmaY=gap_width * 0.5)

    # Adaptive weighting: where gap_prob is high, use more Laplacian
    # Where gap_prob is low, rely on first derivative
    laplacian_weight = np.clip(gap_prob_smooth, 0.2, 0.8)
    first_weight = 1.0 - laplacian_weight

    # Fuse with signed responses preserved
    # For first derivative: keep original sign
    # For Laplacian: convert to signed edge response
    # Laplacian positive peak = bright-to-dark-to-bright = two edges
    # We want to enhance the dark-to-bright transition (positive gradient in our convention)

    enhanced = first_deriv * first_weight + second_deriv * laplacian_weight * 0.5

    return enhanced


def _conservative_fusion(first_deriv: np.ndarray, second_deriv: np.ndarray, config) -> np.ndarray:
    """Conservative fusion: use Laplacian only where first derivative is weak."""

    first_mag = np.abs(first_deriv)
    second_mag = np.abs(second_deriv)

    # Identify weak first derivative regions
    first_threshold = np.percentile(first_mag, 50)
    weak_regions = first_mag < first_threshold

    # In weak regions, supplement with Laplacian
    # In strong regions, use first derivative exclusively
    enhanced = first_deriv.copy()

    # Add scaled Laplacian contribution in weak regions
    # Preserve sign from Laplacian
    laplacian_contribution = second_deriv * 0.3
    enhanced = np.where(weak_regions, enhanced + laplacian_contribution, enhanced)

    return enhanced


def extract_paired_edges(gradient_response: np.ndarray, structure_analysis, config) -> Optional[dict]:
    """For double_bright structure, extract paired edges belonging to same paper.

    Returns:
        Dictionary with paired edge information, or None if not applicable
    """
    if structure_analysis is None or structure_analysis.structure_type != 'double_bright':
        return None

    # For double-bright papers: each paper has two bright edges
    # Need to identify pairs and treat them as a single paper unit

    # This is a placeholder for paired edge logic
    # Full implementation would involve:
    # 1. Detect all candidate edges
    # 2. Group edges into pairs based on spacing and correlation
    # 3. Merge paired edges into single paper detections

    paired_info = {
        'enabled': True,
        'expected_pair_spacing': structure_analysis.period * structure_analysis.bright_width_ratio,
        'pair_correlation_threshold': 0.7
    }

    return paired_info


def adaptive_peak_detection(profile: np.ndarray, period: float, structure_analysis,
                            config, weak: bool = False) -> tuple[list, float]:
    """Adaptive peak detection with structure-aware thresholds.

    Args:
        profile: 1D signal profile
        period: Estimated period
        structure_analysis: Structure characteristics
        config: Algorithm configuration
        weak: Whether to use weak prominence threshold

    Returns:
        (peak_candidates, threshold_used)
    """
    from scipy.signal import find_peaks

    # Compute adaptive threshold based on structure
    if structure_analysis and structure_analysis.structure_type == 'narrow_gap':
        # For narrow gaps: lower threshold to avoid missing weak edges
        prominence_k = config.weak_prominence_k * 0.7 if weak else config.strong_prominence_k * 0.8
    elif structure_analysis and structure_analysis.edge_contrast_ratio > 3.0:
        # High contrast variation: use lower threshold
        prominence_k = config.weak_prominence_k * 0.85 if weak else config.strong_prominence_k * 0.9
    else:
        # Standard thresholds
        prominence_k = config.weak_prominence_k if weak else config.strong_prominence_k

    # Compute robust prominence threshold
    profile_abs = np.abs(profile)
    mad = np.median(np.abs(profile_abs - np.median(profile_abs)))
    mad = max(mad, np.std(profile_abs) * config.mad_floor_ratio)
    threshold = prominence_k * mad

    # Find peaks with adaptive parameters
    min_distance = max(3, int(period * 0.3))

    peaks, properties = find_peaks(profile_abs, prominence=threshold, distance=min_distance)

    # Convert to candidate format expected by downstream code
    candidates = []
    for i, peak_idx in enumerate(peaks):
        candidates.append({
            'position': float(peak_idx),
            'value': float(profile[peak_idx]),
            'prominence': float(properties['prominences'][i])
        })

    return candidates, float(threshold)
