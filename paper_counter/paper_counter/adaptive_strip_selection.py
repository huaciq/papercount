# -*- coding: utf-8 -*-
"""Adaptive strip selection for region detection.

Instead of using fixed center 225px, find the most horizontal strip
to minimize tilt-induced errors.
"""
from __future__ import annotations
import numpy as np
import cv2
from typing import Tuple, Optional


def estimate_strip_horizontality(strip: np.ndarray, sample_rows: int = 20) -> dict:
    """Estimate how horizontal the paper edges are in a strip.

    Args:
        strip: Grayscale image strip (h, w)
        sample_rows: Number of rows to sample for peak tracking

    Returns:
        Dict with horizontality metrics
    """
    h, w = strip.shape

    if h < 50 or w < 50:
        return {'tilt_score': float('inf'), 'valid': False}

    # Method 1: Track peaks across rows and measure slope
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    # Sample rows uniformly
    row_indices = np.linspace(h // 4, 3 * h // 4, min(sample_rows, h // 2), dtype=int)

    peak_positions = []
    for row_idx in row_indices:
        row = strip[row_idx, :]
        smoothed = gaussian_filter1d(row.astype(float), sigma=2.0)

        # Find peaks (bright paper regions)
        peaks, properties = find_peaks(smoothed, prominence=5, distance=10)

        if len(peaks) > 0:
            # Use strongest peak
            strongest = peaks[np.argmax(properties['prominences'])]
            peak_positions.append((row_idx, strongest))

    if len(peak_positions) < 3:
        return {'tilt_score': float('inf'), 'valid': False, 'reason': 'insufficient_peaks'}

    # Linear fit to measure tilt
    rows = np.array([p[0] for p in peak_positions])
    cols = np.array([p[1] for p in peak_positions])

    # Fit line: col = slope * row + intercept
    coeffs = np.polyfit(rows, cols, 1)
    slope = coeffs[0]

    # Measure fit quality
    predicted_cols = np.polyval(coeffs, rows)
    residuals = cols - predicted_cols
    rmse = np.sqrt(np.mean(residuals ** 2))

    # Tilt angle in degrees
    tilt_angle = np.degrees(np.arctan(slope))

    # Tilt score: combine absolute tilt and RMSE
    tilt_score = abs(tilt_angle) + rmse / w * 100

    # Method 2: Hough transform on edges (for validation)
    edges = cv2.Canny((strip * 255 / strip.max()).astype(np.uint8), 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=min(50, h // 4))

    hough_angles = []
    if lines is not None:
        for line in lines[:10]:  # Check top 10 lines
            rho, theta = line[0]
            angle = np.degrees(theta - np.pi / 2)  # Convert to tilt from horizontal
            if -45 < angle < 45:  # Only near-horizontal lines
                hough_angles.append(abs(angle))

    hough_tilt = np.median(hough_angles) if hough_angles else None

    return {
        'tilt_score': float(tilt_score),
        'tilt_angle': float(tilt_angle),
        'peak_rmse': float(rmse),
        'peak_count': len(peak_positions),
        'hough_tilt': float(hough_tilt) if hough_tilt is not None else None,
        'valid': True
    }


def find_best_horizontal_strip(image: np.ndarray,
                              strip_width: int = 225,
                              sample_count: int = 8) -> Tuple[Tuple[int, int], dict]:
    """Find the most horizontal strip in the image.

    Args:
        image: Grayscale image (h, w)
        strip_width: Desired strip width
        sample_count: Number of positions to sample

    Returns:
        ((x_start, x_end), analysis_dict)
    """
    h, w = image.shape

    if w < strip_width:
        # Image too narrow, use full width
        return (0, w), {'reason': 'image_too_narrow', 'used_full_width': True}

    # Sample positions, avoiding extreme edges
    margin = max(20, strip_width // 4)
    x_centers = np.linspace(margin + strip_width // 2,
                           w - margin - strip_width // 2,
                           sample_count)

    candidates = []
    for x_center in x_centers:
        x_start = int(x_center - strip_width // 2)
        x_end = int(x_center + strip_width // 2)

        # Ensure bounds
        x_start = max(0, min(w - strip_width, x_start))
        x_end = x_start + strip_width

        strip = image[:, x_start:x_end]

        # Estimate horizontality
        metrics = estimate_strip_horizontality(strip)

        if metrics['valid']:
            candidates.append({
                'x_range': (x_start, x_end),
                'x_center': (x_start + x_end) // 2,
                'metrics': metrics
            })

    if not candidates:
        # Fallback to center
        x_center = w // 2
        x_start = x_center - strip_width // 2
        x_end = x_start + strip_width
        return (x_start, x_end), {'reason': 'no_valid_candidates', 'used_center': True}

    # Select best candidate (lowest tilt_score)
    best = min(candidates, key=lambda c: c['metrics']['tilt_score'])

    analysis = {
        'selected_x_range': best['x_range'],
        'selected_x_center': best['x_center'],
        'selected_metrics': best['metrics'],
        'candidates_evaluated': len(candidates),
        'all_candidates': candidates
    }

    return best['x_range'], analysis


def adaptive_crop_for_regions(image: np.ndarray,
                              cfg,
                              enable_adaptive: bool = True) -> Tuple[np.ndarray, dict]:
    """Crop image adaptively for region detection.

    Args:
        image: Full grayscale image
        cfg: Counter configuration
        enable_adaptive: If False, use legacy center crop

    Returns:
        (cropped_image, crop_info)
    """
    h, w = image.shape
    target_width = 225  # Maintain compatibility with existing code

    crop_info = {
        'adaptive_enabled': enable_adaptive,
        'original_shape': (h, w),
        'target_width': target_width
    }

    if not enable_adaptive or w <= target_width:
        # Fallback to center crop
        x_center = w // 2
        x_start = max(0, x_center - target_width // 2)
        x_end = min(w, x_start + target_width)
        crop_info.update({
            'method': 'center_crop',
            'x_range': (x_start, x_end)
        })
        return image[:, x_start:x_end], crop_info

    # Adaptive selection
    (x_start, x_end), analysis = find_best_horizontal_strip(image, target_width)

    crop_info.update({
        'method': 'adaptive_horizontal',
        'x_range': (x_start, x_end),
        'analysis': analysis
    })

    return image[:, x_start:x_end], crop_info
