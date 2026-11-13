# -*- coding: utf-8 -*-
"""Adaptive improvements for region-based paper detection.

Addresses two key problems:
1. Narrow-gap papers: Two sheets merged into one box
2. Tilted papers: HBB boxes include background/gaps

Solution approach:
1. Detect narrow gaps using second-derivative (Laplacian) analysis
2. Split merged bands when narrow gap is detected
3. For tilted papers: adjust box boundaries to follow paper body contour
"""
from __future__ import annotations
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks
from typing import Optional


def detect_narrow_gaps_in_profile(profile: np.ndarray, typical_spacing: float) -> dict:
    """Detect narrow dark gaps in the intensity profile using Laplacian.

    Args:
        profile: 1D intensity profile (center column or fused)
        typical_spacing: Estimated spacing between papers

    Returns:
        Dict with gap detection results
    """
    # Compute second derivative (Laplacian approximation)
    # For 1D: Laplacian ≈ second difference
    smoothed = gaussian_filter1d(profile, sigma=0.5)
    second_deriv = np.diff(smoothed, n=2)

    # Pad to match original length
    second_deriv = np.pad(second_deriv, (1, 1), mode='edge')

    # Narrow gaps show up as positive peaks in -second_deriv (concave regions)
    gap_signal = -second_deriv

    # Detect gap candidates
    # Narrow gaps should be much smaller than typical_spacing
    max_gap_width = max(3, typical_spacing * 0.15)

    noise_level = 1.4826 * np.median(np.abs(gap_signal - np.median(gap_signal)))
    threshold = max(0.5, 3.0 * noise_level)

    peaks, properties = find_peaks(
        gap_signal,
        prominence=threshold,
        width=(None, max_gap_width),
        distance=max(2, typical_spacing * 0.3)
    )

    gaps = []
    for peak_idx, prominence, width_samples in zip(
        peaks,
        properties['prominences'],
        properties.get('widths', [None] * len(peaks))
    ):
        if width_samples is None:
            width_samples = 1.0

        # Verify this is actually a dark region in the original profile
        search_width = int(max(2, width_samples * 1.5))
        left = max(0, peak_idx - search_width)
        right = min(len(profile), peak_idx + search_width + 1)

        # Check if original profile is darker than neighbors
        local_profile = profile[left:right]
        if len(local_profile) < 3:
            continue

        center_val = profile[peak_idx]
        neighbor_vals = np.concatenate([
            profile[max(0, left-3):left],
            profile[right:min(len(profile), right+3)]
        ])

        if len(neighbor_vals) > 0:
            contrast = np.median(neighbor_vals) - center_val
        else:
            contrast = 0

        if contrast < 1.0:  # Not actually darker
            continue

        gaps.append({
            'position': int(peak_idx),
            'width': float(width_samples),
            'prominence': float(prominence),
            'contrast': float(contrast),
            'confidence': min(1.0, prominence / (threshold * 2))
        })

    return {
        'gaps': gaps,
        'has_narrow_gaps': len(gaps) > 0,
        'gap_count': len(gaps),
        'typical_gap_width': float(np.median([g['width'] for g in gaps])) if gaps else None,
        'detection_method': 'second_derivative_laplacian'
    }


def should_split_band(band: dict, profile: np.ndarray, typical_spacing: float,
                     gap_analysis: Optional[dict] = None) -> Optional[dict]:
    """Determine if a band should be split due to narrow gap.

    Args:
        band: Band dictionary with 'top', 'bottom', 'center'
        profile: Intensity profile
        typical_spacing: Expected spacing between papers
        gap_analysis: Pre-computed gap analysis (optional)

    Returns:
        Split info dict if should split, None otherwise
    """
    band_width = band['bottom'] - band['top']

    # Only consider splitting wide bands that might contain two papers
    if band_width < typical_spacing * 1.3:
        return None

    # Extract band region
    start = max(0, int(band['top']))
    end = min(len(profile), int(band['bottom']))
    band_profile = profile[start:end]

    if len(band_profile) < 5:
        return None

    # Detect gap within this band
    local_gap_analysis = detect_narrow_gaps_in_profile(
        band_profile,
        typical_spacing / 2  # Half spacing since we're looking within one band
    )

    if not local_gap_analysis['has_narrow_gaps']:
        return None

    # Find the strongest gap
    gaps = local_gap_analysis['gaps']
    best_gap = max(gaps, key=lambda g: g['confidence'])

    # Convert local position to global
    global_gap_pos = start + best_gap['position']

    # Verify the gap is not too close to band edges (could be normal boundary)
    edge_margin = max(2, typical_spacing * 0.2)
    if (global_gap_pos - band['top'] < edge_margin or
        band['bottom'] - global_gap_pos < edge_margin):
        return None

    # Check if split would create reasonable sub-bands
    left_width = global_gap_pos - band['top']
    right_width = band['bottom'] - global_gap_pos
    min_width = max(2, typical_spacing * 0.25)

    if left_width < min_width or right_width < min_width:
        return None

    return {
        'should_split': True,
        'gap_position': global_gap_pos,
        'gap_width': best_gap['width'],
        'gap_confidence': best_gap['confidence'],
        'left_band': {
            'top': band['top'],
            'bottom': global_gap_pos - best_gap['width'] / 2,
            'center': (band['top'] + global_gap_pos) / 2
        },
        'right_band': {
            'top': global_gap_pos + best_gap['width'] / 2,
            'bottom': band['bottom'],
            'center': (global_gap_pos + band['bottom']) / 2
        }
    }


def split_bands_at_narrow_gaps(bands: list, profile: np.ndarray,
                               typical_spacing: float) -> tuple[list, dict]:
    """Split bands that contain narrow gaps.

    Args:
        bands: List of band dictionaries
        profile: Intensity profile
        typical_spacing: Expected spacing between papers

    Returns:
        (modified_bands, debug_info)
    """
    # First, do global gap analysis
    gap_analysis = detect_narrow_gaps_in_profile(profile, typical_spacing)

    if not gap_analysis['has_narrow_gaps']:
        return bands, {
            'splits_performed': 0,
            'gap_analysis': gap_analysis,
            'reason': 'no_narrow_gaps_detected'
        }

    modified_bands = []
    splits_performed = []

    for band in bands:
        split_info = should_split_band(band, profile, typical_spacing, gap_analysis)

        if split_info is None:
            modified_bands.append(band)
        else:
            # Create two new bands from the split
            left_band = band.copy()
            left_band.update(split_info['left_band'])
            left_band['id'] = f"{band['id']}_left"
            left_band['split_from'] = band['id']

            right_band = band.copy()
            right_band.update(split_info['right_band'])
            right_band['id'] = f"{band['id']}_right"
            right_band['split_from'] = band['id']

            # Adjust scores (conservative: reduce since split is uncertain)
            confidence_factor = split_info['gap_confidence']
            left_band['score'] = band['score'] * confidence_factor * 0.9
            right_band['score'] = band['score'] * confidence_factor * 0.9

            modified_bands.extend([left_band, right_band])

            splits_performed.append({
                'original_band_id': band['id'],
                'gap_position': split_info['gap_position'],
                'confidence': split_info['gap_confidence']
            })

    # Re-sort and re-index
    modified_bands.sort(key=lambda b: b['center'])
    for i, band in enumerate(modified_bands):
        if not band['id'].endswith(('_left', '_right')):
            band['index'] = i

    return modified_bands, {
        'splits_performed': len(splits_performed),
        'split_details': splits_performed,
        'gap_analysis': gap_analysis,
        'original_band_count': len(bands),
        'modified_band_count': len(modified_bands)
    }


def adjust_box_boundaries_for_tilt(boxes: list, aligned: np.ndarray,
                                   mapping: np.ndarray, raw: np.ndarray) -> tuple[list, dict]:
    """Adjust box boundaries to follow paper body contours for tilted papers.

    For HBB boxes on tilted papers, adjust top/bottom boundaries to better
    follow the actual paper body instead of including background gaps.

    Args:
        boxes: List of box dictionaries with 'detection_xyxy'
        aligned: Aligned (straightened) image
        mapping: Column-wise vertical mapping from aligned to detection coords
        raw: Raw grayscale image

    Returns:
        (adjusted_boxes, debug_info)
    """
    h, w = aligned.shape
    adjustments = []

    for box in boxes:
        box_id = box['id']
        aligned_top = box['aligned_xyxy'][1]
        aligned_bottom = box['aligned_xyxy'][3]

        # Sample multiple x positions to detect tilt
        sample_x = np.linspace(10, w-10, min(5, w//50), dtype=int)

        # For each column, find where the paper body actually is
        top_adjustments = []
        bottom_adjustments = []

        for x in sample_x:
            col_mapping = mapping[:, x]

            # Find aligned rows that map near the box boundaries
            aligned_rows = np.arange(h)
            detection_y = col_mapping

            # Near top boundary: search for paper body start
            search_top_start = max(0, int(aligned_top) - 10)
            search_top_end = min(h, int(aligned_top) + 10)

            if search_top_end > search_top_start:
                search_region_aligned = aligned[search_top_start:search_top_end, x]
                search_region_raw = raw[search_top_start:search_top_end, x]

                # Find where intensity rises (paper body starts)
                threshold = np.percentile(search_region_raw, 50)
                paper_starts = np.where(search_region_raw > threshold)[0]

                if len(paper_starts) > 0:
                    local_top = search_top_start + paper_starts[0]
                    detection_top = col_mapping[local_top]
                    top_adjustments.append(detection_top)

            # Near bottom boundary: search for paper body end
            search_bottom_start = max(0, int(aligned_bottom) - 10)
            search_bottom_end = min(h, int(aligned_bottom) + 10)

            if search_bottom_end > search_bottom_start:
                search_region_aligned = aligned[search_bottom_start:search_bottom_end, x]
                search_region_raw = raw[search_bottom_start:search_bottom_end, x]

                # Find where intensity drops (paper body ends)
                threshold = np.percentile(search_region_raw, 50)
                paper_ends = np.where(search_region_raw > threshold)[0]

                if len(paper_ends) > 0:
                    local_bottom = search_bottom_start + paper_ends[-1]
                    detection_bottom = col_mapping[local_bottom]
                    bottom_adjustments.append(detection_bottom)

        # Compute adjusted boundaries
        if top_adjustments:
            adjusted_top = float(np.median(top_adjustments))
            top_shift = adjusted_top - box['detection_xyxy'][1]
        else:
            adjusted_top = box['detection_xyxy'][1]
            top_shift = 0

        if bottom_adjustments:
            adjusted_bottom = float(np.median(bottom_adjustments))
            bottom_shift = adjusted_bottom - box['detection_xyxy'][3]
        else:
            adjusted_bottom = box['detection_xyxy'][3]
            bottom_shift = 0

        # Only adjust if shift is significant and reasonable
        max_reasonable_shift = 10  # pixels
        if abs(top_shift) > 1 and abs(top_shift) < max_reasonable_shift:
            box['detection_xyxy'][1] = adjusted_top
        if abs(bottom_shift) > 1 and abs(bottom_shift) < max_reasonable_shift:
            box['detection_xyxy'][3] = adjusted_bottom

        adjustments.append({
            'box_id': box_id,
            'top_shift': float(top_shift),
            'bottom_shift': float(bottom_shift),
            'adjusted': abs(top_shift) > 1 or abs(bottom_shift) > 1
        })

    return boxes, {
        'boxes_adjusted': sum(a['adjusted'] for a in adjustments),
        'adjustment_details': adjustments
    }
