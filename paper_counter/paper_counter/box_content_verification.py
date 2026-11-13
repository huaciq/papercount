# -*- coding: utf-8 -*-
"""Box content verification - verify boxes by analyzing paper body features.

For low-score boxes, verify if they contain actual paper body by analyzing:
1. Grayscale distribution (paper vs background)
2. Horizontal consistency (paper body should be consistent across width)
3. Contrast with neighbors
4. Texture features
"""
from __future__ import annotations
import numpy as np
import cv2
from scipy.stats import mode
from typing import Dict, List, Tuple


def analyze_box_content(box: dict, aligned_image: np.ndarray,
                       aligned_raw: np.ndarray, valid_mask: np.ndarray) -> Dict:
    """Analyze the content of a box to verify if it contains real paper.

    Args:
        box: Box dictionary with aligned_xyxy
        aligned_image: Aligned (normalized) image
        aligned_raw: Aligned raw grayscale image
        valid_mask: Valid pixel mask

    Returns:
        Dict with verification metrics
    """
    # Extract box region
    x1, y1, x2, y2 = box['aligned_xyxy']
    y1_int, y2_int = int(np.floor(y1)), int(np.ceil(y2))

    h, w = aligned_image.shape
    y1_int = max(0, y1_int)
    y2_int = min(h, y2_int)

    if y2_int <= y1_int:
        return {'valid': False, 'reason': 'invalid_coordinates'}

    # Extract regions
    box_region = aligned_raw[y1_int:y2_int, :]
    box_valid = valid_mask[y1_int:y2_int, :]

    if box_region.size == 0:
        return {'valid': False, 'reason': 'empty_region'}

    # Metric 1: Grayscale distribution
    # Paper body should have higher intensity than background
    valid_pixels = box_region[box_valid]

    if len(valid_pixels) < 10:
        return {'valid': False, 'reason': 'insufficient_valid_pixels'}

    mean_intensity = float(np.mean(valid_pixels))
    median_intensity = float(np.median(valid_pixels))
    std_intensity = float(np.std(valid_pixels))

    # Metric 2: Horizontal consistency
    # Paper body should be relatively consistent across width
    column_means = np.mean(box_region, axis=0)
    valid_column_means = column_means[np.any(box_valid, axis=0)]

    if len(valid_column_means) < w * 0.5:
        horizontal_consistency = 0.0
    else:
        # Low std = high consistency
        column_std = np.std(valid_column_means)
        column_range = np.ptp(valid_column_means)
        horizontal_consistency = 1.0 - min(1.0, column_std / (np.mean(valid_column_means) + 1e-6))

    # Metric 3: Vertical profile quality
    # Good paper body should have a clear peak in vertical profile
    row_means = np.mean(box_region, axis=1)
    profile_quality = float(np.ptp(row_means) / (np.mean(row_means) + 1e-6))

    # Metric 4: Coverage
    # What fraction of the box has valid pixels?
    coverage = float(np.sum(box_valid) / box_valid.size)

    return {
        'valid': True,
        'mean_intensity': mean_intensity,
        'median_intensity': median_intensity,
        'std_intensity': std_intensity,
        'horizontal_consistency': horizontal_consistency,
        'profile_quality': profile_quality,
        'coverage': coverage,
    }


def verify_box_is_paper(box: dict, metrics: Dict,
                       neighbor_metrics: List[Dict],
                       background_threshold: float = 30.0) -> Dict:
    """Verify if a box contains real paper based on content analysis.

    Args:
        box: Box to verify
        metrics: Content metrics for this box
        neighbor_metrics: Metrics from neighboring boxes
        background_threshold: Intensity threshold for background

    Returns:
        Dict with verification result and confidence
    """
    if not metrics.get('valid'):
        return {
            'is_paper': False,
            'confidence': 0.0,
            'reason': metrics.get('reason', 'invalid_metrics')
        }

    # Compute neighbor statistics for comparison
    if neighbor_metrics:
        valid_neighbors = [m for m in neighbor_metrics if m.get('valid')]
        if valid_neighbors:
            neighbor_mean_intensity = np.mean([m['mean_intensity'] for m in valid_neighbors])
            neighbor_consistency = np.mean([m['horizontal_consistency'] for m in valid_neighbors])
        else:
            neighbor_mean_intensity = None
            neighbor_consistency = None
    else:
        neighbor_mean_intensity = None
        neighbor_consistency = None

    # Decision factors - More conservative scoring
    factors = []

    # Factor 1: Intensity check (more weight)
    if metrics['mean_intensity'] > background_threshold + 20:
        factors.append(('intensity_well_above_background', 0.35))
    elif metrics['mean_intensity'] > background_threshold:
        factors.append(('intensity_above_background', 0.15))
    else:
        factors.append(('intensity_too_low', -0.6))  # More penalty

    # Factor 2: Consistency check (more weight)
    if metrics['horizontal_consistency'] > 0.8:
        factors.append(('very_high_consistency', 0.25))
    elif metrics['horizontal_consistency'] > 0.7:
        factors.append(('high_consistency', 0.15))
    elif metrics['horizontal_consistency'] < 0.5:
        factors.append(('low_consistency', -0.3))  # More penalty

    # Factor 3: Coverage check (more strict)
    if metrics['coverage'] > 0.7:
        factors.append(('excellent_coverage', 0.2))
    elif metrics['coverage'] > 0.5:
        factors.append(('good_coverage', 0.1))
    elif metrics['coverage'] < 0.4:
        factors.append(('poor_coverage', -0.35))  # More penalty

    # Factor 4: Comparison with neighbors (more weight)
    if neighbor_mean_intensity is not None:
        intensity_ratio = metrics['mean_intensity'] / (neighbor_mean_intensity + 1e-6)
        if 0.8 < intensity_ratio < 1.2:
            # Very similar to neighbors - strong evidence
            factors.append(('very_similar_to_neighbors', 0.35))
        elif 0.7 < intensity_ratio < 1.3:
            # Similar to neighbors - good sign
            factors.append(('similar_to_neighbors', 0.2))
        elif intensity_ratio < 0.5:
            # Much darker than neighbors - suspicious
            factors.append(('much_darker', -0.5))
        elif intensity_ratio < 0.7:
            factors.append(('somewhat_darker', -0.2))

    # Compute confidence score - Start lower
    confidence = 0.4  # Start more skeptical

    reasons = []
    for factor_name, weight in factors:
        confidence += weight
        reasons.append(factor_name)

    confidence = np.clip(confidence, 0.0, 1.0)

    # Higher threshold for decision
    is_paper = confidence > 0.6  # Raised from 0.5 to 0.6

    return {
        'is_paper': is_paper,
        'confidence': float(confidence),
        'factors': reasons,
        'metrics': metrics
    }


def verify_suspicious_boxes(boxes: List[dict], suspicious_indices: List[int],
                           aligned_image: np.ndarray, aligned_raw: np.ndarray,
                           valid_mask: np.ndarray) -> Dict:
    """Verify suspicious boxes using content analysis.

    Args:
        boxes: All boxes
        suspicious_indices: Indices of suspicious boxes to verify
        aligned_image: Aligned normalized image
        aligned_raw: Aligned raw image
        valid_mask: Valid pixel mask

    Returns:
        Dict with verification results for each suspicious box
    """
    results = {}

    # First, analyze all boxes to get baseline
    all_metrics = []
    for box in boxes:
        metrics = analyze_box_content(box, aligned_image, aligned_raw, valid_mask)
        all_metrics.append(metrics)

    # Verify each suspicious box
    for idx in suspicious_indices:
        if idx < 0 or idx >= len(boxes):
            continue

        box = boxes[idx]
        metrics = all_metrics[idx]

        # Get neighbor metrics
        neighbor_indices = []
        if idx > 0:
            neighbor_indices.append(idx - 1)
        if idx < len(boxes) - 1:
            neighbor_indices.append(idx + 1)

        neighbor_metrics = [all_metrics[i] for i in neighbor_indices]

        # Verify
        verification = verify_box_is_paper(box, metrics, neighbor_metrics)

        results[idx] = {
            'box_number': box['number'],
            'box_index': idx,
            'verification': verification
        }

    return results


def should_remove_box_by_content(box: dict, verification: Dict,
                                gap_before: float, gap_after: float,
                                median_gap: float) -> Tuple[bool, str]:
    """Decide if a box should be removed based on content verification.

    Args:
        box: Box to decide on
        verification: Verification result from verify_box_is_paper
        gap_before: Gap before this box
        gap_after: Gap after this box
        median_gap: Median gap between boxes

    Returns:
        (should_remove, reason)
    """
    if not verification.get('is_paper'):
        confidence = verification.get('confidence', 0)

        # Very low confidence + not filling critical gap = remove
        if confidence < 0.3:
            # Check if removing would create a large gap
            if gap_before < median_gap * 1.5 and gap_after < median_gap * 1.5:
                return True, f"Low confidence ({confidence:.2f}) and not filling critical gap"

        # Low confidence but filling a large gap = keep with warning
        if gap_before > median_gap * 2 or gap_after > median_gap * 2:
            return False, f"Low confidence ({confidence:.2f}) but filling large gap, kept"

        # Moderate confidence = keep if score not too low
        if confidence < 0.4 and box['score'] < 1.5:
            return True, f"Low confidence ({confidence:.2f}) + very low score ({box['score']:.2f})"

    # Default: keep
    return False, f"Confidence ({verification.get('confidence', 0):.2f}) acceptable"
