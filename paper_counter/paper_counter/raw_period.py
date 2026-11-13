"""Measured RAW interfaces disambiguate a harmonic search scale.

This module receives pixel profiles and an existing scale, never image names,
ground truth or requested counts. It cannot create candidate locations.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate, find_peaks

from .config import CounterConfig


def raw_interface_period(profiles: np.ndarray, initial: float,
                         cfg: CounterConfig) -> tuple[float, dict]:
    required = max(2, int(np.ceil(cfg.min_track_support_ratio * len(profiles))))
    diagnostic = {'changed': False, 'original_scale': float(initial),
                  'required_strips': required, 'strip_evidence': [],
                  'period_generated_paper_count': 0}
    votes = []
    for s, values in enumerate(profiles):
        values = np.asarray(values, dtype=float)
        if len(values) < 2 * cfg.max_period or not np.all(np.isfinite(values)):
            continue
        # Use unnormalized intensity; broad illumination does not establish
        # paper existence. Local dark interfaces must have measurable contrast.
        signal = gaussian_filter1d(values, 32) - gaussian_filter1d(values, 1)
        ys, props = find_peaks(signal, distance=cfg.min_period,
                              prominence=cfg.strip_sequence_raw_min_contrast)
        if len(ys) < 6:
            continue
        cutoff = max(cfg.strip_sequence_raw_min_contrast,
                     .3 * float(np.median(props['prominences'])))
        ys = ys[props['prominences'] >= cutoff]
        gaps = np.diff(ys)
        gaps = gaps[(gaps >= cfg.min_period) & (gaps <= cfg.max_period)]
        if len(gaps) < 4:
            continue
        measured = float(np.median(gaps))
        centered = signal - signal.mean()
        energy = float(np.dot(centered, centered))
        if energy <= cfg.epsilon:
            continue
        acf = correlate(centered, centered, mode='full', method='fft')[
            len(signal)-1:len(signal)+cfg.max_period] / energy
        peaks, _ = find_peaks(acf)
        matches = [int(p) for p in peaks if cfg.min_period <= p <= cfg.max_period
                   and abs(p-measured) <= .2*measured and acf[p] >= .1]
        if not matches:
            continue
        chosen = max(matches, key=lambda p: acf[p])
        votes.append(chosen)
        diagnostic['strip_evidence'].append({'strip': s, 'raw_gap_median': measured,
            'raw_interface_observations': len(ys), 'raw_acf_period': chosen,
            'raw_acf_score': float(acf[chosen])})
    if len(votes) < required:
        return initial, {**diagnostic, 'reason': 'insufficient_raw_interface_quorum'}
    consensus = float(np.median(votes))
    agreeing = sum(abs(p-consensus) <= .2*consensus for p in votes)
    diagnostic.update(alternative_consensus=consensus, agreeing_strips=agreeing)
    if agreeing >= required and 1.7 <= initial/consensus <= 2.3:
        return consensus, {**diagnostic, 'changed': True, 'selected_scale': consensus,
                           'reason': 'raw_interface_gap_and_acf_consensus'}
    return initial, {**diagnostic, 'selected_scale': float(initial),
                     'reason': 'non_harmonic_or_ambiguous_disagreement'}
