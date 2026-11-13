"""Measured paper bands, independent of stack spacing and image-wide brightness.

Enhancement proposes boundaries; only these unnormalized measurements establish
body identity. Sampling follows an observed opposite edge, not a multiple of T.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def measure_body(line, y, polarity, spacing, minimum_contrast=2.0):
    line = np.asarray(line, dtype=float)
    center = int(round(y))
    empty = {'state': 'unknown', 'observable': False, 'contrast': 0., 'width': None,
             'pair_y': None, 'snr': 0., 'partial': False}
    if not 0 <= center < len(line) or len(line) < 5:
        return empty
    direction = 1 if polarity == 'positive' else -1
    radius = max(4, int(np.ceil(1.25 * spacing)))
    base=max(0,center-radius-4)
    line=line[base:min(len(line),center+radius+5)]
    center-=base
    smooth = gaussian_filter1d(line, .6)
    gradient = np.gradient(smooth)
    lo, hi = max(0, center-radius), min(len(line), center+radius+1)
    local = smooth[lo:hi]
    derivatives = np.diff(local)
    noise = max(.5, 1.4826 * float(np.median(np.abs(derivatives-np.median(derivatives)))))
    signed = gradient * direction
    nearby = np.arange(max(1, center-2), min(len(line)-1, center+3))
    if not len(nearby):
        return empty
    boundary = int(nearby[np.argmax(signed[nearby])])
    opposite, _ = find_peaks(-signed, prominence=max(.5, noise))
    opposite = [int(p) for p in opposite if 2 <= direction*(p-boundary) <= radius]
    choices = []
    for pair in sorted(opposite, key=lambda p: abs(p-boundary)):
        upper, lower = sorted((boundary, pair))
        width = lower-upper
        a, b = upper+max(1, int(.15*width)), lower-max(0, int(.15*width))
        inside = smooth[a:b]
        before = smooth[max(0, upper-3):upper]
        after = smooth[lower+1:min(len(line), lower+4)]
        if len(inside) < 1 or not len(before) or not len(after):
            continue
        body = float(np.median(inside))
        outside = max(float(np.median(before)), float(np.median(after)))
        contrast = body-outside
        interior_dark = inside < outside+max(noise,.15*max(0.,contrast))
        crosses_gap = bool(np.any(interior_dark[:-1]&interior_dark[1:]))
        # Both signs and a bounded bright interior are required. A monotonic
        # reflection tail cannot become a body merely by ending at black.
        if (contrast >= max(minimum_contrast, 2*noise)
                and signed[boundary] > noise and -signed[pair] > noise
                and not crosses_gap
                and np.mean(inside > outside+minimum_contrast*.5) >= .7
                and (contrast>=.3*max(body,minimum_contrast) or minimum_contrast<2.)):
            choices.append({'state': 'confirmed', 'observable': True, 'contrast': contrast,
                'width': float(width), 'pair_y': float(pair+base), 'boundary_y': float(boundary+base),
                'body_level': body, 'outside_level': outside, 'snr': contrast/noise,
                'partial': False, 'sample_bounds': [a+base,b+base]})
    if choices:
        # 纸身上的纹理小坑也有相反导数；优先采用内外对比最大的完整带，
        # 不把遇到的第一个纹理凹槽当成这张纸的另一侧边界。
        return max(choices,key=lambda p:p['contrast'])
    # A visible opposite edge can delimit a body clipped by the image border.
    border = len(line)-1 if direction > 0 else 0
    distance = abs(border-boundary)
    if 2 <= distance <= radius:
        upper, lower = sorted((border, boundary))
        inside = smooth[upper+1:lower]
        outside = smooth[max(0,boundary-4):boundary] if direction > 0 else smooth[boundary+1:boundary+5]
        if len(inside) >= 2 and len(outside):
            contrast = float(np.median(inside)-np.median(outside))
            if contrast >= max(minimum_contrast, 3*noise) and signed[boundary] > noise:
                return {'state':'confirmed','observable':True,'contrast':contrast,
                        'width':float(distance),'pair_y':None,'boundary_y':float(boundary+base),
                        'snr':contrast/noise,'partial':True,'sample_bounds':[upper+1+base,lower+base]}
    complete = center-radius >= 0 and center+radius < len(line)
    paper_side = smooth[max(lo,center-radius):center] if direction < 0 else smooth[center+1:hi]
    # Absence is evidence only when the whole neighborhood is available and
    # the weak transition belongs to the dark tail of a brighter structure.
    peak = float(np.percentile(local,95)) if len(local) else 0.
    level = float(np.median(paper_side)) if len(paper_side) else 0.
    monotone = np.mean(-direction*np.diff(paper_side) <= noise) if len(paper_side)>1 else 0.
    absent = complete and peak >= 4*minimum_contrast and level < .25*peak and monotone >= .75
    return {**empty, 'state':'absent' if absent else 'unknown','observable':bool(complete),
            'noise':noise,'local_level':level,'local_reference':peak,'monotone_tail_ratio':float(monotone)}


def attach_body_evidence(tracks, profiles, offset, spacing_at, preferred, cfg):
    for track in tracks:
        polarity = track.boundary_polarity or preferred
        spacing = max(cfg.min_period, float(spacing_at(track.center_y)))
        samples = []
        periodic=any(p.observation_source=='period_assisted_RAW_band' for p in track.peak_candidates)
        minimum=cfg.strip_sequence_raw_min_contrast*(.5 if periodic else 1.)
        for strip, y in sorted(track.strip_positions.items()):
            if not 0 <= strip < len(profiles):
                continue
            proof = measure_body(profiles[strip], y-offset, polarity, spacing,
                                 minimum)
            if proof.get('pair_y') is not None:proof['pair_y']+=offset
            if proof.get('boundary_y') is not None:proof['boundary_y']+=offset
            if proof.get('sample_bounds') is not None:proof['sample_bounds']=[v+offset for v in proof['sample_bounds']]
            samples.append({'strip':strip, 'y':float(y), **proof})
        confirmed = [s for s in samples if s['state']=='confirmed']
        observable = [s for s in samples if s['observable']]
        absent = [s for s in observable if s['state']=='absent']
        # A strong local segment establishes existence. Missing observations
        # remain unknown and are never counted as negative votes.
        local_anchor = any(s['snr'] >= 6 and s['contrast'] >= 4*cfg.strip_sequence_raw_min_contrast
                           and s['width']>=3 for s in confirmed)
        verified = len(confirmed)>=2 or local_anchor
        # unknown 既不反对存在，也不反对已测到的背景；只统计明确证据。
        # 背景仍需要多个完整窗口，单条带的暗像素不能否决一张纸。
        background = len(absent)>=max(3,int(np.ceil(.25*len(samples)))) and not confirmed
        state = 'confirmed' if verified else 'absent' if background else 'unknown'
        track.local_evidence['paper_body'] = {'state':state,'verified':verified,
            'fully_observable':bool(len(observable)==len(samples) and samples),
            'confirmed_strips':[s['strip'] for s in confirmed],
            'absent_strips':[s['strip'] for s in absent], 'local_anchor':bool(local_anchor),
            'observed_width':float(np.median([s['width'] for s in confirmed])) if confirmed else None,
            'samples':samples,'source':'unnormalized_bounded_band_not_global_brightness'}


def narrow_intensity_profiles(gray, strips, orientation_fields=None):
    """Small measured column neighborhoods retain curved/partial observations."""
    return np.stack([np.median(gray[:,max(a,int((a+b)/2)-2):min(b,int((a+b)/2)+3)],axis=1)
                     for a,b in strips]).astype(np.float32)*(255/np.iinfo(gray.dtype).max)
