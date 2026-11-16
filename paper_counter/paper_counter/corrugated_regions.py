"""Count corrugated bodies using measured transverse seams, never a height grid.

The webs crossing a flute cavity distinguish it from an inter-sheet gap.
Thickness ranks competing observed seams; it cannot create a new boundary.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, binary_closing, binary_dilation
from scipy.signal import find_peaks

from .region_sequence import runs, scale_hypotheses


def corroborate_seam_scale(reference, raw, interval, units, pitch):
    """Reject a seam half-cycle only when raw gray does not repeat at that scale."""
    a,b=interval
    stripes=np.array([np.mean(part,axis=1) for part in np.array_split(raw,3,axis=1)])
    mask=np.zeros(len(raw),bool);mask[a:b]=True
    measured,model=scale_hypotheses(units,stripes,mask,pitch)
    profile=np.mean(stripes,axis=0)
    signal=(profile-gaussian_filter1d(profile,max(12.,min(70.,len(profile)/30))))*mask
    size=1<<(2*len(signal)-1).bit_length()
    spectrum=np.fft.rfft(signal,size)
    acf=np.fft.irfft(spectrum*np.conj(spectrum))[:len(signal)]
    acf=acf/max(float(acf[0]),1e-8)
    short=float(np.interp(reference,np.arange(len(acf)),acf))
    complete=float(np.interp(measured,np.arange(len(acf)),acf))
    info=dict(seam_reference=float(reference),raw_reference=float(measured),
              seam_scale_acf=short,raw_scale_acf=complete,override=False)
    if 1.6 <= measured/reference <= 2.7 and short < .1 and complete > .25 and complete-short > .2:
        info.update(override=True,reason='short_seam_spacing_has_no_raw_recurrence')
        return float(measured),info
    return reference,info


def observe_seams(raw, valid, pitch):
    """Retain spatial column votes so a dark cavity cannot hide its bright web."""
    raw = np.asarray(raw, dtype=np.float32)
    h, w = raw.shape
    smooth = gaussian_filter1d(raw, .65, axis=0)
    threshold, _ = cv2.threshold(np.clip(raw, 0, 255).astype(np.uint8), 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low, high = raw[raw <= threshold], raw[raw > threshold]
    bg = float(np.median(low)) if len(low) else float(np.min(raw))
    fg = float(np.median(high)) if len(high) else bg
    contrast = max(1., fg-bg)
    ink = (raw > threshold) & valid
    occupancy = np.mean(ink, axis=1)
    active = binary_closing(occupancy > .08, structure=np.ones(3))
    intervals = runs(active)
    if not intervals or contrast < 8:
        return [], None, dict(reason='no_credible_paper', candidates=[], contrast=contrast)
    # Background speckles have neither sustained bright area nor membership
    # in a substantial component of the already localized stack.
    masses = [float(occupancy[a:b].sum()) for a, b in intervals]
    largest = max(masses)
    body = [(a, b) for (a, b), mass in zip(intervals, masses) if mass >= .08*largest]
    a, b = int(body[0][0]), int(body[-1][1])
    radius = max(4, int(round(.5 * max(pitch or 14., 4.))))
    left = maximum_filter1d(smooth, 2*radius+1, origin=radius, axis=0)
    right = maximum_filter1d(smooth, 2*radius+1, origin=-radius, axis=0)
    trough = np.minimum(left, right)-smooth
    residual = raw-smooth
    noise = 1.4826*np.median(abs(residual[a:b]), axis=0)
    minimum = np.maximum(max(5., .03*contrast), 3*noise)
    column_vote = (trough > minimum[None, :]) & valid
    coverage = column_vote.sum(axis=1)/np.maximum(valid.sum(axis=1), 1)
    # A transverse bright web inside a genuinely black cavity is physical
    # material, even when the other 80% of the row is dark. A gray contact
    # seam between blurred faces can instead rely on its measured contrast.
    dark_fraction = np.mean(smooth < bg+.12*contrast, axis=1)
    web_fraction = np.mean(smooth > bg+.75*contrast, axis=1)
    porosity = float(np.mean(smooth[a:b] < bg+.12*contrast))
    hollow = porosity > .35
    # Topology is clearer when open cavities occupy substantial image area.
    # With mostly closed/overexposed flutes, require a darker local cavity so
    # a weak real contact seam is not rejected by a few neighboring bright pixels.
    topology_confidence = float(np.clip((porosity-.12)/.16,0,1))
    local_dark_support = .45-.30*topology_confidence
    cavity = (web_fraction > .055) if hollow else ((dark_fraction > local_dark_support) & (web_fraction > .055))
    # Blur creates gray shoulders around a black cavity. Those shoulders are
    # part of the same web-bounded cavity, not weak transverse contact seams.
    shoulder = 2
    shoulders = binary_dilation(cavity, iterations=shoulder)
    response = coverage*(1-topology_confidence*shoulders.astype(float))
    response[cavity] = 0.
    minimum_coverage = .90 if hollow else .48
    peaks, _ = find_peaks(response, height=minimum_coverage, prominence=.10, distance=3)
    candidates = []
    profile = np.mean(smooth, axis=1)
    for p in peaks:
        if not a < p < b:
            continue
        # Refine inside this measured trough only. Nearby plateau maxima are
        # one seam, rather than extra paper boundaries.
        lo, hi = max(a+1, p-2), min(b-1, p+3)
        good = np.flatnonzero(response[lo:hi] >= max(minimum_coverage, response[p]-.08))+lo
        y = int(good[np.argmin(profile[good])]) if len(good) else int(p)
        item = dict(y=y, coverage=float(coverage[y]), gray_depth=float(np.mean(trough[y])),
                    bright_web_fraction=float(web_fraction[y]), dark_fraction=float(dark_fraction[y]),
                    support='strong' if coverage[y] >= .72 else 'weak_observed_contact')
        if candidates and y-candidates[-1]['y'] <= 4:
            if profile[y] < profile[candidates[-1]['y']]:
                candidates[-1] = item
        else:
            candidates.append(item)
    return candidates, (a, b), dict(candidates=candidates, body_interval=[a,b],
        excluded_foreground_components=[dict(interval=[int(lo),int(hi)],mass=mass)
            for (lo,hi),mass in zip(intervals,masses) if hi<=a or lo>=b],
        foreground_threshold=float(threshold), background=bg, foreground=fg,
        contrast=contrast, cavity_area_fraction=porosity, open_flute_structure=bool(hollow),
        topology_confidence=topology_confidence,cavity_shoulder_pixels=shoulder,
        flank_radius=radius, cavity_rows_rejected=int(np.count_nonzero(cavity[a:b])),
        dark_intervals=[[int(lo),int(hi)] for lo,hi in runs(np.percentile(smooth,98,axis=1) < bg+.12*contrast)],
        boundary_rule='transverse_raw_gray_trough_without_bright_flute_web',
        thickness_creates_boundaries=False)


def segment_corrugated_bodies(raw, valid, bands, units, pitch=None):
    centers = np.array(sorted(b['center'] for b in bands), dtype=float)
    lobe_spacing = float(np.median(np.diff(centers))) if len(centers) >= 3 else 7.
    # The global ACF can lock onto fine optical texture. Several independently
    # observed lobes provide enough flank context for a blurred contact seam.
    window_pitch = max(float(pitch or 14.), 4*lobe_spacing)
    seams, interval, evidence = observe_seams(raw, valid, window_pitch)
    audit = dict(algorithm='observed_corrugated_body_sections', before_count=len(units),
                 after_count=0, applied=True, observed_seams=evidence, actions=[],
                 inferred_regions=0, period_creates_boxes=False, excluded_partial_edges=[])
    if interval is None:
        return [], audit
    a, b = interval
    dark = np.zeros(len(raw), dtype=bool)
    for lo, hi in evidence['dark_intervals']:
        dark[lo:hi] = True
    dark_prefix = np.r_[0, np.cumsum(dark)]
    positions = np.array([a]+[s['y'] for s in seams]+[b], dtype=int)
    distances = np.maximum(1, np.diff(positions)-np.diff(dark_prefix[positions]))
    anchors = [s for s in seams if s['coverage'] >= .72]
    if len(anchors) >= 3:
        anchor_positions = np.array([s['y'] for s in anchors])
        internal = np.maximum(1, np.diff(anchor_positions)-np.diff(dark_prefix[anchor_positions])).astype(float)
        # Height-weighted mode prevents several short texture fragments from
        # outvoting a complete repeating body. This only ranks observed cuts.
        logs = np.log(np.maximum(internal, 1.))
        density = np.exp(-.5*((logs[:,None]-logs[None,:])/.22)**2) @ np.sqrt(internal)
        mode = float(internal[np.argmax(density)])
        near = internal[abs(np.log(internal/mode)) < .3]
        reference = float(np.median(near))
    else:
        reference = float(np.median(distances))
    reference = max(reference, 3.)
    reference,corroboration=corroborate_seam_scale(reference,raw,interval,units,pitch)
    audit['scale_corroboration']=corroboration
    for component in evidence['excluded_foreground_components']:
        lo,hi=component['interval']
        ids=[i for i,band in enumerate(bands) if lo<=band['center']<hi]
        if hi-lo < .32*reference and len(ids)<=1:
            audit['excluded_partial_edges'].append(dict(interval=[lo,hi],reference_height=reference,
                band_ids=ids,reason='thin_isolated_outer_liner'))
    for side in ('top', 'bottom'):
        if len(positions) < 3:
            break
        top, bottom = (positions[0], positions[1]) if side == 'top' else (positions[-2], positions[-1])
        ids = [i for i, band in enumerate(bands) if top <= band['center'] < bottom]
        if bottom-top < .32*reference and len(ids) <= 1:
            audit['excluded_partial_edges'].append(dict(interval=[int(top),int(bottom)],
                reference_height=reference,band_ids=ids,reason='incomplete_thin_outer_liner'))
            if side == 'top':
                positions, seams = positions[1:], seams[1:]
            else:
                positions, seams = positions[:-1], seams[:-1]
    a, b = int(positions[0]), int(positions[-1])
    costs = np.full(len(positions), np.inf)
    parent = np.full(len(positions), -1, int)
    costs[0] = 0.
    for end in range(1, len(positions)):
        starts = np.arange(end)
        heights = positions[end]-positions[starts]-dark_prefix[positions[end]]+dark_prefix[positions[starts]]
        # Relative duration grows with a missing body, whereas log-duration
        # lets one giant rectangle become cheaper than many observed sheets.
        z = abs(heights/reference-1.)/.35
        duration = 2*(np.sqrt(1+z*z)-1)
        # Missing seams remain unresolved thick bodies. There is no maximum
        # duration or fallback cut that can tile a seam-free bright rectangle.
        boundary = 0. if end == len(positions)-1 else .5+2*(1-seams[end-1]['coverage'])
        values = costs[starts]+duration+boundary
        pick = int(np.argmin(values))
        costs[end], parent[end] = float(values[pick]), int(starts[pick])
    indices = [len(positions)-1]
    while indices[-1] > 0:
        indices.append(int(parent[indices[-1]]))
    indices = indices[::-1]
    output = []
    audit['background_intervals_excluded'] = []
    dark_intervals = evidence['dark_intervals']
    for top, bottom in zip(positions[indices[:-1]], positions[indices[1:]]):
        original_top, original_bottom = int(top), int(bottom)
        for lo, hi in dark_intervals:
            if lo <= top < hi:
                top = hi
            if lo <= bottom < hi:
                bottom = lo
        if bottom <= top:
            audit['background_intervals_excluded'].append([original_top,original_bottom])
            continue
        ids = [i for i, band in enumerate(bands) if top <= band['center'] < bottom]
        # A tiny exposed liner is an incomplete outside fragment. Never apply
        # this rule to a thin interior paper with two independent neighbors.
        if (top == a or bottom == b) and bottom-top < .32*reference and len(ids) <= 1:
            audit['excluded_partial_edges'].append(dict(interval=[int(top),int(bottom)],
                reference_height=reference,band_ids=ids,reason='incomplete_thin_outer_liner'))
            continue
        sources = [u['id'] for u in units if min(bottom,u['bottom']) > max(top,u['top'])]
        output.append(dict(id=f'corrugated_{len(output):05d}', top=float(top), bottom=float(bottom),
            band_ids=ids, score=float(np.mean([bands[i]['score'] for i in ids])) if ids else .5,
            structure='paper_body', thickness_inferred=False,
            evidence=dict(corrugated_body=dict(source_unit_ids=sources,
                boundary_source='observed_transverse_seams', reference_height=reference,
                optical_bands_are_not_sheets=True))))
    audit.update(after_count=len(output), normal_height=reference, sequence_cost=float(costs[-1]),
        selected_seams=positions[indices[1:-1]].tolist(),
        rejected_seams=[s for i,s in enumerate(seams,1) if i not in indices],
        actions=[dict(action='partition_at_observed_seams_only', bounds=positions[indices].tolist())])
    audit['unresolved_contact_regions'] = [dict(id=u['id'],interval=[u['top'],u['bottom']],
        reason='wide_body_without_another_supported_transverse_seam') for u in output
        if u['bottom']-u['top'] > 1.8*reference]
    return output, audit
