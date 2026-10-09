"""III-D positive ridge regions, one sheet/region, area and peak verification.

    Paper omits numeric thresholds and representative-selection details. The
    relative median thresholds and strongest-peak tie rule here are explicit
    research extensions, not asserted to be the authors' original calibration.
    """
import numpy as np


def ridge_regions(response):
    f = np.asarray(response, dtype=np.float64)
    change = np.diff(np.r_[False, f > 0, False].astype(int))
    regions = []
    for start, end in zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1)):
        # Zero crossing interpolated between adjacent pixel centers. Region
        # integral uses trapezoids, including fractional triangles at crossings.
        xs = np.arange(start, end, dtype=float)
        values = f[start:end]
        if start > 0:
            crossing = start - f[start] / (f[start] - f[start - 1])
            xs, values = np.r_[crossing, xs], np.r_[0., values]
        if end < len(f):
            crossing = end - 1 + f[end - 1] / (f[end - 1] - f[end])
            xs, values = np.r_[xs, crossing], np.r_[values, 0.]
        area = float(np.trapz(values, xs)) if len(xs) > 1 else float(values[0])
        regions.append(dict(start=int(start), end=int(end), area=area, maximum=float(np.max(f[start:end])),
                            truncated=bool(start == 0 or end == len(f))))
    return regions


def verify(response, candidates, area_ratio=.1, peak_ratio=.1, positive_floor=1e-7):
    regions = ridge_regions(response)
    # Reference only regions that contain a gray-gated candidate. Pure floating
    # point zero fluctuations cannot define the verification thresholds.
    occupied = [r for r in regions if any(r['start'] <= c['y'] < r['end'] for c in candidates)]
    area_threshold = area_ratio * float(np.median([r['area'] for r in occupied])) if occupied else 0.
    peak_threshold = max(positive_floor, peak_ratio * float(np.median([r['maximum'] for r in occupied]))) if occupied else positive_floor
    accepted, rejected = [], []
    for index, region in enumerate(regions):
        cc = [c for c in candidates if region['start'] <= c['y'] < region['end']]
        if not cc:
            continue
        enriched = [dict(**c, region=index, region_start=region['start'], region_end=region['end'],
                         region_area=region['area'], region_maximum=region['maximum'],
                         region_truncated=region['truncated']) for c in cc]
        if region['area'] < area_threshold or region['maximum'] < peak_threshold:
            reason = ('low_area_and_peak' if region['area'] < area_threshold and region['maximum'] < peak_threshold
                      else 'low_region_area' if region['area'] < area_threshold else 'low_region_peak')
            rejected.extend(dict(**c, reason=reason) for c in enriched)
            continue
        enriched.sort(key=lambda c: (-c['strength'], c['y']))
        accepted.append(enriched[0])
        rejected.extend(dict(**c, reason='duplicate_same_positive_region') for c in enriched[1:])
    accepted.sort(key=lambda c: c['y'])
    return accepted, rejected, dict(regions=regions, area_threshold=area_threshold,
                                     peak_threshold=peak_threshold, thresholds_published=False)
