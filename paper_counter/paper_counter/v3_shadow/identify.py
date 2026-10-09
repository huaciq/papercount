"""III-D literal sliding window/jump and RAW reference-area intensity gate."""
import numpy as np


def peaks(response, raw, sigma_min, reference_mean, positive_floor=1e-7):
    f, raw = np.asarray(response), np.asarray(raw)
    if f.ndim != 1 or f.shape != raw.shape or not np.isfinite(reference_mean):
        raise ValueError('Aligned finite 1D inputs required')
    radius = max(1, int(np.ceil(sigma_min)))
    c, candidates, rejected = 0, [], []
    while c < len(f):
        window = f[max(0, c - radius):min(len(f), c + radius + 1)]
        # Equal maxima use first encountered sample, followed by the literal
        # paper jump. Region verification resolves remaining plateau duplicates.
        if f[c] > positive_floor and f[c] >= np.max(window):
            record = dict(y=int(c), strength=float(f[c]), raw=float(raw[c]),
                          reference_mean=float(reference_mean), window_radius=radius)
            if raw[c] < reference_mean:
                rejected.append(dict(**record, reason='below_raw_reference_mean'))
            else:
                candidates.append(record)
            c += radius + 1
        else:
            c += 1
    return candidates, rejected
