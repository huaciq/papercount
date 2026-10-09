"""Controlled physical layers and independent metrology/GT for 1D evaluation."""
import numpy as np


def suite(replicates=5):
    cases = []
    for kind in ('standard', 'narrow_gap', 'bimodal', 'weak_sheet', 'wide_gap', 'background_drift', 'noise', 'endpoint', 'texture', 'saturation'):
        for seed in range(replicates):
            rng = np.random.default_rng(17001 + seed)
            n, width, gap = 12, 12., 4.
            if kind == 'narrow_gap':
                gap = 1.
            centers = [30.]
            widths = []
            for j in range(n):
                w = width + rng.uniform(-1., 1.)
                widths.append(w)
                if j:
                    extra = 32. if kind == 'wide_gap' and j == 6 else 0.
                    centers.append(centers[-1] + (widths[j - 1] + w) / 2 + gap + extra)
            if kind == 'endpoint':
                shift = centers[0] - 2.
                centers = [c - shift for c in centers]
            length = int(np.ceil(centers[-1] + 35))
            if kind == 'endpoint':
                length = int(np.ceil(centers[-1] + 3))
            y = np.arange(length)
            signal = np.full(length, 20.)
            truths = []
            for j, (center, w) in enumerate(zip(centers, widths)):
                inside = np.abs(y - center) <= w / 2
                amplitude = 160.
                weak = kind == 'weak_sheet' and j in (3, 7)
                if weak:
                    amplitude = 35.
                if kind == 'bimodal' and j in (3, 7):
                    shape = .9 * np.exp(-.5 * ((y - center - w * .23) / (w * .16))**2)
                    shape += .9 * np.exp(-.5 * ((y - center + w * .23) / (w * .16))**2)
                else:
                    shape = np.exp(-.5 * ((y - center) / (w * .29))**2)
                signal += amplitude * shape * inside
                truths.append(dict(center=float(center), left=float(center - w / 2), right=float(center + w / 2), weak=weak))
            texture_intervals = []
            if kind == 'texture':
                # Two isolated bright speckles in a known non-sheet background.
                for c in (8., centers[-1] + 22.):
                    signal += 40 * np.exp(-.5 * ((y - c) / .5)**2)
                    texture_intervals.append([float(c - 3), float(c + 3)])
            if kind == 'background_drift':
                signal += np.linspace(-8, 45, length) + 12 * np.sin(y / length * np.pi)
            noise = 9. if kind == 'noise' else 1.
            signal += rng.normal(0, noise, length)
            if kind == 'saturation':
                signal = np.clip(signal * 3.5, 0, 255)
            else:
                signal = np.clip(signal, 0, 255)
            # Metrology comes from independently specified physical widths,
            # not their center spacing. No truth center enters the front end.
            scales = np.linspace(min(widths) / 2, max(widths) / 2, 5)
            mean_gap = (gap * (n - 1) + (32 if kind == 'wide_gap' else 0)) / (n - 1)
            cases.append(dict(name=f'{kind}_{seed}', kind=kind, signal=signal, truths=truths,
                              texture_intervals=texture_intervals, scales=scales, sigma_v=mean_gap / 2,
                              seed=17001 + seed, count=n))
    for kind in ('flat', 'linear_drift', 'black_band_only'):
        y = np.arange(240)
        signal = np.full(240, 20.) if kind == 'flat' else 20. + .2 * y
        if kind == 'black_band_only':
            signal = np.full(240, 120.)
            signal[60:180] = 10.
        cases.append(dict(name=kind, kind=kind, signal=signal, truths=[], texture_intervals=[],
                          scales=np.linspace(5.5, 6.5, 5), sigma_v=2., seed=None, count=0))
    return cases


def evidence_metrics(candidates, truths, texture_intervals):
    """Non-overlapping physical body intervals define independent associations.

    At most one TP per body. Extra candidates in that body are duplicates;
    unmatched candidates are FP. This is 1D localization, not track accuracy.
    """
    hits = [sum(t['left'] <= c['y'] <= t['right'] for c in candidates) for t in truths]
    tp = sum(h > 0 for h in hits)
    duplicate = sum(max(0, h - 1) for h in hits)
    weak = [h for h, t in zip(hits, truths) if t['weak']]
    texture = sum(any(a <= c['y'] <= b for a, b in texture_intervals) for c in candidates)
    return dict(gt_layers=len(truths), tp=tp, fp=len(candidates) - tp, fn=len(truths) - tp,
                matched_weak=sum(h > 0 for h in weak), weak_layers=len(weak), duplicate_candidates=duplicate,
                candidate_count=len(candidates), texture_fp=texture, texture_objects=len(texture_intervals),
                physical_sheet_precision=tp / len(candidates) if candidates else None,
                physical_sheet_recall=tp / len(truths) if truths else None,
                ridge_duplicate_rate=duplicate / len(candidates) if candidates else None,
                weak_sheet_recall=sum(h > 0 for h in weak) / len(weak) if weak else None)
