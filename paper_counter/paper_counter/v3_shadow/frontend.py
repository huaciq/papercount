"""Standalone front-end. Its signature has no GT, image name, or target count."""
import time
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .config import ShadowConfig
from .ridge_valley import multiscale, valley_likelihood, descriptor
from .identify import peaks
from .verification import verify


def intervals(mask):
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def measure_widths(raw, cfg):
    """Pixel brightness-run widths, never center-to-center period.

    RAW thresholding and robust 10/90 percentiles are operational extensions.
    Full observed widths and extrema remain in the diagnostics.
    """
    smooth = gaussian_filter1d(raw, cfg.width_measure_smoothing_px, mode='reflect')
    low, high = np.percentile(smooth, [10, 90])
    if high - low < 1e-6:
        return None, dict(reason='no_raw_brightness_contrast')
    cut = float((low + high) / 2)
    bright = intervals(smooth > cut)
    body = [int(b - a) for a, b in bright if b - a >= 2 and a > 0 and b < len(raw)]
    gaps = [int(b - a) for a, b in intervals(smooth <= cut) if a > 0 and b < len(raw)]
    diag = dict(threshold=cut, body_widths=body, gap_widths=gaps, source='raw_bright_runs_not_period',
                body_intervals=[[int(a), int(b)] for a, b in bright], width_quantiles=[.1, .9])
    if not body or not gaps:
        return None, dict(**diag, reason='no_separate_measurable_body_and_gap')
    wmin, wmax = np.quantile(body, [.1, .9])
    smin, smax = max(cfg.min_sigma_px, float(wmin) / 2), max(cfg.min_sigma_px, float(wmax) / 2)
    sv = max(cfg.min_sigma_px, float(np.mean(gaps)) / 2)
    sigmas = np.unique(np.linspace(smin, smax, cfg.scales))
    diag.update(observed_min_body_width=min(body), observed_max_body_width=max(body),
                sigma_min=smin, sigma_max=smax, sigma_v=sv, mean_gap_width=float(np.mean(gaps)))
    return (sigmas, sv), diag


def analyze(signal, raw=None, config=None, *, measured_scales=None, reference_values=None, valid=None, intensity_ceiling=None):
    cfg = config or ShadowConfig()
    started = time.perf_counter()
    signal = np.asarray(signal, dtype=np.float64)
    raw = signal.copy() if raw is None else np.asarray(raw, dtype=np.float64)
    if signal.ndim != 1 or signal.shape != raw.shape or len(signal) < 3 or not np.all(np.isfinite(signal)) or not np.all(np.isfinite(raw)):
        raise ValueError('Finite aligned 1D signal and RAW arrays of at least 3 samples required')
    valid = np.ones(len(raw), bool) if valid is None else np.asarray(valid, dtype=bool)
    if valid.shape != raw.shape:
        raise ValueError('Observability must match profile coordinates')
    measured, widths = measure_widths(raw, cfg)
    if measured_scales is not None:
        measured = (np.asarray(measured_scales[0], dtype=float), float(measured_scales[1]))
        widths['scale_source'] = 'externally_measured_body_and_gap_widths; no_period'
    else:
        widths['scale_source'] = 'raw_run_measurement'
    if measured is None or not valid.any():
        return dict(stage=cfg.stage, candidates=[], identified=[], rejected=[], status='insufficient_evidence',
                    review_required=True, widths=widths, config=cfg.to_dict(), seconds=time.perf_counter() - started,
                    arrays={}, kernels=[], verification={})
    sigmas, sv = measured
    m = multiscale(signal, sigmas, 'gaussian' if cfg.stage == 'A1' else 'bi', cfg.rho, cfg.gamma, cfg.tail_sigma)
    fv, vk = valley_likelihood(signal, sv, cfg.gamma, cfg.tail_sigma)
    f = descriptor(m['maximum'], fv, cfg.eta) if cfg.stage in ('A3', 'A4') else m['maximum']
    k = max(1, int(np.ceil(len(raw) * cfg.reference_end_fraction)))
    reference_values = np.r_[raw[:k], raw[-k:]] if reference_values is None else np.asarray(reference_values)
    if not reference_values.size or not np.all(np.isfinite(reference_values)):
        raise ValueError('Finite nonempty RAW reference area required')
    reference_mean = float(np.mean(reference_values))
    identified, rejected_gray = peaks(f, raw, float(np.min(sigmas)), reference_mean, cfg.absolute_peak_floor)
    # Actual data support, not a probability. Reflect-padding samples never
    # count as observed pixels. Unobserved peaks remain visible with a flag.
    radius = max(len(t['weights']) // 2 for t in m['kernels'])
    if cfg.stage in ('A3', 'A4'):
        radius = max(radius, len(vk['weights']) // 2)
    for c in identified:
        y = c['y']
        lo, hi = max(0, y - radius), min(len(raw), y + radius + 1)
        c.update(best_sigma=float(m['best_sigma'][y]), complete_support=bool(y >= radius and y + radius < len(raw) and valid[lo:hi].all()),
                 observable=bool(valid[y]), source='ridge_center', kernel_support_radius=radius)
    if cfg.stage == 'A4':
        candidates, rejected_region, verification = verify(f, identified, cfg.region_area_ratio, cfg.region_peak_ratio, cfg.absolute_peak_floor)
    else:
        candidates, rejected_region, verification = [dict(c) for c in identified], [], {}
    saturation = float(np.mean(raw >= intensity_ceiling)) if intensity_ceiling is not None else None
    review = any(not c['complete_support'] or not c['observable'] for c in candidates) or saturation is not None and saturation >= cfg.max_saturated_fraction
    if not candidates:
        review = True
    return dict(stage=cfg.stage, candidates=candidates, identified=identified, rejected=rejected_gray + rejected_region,
                status='review_required' if review else 'shadow_observations', review_required=bool(review),
                reference_mean=reference_mean, widths=widths, saturated_fraction=saturation, config=cfg.to_dict(),
                arrays=dict(raw=raw, signal=signal, response=f, ridge=m['maximum'], valley=fv,
                            scale_responses=m['levels'], sigmas=m['sigmas'], best_sigma=m['best_sigma'],
                            best_index=m['best_index'], valid=valid), kernels=m['kernels'] + [dict(**vk, kind='valley')],
                verification=verification, seconds=time.perf_counter() - started)
