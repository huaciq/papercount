from dataclasses import replace
import numpy as np

from paper_counter import CounterConfig
from paper_counter.local_detection import detect_strip_peaks
from paper_counter.preprocessing import signed_gradient, signed_gradient_levels
from paper_counter.profiles import make_strip_profiles_diagnostic


def test_default_gradient_is_exact_legacy_gradient():
    image = np.arange(120, dtype=np.float32).reshape(12, 10)
    levels, native = signed_gradient_levels(image, CounterConfig())
    assert len(levels) == 1
    assert levels[0][2] == "native"
    np.testing.assert_array_equal(levels[0][1], signed_gradient(image))
    np.testing.assert_array_equal(native, signed_gradient(image))


def test_local_slope_alignment_recovers_smeared_edge_response():
    h, w = 240, 100
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    gradient = np.exp(-0.5 * ((y - (100 + 0.12 * x)) / 1.3) ** 2).astype(np.float32)
    base = CounterConfig(num_strips=1)
    median, _, _ = make_strip_profiles_diagnostic(gradient, "positive", base)
    aligned, _, diagnostics = make_strip_profiles_diagnostic(
        gradient, "positive", replace(base, strip_aggregation_mode="slope_aligned",
                                      strip_alignment_max_shift=10))
    assert aligned.max() > 2 * median.max()
    assert diagnostics[0]["max_observed_lag_range"] > 0


def test_enhanced_only_peak_cannot_be_observation():
    y = np.arange(240)
    enhanced = np.exp(-0.5 * ((y - 120) / 2) ** 2).astype(np.float32)[None, :]
    raw = np.zeros_like(enhanced)
    sources = np.ones_like(enhanced)
    cfg = replace(CounterConfig(num_strips=1), multiscale_gradient_enabled=True,
                  min_absolute_track_support=1)
    candidates, _ = detect_strip_peaks(enhanced, 20, cfg, raw_profiles=raw,
                                       scale_sources=sources)
    assert candidates[0]
    assert candidates[0][0].reason == "enhanced_only_proposal"
    assert candidates[0][0].raw_support == 0


def test_multiscale_only_does_not_inherit_directional_sigma_x():
    image = np.zeros((31, 31), dtype=np.float32)
    image[15:, 10:21] = 1
    first = replace(CounterConfig(), multiscale_gradient_enabled=True,
                    gradient_scales=[0.0, 1.0], directional_sigma_x=1.0)
    second = replace(first, directional_sigma_x=7.0)
    levels_a, _ = signed_gradient_levels(image, first)
    levels_b, _ = signed_gradient_levels(image, second)
    np.testing.assert_array_equal(levels_a[1][1], levels_b[1][1])
