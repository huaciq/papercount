"""Independent mathematical identities and behavior for the paper 1D Shadow."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.integrate import quad
from paper_counter.v3_shadow.config import ShadowConfig
from paper_counter.v3_shadow.gaussian import gaussian, gaussian_second, sampled_kernel as gkernel, response
from paper_counter.v3_shadow.bigaussian import bi_gaussian, bi_first, bi_second, bi_ridge_filter, sampled_kernel as bkernel
from paper_counter.v3_shadow.ridge_valley import multiscale, valley_likelihood, descriptor
from paper_counter.v3_shadow.identify import peaks
from paper_counter.v3_shadow.verification import verify, ridge_regions
from paper_counter.v3_shadow.frontend import analyze, measure_widths
from paper_counter.v3_shadow.synthetic import suite, evidence_metrics


@pytest.mark.parametrize('sigma', [1., 2.5, 4., 11.])
def test_rho_one_collapses_to_gaussian(sigma):
    x = np.linspace(-30, 30, 6001)
    np.testing.assert_allclose(bi_gaussian(x, sigma, 1), gaussian(x, sigma), atol=1e-15)
    np.testing.assert_allclose(bi_second(x, sigma, 1), gaussian_second(x, sigma), atol=1e-15)
    np.testing.assert_allclose(bkernel(sigma, 1)[1], gkernel(sigma)[1], atol=1e-15)


@pytest.mark.parametrize('sign', [-1, 1])
def test_piecewise_join_is_c2(sign):
    sigma, delta = 4., 1e-7
    x = np.array([sign * (sigma - delta), sign * (sigma + delta)])
    assert abs(np.diff(bi_gaussian(x, sigma))[0]) < 1e-8
    assert abs(np.diff(bi_first(x, sigma))[0]) < 1e-8
    assert abs(np.diff(bi_second(x, sigma))[0]) < 1e-7


def test_analytic_derivatives_match_numeric_function_not_dog():
    # Differentiate the piecewise function numerically, away from its joins.
    h = 1e-4
    for x in (-9., -5., -2., 0., 2., 5., 9.):
        derivative = (bi_gaussian(x + h, 4) - 2 * bi_gaussian(x, 4) + bi_gaussian(x - h, 4)) / h**2
        assert float(bi_second(x, 4)) == pytest.approx(float(derivative), abs=2e-8)
    assert float(bi_gaussian(0, 4)) != pytest.approx(float(gaussian(0, 4)))


@pytest.mark.parametrize('sigma', [.5, 1., 4., 7.3])
def test_exact_bin_integrals_and_no_unit_amplitude_normalization(sigma):
    x, weights, correction = bkernel(sigma)
    center = len(weights) // 2
    for j in (center - 1, center, center + 1):
        numerical = quad(lambda z: float(bi_ridge_filter(z, sigma)), x[j] - .5, x[j] + .5,
                         points=[p for p in (-sigma, sigma) if x[j] - .5 < p < x[j] + .5], epsabs=1e-9)[0]
        assert weights[j] + (correction if j == center else 0) == pytest.approx(numerical, abs=1e-8)
    assert abs(weights.sum()) < 1e-12
    np.testing.assert_allclose(weights, weights[::-1], atol=1e-14)
    np.testing.assert_allclose(response(np.ones(200), weights), 0., atol=1e-12)


def test_scale_normalization_continuous_invariance():
    x = np.linspace(-10, 10, 2001)
    # gamma=1 gives h(a*x,a*sigma)=h(x,sigma)/a.
    np.testing.assert_allclose(3 * bi_ridge_filter(3 * x, 12), bi_ridge_filter(x, 4), atol=1e-14)


def test_independent_valley_sign_and_exact_unclipped_descriptor():
    y = np.arange(201) - 100
    dark = 100 - 70 * np.exp(-.5 * (y / 2)**2)
    fv, vk = valley_likelihood(dark, 2)
    assert fv[100] > 0 and fv[90] < 0
    fb = multiscale(dark, [4, 5], kind='bi')['maximum']
    np.testing.assert_allclose(descriptor(fb, fv, .8), fb - .8 * fv)
    assert not np.allclose(fv, -fb)
    assert vk['sigma'] == 2


def test_best_scale_matches_independent_pointwise_maximum():
    case = suite(1)[0]
    m = multiscale(case['signal'], case['scales'])
    np.testing.assert_allclose(m['maximum'], m['levels'].max(axis=0))
    np.testing.assert_array_equal(m['best_sigma'], m['sigmas'][np.argmax(m['levels'], axis=0)])


def test_literal_peak_jump_and_raw_reference_gate():
    f = np.array([0., 2, 0, 3, 0, 2, 0, 1, 0])
    raw = np.array([0., 5, 0, 20, 0, 20, 0, 20, 0])
    accepted, rejected = peaks(f, raw, 1, 10)
    assert [c['y'] for c in accepted] == [3, 5, 7]
    assert [c['y'] for c in rejected] == [1]


def test_region_single_identity_and_energy_rejection():
    f = np.array([-1., 2, 1, 3, -1, .001, -1, 2, -1])
    candidates = [dict(y=y, strength=f[y]) for y in (1, 3, 5, 7)]
    accepted, rejected, diagnostic = verify(f, candidates, .1, .1)
    assert [c['y'] for c in accepted] == [3, 7]
    assert {c['reason'] for c in rejected} == {'duplicate_same_positive_region', 'low_area_and_peak'}
    assert diagnostic['thresholds_published'] is False
    assert len(ridge_regions(f)) == 3


def test_measured_width_is_not_center_spacing():
    a = np.full(240, 20.)
    for center in (30, 70, 110, 150, 190):
        a[center - 4:center + 4] = 200
    m, d = measure_widths(a, ShadowConfig())
    assert max(m[0]) == pytest.approx(4.)
    assert d['mean_gap_width'] == pytest.approx(32.)
    assert not np.isclose(m[0][0], 40 / 2)


def test_shift_invariance_no_image_or_gt_special_cases():
    case = suite(1)[0]
    cfg = ShadowConfig()
    kwargs = dict(config=cfg, measured_scales=(case['scales'], case['sigma_v']), reference_values=[20.])
    original = analyze(case['signal'], **kwargs)
    shifted = analyze(np.r_[np.full(21, 20.), case['signal']], **kwargs)
    assert [c['y'] + 21 for c in original['candidates']] == [c['y'] for c in shifted['candidates']]


def test_flat_data_insufficient_and_endpoint_support_not_fabricated():
    out = analyze(np.full(200, 20.))
    assert out['status'] == 'insufficient_evidence' and not out['candidates']
    case = next(c for c in suite(1) if c['kind'] == 'endpoint')
    out = analyze(case['signal'], measured_scales=(case['scales'], case['sigma_v']))
    assert out['review_required']
    assert any(not c['complete_support'] for c in out['candidates'])
    assert all(0 <= c['y'] < len(case['signal']) for c in out['candidates'])


def test_negative_signal_never_auto_inserts_paper_from_gap():
    out = analyze(np.zeros(400), measured_scales=([4., 5.], 2.))
    assert not out['candidates']
    with pytest.raises(ValueError):
        ShadowConfig(stage='A6')


def test_synthetic_metrics_separate_duplicate_fp_and_weak_recall():
    m = evidence_metrics([dict(y=y) for y in (4, 5, 14, 22)],
                         [dict(left=2, right=7, weak=False), dict(left=12, right=17, weak=True)], [[20, 25]])
    assert (m['tp'], m['fp'], m['duplicate_candidates'], m['texture_fp']) == (2, 2, 1, 1)
    assert m['weak_sheet_recall'] == 1


@pytest.mark.parametrize('bad', [np.array([1, np.nan, 2]), np.array([1, 2]), np.ones((4, 4))])
def test_invalid_observations_rejected(bad):
    with pytest.raises(ValueError):
        analyze(bad)
