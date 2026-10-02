"""Positive boundaries must use their own observations and paper-side context."""
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from paper_counter import CounterConfig, PaperCounter
from paper_counter.track_identity import extract_track_aligned_profiles, _positive_pair
from paper_counter.geometry_recovery import recover_geometry_candidates
from test_counter import synthetic_papers
from test_track_identity import paper_track


def test_positive_identity_profiles_match_reflected_negative_boundary():
    length = 301
    y = np.arange(length)
    raw = np.tile(40 + 160 / (1 + np.exp(-(y - 120) / 2)), (3, 1))
    positive = np.tile(np.exp(-.5 * ((y - 120) / 2) ** 2) * 10, (3, 1))
    negative = np.tile(np.exp(-.5 * ((y - 136) / 2) ** 2) * 9, (3, 1))
    pos_cfg = CounterConfig(gradient_polarity="positive", num_strips=3)
    neg_cfg = replace(pos_cfg, gradient_polarity="negative")
    pos_track = paper_track("pos", {0: 120, 1: 120, 2: 120})
    neg_track = paper_track("neg", {0: 180, 1: 180, 2: 180})
    pos = extract_track_aligned_profiles(pos_track, 30, raw, raw, negative, positive, 0, pos_cfg)
    neg = extract_track_aligned_profiles(neg_track, 30, raw[:, ::-1], raw[:, ::-1],
                                       positive[:, ::-1], negative[:, ::-1], 0, neg_cfg)
    for name in ("raw", "normalized", "negative", "positive", "negative_by_strip"):
        np.testing.assert_allclose(getattr(pos, name), getattr(neg, name))
    assert pos.negative.max() > 9
    pair_pos = _positive_pair(pos_track, 30, negative, 0, pos_cfg)
    pair_neg = _positive_pair(neg_track, 30, negative[:, ::-1], 0, neg_cfg)
    assert pair_pos == pair_neg
    assert pair_pos["support"] == 1
    assert pair_pos["distance"] > 0


def test_positive_pipeline_uses_positive_native_period_and_oriented_verifier():
    image, centers = synthetic_papers(count=20)
    cfg = CounterConfig(gradient_polarity="positive", multiscale_gradient_enabled=True,
                        final_track_verifier_shadow_mode=False)
    result = PaperCounter(cfg).process(image)
    assert result.count == len(centers)
    assert result.gradient_polarity == "positive"
    assert result.debug_info["counted_boundary"] == "paper_top"
    assert result.debug_info["initial_period_source"] == "native_gradient_frontend_lock"
    identity = result.debug_info["track_identity"]
    assert identity["primary_polarity"] == "positive"
    assert identity["physical_paper_side"] == "below"
    negative = PaperCounter(replace(cfg, gradient_polarity="negative")).process(image)
    assert all(p < n for p, n in zip(result.peak_positions, negative.peak_positions))


def test_positive_debug_exports_actual_and_canonical_channels(tmp_path):
    from paper_counter.visualization import save_debug
    image, _ = synthetic_papers(count=12)
    cfg = CounterConfig(gradient_polarity="positive", multiscale_gradient_enabled=True)
    result = PaperCounter(cfg).process(image)
    save_debug(result, tmp_path, cfg)
    assert (tmp_path / "positive_gradient.png").is_file()
    assert (tmp_path / "negative_gradient.png").is_file()
    assert (tmp_path / "track_identity_profiles.png").is_file()


def test_positive_recovery_is_not_blocked_by_legacy_negative_gate():
    cfg = CounterConfig(gradient_polarity="positive")
    hypothesis = {"search": True, "local_period": 20, "predicted_y": 120,
                  "left": 100, "right": 140, "left_strip_positions": {0: 100},
                  "right_strip_positions": {0: 140}, "predicted_strip_positions": {0: 120},
                  "search_radius_by_strip": {0: 5}}
    *_, attempts = recover_geometry_candidates([hypothesis], [[]], [], [], 1, cfg,
                                               negative_evidence=False, primary_evidence=True)
    assert attempts[0]["reason"] == "recovery_no_image_evidence"
    *_, blocked = recover_geometry_candidates([hypothesis], [[]], [], [], 1, cfg,
                                              negative_evidence=False, primary_evidence=False)
    assert blocked[0]["reason"] == "recovery_primary_evidence_required"


def test_blank_positive_multiscale_does_not_invent_papers():
    result = PaperCounter(CounterConfig(gradient_polarity="positive",
                          multiscale_gradient_enabled=True)).process(np.zeros((500, 96), np.uint8))
    assert result.count == 0


def test_auto_resolves_actual_polarity_for_identity_verifier():
    image, _ = synthetic_papers(count=12)
    from paper_counter import track_counter
    original = track_counter.analyze_track_identity
    with patch.object(track_counter, "analyze_track_identity", wraps=original) as analyze:
        result = PaperCounter(CounterConfig(gradient_polarity="auto")).process(image)
    assert analyze.call_args.args[-1].gradient_polarity == result.gradient_polarity
