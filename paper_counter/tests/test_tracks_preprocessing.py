import numpy as np
from types import SimpleNamespace

from paper_counter import CounterConfig, PaperCounter, PaperDetector
from paper_counter.preprocessing import preprocess_tracks_signal
from paper_counter.paper_geometry import prepare_geometry
from paper_counter.track_postprocessing import apply_horizontal_gap_postprocessing


def _synthetic_stack(count=16, period=20, width=96):
    y = np.arange((count + 4) * period)
    signal = np.full(len(y), 70.0)
    for center in np.arange(count) * period + 2 * period:
        signal += 120 * np.exp(-0.5 * ((y - center) / 2.0) ** 2)
    return np.clip(signal[:, None] * np.linspace(.8, 1.1, width)[None, :], 0, 255).astype(np.uint8)


def test_tracks_preprocessor_keeps_raw_evidence_separate():
    image = _synthetic_stack()
    cfg = CounterConfig(tracks_preprocessing_enabled=True,
                        tracks_gap_enhancement_enabled=True)
    prepared, debug = preprocess_tracks_signal(image, cfg)
    assert prepared.dtype == np.float32
    assert prepared.shape == image.shape
    assert debug["raw_evidence_source"] == "original_gray"
    assert debug["gap_enhancement"]["enabled"]
    assert np.array_equal(image, image.copy())


def test_horizontal_tracks_variant_preserves_known_count_and_reports_stages():
    result = PaperDetector(preset="tracks_preprocessed").detect(_synthetic_stack())
    assert result.count == 16
    assert result.raw_result.debug_info["preprocessing"]["enabled"]
    assert result.raw_result.debug_info["horizontal_gap_postprocessing"]["enabled"]


def test_preprocessing_disabled_is_bitwise_compatible_at_frontend():
    image = _synthetic_stack(count=8)
    cfg = CounterConfig()
    prepared, debug = preprocess_tracks_signal(image, cfg)
    assert prepared is image
    assert debug["enabled"] is False
    result = PaperCounter(cfg).process(image)
    assert result.count == 8


def test_tracks_geometry_preserves_full_extent_when_requested():
    image = _synthetic_stack(count=8)
    prepared, info = prepare_geometry(image, preserve_extent=True)
    assert prepared.shape[:2] == image.shape
    assert info["preserve_extent"] is True
    assert info["crop_xyxy"] == [0, 0, image.shape[1], image.shape[0]]


def test_horizontal_postprocessing_only_removes_a_shared_near_duplicate():
    cfg = CounterConfig(tracks_postprocessing_enabled=True, tracks_post_min_gap_ratio=.3)
    def track(track_id, y, score):
        return SimpleNamespace(track_id=track_id, center_y=y,
            strip_positions={0: y, 1: y + .2, 2: y - .1}, total_score=score,
            support_ratio=.9, smoothness_score=.9, raw_evidence_ratio=.9,
            reason=None, sequence_decision=None)
    kept, removed, debug = apply_horizontal_gap_postprocessing(
        [track("a", 100, .9), track("b", 103, .5)], 20, cfg)
    assert [t.track_id for t in kept] == ["a"]
    assert [t.track_id for t in removed] == ["b"]
    assert debug["decisions"][0]["action"] == "suppress_near_duplicate"
