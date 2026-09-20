"""Deterministic signal and image regression tests; no learned models."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from paper_counter import CounterConfig, PaperCounter
from paper_counter.models import PeakCandidate
from paper_counter.peaks import detect_candidates, remove_duplicates, strip_peak_indices
from paper_counter.period import estimate_period
from paper_counter.repair import repair_missing_peaks
from paper_counter.visualization import save_debug


def synthetic_papers(count=30, period=20, width=96):
    """Known bright paper ridges under a horizontal lighting gradient."""
    centers = np.arange(count) * period + 2 * period
    y = np.arange((count + 4) * period)
    signal = np.full(len(y), 60.0)
    for center in centers:
        signal += 130 * np.exp(-0.5 * ((y - center) / 2.0) ** 2)
    image = signal[:, None] * np.linspace(0.75, 1.1, width)[None, :]
    return np.clip(image, 0, 255).astype(np.uint8), centers


class CounterTests(unittest.TestCase):
    def test_full_pipeline_counts_one_edge_per_ridge_for_both_signs(self):
        image, centers = synthetic_papers()
        for sign in ("positive", "negative", "auto"):
            result = PaperCounter(CounterConfig(gradient_polarity=sign)).process(image)
            self.assertEqual(result.count, len(centers), sign)
            self.assertAlmostEqual(result.period, 20, delta=1)
            self.assertTrue(all(abs(y - center) <= 4 for y, center in zip(result.peak_positions, centers)))
            self.assertTrue(0 <= result.confidence <= 1)

    def test_large_line_scan_native_coordinates(self):
        image, centers = synthetic_papers(count=300, period=25)
        result = PaperCounter().process(image)
        self.assertEqual(result.count, 300)
        self.assertGreater(result.peak_positions[-1], 7500)
        self.assertLessEqual(abs(result.peak_positions[-1] - centers[-1]), 4)

    def test_blank_image_returns_no_validated_edges(self):
        for intensity in (0, 1, 255):
            result = PaperCounter().process(np.full((500, 96), intensity, np.uint8))
            self.assertEqual(result.count, 0)
            self.assertIsNone(result.period)
            self.assertEqual(result.confidence, 0)
            json.dumps(result.debug_info, allow_nan=False)

    def test_noise_is_not_high_confidence(self):
        image = np.random.default_rng(19).integers(0, 256, (2000, 96), dtype=np.uint8)
        result = PaperCounter().process(image)
        self.assertLess(result.confidence, 0.4)

    def test_two_corrupt_strips_do_not_break_majority(self):
        image, centers = synthetic_papers()
        image[:, 6:27] = np.random.default_rng(3).integers(0, 256, (len(image), 21), dtype=np.uint8)
        result = PaperCounter().process(image)
        self.assertEqual(result.count, len(centers))

    def test_roi_coordinates_are_original(self):
        image, _ = synthetic_papers(count=10)
        padded = np.pad(image, ((70, 20), (10, 10)), constant_values=60)
        roi = {"x1": 10, "x2": 106, "y1": 70, "y2": 70 + len(image)}
        result = PaperCounter(CounterConfig(roi=roi)).process(padded)
        reference = PaperCounter().process(image)
        self.assertEqual(result.peak_positions, [y + 70 for y in reference.peak_positions])

    def test_roi_bounds_and_narrow_image_are_rejected(self):
        image = np.zeros((50, 20), np.uint8)
        for roi in ({"x1": -1, "x2": 10, "y1": 0, "y2": 20},
                    {"x1": 0, "x2": 21, "y1": 0, "y2": 20}):
            with self.assertRaises(ValueError):
                PaperCounter(CounterConfig(roi=roi)).process(image)
        with self.assertRaises(ValueError):
            PaperCounter().process(np.zeros((100, 4), np.uint8))

    def test_period_bounds_and_harmonics(self):
        y = np.arange(1000)
        profile = np.cos(2 * np.pi * y / 20)
        self.assertEqual(estimate_period(profile, 5, 50)[0], 20)
        self.assertIsNone(estimate_period(profile, 5, 12)[0])
        self.assertIsNone(estimate_period(np.zeros(100), 5, 50)[0])

    def test_no_evidence_never_repairs(self):
        config = CounterConfig()
        anchors = [PeakCandidate(20, 5, 5, 2), PeakCandidate(60, 5, 5, 2)]
        repaired, attempts = repair_missing_peaks(anchors, [], [np.array([])] * 8, 20, config, set())
        self.assertEqual(repaired, [])
        self.assertEqual(attempts[0]["reason"], "no_image_evidence")

    def test_weak_real_peak_can_repair_but_single_strip_cannot(self):
        config = CounterConfig()
        anchors = [PeakCandidate(20, 5, 5, 2), PeakCandidate(60, 5, 5, 2)]
        weak = [PeakCandidate(41, 1, 1.5, 2, source="repaired")]
        for votes, expected in ((8, 1), (1, 0)):
            strips = [np.array([41]) if i < votes else np.array([]) for i in range(8)]
            repaired, _ = repair_missing_peaks(anchors, weak, strips, 20, config, set())
            self.assertEqual(len(repaired), expected)

    def test_adaptive_weak_threshold_repair(self):
        cfg = CounterConfig()
        y = np.arange(200)
        signal = sum(amplitude * np.exp(-0.5 * ((y - center) / 3) ** 2)
                     for center, amplitude in ((40, 5), (60, 5), (80, 0.8), (100, 5), (120, 5)))
        strong, threshold = detect_candidates(signal, 20, cfg)
        weak, _ = detect_candidates(signal, 20, cfg, weak=True)
        self.assertGreater(threshold, 0)
        self.assertTrue(len(weak) >= len(strong))
        self.assertTrue(all(p.width > 0 for p in weak))

    def test_duplicates_keep_stronger_candidate_and_record_reason(self):
        candidates = [PeakCandidate(20, 1, 1, 2, 0.5), PeakCandidate(24, 4, 4, 2, 1)]
        kept, removed = remove_duplicates(candidates, 20, CounterConfig())
        self.assertEqual([p.y for p in kept], [24])
        self.assertEqual(removed[0].reason, "duplicate")

    def test_uint16_and_bgra_supported(self):
        image, centers = synthetic_papers(count=10)
        for data in (image.astype(np.uint16) * 257,
                     np.dstack((image, image, image, np.full_like(image, 255)))):
            self.assertEqual(PaperCounter().process(data).count, len(centers))

    def test_debug_outputs_even_with_no_period(self):
        cfg = CounterConfig()
        result = PaperCounter(cfg).process(np.zeros((100, 96), np.uint8))
        with tempfile.TemporaryDirectory() as directory:
            save_debug(result, directory, cfg)
            for name in ("result_overlay.png", "profile.png", "autocorrelation.png", "strip_profiles.png",
                         "debug_info.json", "positive_gradient.png", "negative_gradient.png", "profiles.npz"):
                self.assertGreater((Path(directory) / name).stat().st_size, 0)
            with (Path(directory) / "debug_info.json").open(encoding="utf-8") as stream:
                self.assertEqual(json.load(stream)["count"], 0)


if __name__ == "__main__":
    unittest.main()
