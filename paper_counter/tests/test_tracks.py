"""Behavioral tests for curved tracks, brightness drift and evidence-only recovery."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np
from paper_counter import CounterConfig, PaperCounter
from paper_counter.models import StripPeakCandidate
from paper_counter.local_detection import detect_strip_peaks
from paper_counter.tracking import associate_strip_peaks, validate_tracks, make_track, deduplicate_tracks
from paper_counter.local_period import estimate_local_period, missing_edge_hypotheses
from paper_counter.track_recovery import recover_weak_tracks
from paper_counter.active_roi import detect_active_stack_roi


def peaks_at(positions, strength="strong", prominence=5.0):
    """Construct measured strip candidates, with optional missing observations."""
    return [[StripPeakCandidate(i, float(y), prominence, prominence, 2.0, 2.0, strength, 0.8)]
            if y is not None else [] for i, y in enumerate(positions)]


def horizontal_track(y, cfg):
    return make_track([s[0] for s in peaks_at([y] * cfg.num_strips)], 26, cfg.num_strips, cfg)


class TrackTests(unittest.TestCase):
    def setUp(self):
        self.cfg = CounterConfig(num_strips=8)

    def link(self, ys):
        return validate_tracks(associate_strip_peaks(peaks_at(ys), 26, self.cfg), self.cfg)

    def test_curved_example_is_one_track(self):
        accepted, _ = self.link([100, 101, 102, 104, 105, 104, 103, 101])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0].support_count, 8)
        self.assertEqual(accepted[0].center_y, 102.5)

    def test_two_missing_strips_allowed(self):
        accepted, _ = self.link([100, 101, 102, None, None, 104, 103, 101])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0].support_count, 6)
        self.assertEqual(accepted[0].longest_contiguous_run, 3)

    def test_random_large_jumps_rejected(self):
        accepted, rejected = self.link([100, 170, 61, 240, 31, 310, 5, 400])
        self.assertFalse(accepted)
        self.assertTrue(rejected)

    def test_smooth_nonzero_slope_preserved(self):
        accepted, _ = self.link([100, 103, 106, 108, 109, 108, 106, 103])
        self.assertEqual(len(accepted), 1)
        self.assertGreater(accepted[0].y_range, 5)

    def test_oscillating_path_fails_curvature(self):
        candidate = make_track([s[0] for s in peaks_at([100, 105, 100, 105, 100, 105, 100, 105])], 26, 8, self.cfg)
        accepted, rejected = validate_tracks([candidate], self.cfg)
        self.assertFalse(accepted)
        self.assertEqual(rejected[0].reason, "low_smoothness")

    def test_one_pixel_peak_localization_jitter_is_not_a_geometry_failure(self):
        candidate = make_track(
            [s[0] for s in peaks_at([100, 101, 100, 101, 100, 101, 100, 101])],
            26, 8, self.cfg,
        )
        accepted, rejected = validate_tracks([candidate], self.cfg)
        self.assertEqual(len(accepted), 1)
        self.assertFalse(rejected)
        self.assertGreater(candidate.smoothness_score, self.cfg.min_track_smoothness)

    def test_local_threshold_recalls_dim_half(self):
        # Smooth periodic contrast changes by 20x; a single whole-image MAD
        # suppresses the dim section while window statistics follow its scale.
        y = np.arange(4000)
        amplitude = np.where(y < 2800, 20.0, 1.0)
        signal = (amplitude * (1 + np.cos(2 * np.pi * (y - 13) / 26))).astype(np.float32)
        cfg = self.cfg
        peaks, stats = detect_strip_peaks(signal[None, :], 26, cfg)
        dim = [p for p in peaks[0] if 3100 < p.y < 3800 and p.strength_type == "strong"]
        from scipy.signal import find_peaks
        mad = np.median(np.abs(signal - np.median(signal)))
        global_peaks, _ = find_peaks(signal, prominence=cfg.strong_prominence_k * mad)
        self.assertGreater(len(dim), 20)
        self.assertLess(sum(3100 < p < 3800 for p in global_peaks), len(dim))
        self.assertGreater(stats[0]["strong_threshold"][1000], stats[0]["strong_threshold"][3500])

    def test_normal_spacing_has_no_missing_hypotheses(self):
        tracks = [horizontal_track(y, self.cfg) for y in [100, 126, 151, 178, 204]]
        local = estimate_local_period(tracks, 26, self.cfg)
        self.assertAlmostEqual(local.at(151), 26, delta=0.5)
        self.assertEqual(missing_edge_hypotheses(tracks, local, self.cfg), [])

    def test_weak_recovery_requires_real_track(self):
        tracks = [horizontal_track(y, self.cfg) for y in [100, 126, 178, 204]]
        local = estimate_local_period(tracks, 26, self.cfg)
        hypotheses = missing_edge_hypotheses(tracks, local, self.cfg)
        weak = peaks_at([152, 153, 154, 154, 154, 153, 152, 151], "weak", 1.2)
        recovered, _, attempts = recover_weak_tracks(hypotheses, weak, tracks, 8, self.cfg)
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0].source, "recovered")
        self.assertEqual(recovered[0].weak_support_count, 8)
        self.assertEqual(attempts[0]["reason"], "weak_multistrip_evidence")

    def test_single_strip_and_positive_only_cannot_recover(self):
        tracks = [horizontal_track(y, self.cfg) for y in [100, 126, 178, 204]]
        local = estimate_local_period(tracks, 26, self.cfg)
        hypotheses = missing_edge_hypotheses(tracks, local, self.cfg)
        for weak in (peaks_at([152] + [None] * 7, "weak", 1.2), [[] for _ in range(8)]):
            recovered, _, attempts = recover_weak_tracks(hypotheses, weak, tracks, 8, self.cfg,
                                                        peaks_at([152] * 8))
            self.assertFalse(recovered)
            self.assertEqual(attempts[0]["reason"], "no_recovery_evidence")

    def test_local_period_tracks_drift_and_ignores_double_gap(self):
        centers = np.r_[0, np.cumsum(np.linspace(28, 24, 80))]
        centers = np.delete(centers, 40)
        tracks = [horizontal_track(float(y), self.cfg) for y in centers]
        local = estimate_local_period(tracks, 26, self.cfg)
        self.assertAlmostEqual(local.at(float(centers[4])), 28, delta=0.5)
        self.assertAlmostEqual(local.at(float(centers[-5])), 24, delta=0.5)
        self.assertLess(np.max(local.values), 29)

    def test_active_roi_excludes_large_black_regions(self):
        image = np.zeros((2500, 96), np.uint8)
        y = np.arange(780)
        image[800:1580] = (100 + 60 * np.cos(2 * np.pi * y / 26))[:, None]
        top, bottom, info = detect_active_stack_roi(image, 26, self.cfg)
        self.assertGreater(top, 500)
        self.assertLess(bottom, 1800)
        self.assertLessEqual(top, 800)
        self.assertGreaterEqual(bottom, 1580)

    def test_raw_mask_does_not_bypass_active_roi(self):
        image = np.zeros((900, 96), np.uint8)
        image[250:750] = 128
        cfg = replace(self.cfg, raw_stack_mask_enabled=True, auto_stack_roi=True)
        with patch("paper_counter.track_counter.detect_active_stack_roi",
                   return_value=(250, 750, {"reason": "test_sustained_region"})) as detector:
            result = PaperCounter(cfg).process(image)
        detector.assert_called_once()
        self.assertEqual(result.debug_info["requested_roi"]["y1"], 0)
        self.assertEqual(result.debug_info["active_stack_roi"]["y1"], 250)
        self.assertEqual(result.debug_info["active_stack_roi"]["y2"], 750)
        self.assertEqual(result.debug_info["active_roi_diagnostics"]["reason"],
                         "test_sustained_region")

    def test_partial_first_sheet_is_eligible(self):
        y = np.arange(520)
        profile = sum(5 * np.exp(-0.5 * ((y - center) / 2) ** 2) for center in range(0, 520, 26)).astype(np.float32)
        candidates, _ = detect_strip_peaks(np.tile(profile, (8, 1)), 26, self.cfg)
        accepted, _ = validate_tracks(associate_strip_peaks(candidates, 26, self.cfg), self.cfg)
        self.assertTrue(any(t.center_y <= 1 for t in accepted))

    def test_weak_only_cannot_seed_normal_track(self):
        self.assertEqual(associate_strip_peaks(peaks_at([100] * 8, "weak", 1.2), 26, self.cfg), [])

    def test_coherent_tiny_peak_cannot_pass_on_geometry_alone(self):
        nodes = [s[0] for s in peaks_at([100] * 8, prominence=1.8)]
        for node in nodes:
            node.local_threshold = 1.0
            node.local_prominence_reference = 30.0
        track = make_track(nodes, 26, 8, self.cfg)
        kept, rejected = validate_tracks([track], self.cfg)
        self.assertFalse(kept)
        self.assertEqual(rejected[0].reason, "low_gradient_score")

    def test_duplicate_requires_shared_geometry(self):
        a, b = horizontal_track(100, self.cfg), horizontal_track(102, self.cfg)
        kept, removed = deduplicate_tracks([a, b], lambda y: 26, self.cfg)
        self.assertEqual(len(kept), 1)
        self.assertEqual(removed[0].reason, "duplicate_track")

    def test_config_validation(self):
        for kwargs in ({"num_strips": True}, {"strong_prominence_k": "2"},
                       {"weak_prominence_k": 3.0}, {"auto_stack_roi": "true"},
                       {"local_period_neighbors": 0}, {"min_track_score": 2},
                       {"track_shift_penalty": float("nan")}):
            with self.assertRaises(ValueError):
                CounterConfig(**kwargs)

    def test_8_12_16_strip_counts(self):
        for n in (8, 12, 16):
            cfg = replace(self.cfg, num_strips=n)
            paths = associate_strip_peaks(peaks_at([100 + np.sin(i / 3) for i in range(n)]), 26, cfg)
            self.assertEqual(len(validate_tracks(paths, cfg)[0]), 1)

    def test_full_image_curved_edges_and_roi_offset(self):
        # Each strip's edge position moves smoothly; image coordinates must survive cropping.
        h, w = 780, 120
        y = np.arange(h)[:, None]
        shift = (5 * np.sin(np.linspace(0, np.pi, w)))[None, :]
        image = np.full((h, w), 60.0)
        for center in np.arange(26, 754, 26):
            image += 120 * np.exp(-0.5 * ((y - center - shift) / 2) ** 2)
        image = np.clip(image, 0, 255).astype(np.uint8)
        padded = np.pad(image, ((100, 80), (10, 10)), constant_values=60)
        cfg = replace(self.cfg, roi={"x1": 10, "x2": 130, "y1": 100, "y2": 880})
        result = PaperCounter(cfg).process(padded)
        self.assertEqual(result.count, 28)
        self.assertTrue(all(t.y_range >= 3 for t in result.paper_edge_tracks))
        self.assertTrue(all(y >= 100 for t in result.paper_edge_tracks for y in t.strip_positions.values()))

    def test_strip_period_consensus_when_fusion_has_no_period(self):
        from unittest.mock import patch
        from test_counter import synthetic_papers
        from paper_counter.period import score_polarity
        def washed_out(profile, config):
            assessment = score_polarity(profile, config)
            assessment["period"] = None
            return assessment
        with patch("paper_counter.track_counter.score_polarity", side_effect=washed_out):
            result = PaperCounter(self.cfg).process(synthetic_papers(count=10)[0])
        self.assertEqual(result.debug_info["initial_period_source"], "strip_acf_consensus")
        self.assertEqual(result.count, 10)


if __name__ == "__main__":
    unittest.main()
