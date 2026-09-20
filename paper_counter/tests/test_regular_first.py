"""Regular-first hybrid unit coverage (evidence remains mandatory)."""
import json
import unittest
from dataclasses import replace

import numpy as np

from paper_counter import CounterConfig
from paper_counter.ground_truth import evaluate_result
from paper_counter.models import PaperCountResult, PeakCandidate
from paper_counter.raw_mask import StackState, detect_raw_stack_mask, refine_stack_mask_with_strong_tracks
from paper_counter.regularity import estimate_local_regularity, evaluate_middle_track_competition
from paper_counter.track_geometry import geometry_pair_score
from paper_counter.vertical_sequence import optimize_vertical_sequence, transition_score
from paper_counter.geometry_recovery import geometry_anomalies
from paper_counter.tracking import classify_tracks

from test_vertical_sequence import track, prior


class RegularFirstTests(unittest.TestCase):
    def setUp(self):
        self.cfg = CounterConfig()

    def regular_tracks(self, n=10, spacing=25):
        return [track(100 + i * spacing, .82) for i in range(n)]

    def test_standard_regular_stack_scores_high(self):
        m = estimate_local_regularity(self.regular_tracks(), None, 25, self.cfg)
        self.assertGreater(np.median(m.final_regularity), .70)
        self.assertEqual(m.region_ratios()["REGULAR"], 1.0)

    def test_robust_spacing_rejects_texture_and_missing_gap(self):
        ys = [100, 125, 151, 164, 189, 241, 265, 291]
        m = estimate_local_regularity([track(y, .82) for y in ys], None, 25, self.cfg)
        self.assertLess(abs(float(np.median(m.local_spacing)) - 25), 4)

    def test_local_acf_component_is_bounded(self):
        tracks = self.regular_tracks(12, 25)
        profile = np.cos(2 * np.pi * np.arange(500) / 25)
        m = estimate_local_regularity(tracks, profile, 25, self.cfg)
        self.assertTrue(np.all((m.acf_score >= 0) & (m.acf_score <= 1)))
        self.assertGreater(np.median(m.acf_score), .4)

    def test_irregular_tracks_are_not_regular(self):
        ys = [100, 120, 163, 187, 238, 266, 303]
        m = estimate_local_regularity([track(y, .7) for y in ys], None, 25, self.cfg)
        self.assertLess(np.median(m.final_regularity), .70)

    def test_mixed_transition_is_continuous(self):
        tracks = [track(100 + i * (25 if i < 5 else 33), .82) for i in range(10)]
        m = estimate_local_regularity(tracks, None, 25, self.cfg)
        self.assertLess(np.max(np.abs(np.diff(m.final_regularity))), .8)

    def test_adaptive_large_gap_penalty_grows_with_regularity(self):
        a, b = track(100), track(155)
        regular = estimate_local_regularity(self.regular_tracks(), None, 25, self.cfg)
        low = replace(regular, final_regularity=np.zeros_like(regular.final_regularity))
        e_hi = transition_score(a, b, prior(25), self.cfg, regular)
        e_lo = transition_score(a, b, prior(25), self.cfg, low)
        self.assertLessEqual(e_hi["pair_score"], e_lo["pair_score"] + 1e-6)

    def test_short_gap_penalty_remains_strong_in_irregular_zone(self):
        a = track(100)
        edge = transition_score(a, track(110), prior(25), self.cfg)
        self.assertGreater(edge["short_gap_penalty"], edge["long_gap_penalty"])

    def test_secondary_texture_competition(self):
        row = evaluate_middle_track_competition(track(100), track(114, .9), track(150), 50, self.cfg, .9)
        self.assertTrue(row["secondary_texture_likely"])

    def test_true_close_layer_can_survive_geometry(self):
        self.assertFalse(geometry_pair_score(track(100), track(118, .95), prior(50), self.cfg)["hard_reject"])

    def test_empty_regularity_is_safe(self):
        m = estimate_local_regularity([], None, 25, self.cfg)
        self.assertEqual(m.at(100), .5)
        json.dumps(m.to_dict())

    def test_internal_black_gap_is_not_invalid(self):
        image = np.full((300, 64), 150, np.uint8)
        image[145:155] = 0
        mask = detect_raw_stack_mask(image, self.cfg, 25)
        self.assertNotEqual(mask.state_at(150), StackState.INVALID)

    def test_edge_black_run_is_invalid(self):
        image = np.full((300, 64), 150, np.uint8)
        image[:40] = 0; image[-40:] = 0
        mask = detect_raw_stack_mask(image, self.cfg, 25)
        self.assertEqual(mask.state_at(10), StackState.INVALID)
        self.assertEqual(mask.state_at(290), StackState.INVALID)

    def test_strong_track_can_refine_mask_once(self):
        image = np.full((300, 64), 150, np.uint8); image[:80] = 0
        mask = detect_raw_stack_mask(image, self.cfg, 25)
        t = track(75, .95); t.raw_evidence_ratio = 1.0; t.raw_mask_state = "MARGIN"
        strong, _, _ = classify_tracks([t], self.cfg)
        info = refine_stack_mask_with_strong_tracks(mask, strong, self.cfg, 25)
        self.assertIn("strong_track_above_initial_boundary", info["mask_expansion_reason"])

    def test_marginal_track_does_not_refine_mask(self):
        image = np.full((300, 64), 150, np.uint8); image[:80] = 0
        mask = detect_raw_stack_mask(image, self.cfg, 25)
        t = track(75, .53); t.candidate_class = "marginal"; t.raw_evidence_ratio = 1.0
        info = refine_stack_mask_with_strong_tracks(mask, [t], self.cfg, 25)
        self.assertEqual(info["supporting_track_ids"], [])

    def test_regular_large_gap_is_searchable(self):
        class R:
            def at(self, y): return .9
        rows = geometry_anomalies([track(100), track(155), track(180)], prior(25), self.cfg, R())
        self.assertTrue(any(x.get("search") for x in rows))

    def test_irregular_large_gap_is_review_only(self):
        class R:
            def at(self, y): return .1
        rows = geometry_anomalies([track(100), track(155), track(180)], prior(25), self.cfg, R())
        self.assertFalse(any(x.get("search") for x in rows))

    def test_no_evidence_does_not_create_regular_paper(self):
        strong, marginal, _ = classify_tracks([track(100), track(150)], self.cfg)
        result = optimize_vertical_sequence(strong + marginal, prior(25), self.cfg)
        self.assertEqual(len(result.selected), 2)

    def test_curve_geometry_is_preserved_for_sequence(self):
        a = track(100, ys=100 + np.arange(12) * .5)
        b = track(150, ys=150 + np.arange(12) * .5)
        self.assertIsNotNone(transition_score(a, b, prior(50), self.cfg))
        self.assertTrue(a.curve_geometry or b.curve_geometry)

    def test_ground_truth_count_metrics(self):
        result = PaperCountResult(3, [100, 125, 150], [], [], [], 25, "negative", .8, {})
        metrics = evaluate_result(result, {"sample.png": {"count": 3, "edge_y": [100, 125, 150]}}, "sample.png")
        self.assertEqual(metrics["count_error"], 0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_ground_truth_missing_is_optional(self):
        result = PaperCountResult(0, [], [], [], [], None, "negative", 0, {})
        self.assertFalse(evaluate_result(result, {}, "x.png")["available"])

    def test_margin_defaults_are_enabled_but_evidence_gated(self):
        self.assertTrue(self.cfg.margin_allow_marginal)
        self.assertTrue(self.cfg.margin_allow_weak_recovery)
        self.assertGreater(self.cfg.margin_min_support_ratio, .5)

    def test_regular_map_serializes_thresholds_and_regions(self):
        m = estimate_local_regularity(self.regular_tracks(), None, 25, self.cfg)
        data = m.to_dict()
        self.assertIn("regular_regions", data)
        self.assertIn("irregular_threshold", data)

    def test_sequence_accepts_regularity_argument(self):
        m = estimate_local_regularity(self.regular_tracks(), None, 25, self.cfg)
        strong, marginal, _ = classify_tracks(self.regular_tracks(), self.cfg)
        result = optimize_vertical_sequence(strong + marginal, prior(25), self.cfg, m)
        self.assertGreaterEqual(len(result.selected), 8)

    def test_period_multiple_mode_remains_available(self):
        cfg = replace(self.cfg, sequence_period_multiple_mode=True)
        strong, marginal, _ = classify_tracks([track(100), track(125), track(150)], cfg)
        result = optimize_vertical_sequence(strong + marginal, prior(25), cfg)
        self.assertEqual(len(result.selected), 3)


if __name__ == "__main__":
    unittest.main()
