"""RAW permissions, curved ordered geometry and nonperiodic physical gaps."""
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from paper_counter import CounterConfig, PaperCounter
from paper_counter.raw_mask import (detect_raw_stack_mask, RawStackMask, StackState,
                                    filter_mask_peaks, track_mask_permission)
from paper_counter.models import StripPeakCandidate
from paper_counter.track_geometry import fit_track_curve, gap_profile, geometry_pair_score, estimate_local_spacing
from paper_counter.tracking import classify_tracks, validate_tracks, associate_strip_peaks
from paper_counter.vertical_sequence import optimize_vertical_sequence, transition_score
from paper_counter.sequence_recovery import recovery_hypotheses, recover_sequence_candidates
from test_vertical_sequence import track, prior, solve
from test_counter import synthetic_papers


class GeometrySequenceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = CounterConfig()

    def test_black_background_fake_gradient_has_no_permission(self):
        body, _ = synthetic_papers(count=12)
        image = np.pad(body, ((300, 300), (0, 0)), constant_values=0)
        mask = detect_raw_stack_mask(image, self.cfg)
        self.assertEqual(mask.state_at(50), StackState.INVALID)
        self.assertEqual(mask.state_at(len(image) - 50), StackState.INVALID)
        from paper_counter.local_detection import detect_strip_peaks
        def fake_after_normalization(*args, **kwargs):
            candidates, statistics = detect_strip_peaks(*args, **kwargs)
            for i, strip in enumerate(candidates):
                strip.append(StripPeakCandidate(i, 50., 100, 100, 2, 1, "strong"))
            return candidates, statistics
        with patch("paper_counter.track_counter.detect_strip_peaks", side_effect=fake_after_normalization):
            result = PaperCounter(self.cfg).process(image)
        self.assertGreater(result.debug_info["raw_mask_invalid_peaks_removed"], 0)
        self.assertEqual(result.debug_info["invalid_selected_track_count"], 0)
        self.assertFalse(any(t.center_y == 50 for t in result.paper_edge_tracks))

    def test_boundary_strong_raw_edge_allowed(self):
        image = np.zeros((400, 96), np.uint8)
        image[100:300] = 150
        mask = detect_raw_stack_mask(image, self.cfg, 30)
        t = track(299, .9)
        classify_tracks([t], self.cfg)
        self.assertEqual(mask.state_at(299), StackState.MARGIN)
        self.assertTrue(track_mask_permission(t, mask, self.cfg))
        self.assertEqual(len(solve([t], 30).selected), 1)

    def test_margin_marginal_not_rescued_by_sequence(self):
        t = track(130, .58)
        t.raw_mask_state = "MARGIN"
        result = solve([track(100), t, track(160)], 30)
        self.assertNotIn(t.track_id, [s.track_id for s in result.selected])
        self.assertEqual(t.reason, "raw_margin_marginal_forbidden")

    def test_large_gap_is_valid_adjacency(self):
        result = solve([track(y) for y in (100, 130, 185, 215)], 30)
        self.assertEqual(len(result.selected), 4)
        self.assertTrue(any(e["gap_median"] == 55 for e in result.transitions))
        self.assertTrue(all(e["period_multiple_interpretation"] == "disabled" for e in result.transitions))

    def test_real_marginal_in_large_gap_uses_evidence_not_bonus(self):
        cfg = replace(self.cfg, sequence_rescue_bonus=100.)
        result = solve([track(100), track(130), track(158, .58), track(185), track(215)], 30, cfg)
        self.assertEqual(len(result.selected), 5)
        row = next(r for r in result.decisions if r["center_y"] == 158)
        self.assertAlmostEqual(row["unary_score"], .58 - cfg.sequence_suppress_margin)

    def test_thick_layer_gap_not_decomposed(self):
        ys = np.r_[100, 100 + np.cumsum([24, 25, 23, 46, 26, 24])]
        result = solve([track(y) for y in ys], 25)
        self.assertEqual(len(result.selected), len(ys))
        self.assertEqual(sum(e["gap_median"] == 46 for e in result.transitions), 1)

    def test_parallel_curved_secondary_texture_suppressed(self):
        curve = np.array([0, 2, 5, 9, 7, 4, 7, 11, 14, 12, 9, 10])
        result = solve([track(100, .9, ys=100+curve), track(116, .8, ys=116+curve),
                        track(152, .9, ys=152+curve)], 52)
        self.assertEqual(len(result.selected), 2)
        self.assertEqual(result.suppressed[0].reason, "sequence_suppressed_short_gap")

    def test_two_true_close_layers_not_hard_rejected(self):
        result = solve([track(y, .95) for y in (100, 130, 151, 181)], 30)
        self.assertEqual(len(result.selected), 4)

    def assert_curve_links(self, ys):
        candidates = [[StripPeakCandidate(i, float(y), 5, 5, 2, 1, "strong")] for i, y in enumerate(ys)]
        kept, _ = validate_tracks(associate_strip_peaks(candidates, 26, self.cfg), self.cfg)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].support_count, 12)

    def test_stable_tilt_not_penalized_as_nonhorizontal(self):
        self.assert_curve_links(500 + np.arange(12) * 7)

    def test_smooth_curved_track_allowed(self):
        self.assert_curve_links([500, 502, 505, 508, 511, 513, 513, 511, 508, 505, 502, 500])

    def test_local_undulation_allowed(self):
        self.assert_curve_links([500, 502, 505, 509, 507, 504, 507, 511, 514, 512, 509, 510])

    def test_random_large_jumps_geometry_rejected(self):
        t = track(500, ys=[500,530,488,525,480,531,488,536,479,528,482,535])
        strong, marginal, garbage = classify_tracks([t], self.cfg)
        self.assertFalse(strong or marginal)
        self.assertEqual(garbage[0].reason, "hard_geometry_failure")

    def test_crossing_transition_forbidden(self):
        a = track(100, ys=np.arange(12)*4 + 100)
        b = track(120, ys=140 - np.arange(12)*3)
        # Order by observed medians, just as the production DAG does.
        a, b = sorted([a, b], key=lambda t: t.center_y)
        self.assertIsNone(transition_score(a, b, prior(30), self.cfg))
        self.assertTrue(gap_profile(a, b, self.cfg, 30)["crossing_hard_reject"])

    def test_nonuniform_gap_profile_allowed(self):
        a = track(100, support=8, ys=np.arange(8)+100)
        b = track(130, support=8, ys=np.arange(8)+100+np.array([22,24,28,35,33,29,25,23]))
        edge = transition_score(a, b, prior(28), self.cfg)
        self.assertIsNotNone(edge)
        self.assertEqual(edge["non_crossing_ratio"], 1.)
        self.assertGreater(edge["gap_max"], edge["gap_min"])

    def test_invalid_region_cannot_recover(self):
        strong, _, _ = classify_tracks([track(100), track(204)], self.cfg)
        mask = RawStackMask(np.zeros(300, np.uint8), np.zeros(300, bool), {}, {})
        weak = [[StripPeakCandidate(i, 152, 1., 1., 2, 1., "weak")] for i in range(12)]
        added, _, _, attempts = recover_sequence_candidates(recovery_hypotheses(strong, prior(), self.cfg),
            weak, strong, strong, 12, self.cfg, raw_mask=mask)
        self.assertFalse(added)
        self.assertEqual(attempts[0]["reason"], "recovery_raw_mask_forbidden")

    def test_two_point_one_scale_without_negative_evidence_no_paper(self):
        strong, _, _ = classify_tracks([track(100), track(163)], self.cfg)
        plans = recovery_hypotheses(strong, prior(30), self.cfg)
        self.assertEqual(len(plans), 1)
        self.assertNotIn("suspected_missing_count", plans[0])
        added, _, _, _ = recover_sequence_candidates(plans, [[] for _ in range(12)], strong, strong, 12, self.cfg)
        self.assertFalse(added)
        self.assertEqual(len(optimize_vertical_sequence(strong, prior(30), self.cfg).selected), 2)

    def test_real_weak_negative_curve_recovers(self):
        strong, _, _ = classify_tracks([track(100), track(163)], self.cfg)
        weak = [[StripPeakCandidate(i, 129 + np.sin(i), 1., 1., 2, 1., "weak")] for i in range(12)]
        added, _, _, _ = recover_sequence_candidates(recovery_hypotheses(strong, prior(30), self.cfg),
            weak, strong, strong, 12, self.cfg)
        self.assertEqual(len(added), 1)
        self.assertEqual(len(optimize_vertical_sequence(strong + added, prior(30), self.cfg).selected), 3)

    def test_asymmetric_spacing_penalty(self):
        a = track(100)
        short = geometry_pair_score(a, track(112), prior(30), self.cfg)
        long = geometry_pair_score(a, track(154), prior(30), self.cfg)
        self.assertGreater(short["short_gap_penalty"], 10 * long["long_gap_penalty"])

    def test_wrong_initial_spacing_does_not_decide_count(self):
        for d in (20, 30, 60):
            self.assertEqual(len(solve([track(y, .95) for y in (100,130,160,190,220)], d).selected), 5)
        tracks = [track(y) for y in (100,130,160,190,220)]
        self.assertEqual(estimate_local_spacing(tracks, 90, self.cfg).at(150), 30)

    def test_gap_profile_distinguishes_same_center_gap(self):
        ys = np.arange(12)*20 + 100
        a = track(100, ys=ys)
        texture = track(130, ys=ys + np.array([3,3,3,3,37,37,37,37,3,3,3,3]))
        real = track(130, ys=ys + (texture.center_y - a.center_y))
        self.assertEqual(texture.center_y, real.center_y)
        bad = geometry_pair_score(a, texture, prior(52), self.cfg)
        good = geometry_pair_score(a, real, prior(52), self.cfg)
        self.assertGreater(bad["small_gap_ratio"], good["small_gap_ratio"])
        self.assertGreater(good["pair_score"], bad["pair_score"] + 1)

    def test_curve_interpolation_does_not_extrapolate_or_invent_support(self):
        t = track(100, support=8)
        t.strip_positions.pop(3)
        before = dict(t.strip_positions)
        curve = fit_track_curve(t, self.cfg)
        self.assertTrue(curve.valid[3])
        self.assertFalse(curve.observed[3])
        self.assertFalse(curve.valid[11])
        self.assertEqual(t.strip_positions, before)

    def test_raw_mask_uint16_scale_invariance(self):
        image = np.zeros((400, 60), np.uint8)
        image[100:300] = 120
        a = detect_raw_stack_mask(image, self.cfg)
        b = detect_raw_stack_mask(image.astype(np.uint16)*200, self.cfg)
        np.testing.assert_array_equal(a.states, b.states)


if __name__ == "__main__":
    unittest.main()
