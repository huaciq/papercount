"""Sequence decisions use observed tracks, never manufactured periodic nodes."""
import unittest
from dataclasses import replace
import numpy as np
from paper_counter.config import CounterConfig
from paper_counter.models import StripPeakCandidate
from paper_counter.tracking import make_track, classify_tracks, validate_tracks
from paper_counter.local_period import LocalPeriod
from paper_counter.vertical_sequence import optimize_vertical_sequence, unary_score, transition_score
from paper_counter.sequence_recovery import (adaptive_search_radius, spacing_anomalies,
    recovery_hypotheses, recover_sequence_candidates)
from paper_counter.tracking import deduplicate_obvious_tracks


def track(y, score=0.8, support=12, ys=None):
    cfg = CounterConfig()
    peaks = [StripPeakCandidate(i, float(y if ys is None else ys[i]), 5, 5, 2, 1, "strong")
             for i in range(support)]
    t = make_track(peaks, 52, 12, cfg)
    t.total_score = score
    return t


def prior(d=52, mad=0):
    return LocalPeriod(np.array([0., 10000.]), np.array([d, d]), np.array([mad, mad]), d)


def solve(tracks, d=52, cfg=None):
    cfg = cfg or CounterConfig()
    strong, marginal, _ = classify_tracks(tracks, cfg)
    return optimize_vertical_sequence(strong + marginal, prior(d), cfg)


class VerticalSequenceTests(unittest.TestCase):
    def test_short_neighbor_suppressed(self):
        result = solve([track(100), track(116, .8), track(152)])
        self.assertEqual([t.center_y for t in result.selected], [100, 152])
        self.assertEqual(result.suppressed[0].reason, "sequence_suppressed_short_gap")

    def test_marginal_true_track_rescued(self):
        result = solve([track(100), track(151, .58), track(203)])
        self.assertEqual([t.center_y for t in result.selected], [100, 151, 203])
        self.assertEqual(result.rescued[0].sequence_decision, "rescued")

    def test_strong_does_not_mean_final(self):
        result = solve([track(100, .9), track(116, .8), track(152, .9)])
        self.assertEqual(result.suppressed[0].candidate_class, "strong")

    def test_marginal_unary_stays_below_strong(self):
        cfg = CounterConfig()
        marginal, strong = track(100, .599), track(152, .60)
        classify_tracks([marginal, strong], cfg)
        self.assertLess(unary_score(marginal, cfg), unary_score(strong, cfg))

    def test_endpoint_after_uncertain_missing_period_is_retained(self):
        result = solve([track(25.5, .82), track(52, .8), track(85, .8),
                        track(112, .83), track(157, .86)], d=27)
        self.assertIn(157, [t.center_y for t in result.selected])
        self.assertEqual(len(result.selected), 5)

    def test_dp_matches_exhaustive_subsets(self):
        from itertools import combinations
        from paper_counter.vertical_sequence import huber
        for weight in (0.0, .1):
            cfg = replace(CounterConfig(), sequence_spacing_change_weight=weight)
            tracks = [track(y, s) for y, s in ((100, .8), (116, .85), (151, .58),
                       (178, .50), (204, .82), (229, .62), (256, .85))]
            strong, marginal, _ = classify_tracks(tracks, cfg)
            best = float("-inf")
            for size in range(1, len(tracks) + 1):
                for subset in combinations(tracks, size):
                    if not any(t.candidate_class == "strong" for t in subset):
                        continue
                    edges = [transition_score(a, b, prior(), cfg) for a, b in zip(subset, subset[1:])]
                    if any(e is None for e in edges):
                        continue
                    value = sum(unary_score(t, cfg) for t in subset) + sum(e["pair_score"] for e in edges)
                    for a, b in zip(edges, edges[1:]):
                        delta = b["gap"] / b["period_multiple"] / b["local_period"] - a["gap"] / a["period_multiple"] / a["local_period"]
                        value -= weight * huber(delta / cfg.sequence_spacing_scale_ratio)
                    best = max(best, value)
            self.assertAlmostEqual(optimize_vertical_sequence(strong + marginal, prior(), cfg).total_sequence_score, best)

    def test_marginal_noise_not_all_retained(self):
        result = solve([track(100), track(112, .52), track(124, .50), track(152)])
        self.assertEqual(len(result.selected), 2)
        self.assertEqual(len(result.suppressed), 2)

    def test_low_total_is_not_low_gradient(self):
        t = track(100, .58)
        _, rejected = validate_tracks([t], CounterConfig())
        self.assertEqual(rejected[0].reason, "low_total_score")

    def test_no_strong_anchor_no_sequence(self):
        result = solve([track(100, .55), track(152, .55), track(204, .55)])
        self.assertEqual(result.selected, [])

    def test_missing_period_does_not_create_track(self):
        # Explicit comparison mode retains the historical n-period diagnostic.
        result = solve([track(100), track(204)], cfg=CounterConfig(sequence_period_multiple_mode=True))
        self.assertEqual(len(result.selected), 2)
        self.assertEqual(result.transitions[0]["period_multiple"], 2)

    def test_second_order_score_accounting(self):
        cfg = replace(CounterConfig(), sequence_spacing_change_weight=.1)
        result = solve([track(100), track(151, .58), track(203), track(256)], cfg=cfg)
        objective = sum(r["unary_score"] for r in result.decisions if r["decision"] != "suppressed")
        objective += sum(e["pair_score"] + e["spacing_change_score"] for e in result.transitions)
        self.assertAlmostEqual(result.total_sequence_score, objective)

    def test_invalid_configuration(self):
        for kwargs in ({"track_proposal_floor": .61}, {"recovery_max_search_ratio": .1},
                       {"sequence_spacing_scale_ratio": 0}, {"max_sequence_gap_periods": 0},
                       {"sequence_short_gap_penalty": float("nan")}):
            with self.assertRaises(ValueError):
                CounterConfig(**kwargs)

    def test_reuse_ten_strip_candidate_preserves_geometry(self):
        cfg = CounterConfig()
        left, middle, right = track(100), track(152, .58, 10), track(204)
        strong, marginal, _ = classify_tracks([left, middle, right], cfg)
        geometry = dict(middle.strip_positions)
        hypotheses = recovery_hypotheses(strong, prior(), cfg)
        added, _, reuse, weak = recover_sequence_candidates(hypotheses, [[] for _ in range(12)],
            strong, strong + marginal, 12, cfg)
        self.assertEqual(added, [])
        self.assertEqual(weak, [])
        self.assertEqual(reuse[0]["reason"], "recovery_reused_marginal_track")
        self.assertEqual(reuse[0]["candidates"][0]["support_count"], 10)
        self.assertEqual(middle.strip_positions, geometry)
        self.assertEqual(reuse[0]["candidates"][0]["strip_positions"], geometry)

    def test_no_pool_candidate_triggers_weak_search(self):
        cfg = CounterConfig()
        strong, _, _ = classify_tracks([track(100), track(204)], cfg)
        candidates = [[StripPeakCandidate(i, 152, .5, .5, 2, 1, "weak")] for i in range(12)]
        added, _, reuse, weak = recover_sequence_candidates(recovery_hypotheses(strong, prior(), cfg),
            candidates, strong, strong, 12, cfg)
        self.assertEqual(reuse[0]["reason"], "recovery_no_existing_candidate")
        self.assertEqual(len(weak), 1)
        self.assertEqual(len(added), 1)
        result = optimize_vertical_sequence(strong + added, prior(), cfg)
        self.assertEqual(len(result.selected), 3)

    def test_noninteger_anomaly_does_not_invent_evidence(self):
        cfg = CounterConfig()
        # Include measured context on both sides of the anomalous 41px gap.
        strong, _, _ = classify_tracks([track(74), track(100), track(141), track(167)], cfg)
        self.assertEqual(spacing_anomalies(strong, prior(26), cfg)[0]["gap"], 41)
        added, _, _, weak = recover_sequence_candidates(recovery_hypotheses(strong, prior(26), cfg),
            [[] for _ in range(12)], strong, strong, 12, cfg)
        self.assertEqual(added, [])
        self.assertEqual(weak[0]["reason"], "recovery_no_image_evidence")
        self.assertEqual(len(optimize_vertical_sequence(strong, prior(26), cfg).selected), 4)

    def test_curved_weak_recovery_follows_per_strip_prediction(self):
        cfg = CounterConfig()
        curve = np.array([0, 2, 5, 8, 11, 14, 14, 11, 8, 5, 2, 0])
        strong, _, _ = classify_tracks([track(100, ys=100 + curve), track(204, ys=204 + curve)], cfg)
        candidates = [[StripPeakCandidate(i, float(152 + curve[i]), .5, .5, 2, 1, "weak")] for i in range(12)]
        hypotheses = recovery_hypotheses(strong, prior(), cfg)
        self.assertGreater(np.ptp(list(hypotheses[0]["predicted_strip_positions"].values())), 0)
        added, _, _, _ = recover_sequence_candidates(hypotheses, candidates, strong, strong, 12, cfg)
        self.assertEqual(added[0].support_count, 12)
        self.assertEqual(added[0].strip_positions, {i: float(152 + curve[i]) for i in range(12)})

    def test_adaptive_radius_grows_and_is_bounded(self):
        cfg = CounterConfig()
        self.assertGreater(adaptive_search_radius(52, 5, 0, cfg), adaptive_search_radius(52, 0, 0, cfg))
        self.assertGreater(adaptive_search_radius(52, 0, 6, cfg), adaptive_search_radius(52, 0, 0, cfg))
        self.assertEqual(adaptive_search_radius(52, 100, 100, cfg), cfg.recovery_max_search_ratio * 52)

    def test_near_independent_track_survives_early_duplicate_cleanup(self):
        kept, rejected = deduplicate_obvious_tracks([track(100), track(116)])
        self.assertEqual(len(kept), 2)
        kept, rejected = deduplicate_obvious_tracks([track(100), track(100.5)])
        self.assertEqual(len(kept), 1)

    def test_positive_only_cannot_create_sequence_recovery(self):
        cfg = CounterConfig()
        strong, _, _ = classify_tracks([track(100), track(204)], cfg)
        auxiliary = [[StripPeakCandidate(i, 152, 8, 8, 2, 1, "strong")] for i in range(12)]
        added, _, _, _ = recover_sequence_candidates(recovery_hypotheses(strong, prior(), cfg),
            [[] for _ in range(12)], strong, strong, 12, cfg, auxiliary)
        self.assertEqual(added, [])

    def test_search_window_miss_has_distinct_reason(self):
        # Historical narrow-window behavior remains available for comparison.
        cfg = CounterConfig(sequence_period_multiple_mode=True)
        strong, _, _ = classify_tracks([track(100), track(204)], cfg)
        outside = [[StripPeakCandidate(i, 115, .5, .5, 2, 1, "weak")] for i in range(12)]
        added, _, _, weak = recover_sequence_candidates(recovery_hypotheses(strong, prior(), cfg),
            outside, strong, strong, 12, cfg)
        self.assertEqual(added, [])
        self.assertEqual(weak[0]["reason"], "recovery_search_window_miss")


if __name__ == "__main__":
    unittest.main()
