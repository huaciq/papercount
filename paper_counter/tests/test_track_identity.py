"""Coordinate and partial annotation regression tests for the final verifier."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from paper_counter import CounterConfig
from paper_counter.coordinates import (
    OverlayTransform, detection_to_original, original_to_detection,
    reconstruct_render_metadata,
)
from paper_counter.track_annotations import (
    load_track_annotations, match_false_track_annotations, resolve_annotation_image,
)
from paper_counter.models import PaperEdgeTrack
from paper_counter.track_identity import (
    AlignedTrackProfiles, apply_final_track_verifier, background_false_evidence,
    build_edge_template, conservative_identity_decision, extract_track_aligned_profiles,
)


def track(track_id: str, y: float, score: float = .8) -> dict:
    return {"track_id": track_id, "center_y": y, "strip_positions": {0: y - 1, 1: y + 1},
            "total_score": score, "source": "detected"}


class CoordinateTests(unittest.TestCase):
    def test_overlay_title_bar_coordinate_mapping(self):
        transform = OverlayTransform(116, .8, .8, 0, 20, 0, 116)
        overlay = transform.detection_to_overlay(100, 220)
        self.assertEqual(overlay, (80, 276))
        self.assertEqual(transform.overlay_to_detection(*overlay), (100, 220))

    def test_overlay_scale_not_one(self):
        transform = OverlayTransform(50, .5, .25, 10, 40, 7, 50)
        point = transform.detection_to_overlay(110, 440)
        self.assertEqual(point, (57, 150))
        self.assertEqual(transform.overlay_to_detection(*point), (110, 440))

    def test_rotated_image_mapping(self):
        original = detection_to_original(20, 70, 90, 200, 100)
        self.assertEqual(original, (129, 20))
        self.assertEqual(original_to_detection(*original, 90, 200, 100), (20, 70))

    def test_legacy_mapping_derives_title_height(self):
        debug = {"roi": {"x1": 0, "x2": 1282, "y1": 0, "y2": 5000},
                 "image_shape": [5000, 1282], "input_original_shape": [1282, 5000],
                 "rotation_ccw_degrees": 90,
                 "config": {"display_max_width": 1600, "display_max_height": 4000}}
        metadata = reconstruct_render_metadata(debug, 1026, 4116)
        self.assertEqual(metadata["mapping_status"], "reconstructed")
        self.assertEqual(metadata["title_height_px"], 116)
        self.assertAlmostEqual(metadata["preview_scale_y"], .8)


class AnnotationTests(unittest.TestCase):
    def test_approx_annotation_matches_correct_track(self):
        transform = OverlayTransform(100, 1, 1, 0, 0, 0, 100)
        result = match_false_track_annotations([305], [track("a", 200), track("b", 350)], transform, 8)
        self.assertEqual(result[0].status, "matched")
        self.assertEqual(result[0].matched_track_id, "a")

    def test_two_tracks_in_tolerance_are_ambiguous(self):
        transform = OverlayTransform(0, 1, 1)
        result = match_false_track_annotations([104], [track("a", 101), track("b", 107)], transform, 8, 2)
        self.assertEqual(result[0].status, "ambiguous_annotation_match")
        self.assertEqual(len(result[0].candidates), 2)

    def test_duplicate_false_track_keys_are_merged(self):
        text = '{"images":{"x.bmp":{"false_tracks":[10],"false_tracks":[20,30]}}}'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.json"
            path.write_text(text, encoding="utf-8")
            loaded = load_track_annotations(path)
        self.assertEqual(loaded["images"]["x.bmp"]["false_tracks"], [10., 20., 30.])

    def test_extended_annotation_categories_are_normalized(self):
        text = ('{"images":{"x.bmp":{"true_tracks":10,'
                '"missed_true_tracks":[20],"correctly_rejected_tracks":30}}}')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.json"
            path.write_text(text, encoding="utf-8")
            loaded = load_track_annotations(path)["images"]["x.bmp"]
        self.assertEqual(loaded["true_tracks"], [10.])
        self.assertEqual(loaded["missed_true_tracks"], [20.])
        self.assertEqual(loaded["correctly_rejected_tracks"], [30.])
        self.assertEqual(loaded["false_tracks"], [])

    def test_filename_alias_emits_warning(self):
        resolved, warning = resolve_annotation_image("金卡纸-2.bmp", ["金卡纸2.bmp"])
        self.assertEqual(resolved, "金卡纸2.bmp")
        self.assertIn("alias", warning)


def paper_track(track_id: str, positions: dict[int, float]) -> PaperEdgeTrack:
    count = len(positions)
    center = float(np.median(list(positions.values())))
    return PaperEdgeTrack(center, positions, {key: 5.0 for key in positions}, count,
                          count / 3, count, count / 3, center, 1.0,
                          float(np.ptp(list(positions.values()))), .9, .8, count / 3,
                          count / 3, .8, .8, "detected", count, 0,
                          track_id=track_id, candidate_class="strong",
                          sequence_decision="selected")


class AlignedProfileTests(unittest.TestCase):
    def aligned(self, positions):
        cfg = CounterConfig(num_strips=3, identity_profile_samples=81)
        length = 300
        y = np.arange(length)
        negative = np.stack([np.exp(-.5 * ((y - positions[i]) / 2) ** 2) * 10 for i in range(3)])
        positive = np.stack([np.exp(-.5 * ((y - (positions[i] - 8)) / 2) ** 2) * 8 for i in range(3)])
        raw = np.stack([80 + 20 * np.tanh((positions[i] - y) / 3) for i in range(3)])
        normalized = raw / 80
        return extract_track_aligned_profiles(paper_track("t", positions), 20, raw,
                                              normalized, negative, positive, 0, cfg)

    def test_curved_track_aligned_profile_extraction(self):
        result = self.aligned({0: 100, 1: 109, 2: 102})
        self.assertLessEqual(abs(int(np.argmax(result.negative)) - 40), 1)

    def test_tilted_track_aligned_profile_extraction(self):
        result = self.aligned({0: 90, 1: 105, 2: 120})
        self.assertLessEqual(abs(int(np.argmax(result.negative)) - 40), 1)

    def test_template_uses_median_and_outlier_does_not_poison_it(self):
        cfg = CounterConfig(identity_template_min_tracks=2)
        relative = np.linspace(-.6, .6, cfg.identity_profile_samples)
        profiles = {}
        for name, value in (("a", 1.), ("b", 1.), ("false", 100.)):
            array = np.full(cfg.identity_profile_samples, value)
            matrix = array[None, :]
            profiles[name] = AlignedTrackProfiles(relative, array, array, array, array,
                                                   matrix, matrix, matrix, matrix, 0.)
        template = build_edge_template(profiles, list(profiles), cfg)
        self.assertTrue(np.all(template.raw == 1.0))
        self.assertTrue(np.all(template.negative == 1.0))

    def test_false_outlier_does_not_shift_profile_mad(self):
        cfg = CounterConfig(identity_template_min_tracks=2)
        relative = np.linspace(-.6, .6, cfg.identity_profile_samples)
        profiles = {}
        for name, value in (("a", 2.), ("b", 2.), ("false", -40.)):
            array = np.full(cfg.identity_profile_samples, value)
            matrix = array[None, :]
            profiles[name] = AlignedTrackProfiles(relative, array, array, array, array,
                                                   matrix, matrix, matrix, matrix, 0.)
        template = build_edge_template(profiles, list(profiles), cfg)
        self.assertLess(float(np.max(template.raw_mad)), 1e-9)
        self.assertEqual(len(template.reference_track_ids), 3)


class VerifierDecisionTests(unittest.TestCase):
    def test_background_false_edge_needs_no_paper_above(self):
        cfg = CounterConfig()
        self.assertTrue(background_false_evidence(.05, True, "INVALID", 2, 20, cfg))

    def test_true_last_sheet_above_paper_below_black_is_not_background_false(self):
        cfg = CounterConfig()
        self.assertFalse(background_false_evidence(.9, False, "CORE", 2, 20, cfg))

    def test_secondary_texture_with_poor_identity_is_rejected(self):
        family, decision = conservative_identity_decision(["competition", "template"], True, -5, CounterConfig())
        self.assertEqual((family, decision), ("SECONDARY_TEXTURE", "REJECT_SECONDARY_TEXTURE"))

    def test_positive_pair_missing_alone_cannot_reject(self):
        _, decision = conservative_identity_decision([], True, -1, CounterConfig())
        self.assertEqual(decision, "SUSPICIOUS_KEEP")

    def test_two_independent_false_evidence_families_reject(self):
        _, decision = conservative_identity_decision(["template", "negative_shape"], True, -5, CounterConfig())
        self.assertTrue(decision.startswith("REJECT"))

    def test_ambiguous_identity_keeps(self):
        _, decision = conservative_identity_decision(["negative_shape"], True, -5, CounterConfig())
        self.assertEqual(decision, "AMBIGUOUS_KEEP")

    class Feature:
        def __init__(self, decision):
            self.proposed_decision = decision
            self.identity_score = .2
            self.identity_z_score = -5
            self.false_family = "SECONDARY_TEXTURE"

        def to_dict(self):
            return vars(self)

    def test_shadow_mode_does_not_change_count(self):
        tracks = [paper_track("a", {0: 10, 1: 10, 2: 10}),
                  paper_track("b", {0: 30, 1: 30, 2: 30})]
        features = {"a": self.Feature("TRUSTED"), "b": self.Feature("REJECT_SECONDARY_TEXTURE")}
        kept, rejected, debug = apply_final_track_verifier(tracks, features, CounterConfig())
        self.assertEqual((len(kept), len(rejected)), (2, 0))
        self.assertFalse(debug["recovery_after_verifier"])

    def test_active_mode_updates_count_without_recovery(self):
        tracks = [paper_track("a", {0: 10, 1: 10, 2: 10}),
                  paper_track("b", {0: 30, 1: 30, 2: 30})]
        features = {"a": self.Feature("TRUSTED"), "b": self.Feature("REJECT_SECONDARY_TEXTURE")}
        cfg = CounterConfig(final_track_verifier_shadow_mode=False)
        kept, rejected, debug = apply_final_track_verifier(tracks, features, cfg)
        self.assertEqual((len(kept), len(rejected)), (1, 1))
        self.assertEqual(debug["count_after"], 1)
        self.assertFalse(debug["recovery_after_verifier"])

    def test_close_pair_keeps_better_boundary_morphology(self):
        tracks = [paper_track("body", {0: 100, 1: 100, 2: 100}),
                  paper_track("edge", {0: 114, 1: 114, 2: 114})]
        body = self.Feature("AMBIGUOUS_KEEP")
        edge = self.Feature("REJECT_SECONDARY_TEXTURE")
        for feature, raw, normalized, positive, negative, families in (
                (body, .50, .55, -.10, .60, ["template"]),
                (edge, .70, .75, .45, .65, ["paper_support", "template"])):
            feature.raw_template_similarity = raw
            feature.normalized_template_similarity = normalized
            feature.positive_profile_corr = positive
            feature.negative_template_similarity = negative
            feature.template_outlier_z = 8.0
            feature.local_spacing = 29.0
            feature.evidence_families = families
        cfg = CounterConfig(final_track_verifier_shadow_mode=False)
        kept, rejected, debug = apply_final_track_verifier(
            tracks, {"body": body, "edge": edge}, cfg)
        self.assertEqual([track.track_id for track in kept], ["edge"])
        self.assertEqual([track.track_id for track in rejected], ["body"])
        self.assertEqual(debug["pair_adjudications"][0]["reason"],
                         "close_pair_boundary_morphology_swap")


if __name__ == "__main__":
    unittest.main()
