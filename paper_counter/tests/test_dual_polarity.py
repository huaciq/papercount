import unittest
from pathlib import Path
from paper_counter.dual_polarity import compare_positions, process_dual
from paper_counter import CounterConfig
from test_counter import synthetic_papers


class DualPolarityTests(unittest.TestCase):
    def test_default_yaml_dual_keeps_terminal_guard_only_on_primary(self):
        config = CounterConfig.from_yaml(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
        image, _ = synthetic_papers(count=12)
        negative, positive, audit = process_dual(image, config)
        self.assertTrue(config.final_track_verifier_enabled)
        self.assertTrue(config.raw_final_observation_guard_enabled)
        self.assertTrue(negative.debug_info["config"]["raw_final_observation_guard_enabled"])
        self.assertFalse(positive.debug_info["config"]["final_track_verifier_enabled"])
        self.assertFalse(positive.debug_info["config"]["raw_final_observation_guard_enabled"])
        self.assertEqual(audit["negative_final_count"], negative.count)

    def test_phase_shift_is_not_double_counted(self):
        result = compare_positions([10, 30, 50], [5, 25, 45])
        self.assertEqual(result['count_difference_positive_minus_negative'], 0)
        self.assertEqual(result['interval_agreement'], 1)

    def test_equal_totals_can_hide_local_disagreements(self):
        result = compare_positions([10, 30, 50], [15, 20, 55])
        self.assertEqual(result['count_difference_positive_minus_negative'], 0)
        self.assertEqual(result['interval_agreement'], 0)

    def test_empty_has_no_agreement_claim(self):
        self.assertIsNone(compare_positions([], [])['interval_agreement'])

    def test_both_signs_detect_synthetic_sheets(self):
        image, _ = synthetic_papers(count=12)
        neg, pos, audit = process_dual(image, CounterConfig())
        self.assertEqual((neg.count, pos.count), (12, 12))
        self.assertEqual(audit['negative_pre_verifier_count'], 12)
        self.assertEqual(audit['negative_roi'], audit['positive_roi'])
