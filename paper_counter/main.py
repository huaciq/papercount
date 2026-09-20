"""Offline single-image paper counting CLI."""

import argparse
import sys
from pathlib import Path
import numpy as np

from paper_counter import CounterConfig, PaperCounter
from paper_counter.preprocessing import read_image
from paper_counter.visualization import save_debug
from paper_counter.ground_truth import load_ground_truth, evaluate_result


def main() -> int:
    """Load one image, run the counter and save all diagnostics."""
    parser = argparse.ArgumentParser(description="Traditional paper-edge counting at native image resolution")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs" / "default.yaml")
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--rotate-ccw", action="store_true", help="Rotate input counter-clockwise 90 degrees before detection")
    parser.add_argument("--ground-truth", type=Path, default=None, help="Optional ground_truth.json for count/edge metrics")
    args = parser.parse_args()
    try:
        config = CounterConfig.from_yaml(args.config)
        image = read_image(args.image)
        original_shape = list(image.shape)
        if args.rotate_ccw:
            image = np.rot90(image, 1).copy()
        result = PaperCounter(config).process(image)
        result.debug_info["input_image"] = str(args.image.resolve())
        result.debug_info["input_original_shape"] = original_shape
        result.debug_info["rotation_ccw_degrees"] = 90 if args.rotate_ccw else 0
        result.debug_info["coordinate_system"] = "detection image after requested rotation; x2/y2 exclusive"
        if args.rotate_ccw:
            result.debug_info["coordinate_mapping_to_original"] = {
                "original_x": "original_width - 1 - detection_y", "original_y": "detection_x"}
        if args.ground_truth:
            result.debug_info["ground_truth"] = evaluate_result(
                result, load_ground_truth(args.ground_truth), args.image.name)
        save_debug(result, args.output, config)
    except (ValueError, OSError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    debug = result.debug_info
    items = {"Image": str(args.image.resolve()), "ROI": debug["roi"],
             "Gradient polarity": result.gradient_polarity,
             "Positive score": debug["polarity_scores"]["positive"]["score"],
             "Negative score": debug["polarity_scores"]["negative"]["score"],
             "Estimated period": result.period, "Mean peak spacing": debug["mean_peak_spacing"],
             "Raw candidate count": debug["raw_candidate_count"], "Rejected count": len(result.rejected_peaks),
             "Repaired count": len(result.repaired_peaks), "Final count": result.count,
             "Confidence": round(result.confidence, 4), "Status": debug["status"],
             "Output": str(args.output.resolve())}
    for name, value in items.items():
        print(f"{name}: {value}")
    if debug.get("algorithm") == "multi_strip_tracks":
        for label, key in (("Active stack ROI", "active_stack_roi"), ("Initial global period", "initial_global_period"),
                           ("Median local period", "median_local_period"), ("Strong strip candidate count", "strong_strip_candidate_count"),
                           ("Weak strip candidate count", "weak_strip_candidate_count"), ("Raw track count", "raw_track_count")):
            print(f"{label}: {debug[key]}")
        print(f"Rejected track count: {len(result.rejected_tracks)}")
        print(f"Strong track count: {len(debug.get('strong_tracks', []))}")
        print(f"Marginal track count: {len(debug.get('marginal_tracks', []))}")
        print(f"Garbage track count: {len(debug.get('garbage_tracks', []))}")
        sequence = debug.get("vertical_sequence", {})
        print(f"Sequence selected: {len(sequence.get('selected_track_ids', []))}")
        print(f"Sequence rescued: {len(sequence.get('rescued_track_ids', []))}")
        print(f"Sequence suppressed: {len(sequence.get('suppressed_track_ids', []))}")
        print(f"Sequence Total Score: {sequence.get('total_sequence_score', 0):.6f}")
        print(f"Suspicious gaps after sequence optimization: {len(debug.get('remaining_suspicious_gaps', []))}")
        reused_ids = {track_id for a in debug.get("candidate_reuse_recovery_attempts", [])
                      for track_id in a.get("final_selected_track_ids", [])}
        marginal_ids = {t.track_id for t in result.paper_edge_tracks if t.candidate_class == "marginal"}
        print(f"Candidate-reuse recoveries: {len(reused_ids & marginal_ids)}")
        print(f"Weak-image recoveries: {sum(t.source == 'recovered' for t in result.paper_edge_tracks)}")
        print(f"Recovered track count: {sum(t.source == 'recovered' for t in result.paper_edge_tracks)}")
        print(f"Sequence rescued count: {len(debug.get('sequence_rescued_tracks', []))}")
        print(f"Sequence suppressed count: {len(debug.get('sequence_suppressed_tracks', []))}")
        print(f"Final track count: {len(result.paper_edge_tracks)}")
        print(f"Final Count: {result.count}")
        regularity = debug.get("regularity", {})
        ratios = regularity.get("region_ratios", {})
        print(f"Regular region ratio: {ratios.get('REGULAR', 0):.3f}")
        print(f"Mixed region ratio: {ratios.get('MIXED', 0):.3f}")
        print(f"Irregular region ratio: {ratios.get('IRREGULAR', 0):.3f}")
        mask_thresholds = debug.get("raw_stack_mask", {}).get("thresholds", {})
        refinement = debug.get("mask_refinement", {})
        print(f"Initial stack ROI: {mask_thresholds.get('initial_top')}..{mask_thresholds.get('initial_bottom')}")
        print(f"Refined stack ROI: {refinement.get('refined_top')}..{refinement.get('refined_bottom')}")
        print(f"Regular missing hypotheses: {len(debug.get('regular_missing_hypotheses', []))}")
        print(f"Period-assisted recoveries: {len(debug.get('regular_period_assisted_recoveries', []))}")
        print(f"Geometry-only recoveries: {len(debug.get('geometry_only_recoveries', []))}")
        print(f"Short-gap competitions: {len(debug.get('secondary_texture_competitions', []))}")
        print(f"Strong texture suppressions: {sum(1 for t in debug.get('sequence_suppressed_tracks', []) if t.get('reason') == 'sequence_suppressed_short_gap')}")
        print(f"Unresolved regular gaps: {len(debug.get('unresolved_regular_gaps', []))}")
        if debug.get("ground_truth", {}).get("available"):
            print(f"Ground-truth metrics: {debug['ground_truth']}")
        for i, track in enumerate(result.paper_edge_tracks, 1):
            print(f"Track #{i} center_y: {track.center_y:.2f} support: {track.support_count}/{config.num_strips} "
                  f"({track.support_ratio:.3f}) continuity: {track.continuity_ratio:.3f} y_mad: {track.y_mad:.3f} "
                  f"gradient_score: {track.gradient_score:.3f} smoothness_score: {track.smoothness_score:.3f} "
                  f"total_score: {track.total_score:.3f} source: {track.source} decision: {track.sequence_decision}")
    for warning in debug["warnings"]:
        print(f"Warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
