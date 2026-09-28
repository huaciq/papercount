"""Independent polarity audit; interval agreement is evidence, not ground truth."""
from dataclasses import replace
import numpy as np
from .track_counter import process_tracks


def compare_positions(negative, positive):
    """Count opposite edges in open inter-edge intervals, without merging counts.

    One positive edge per negative interval is expected for simple bright/dark
    layers. Texture, missed boundaries and boundary phase can violate this.
    """
    negative = np.sort(np.asarray(negative, dtype=float))
    positive = np.sort(np.asarray(positive, dtype=float))
    intervals = []
    for left, right in zip(negative[:-1], negative[1:]):
        inside = positive[(positive > left) & (positive < right)]
        intervals.append({"from_y": float(left), "to_y": float(right),
                          "positive_positions": inside.tolist(),
                          "positive_count": len(inside),
                          "status": "one_opposite_edge" if len(inside) == 1 else
                                    "missing_opposite_edge" if not len(inside) else
                                    "multiple_opposite_edges"})
    good = sum(r["positive_count"] == 1 for r in intervals)
    return {"negative_count": len(negative), "positive_count": len(positive),
            "count_difference_positive_minus_negative": len(positive) - len(negative),
            "interval_agreement": good / len(intervals) if intervals else None,
            "intervals": intervals,
            "interpretation": "review evidence only; equal counts do not prove correct edges"}


def process_dual(image, config):
    """Keep production negative output, run positive without negative verifier.

    Pre-verifier negative tracks remove verifier differences from comparison.
    Both branches use identical input, ROI algorithm and tracking parameters.
    """
    if config.algorithm != "tracks":
        raise ValueError("Dual polarity audit requires the tracks algorithm")
    negative = process_tracks(image, replace(config, gradient_polarity="negative"))
    positive = process_tracks(image, replace(config, gradient_polarity="positive",
                              final_track_verifier_enabled=False, positive_aux_weight=0.0,
                              recovery_require_negative_evidence=False))
    before_ids = set(negative.debug_info["pre_verifier_sequence"]["track_ids"])
    candidates = negative.paper_edge_tracks + negative.rejected_tracks
    before = {t.track_id: t.center_y for t in candidates if t.track_id in before_ids}
    audit = compare_positions(list(before.values()), [t.center_y for t in positive.paper_edge_tracks])
    audit.update(negative_final_count=negative.count,
                 negative_pre_verifier_count=len(before),
                 negative_confidence=negative.confidence,
                 positive_confidence=positive.confidence,
                 negative_roi=negative.debug_info["roi"], positive_roi=positive.debug_info["roi"],
                 negative_period=negative.period, positive_period=positive.period,
                 positive_tracks=[t.to_dict() for t in positive.paper_edge_tracks],
                 negative_auxiliary_weight=config.positive_aux_weight,
                 branch_difference="negative retains positive recovery auxiliary; positive uses its own evidence",
                 positive_verifier="disabled: existing verifier assumes negative boundaries",
                 coordinate_system="rotated detection image, not overlay pixels")
    negative.debug_info["dual_polarity"] = audit
    return negative, positive, audit
