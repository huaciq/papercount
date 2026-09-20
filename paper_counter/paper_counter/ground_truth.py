"""Optional ground-truth loading and edge/count metrics."""
from __future__ import annotations

import json
from pathlib import Path
import re
import numpy as np


def load_ground_truth(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("ground_truth.json must be an object keyed by image name")
    return data


def load_count_text(path: str | Path) -> int:
    """Read a simple count truth file such as ``28张`` or ``28``.

    The truth files used for the image batch are intentionally kept outside
    the detector output.  This helper accepts UTF-8/UTF-8-BOM text and uses
    the first integer in the file as the expected sheet count.
    """
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    match = re.search(r"[-+]?\d+", text)
    if not match:
        raise ValueError(f"No integer count found in truth file: {path}")
    value = int(match.group(0))
    if value < 0:
        raise ValueError(f"Truth count must be non-negative: {path}")
    return value


def evaluate_result(result, truth: dict | None, image_name: str, tolerance_ratio: float = .35) -> dict:
    entry = (truth or {}).get(image_name)
    if entry is None:
        return {"available": False}
    if isinstance(entry, (int, float)):
        entry = {"count": int(entry)}
    expected = entry.get("count")
    metrics = {"available": True, "expected_count": expected,
               "predicted_count": int(result.count),
               "count_error": None if expected is None else int(result.count) - int(expected)}
    expected_edges = entry.get("edge_y")
    if expected_edges is None:
        return metrics
    truth_edges = sorted(float(x) for x in expected_edges)
    observed = result.edge_positions
    if not observed and getattr(result, "peak_positions", None):
        observed = result.peak_positions
    predicted = sorted(float(x) for x in observed)
    spacing = np.diff(predicted)
    tolerance = float(tolerance_ratio * (np.median(spacing) if len(spacing) else 1.0))
    used = set(); matched = 0
    for y in truth_edges:
        choices = [(abs(y - p), i) for i, p in enumerate(predicted) if i not in used and abs(y - p) <= tolerance]
        if choices:
            _, i = min(choices); used.add(i); matched += 1
    fp = len(predicted) - matched; fn = len(truth_edges) - matched
    precision = matched / max(1, matched + fp); recall = matched / max(1, matched + fn)
    metrics.update({"matched": matched, "false_positive": fp, "false_negative": fn,
                    "precision": precision, "recall": recall,
                    "f1": 2 * precision * recall / max(1e-9, precision + recall),
                    "edge_tolerance": tolerance})
    return metrics


__all__ = ["load_ground_truth", "load_count_text", "evaluate_result"]
