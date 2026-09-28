"""Compare saved strip responses at reviewer-labelled overlay coordinates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_counter.coordinates import transform_from_metadata


def load_case(folder: Path, label: str) -> dict:
    debug = json.loads((folder/"debug_info.json").read_text(encoding="utf-8"))
    metadata = json.loads((folder/"render_metadata.json").read_text(encoding="utf-8"))
    with np.load(folder/"profiles.npz", allow_pickle=False) as archive:
        profiles = archive["negative_profiles"].copy()
        y = archive["original_y"].copy()
    candidates = debug.get("strip_peak_candidates", [])
    return {"label": label, "debug": debug, "transform": transform_from_metadata(metadata),
            "profiles": profiles, "y": y, "candidates": candidates}


def candidate_coverage(case: dict, detection_y: float, radius: float) -> int:
    count = 0
    for strip in case["candidates"]:
        rows = strip if isinstance(strip, list) else [strip]
        if any(abs(float(row.get("y", -1e9))-detection_y) <= radius for row in rows):
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", nargs=2, action="append", metavar=("LABEL", "IMAGE_OUTPUT"), required=True)
    parser.add_argument("--overlay-y", nargs="+", type=float, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--half-window", type=int, default=35)
    args = parser.parse_args()
    cases = [load_case(Path(folder), label) for label, folder in args.case]
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    fig, axes = plt.subplots(len(args.overlay_y), 1, figsize=(11, 3.4*len(args.overlay_y)), squeeze=False)
    for ax, overlay_y in zip(axes[:, 0], args.overlay_y):
        for case in cases:
            _, detection_y = case["transform"].overlay_to_detection(0, overlay_y)
            mask = np.abs(case["y"]-detection_y) <= args.half_window
            fused = np.median(case["profiles"], axis=0)
            ax.plot(case["y"][mask], fused[mask], lw=1.25, label=case["label"])
            period = float(case["debug"].get("period") or 20)
            coverage = candidate_coverage(case, detection_y, max(2.0, .2*period))
            summary.append({"overlay_y": overlay_y, "label": case["label"],
                            "detection_y": detection_y, "candidate_strip_coverage": coverage,
                            "num_strips": int(case["profiles"].shape[0]),
                            "local_response_max": float(np.max(fused[mask])) if np.any(mask) else None})
        ax.axvline(cases[0]["transform"].overlay_to_detection(0, overlay_y)[1], color="black", ls=":", lw=1)
        ax.set_title(f"review coordinate overlay Y={overlay_y:g}")
        ax.set_xlabel("detection Y")
        ax.set_ylabel("median normalized negative response")
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.output/"aggregation_local_response.png", dpi=160)
    plt.close(fig)
    (args.output/"aggregation_candidate_coverage.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Strip aggregation comparison", "",
             "The vertical marker is the reviewer coordinate after exact overlay-to-detection mapping.", "",
             "| overlay Y | method | detection Y | candidate strips | total strips | local max |",
             "|---:|---|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['overlay_y']:.1f} | {row['label']} | {row['detection_y']:.2f} | "
                     f"{row['candidate_strip_coverage']} | {row['num_strips']} | {row['local_response_max']:.3f} |")
    (args.output/"aggregation_candidate_coverage.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
