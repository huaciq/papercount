"""Compare the historical tracks preset with the horizontal-stack variant."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_counter import PaperDetector
from paper_counter.preprocessing import read_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] / "数据")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--match", default="", help="ASCII substring for a dataset directory")
    parser.add_argument("--no-geometry", action="store_true")
    parser.add_argument("--always-rotate", action="store_true")
    parser.add_argument("--never-rotate", action="store_true")
    parser.add_argument("--no-gap", action="store_true")
    parser.add_argument("--gap-gain", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("output/preprocessed_compare.json"))
    args = parser.parse_args()
    # Keep this script independent of filenames and Unicode command-line quoting.
    search_root = args.root
    if args.match:
        matches = sorted(p for p in args.root.rglob("*") if p.is_dir() and args.match in p.name
                         and any(p.glob("*.bmp")))
        if matches:
            search_root = matches[0]
    files = sorted(search_root.rglob("*.bmp"))[:args.limit]
    base = PaperDetector(preset="tracks")
    if args.no_geometry:
        from paper_counter import CounterConfig
        config_data = PaperDetector(preset="tracks_preprocessed").config.to_dict()
        config_data["auto_geometry_enabled"] = False
        if args.no_gap:
            config_data["tracks_gap_enhancement_enabled"] = False
        if args.gap_gain is not None:
            config_data["tracks_gap_gain"] = args.gap_gain
        prepared = PaperDetector(config=CounterConfig(**config_data))
    else:
        if args.no_gap:
            from paper_counter import CounterConfig
            config_data = PaperDetector(preset="tracks_preprocessed").config.to_dict()
            config_data["tracks_gap_enhancement_enabled"] = False
            if args.gap_gain is not None:
                config_data["tracks_gap_gain"] = args.gap_gain
            prepared = PaperDetector(config=CounterConfig(**config_data))
        else:
            prepared = PaperDetector(preset="tracks_preprocessed")
    rows = []
    for path in files:
        shape = read_image(path).shape
        # The historical test captures are mixed: wide originals need the
        # documented quarter-turn, while already rotated tall captures do not.
        rotate = False if args.never_rotate else (True if args.always_rotate else bool(shape[0] < shape[1]))
        old = base.detect(path, rotate_ccw=rotate)
        new = prepared.detect(path, rotate_ccw=rotate)
        rows.append({"image": str(path), "tracks": old.count, "tracks_preprocessed": new.count,
                     "period": new.period, "geometry": new.raw_result.debug_info.get("geometry_preprocessing", {}),
                     "preprocessing": new.raw_result.debug_info.get("preprocessing", {}),
                     "rotate_ccw": rotate})
        print(path.name, old.count, new.count, "rotate" if rotate else "native", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
