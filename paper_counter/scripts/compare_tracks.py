"""Run the historical baseline and new engine, preserving all prior output."""
from __future__ import annotations
import contextlib
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from paper_counter import CounterConfig, PaperCounter
from paper_counter.legacy_counter import PaperCounter as LegacyCounter
from paper_counter.preprocessing import read_image


def main() -> None:
    """Keep old and new configs/results together, with exact executed CLI logs."""
    root = Path(__file__).resolve().parents[1]
    output = root / ("output_tracks_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir()
    rows = []
    for name in ("2.bmp", "4.bmp", "3.png"):
        cfg = CounterConfig.from_yaml(root / "configs" / "default.yaml")
        # 4.bmp needs the already user-authorized 80px search upper bound.
        if name == "4.bmp":
            cfg = replace(cfg, max_period=80)
        image = read_image(root.parent / "数据" / name)
        old_cfg = CounterConfig(algorithm="legacy", num_strips=8, gradient_polarity="auto", max_period=cfg.max_period)
        old = LegacyCounter(old_cfg).process(image)
        target = output / Path(name).stem
        target.mkdir()
        with (target / "config.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(cfg.to_dict(), stream, sort_keys=False)
        with (target / "legacy_debug_info.json").open("w", encoding="utf-8") as stream:
            json.dump(old.debug_info, stream, indent=2, ensure_ascii=False, allow_nan=False)
        started = time.perf_counter()
        with (target / "cli.log").open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-X", "utf8=0", str(root / "main.py"),
                "--image", str(root.parent / "数据" / name), "--config", str(target / "config.yaml"),
                "--output", str(target)], stdout=stream, stderr=subprocess.STDOUT, check=True,
                env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
        with (target / "debug_info.json").open(encoding="utf-8") as stream:
            new = json.load(stream)
        assert new["raw_track_count"] == len(new["paper_edge_tracks"]) + len(new["rejected_tracks"]), "Track accounting mismatch"
        row = {"image": name, "old_count": old.count, "new_count": new["count"],
               "detected": sum(t["source"] == "detected" for t in new["paper_edge_tracks"]),
               "recovered": sum(t["source"] == "recovered" for t in new["paper_edge_tracks"]),
               "rejected": len(new["rejected_tracks"]), "initial_period": new["initial_global_period"],
               "median_local_period": new["median_local_period"], "active_roi": new["active_stack_roi"],
               "confidence": new["confidence"], "seconds_including_plots": time.perf_counter() - started}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)
    with (output / "comparison.json").open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2, ensure_ascii=False)
    print("OUTPUT:", output, flush=True)


if __name__ == "__main__":
    main()
