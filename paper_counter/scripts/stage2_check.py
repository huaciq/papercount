"""Stages 2/3 checkpoint: link and validate, keeping legacy production count."""
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_counter.config import CounterConfig
from paper_counter.legacy_counter import PaperCounter
from paper_counter.preprocessing import read_image
from paper_counter.local_detection import detect_strip_peaks
from paper_counter.tracking import associate_strip_peaks, validate_tracks
from paper_counter.track_visualization import plot_tracks

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = root / ("output_track_migration_stage2_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(exist_ok=False)
    cfg = CounterConfig(algorithm="legacy")
    result = PaperCounter(cfg).process(read_image(root.parent / "数据" / "2.bmp"))
    profiles = result.artifacts["polarities"]["negative"]["profiles"]
    peaks, _ = detect_strip_peaks(profiles, result.period, cfg)
    raw = associate_strip_peaks(peaks, result.period, cfg)
    kept, rejected = validate_tracks(raw, cfg)
    plot_tracks(kept, rejected, None, result.period, output)
    print("Legacy count:", result.count, "Raw tracks:", len(raw), "Validated:", len(kept), "Rejected:", len(rejected))
