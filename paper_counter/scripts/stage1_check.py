"""Stage 1 checkpoint: legacy count plus independent local strip candidates."""
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_counter.config import CounterConfig
from paper_counter.legacy_counter import PaperCounter
from paper_counter.preprocessing import read_image
from paper_counter.local_detection import detect_strip_peaks
from paper_counter.track_visualization import plot_strip_candidates

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = root / ("output_track_migration_stage1_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(exist_ok=False)
    cfg = CounterConfig(num_strips=8, gradient_polarity="auto", algorithm="legacy")
    result = PaperCounter(cfg).process(read_image(root.parent / "数据" / "2.bmp"))
    profiles = result.artifacts["polarities"][result.gradient_polarity]["profiles"]
    candidates, stats = detect_strip_peaks(profiles, result.period, cfg)
    plot_strip_candidates(candidates, stats, profiles, 0, output)
    print("Stage 1 legacy count:", result.count)
    print("Independent candidates:", sum(map(len, candidates)))
