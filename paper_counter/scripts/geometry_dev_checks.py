"""Compact synthetic failure diagnostics for geometry migration."""
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from test_counter import synthetic_papers
from paper_counter import PaperCounter, CounterConfig
from test_vertical_sequence import solve, track

image, centers = synthetic_papers()
image[:, 6:27] = np.random.default_rng(3).integers(0, 256, (len(image), 21), dtype=np.uint8)
for cfg in (CounterConfig(), CounterConfig(track_max_neighbor_jump_ratio=.25), CounterConfig(track_continuity_geometry_enabled=False)):
    result = PaperCounter(cfg).process(image)
    print("CONFIG", cfg.track_continuity_geometry_enabled, cfg.track_max_neighbor_jump_ratio, "COUNT", result.count, "MASK", result.debug_info["raw_stack_mask"]["intervals"])
    print("selected", result.edge_positions)
    print("rejected usable", [(t.center_y, t.reason, round(t.total_score, 3)) for t in result.rejected_tracks if t.candidate_class != "garbage"])
noise = PaperCounter().process(np.random.default_rng(19).integers(0, 256, (2000, 96), dtype=np.uint8))
print("NOISE", noise.count, noise.debug_info["confidence_components"])
result = solve([track(100), track(112, .52), track(124, .50), track(152)])
print("MARGINAL", [(t.center_y, t.total_score) for t in result.selected])
print("EDGES", result.transitions)
