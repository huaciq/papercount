"""Evaluate classification and sequence-only DP on saved horizontal evidence."""
from pathlib import Path
from datetime import datetime
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_counter.config import CounterConfig
from paper_counter.models import PaperEdgeTrack, StripPeakCandidate
from paper_counter.tracking import classify_tracks
from paper_counter.local_period import estimate_local_period
from paper_counter.vertical_sequence import optimize_vertical_sequence
from paper_counter.track_visualization import plot_vertical_sequence


def main():
    output = ROOT / ("output_sequence_stages_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(exist_ok=False)
    for name in ("2", "4", "3"):
        old = json.loads((ROOT / "output_tracks_20260914_100847_868173" / name / "debug_info.json").read_text(encoding="utf-8"))
        cfg = CounterConfig(**old["config"])
        tracks = []
        for row in old["paper_edge_tracks"] + old["rejected_tracks"]:
            if row["source"] != "detected":
                continue
            row["strip_positions"] = {int(k): v for k, v in row["strip_positions"].items()}
            row["strip_peak_strengths"] = {int(k): v for k, v in row["strip_peak_strengths"].items()}
            row["peak_candidates"] = [StripPeakCandidate(**p) for p in row["peak_candidates"]]
            tracks.append(PaperEdgeTrack(**row))
        tracks.sort(key=lambda t: t.center_y)
        strong, marginal, garbage = classify_tracks(tracks, cfg)
        local = estimate_local_period(strong, old["period"], cfg)
        sequence = optimize_vertical_sequence(strong + marginal, local, cfg)
        target = output / name
        target.mkdir()
        info = {"phase": "classification and sequence-only; no recovery; production count unchanged",
                "old_count": old["count"], "strong": len(strong), "marginal": len(marginal),
                "garbage": len(garbage), "sequence_count": len(sequence.selected), **sequence.to_dict()}
        (target / "checkpoint.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        plot_vertical_sequence(sequence.to_dict(), target)
        print(name, "old", old["count"], "pool", len(strong), len(marginal), "sequence", len(sequence.selected))
        for row in sequence.decisions:
            if name == "4" and 1960 < row["center_y"] < 2035 or name == "2" and row["center_y"] in (1480.5, 1680, 1834):
                print(json.dumps(row))
    print("OUTPUT", output)


if __name__ == "__main__":
    main()
