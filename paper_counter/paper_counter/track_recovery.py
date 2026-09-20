"""Recover weak edges using observed multi-strip paths in hypothesized gaps."""
from __future__ import annotations
from .config import CounterConfig
from .models import StripPeakCandidate, PaperEdgeTrack
from .tracking import associate_strip_peaks, validate_tracks


def positive_aux_score(track: PaperEdgeTrack, auxiliary: list[list[StripPeakCandidate]],
                       period: float, config: CounterConfig) -> float:
    """Bounded auxiliary support; cannot create a track or waive structural gates."""
    supported = 0
    for strip, y in track.strip_positions.items():
        supported += any(not p.reason and abs(p.y - y) <= config.positive_aux_search_ratio * period
                         for p in auxiliary[strip])
    return supported / config.num_strips


def recover_weak_tracks(hypotheses: list[dict], candidates: list[list[StripPeakCandidate]],
                        existing: list[PaperEdgeTrack], valid_count: int, config: CounterConfig,
                        auxiliary: list[list[StripPeakCandidate]] | None = None
                        ) -> tuple[list[PaperEdgeTrack], list[PaperEdgeTrack], list[dict]]:
    """Re-link evidence per strip; never invent Y positions or count predictions."""
    recovered, rejected, attempts = [], [], []
    occupied = {(p.strip_index, p.y) for t in existing for p in t.peak_candidates}
    for hypothesis in hypotheses:
        if not hypothesis["search"]:
            continue
        period = hypothesis["local_period"]
        radius = config.recovery_search_ratio * period
        selected = [[p for p in strip if not p.reason and
                     (p.strip_index, p.y) not in occupied and
                     abs(p.y - hypothesis["predicted_strip_positions"][p.strip_index]) <= radius]
                    for strip in candidates]
        raw = associate_strip_peaks(selected, period, config, valid_count, recovery=True)
        # Auxiliary evidence can lift a borderline total score, but no other gate.
        for track in raw:
            track.predicted_y = hypothesis["predicted_y"]
            if auxiliary is not None:
                track.positive_aux_score = positive_aux_score(track, auxiliary, period, config)
                track.total_score = min(1.0, track.total_score + config.positive_aux_weight * track.positive_aux_score)
                track.confidence = track.total_score
        accepted, failures = validate_tracks(raw, config, recovery=True)
        rejected.extend(failures)
        accepted = sorted(accepted, key=lambda t: t.total_score, reverse=True)
        best = None
        for track in accepted:
            if best is None and not any(abs(track.center_y - other.center_y) < config.duplicate_track_ratio * period
                                        for other in existing + recovered):
                best = track
            else:
                track.reason = "duplicate_track"
                rejected.append(track)
        attempt = {"predicted_y": hypothesis["predicted_y"], "local_period": period,
                   "candidate_count": sum(map(len, selected)), "raw_track_count": len(raw), "actual_center_y": None,
                   "reason": "no_recovery_evidence"}
        if best is not None:
            best.reason = "weak_multistrip_evidence"
            recovered.append(best)
            occupied.update((p.strip_index, p.y) for p in best.peak_candidates)
            attempt.update(actual_center_y=best.center_y, support_count=best.support_count,
                           weak_support_count=best.weak_support_count, recovery_score=best.total_score,
                           positive_aux_score=best.positive_aux_score, reason=best.reason)
        attempts.append(attempt)
    return recovered, rejected, attempts
