"""Select observed horizontal tracks with a soft vertical spacing objective.

No predicted location becomes a node. A DAG edge may span missing periods;
such edges cost evidence rather than fabricating a count. With a nonzero
spacing-change weight, states retain the previous edge (second-order DP).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .config import CounterConfig
from .models import PaperEdgeTrack
from .local_period import LocalPeriod
from .track_geometry import geometry_pair_score, fit_track_curve
from .regularity import evaluate_middle_track_competition


def huber(error: float) -> float:
    error = abs(error)
    return 0.5 * error * error if error <= 1 else error - 0.5


def period_mad_at(local: LocalPeriod, y: float) -> float:
    return float(np.interp(y, local.y, local.mad)) if len(local.y) else 0.0


def unary_score(track: PaperEdgeTrack, config: CounterConfig) -> float:
    # Existing track scores are already normalized in [0, 1]. The small cost
    # discourages adding a candidate whose improvement is only numerical noise.
    reward = config.sequence_unary_weight * float(np.clip(track.total_score, 0, 1))
    if track.candidate_class == "marginal":
        # Even a marginal proposal with high coverage must not receive more
        # unary evidence than the weakest fully qualified strong candidate.
        reward = min(reward + (config.sequence_rescue_bonus if config.sequence_period_multiple_mode else 0),
                     config.sequence_unary_weight * (config.min_track_score - config.epsilon))
    return reward - config.sequence_suppress_margin


def transition_score(left: PaperEdgeTrack, right: PaperEdgeTrack, local: LocalPeriod,
                     config: CounterConfig, regularity=None) -> dict | None:
    midpoint = (left.center_y + right.center_y) / 2
    period = local.at(midpoint)
    gap = right.center_y - left.center_y
    if not config.sequence_period_multiple_mode:
        geometry = geometry_pair_score(left, right, local, config)
        if gap <= 0 or geometry["hard_reject"]:
            return None
        # Continuous regularity adapts the large-gap and spacing prior.  The
        # default direct-call behavior retains the geometry-only objective.
        r = float(np.clip(regularity.at(midpoint), 0, 1)) if regularity is not None else 0.0
        base_long = geometry["long_gap_penalty"]
        base_prior = geometry["spacing_prior_penalty"]
        ratio = max(geometry["gap_median"], config.epsilon) / max(period, config.epsilon)
        adaptive_long = config.sequence_long_spacing_weight * (0.10 + 0.90 * r) * (max(0., ratio - 1) / max(1., ratio))
        adaptive_prior_weight = config.irregular_spacing_prior_weight + r * (
            config.regular_spacing_prior_weight - config.irregular_spacing_prior_weight)
        adaptive_prior = adaptive_prior_weight * min(1., abs(np.log(max(ratio, config.epsilon))))
        geometry["regularity"] = r
        geometry["adaptive_spacing_prior_weight"] = adaptive_prior_weight
        geometry["long_gap_penalty"] = adaptive_long
        geometry["spacing_prior_penalty"] = adaptive_prior
        geometry["pair_score"] += (base_long - adaptive_long) + (base_prior - adaptive_prior)
        return {**geometry, "from_track_id": left.track_id, "to_track_id": right.track_id,
            "from_y": left.center_y, "to_y": right.center_y, "gap": geometry["gap_median"],
            "local_period": period, "period_multiple": 1, "period_multiple_interpretation": "disabled",
            "spacing_error": geometry["gap_median"] - period, "spacing_change_score": 0.0}
    if gap <= 0 or gap > config.max_sequence_gap_periods * period:
        return None
    mad = period_mad_at(local, midpoint)
    scale = max(1.0, config.sequence_spacing_scale_ratio * period, 1.4826 * mad)
    # A gap spanning n periods accumulates spacing uncertainty. Using the
    # one-period sigma for a two-period gap over-penalizes legitimate endpoint
    # evidence; independent spacing deviations accumulate as sqrt(n).
    costs = [(config.sequence_spacing_weight * huber((gap - n * period) / (scale * np.sqrt(n)))
              + config.sequence_missing_period_penalty * (n - 1), n)
             for n in range(1, config.max_sequence_gap_periods + 1)]
    cost, repeats = min(costs)
    short = max(0.0, 1 - gap / (config.sequence_min_spacing_ratio * period))
    cost += config.sequence_short_gap_penalty * short
    return {"from_track_id": left.track_id, "to_track_id": right.track_id,
            "from_y": left.center_y, "to_y": right.center_y, "gap": gap,
            "local_period": period, "period_mad": mad, "single_period_spacing_scale": scale,
            "spacing_scale": float(scale * np.sqrt(repeats)),
            "period_multiple": repeats, "spacing_error": gap - repeats * period,
            "pair_score": -cost, "spacing_change_score": 0.0}


@dataclass
class SequenceResult:
    selected: list[PaperEdgeTrack] = field(default_factory=list)
    suppressed: list[PaperEdgeTrack] = field(default_factory=list)
    rescued: list[PaperEdgeTrack] = field(default_factory=list)
    total_sequence_score: float = 0.0
    transitions: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    rejected_transitions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"selected_track_ids": [t.track_id for t in self.selected],
                "suppressed_track_ids": [t.track_id for t in self.suppressed],
                "rescued_track_ids": [t.track_id for t in self.rescued],
                "total_sequence_score": self.total_sequence_score,
                "transitions": self.transitions, "track_decisions": self.decisions,
                "rejected_transitions": self.rejected_transitions}


def optimize_vertical_sequence(tracks: list[PaperEdgeTrack], local: LocalPeriod,
                               config: CounterConfig, regularity=None) -> SequenceResult:
    """Maximize sum(unary) + sum(pair) + optional spacing-change costs.

    Each component is anchored by at least one strong candidate. Starts/ends
    are free: real partial edges at the stack boundary need no invented neighbor.
    Components separated by more than the bounded transition range are solved
    independently and recorded without a fabricated transition across the break.
    """
    ordered = sorted((t for t in tracks if t.candidate_class in {"strong", "marginal"}),
                     key=lambda t: (t.center_y, t.track_id))
    for i, track in enumerate(ordered):
        if not track.track_id:
            track.track_id = f"candidate_{i:05d}"
        fit_track_curve(track, config)
    # Standalone callers get the same RAW permissions as the production flow.
    forbidden = [t for t in ordered if t.raw_mask_state == "INVALID" or
        (t.raw_mask_state == "MARGIN" and t.candidate_class == "marginal" and
         (not config.margin_allow_marginal or t.raw_evidence_ratio < config.raw_mask_min_track_evidence_ratio or
          t.gradient_score < config.margin_min_gradient_score or t.support_ratio < config.margin_min_support_ratio or
          t.total_score < config.margin_min_track_score))]
    forbidden_ids = {t.track_id for t in forbidden}
    ordered = [t for t in ordered if t.track_id not in forbidden_ids]
    result = SequenceResult()
    components: list[list[PaperEdgeTrack]] = []
    for track in ordered:
        if not components or (config.sequence_period_multiple_mode and track.center_y - components[-1][-1].center_y >
                config.max_sequence_gap_periods * local.at((track.center_y + components[-1][-1].center_y) / 2)):
            components.append([])
        components[-1].append(track)
    for component in components:
        # (previous node, has strong anchor) -> (value, full index path).
        states: list[dict] = []
        edges: dict[tuple[int, int], dict] = {}
        for j, track in enumerate(component):
            reward = unary_score(track, config)
            strong = track.candidate_class == "strong"
            current = {(-1, strong): (reward, (j,))}
            for i in range(j - 1, -1, -1):
                edge = transition_score(component[i], track, local, config, regularity)
                if edge is None:
                    if not config.sequence_period_multiple_mode:
                        result.rejected_transitions.append(geometry_pair_score(component[i], track, local, config))
                    continue
                edges[i, j] = edge
                for (previous, anchored), (score, path) in states[i].items():
                    change = 0.0
                    if previous >= 0 and config.sequence_spacing_change_weight:
                        old = edges[previous, i]
                        # Compare deviations relative to each local prior, so a
                        # measured gradual thickness change is not itself penalized.
                        a = old["gap"] / old["period_multiple"] / old["local_period"]
                        b = edge["gap"] / edge["period_multiple"] / edge["local_period"]
                        change = -config.sequence_spacing_change_weight * huber(
                            (b - a) / config.sequence_spacing_scale_ratio)
                    competition = 0.0
                    if previous >= 0 and regularity is not None:
                        competition = evaluate_middle_track_competition(
                            component[previous], component[i], track,
                            float(local.at(component[i].center_y)), config,
                            float(regularity.at(component[i].center_y)))
                        # Competition is a soft cost; image unary and direct
                        # geometry can still retain a genuine close layer.
                        competition_cost = max(0.0, competition["competition_score"])
                    else:
                        competition = None
                        competition_cost = 0.0
                    value = score + reward + edge["pair_score"] - competition_cost + change
                    key = (i if config.sequence_spacing_change_weight else 0, anchored or strong)
                    if key not in current or value > current[key][0]:
                        current[key] = (value, path + (j,))
            states.append(current)
        anchored_paths = [entry for state in states for (_, anchor), entry in state.items() if anchor]
        if not anchored_paths:
            continue
        score, path = max(anchored_paths, key=lambda entry: entry[0])
        if score <= 0:
            continue
        result.total_sequence_score += score
        result.selected.extend(component[i] for i in path)
        for k, (i, j) in enumerate(zip(path, path[1:])):
            edge = dict(edges[i, j])
            if regularity is not None and k >= 1:
                comp = evaluate_middle_track_competition(
                    component[path[k - 1]], component[i], component[j],
                    float(local.at(component[i].center_y)), config,
                    float(regularity.at(component[i].center_y)))
                edge["competition_penalty"] = max(0.0, comp["competition_score"])
                edge["competition_score"] = comp["competition_score"]
            else:
                edge["competition_penalty"] = 0.0
            if k and config.sequence_spacing_change_weight:
                old = edges[path[k - 1], i]
                a = old["gap"] / old["period_multiple"] / old["local_period"]
                b = edge["gap"] / edge["period_multiple"] / edge["local_period"]
                edge["spacing_change_score"] = -config.sequence_spacing_change_weight * huber(
                    (b - a) / config.sequence_spacing_scale_ratio)
            result.transitions.append(edge)
    selected_ids = {t.track_id for t in result.selected}
    # A marginal proposal that fills an ordinary observed layer gap on both
    # sides is evidence-driven recovery. Keep it even when its unary alone is
    # slightly below the global path margin; short competing textures do not
    # satisfy these two geometry checks.
    for candidate in ordered:
        if candidate.track_id in selected_ids or candidate.candidate_class != "marginal":
            continue
        left = next((t for t in reversed(result.selected) if t.center_y < candidate.center_y), None)
        right = next((t for t in result.selected if t.center_y > candidate.center_y), None)
        if left is None or right is None:
            continue
        r = float(regularity.at(candidate.center_y)) if regularity is not None else 0.5
        before = transition_score(left, candidate, local, config, regularity) or geometry_pair_score(left, candidate, local, config)
        after = transition_score(candidate, right, local, config, regularity) or geometry_pair_score(candidate, right, local, config)
        ordinary_gap = (before["gap_median"] > config.sequence_short_gap_ratio * local.at(candidate.center_y)
                and after["gap_median"] > config.sequence_short_gap_ratio * local.at(candidate.center_y)
                and (regularity is None or candidate.support_ratio >= config.min_track_support_ratio * .75))
        # A low-support candidate may bridge a deliberately corrupted strip
        # set; it still needs positive, non-crossing observations on both
        # sides. High-support short competitors remain suppressed.
        regular_relaxed = regularity is not None and r >= config.regularity_regular_threshold
        irregular_guard = regularity is not None and r <= config.regularity_irregular_threshold
        low_support_bridge = regularity is None and candidate.support_ratio < config.min_track_support_ratio * .6
        if ((ordinary_gap or (regular_relaxed and candidate.support_ratio >= config.min_track_support_ratio * config.regular_recovery_support_relaxation)
                or low_support_bridge)
                and before["non_crossing_ratio"] >= 1.0 and after["non_crossing_ratio"] >= 1.0
                and candidate.support_ratio >= config.min_track_support_ratio * (0.75 if irregular_guard else .5)
                and (not irregular_guard or candidate.total_score >= config.min_track_score)):
            candidate.sequence_decision = "rescued"
            candidate.reason = "sequence_rescued_geometry"
            result.rescued.append(candidate)
            result.selected.append(candidate)
            selected_ids.add(candidate.track_id)
    result.selected.sort(key=lambda t: t.center_y)
    for track in ordered:
        left = next((t for t in reversed(result.selected) if t.center_y < track.center_y), None)
        right = next((t for t in result.selected if t.center_y > track.center_y), None)
        incoming = transition_score(left, track, local, config, regularity) if left else None
        outgoing = transition_score(track, right, local, config, regularity) if right else None
        gaps = [abs(track.center_y - t.center_y) for t in (left, right) if t is not None]
        selected = track.track_id in selected_ids
        if selected:
            rescue = track.candidate_class == "marginal" or track.source == "recovered"
            track.sequence_decision = "rescued" if rescue else "selected"
            track.reason = "sequence_rescued" if rescue else None
            if rescue:
                result.rescued.append(track)
        else:
            track.sequence_decision = "suppressed"
            narrow = any(e.get("short_gap_warning", False) for e in (incoming, outgoing) if e)
            crossing = any(geometry_pair_score(a, b, local, config)["hard_reject"] for a, b in
                           ((left, track), (track, right)) if a is not None and b is not None)
            track.reason = ("sequence_crossing_conflict" if crossing and not config.sequence_period_multiple_mode else
                "sequence_suppressed_short_gap" if narrow or (config.sequence_period_multiple_mode and gaps and min(gaps) <
                config.sequence_min_spacing_ratio * local.at(track.center_y)) else "sequence_spacing_conflict")
            result.suppressed.append(track)
        # Counterfactual local insertion/removal delta; not a substitute for the
        # global DP score. Useful for explaining why the two adjacent edges lose.
        bridge = transition_score(left, right, local, config, regularity) if left and right else None
        delta = unary_score(track, config) + sum(e["pair_score"] for e in (incoming, outgoing) if e)
        if bridge:
            delta -= bridge["pair_score"]
        result.decisions.append({"track_id": track.track_id, "center_y": track.center_y,
            "candidate_class": track.candidate_class, "total_score": track.total_score,
            "classification_reason": track.classification_reason,
            "unary_score": unary_score(track, config),
            "incoming_gap": track.center_y - left.center_y if left else None,
            "outgoing_gap": right.center_y - track.center_y if right else None,
            "incoming": incoming, "outgoing": outgoing,
            "local_period": local.at(track.center_y),
            "spacing_error": [e["spacing_error"] for e in (incoming, outgoing) if e],
            "neighbor_selected_track_ids": [t.track_id for t in (left, right) if t],
            "raw_mask_state": track.raw_mask_state, "curve_geometry": track.curve_geometry,
            "local_insertion_delta_without_second_order": delta,
            "decision": track.sequence_decision, "reason": track.reason or "sequence_selected"})
    for track in forbidden:
        track.sequence_decision = "suppressed"
        track.reason = "raw_mask_invalid" if track.raw_mask_state == "INVALID" else "raw_margin_marginal_forbidden"
        result.suppressed.append(track)
        result.decisions.append({"track_id": track.track_id, "center_y": track.center_y,
            "unary_score": unary_score(track, config), "candidate_class": track.candidate_class,
            "decision": "suppressed", "reason": track.reason, "raw_mask_state": track.raw_mask_state})
    return result
