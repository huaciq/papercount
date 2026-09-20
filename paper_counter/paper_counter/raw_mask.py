"""Three-state permissions derived exclusively from unnormalized grayscale."""
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
import cv2
import numpy as np
from .config import CounterConfig


class StackState(IntEnum):
    INVALID = 0
    MARGIN = 1
    CORE = 2


def intervals(mask: np.ndarray) -> list[list[int]]:
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return [[int(a), int(b)] for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))]


@dataclass
class RawStackMask:
    states: np.ndarray
    evidence: np.ndarray
    statistics: dict
    thresholds: dict
    offset: int = 0

    def state_at(self, y: float) -> int:
        i = int(round(y)) - self.offset
        return int(self.states[i]) if 0 <= i < len(self.states) else int(StackState.INVALID)

    def evidence_at(self, y: float) -> bool:
        i = int(round(y)) - self.offset
        return bool(self.evidence[i]) if 0 <= i < len(self.evidence) else False

    def to_dict(self) -> dict:
        return {"coordinate_offset": self.offset, "state_codes": {s.name: int(s) for s in StackState},
            "intervals": {s.name: [[a + self.offset, b + self.offset] for a, b in intervals(self.states == s)]
                          for s in StackState},
            "row_counts": {s.name: int(np.sum(self.states == s)) for s in StackState},
            "thresholds": self.thresholds, "source": "raw_gray_before_normalization"}


def detect_raw_stack_mask(gray: np.ndarray, config: CounterConfig, spacing_scale: float | None = None,
                          offset: int = 0) -> RawStackMask:
    """Build a conservative mask from RAW gray values.

    ``INVALID`` is reserved for high-confidence black runs connected to the
    top/bottom image border.  Internal dark paper seams remain usable rows.
    This keeps the mask a background suppressor rather than a stack locator.
    """
    scale = spacing_scale or (config.min_period + config.max_period) / 2
    height = len(gray)
    med, low, high, texture = [np.empty(height, dtype=np.float32) for _ in range(4)]
    raw_gradient = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    for start in range(0, height, config.debug_chunk_rows):
        end = min(height, start + config.debug_chunk_rows)
        low[start:end], med[start:end], high[start:end] = np.percentile(gray[start:end], [10, 50, 90], axis=1)
        texture[start:end] = np.median(raw_gradient[start:end], axis=1)
    del raw_gradient
    def threshold(values, k):
        background = values[values <= np.percentile(values, config.raw_mask_background_percentile)]
        base = float(np.median(background))
        mad = float(np.median(np.abs(background - base)))
        contrast = float(np.percentile(values, 95) - base)
        noise = max(1.4826 * mad, config.raw_mask_relative_floor * contrast, config.epsilon)
        return base + k * noise, base, mad
    brightness_cut, black, brightness_mad = threshold(med, config.mask_high_confidence_black_k)
    texture_cut, texture_base, texture_mad = threshold(texture, config.raw_mask_texture_k)
    # Median/P90/texture conjunction deliberately rejects isolated bright
    # pixels and low-contrast black seams.  The thresholds remain native-scale
    # robust values, so uint8/uint16 captures share the same behavior.
    black = (med <= brightness_cut) & (high <= brightness_cut) & (texture <= texture_cut)
    active = ~black
    states = np.full(height, StackState.CORE, dtype=np.uint8)
    initial_top, initial_bottom = 0, height
    if not config.raw_stack_mask_enabled:
        active[:] = True
    elif config.mask_edge_growth_enabled:
        run = max(1, int(config.mask_min_black_run))
        top = 0
        while top < height and black[top]:
            top += 1
        bottom = height
        while bottom > 0 and black[bottom - 1]:
            bottom -= 1
        # A tiny isolated run at an edge is texture, not background.  Only a
        # run meeting the minimum length gets INVALID permission.
        if top < run:
            top = 0
        if height - bottom < run:
            bottom = height
        initial_top, initial_bottom = top, bottom
        states[:top] = StackState.INVALID
        states[bottom:] = StackState.INVALID
        margin = int(np.ceil(config.raw_mask_margin_spacing_ratio * scale))
        pad = max(1, int(np.ceil(config.raw_mask_boundary_padding_ratio * scale)))
        if top < bottom:
            states[max(top - margin, 0):min(top + pad, height)] = StackState.MARGIN
            states[max(bottom - pad, 0):min(bottom + margin, height)] = StackState.MARGIN
            states[:top] = StackState.INVALID
            states[bottom:] = StackState.INVALID
    else:
        # Legacy active evidence is retained for callers that explicitly turn
        # edge growth off, but no internal component is made INVALID.
        states[:] = StackState.CORE
    return RawStackMask(states, active, {"row_median": med, "row_percentile_10": low,
        "row_percentile_90": high, "raw_row_texture": texture},
        {"brightness": brightness_cut, "texture": texture_cut, "background_fraction": float(np.mean(black)),
         "brightness_background_mad": brightness_mad, "texture_background": texture_base,
         "texture_background_mad": texture_mad, "bootstrap_spacing_scale": scale,
         "initial_top": int(initial_top + offset), "initial_bottom": int(initial_bottom + offset),
         "edge_connected_background": True}, offset)


def refine_stack_mask_with_strong_tracks(mask: RawStackMask, tracks: list, config: CounterConfig,
                                         spacing_scale: float | None = None) -> dict:
    """One-pass boundary expansion supported only by strong image tracks."""
    initial_abs_top = int(mask.thresholds.get("initial_top", mask.offset))
    initial_abs_bottom = int(mask.thresholds.get("initial_bottom", mask.offset + len(mask.states)))
    initial = {"initial_top": initial_abs_top, "initial_bottom": initial_abs_bottom,
               "refined_top": initial_abs_top, "refined_bottom": initial_abs_bottom,
               "mask_expansion_reason": [], "supporting_track_ids": []}
    if not config.mask_refine_with_strong_tracks or not tracks:
        return initial
    scale = float(spacing_scale or config.min_period)
    max_expand = max(1, int(round(config.mask_refine_max_spacing_ratio * scale)))
    valid = [t for t in tracks if getattr(t, "candidate_class", "") == "strong"
             and getattr(t, "support_ratio", 0) >= config.min_track_support_ratio
             and getattr(t, "gradient_score", 0) >= config.min_track_gradient_score
             and getattr(t, "raw_evidence_ratio", 0) >= config.raw_mask_min_track_evidence_ratio]
    top, bottom = initial_abs_top - mask.offset, initial_abs_bottom - mask.offset
    for t in valid:
        y = int(round(t.center_y)) - mask.offset
        if y < top and top - y <= max_expand:
            top = max(0, y - 1); initial["supporting_track_ids"].append(t.track_id)
            initial["mask_expansion_reason"].append("strong_track_above_initial_boundary")
        if y >= bottom and y - bottom <= max_expand:
            bottom = min(len(mask.states), y + 2); initial["supporting_track_ids"].append(t.track_id)
            initial["mask_expansion_reason"].append("strong_track_below_initial_boundary")
    if top != initial_abs_top - mask.offset or bottom != initial_abs_bottom - mask.offset:
        mask.states[top:bottom] = np.where(mask.states[top:bottom] == StackState.INVALID,
                                            StackState.MARGIN, mask.states[top:bottom])
        initial["refined_top"], initial["refined_bottom"] = top + mask.offset, bottom + mask.offset
    return initial


def filter_mask_peaks(candidates: list, mask: RawStackMask) -> list:
    """INVALID observations never reach horizontal association."""
    return [[p for p in strip if mask.state_at(p.y) != StackState.INVALID] for strip in candidates]


def track_mask_permission(track, mask: RawStackMask, config: CounterConfig, recovery: bool = False) -> bool:
    """All observed positions must be valid; margin needs direct raw evidence."""
    states = [mask.state_at(y) for y in track.strip_positions.values()]
    track.raw_valid_ratio = sum(s != StackState.INVALID for s in states) / max(1, len(states))
    track.raw_evidence_ratio = sum(mask.evidence_at(y) for y in track.strip_positions.values()) / max(1, len(states))
    state = min(states, default=StackState.INVALID)
    track.raw_mask_state = StackState(state).name
    reason = None
    if state == StackState.INVALID:
        reason = "raw_mask_invalid"
    elif state == StackState.MARGIN:
        if recovery and not config.margin_allow_weak_recovery:
            reason = "raw_margin_recovery_forbidden"
        elif track.raw_evidence_ratio < max(config.raw_mask_min_track_evidence_ratio,
                                            config.margin_recovery_negative_support if recovery else 0.0):
            reason = "raw_margin_insufficient_evidence"
        elif track.candidate_class != "strong" and (not config.margin_allow_marginal or
                track.gradient_score < config.margin_min_gradient_score or track.support_ratio < config.margin_min_support_ratio or
                track.total_score < config.margin_min_track_score):
            reason = "raw_margin_marginal_forbidden"
    if reason:
        track.candidate_class = "garbage"
        track.classification_reason = track.reason = reason
    return reason is None
