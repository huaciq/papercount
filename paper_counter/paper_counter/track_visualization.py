"""Track-specific plots in original-image coordinates."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_track_identity_profiles(identity: dict, identity_debug: dict, output: Path) -> None:
    """Plot per-image robust template and suspicious track-aligned profiles."""
    profiles = identity.get("profiles", {})
    template = identity.get("template")
    relative = np.asarray(identity.get("relative_y", []), dtype=float)
    if template is None or not len(relative):
        return
    decisions = {row["track_id"]: row for row in identity_debug.get("features", [])}
    suspicious = [track_id for track_id, row in decisions.items()
                  if row.get("needs_final_verification")]
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True, constrained_layout=True)
    for ax, name, label in zip(axes, ("raw", "normalized", "negative", "positive"),
                               ("RAW gray", "Normalized intensity", "Negative gradient", "Positive gradient")):
        ax.plot(relative, getattr(template, name), color="black", lw=2, label="per-image median template")
        for track_id in suspicious[:20]:
            if track_id not in profiles:
                continue
            row = decisions[track_id]
            color = "magenta" if row.get("proposed_decision", "").startswith("REJECT") else "orange"
            ax.plot(relative, getattr(profiles[track_id], name), color=color, alpha=.35, lw=.8)
        ax.axvline(0, color="steelblue", ls=":", lw=.8)
        ax.set_ylabel(label)
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("Relative Y / local spacing")
    fig.suptitle("Track-aligned boundary identity profiles (black template; suspicious tracks overlaid)")
    fig.savefig(output / "track_identity_profiles.png", dpi=145)
    plt.close(fig)


def plot_vertical_sequence(sequence: dict, output: Path) -> None:
    """Candidate unary decisions and selected pairwise spacing on shared Y axis."""
    fig, axes = plt.subplots(2, 1, figsize=(17, 8), sharex=True, constrained_layout=True)
    rows = sequence.get("track_decisions", [])
    for decision, color in (("selected", "green"), ("rescued", "goldenrod"), ("suppressed", "red")):
        group = [r for r in rows if r["decision"] == decision]
        if group:
            axes[0].scatter([r["center_y"] for r in group], [r["unary_score"] for r in group],
                            color=color, s=18, label=decision)
    edges = sequence.get("transitions", [])
    if edges:
        ys = [(e["from_y"] + e["to_y"]) / 2 for e in edges]
        axes[1].plot(ys, [e.get("gap_median", e["gap"]) for e in edges], ".-", lw=0.8, label="Selected neighbor gap median")
        axes[1].plot(ys, [e["local_period"] for e in edges], "--", label="Local typical spacing scale")
        for key, color, marker, label in (("short_gap_warning", "red", "x", "Short-gap competition"),
                ("large_gap_anomaly", "purple", "^", "Large gap: anomaly, NOT missing paper")):
            chosen = [(y, e) for y, e in zip(ys, edges) if e.get(key)]
            if chosen:
                axes[1].scatter([y for y, e in chosen], [e["gap_median"] for y, e in chosen],
                                c=color, marker=marker, s=30, label=label)
    axes[0].set(ylabel="Unary evidence score", title="Vertical sequence: observed candidates and decisions")
    axes[1].set(xlabel="Detection image Y (px)", ylabel="Gap / period (px)")
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        ax.grid(alpha=0.15)
    fig.savefig(output / "vertical_sequence.png", dpi=140)
    plt.close(fig)


def plot_regularity(regularity: dict, output: Path) -> None:
    """Render the continuous local regularity map and its component signals."""
    samples = regularity.get("samples", []) if regularity else []
    fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True, constrained_layout=True)
    if samples:
        y = [s["y"] for s in samples]
        axes[0].plot(y, [s["final_regularity"] for s in samples], color="black", label="R(y)")
        axes[0].plot(y, [s["spacing_regularity"] for s in samples], "--", label="spacing")
        axes[0].plot(y, [s["acf_score"] for s in samples], "--", label="local ACF")
        axes[0].plot(y, [s["track_regularity"] for s in samples], "--", label="track quality")
        axes[1].plot(y, [s["local_spacing"] for s in samples], label="local spacing")
        axes[1].plot(y, [s["spacing_mad"] for s in samples], label="spacing MAD")
        for threshold, color, label in ((regularity.get("regular_threshold", .7), "green", "REGULAR"),
                                         (regularity.get("irregular_threshold", .4), "red", "IRREGULAR")):
            axes[0].axhline(threshold, color=color, ls=":", label=label)
    axes[0].set_ylim(-.02, 1.02)
    axes[0].set_ylabel("Regularity")
    axes[1].set(xlabel="Detection image Y (px)", ylabel="Spacing / MAD")
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(output / "regularity_map.png", dpi=150)
    plt.close(fig)


def plot_hybrid_sequence(sequence: dict, regularity: dict, output: Path) -> None:
    """Show final sequence decisions over the regular/mixed/irregular map."""
    fig, ax = plt.subplots(figsize=(15, 4), constrained_layout=True)
    samples = regularity.get("samples", []) if regularity else []
    if samples:
        y = np.asarray([s["y"] for s in samples]); r = np.asarray([s["final_regularity"] for s in samples])
        ax.fill_between(y, 0, r, where=r >= regularity.get("regular_threshold", .7), color="green", alpha=.12)
        ax.fill_between(y, 0, r, where=r <= regularity.get("irregular_threshold", .4), color="red", alpha=.12)
        ax.plot(y, r, color="steelblue", lw=1, label="regularity")
    for decision, color in (("selected", "green"), ("rescued", "goldenrod"), ("suppressed", "red")):
        rows = [x for x in sequence.get("track_decisions", []) if x.get("decision") == decision]
        if rows:
            ax.scatter([x["center_y"] for x in rows], [0.95 if decision != "suppressed" else .1] * len(rows),
                       c=color, s=18, label=decision)
    for gap in sequence.get("unresolved_regular_gaps", []):
        ax.axvline(gap.get("midpoint", gap.get("left", 0)), color="blue", ls=":")
    ax.set(xlabel="Detection image Y (px)", ylabel="R / decision", ylim=(0, 1.05), title="Regular-first hybrid sequence")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(output / "hybrid_sequence.png", dpi=150)
    plt.close(fig)


def plot_raw_stack_mask(raw_preview, mask, output):
    """Audit RAW statistics and three-state permissions before normalization."""
    from .raw_mask import StackState, intervals
    fig, axes = plt.subplots(1, 4, figsize=(15, 9), sharey=True, constrained_layout=True,
                             gridspec_kw={"width_ratios": [2, 1, 1, .5]})
    offset, end = mask.offset, mask.offset + len(mask.states)
    axes[0].imshow(raw_preview, cmap="gray", extent=[0, raw_preview.shape[1], end, offset], aspect="auto")
    colors = {StackState.CORE: "cyan", StackState.MARGIN: "orange", StackState.INVALID: "gray"}
    for state, color in colors.items():
        for a, b in intervals(mask.states == state):
            axes[0].axhspan(a + offset, b + offset, color=color, alpha=.15)
            axes[3].axhspan(a + offset, b + offset, color=color)
    y = np.arange(offset, end)
    for key, label in (("row_median", "median"), ("row_percentile_10", "p10"), ("row_percentile_90", "p90")):
        axes[1].plot(mask.statistics[key], y, lw=.7, label=label)
    axes[1].axvline(mask.thresholds["brightness"], ls="--", color="red", label="adaptive cutoff")
    axes[2].plot(mask.statistics["raw_row_texture"], y, lw=.7, label="Raw Sobel Y texture")
    axes[2].axvline(mask.thresholds["texture"], ls="--", color="red", label="adaptive cutoff")
    for ax in axes[1:3]:
        ax.legend(fontsize=7)
    axes[0].set(title="RAW gray + permission mask", ylabel="Detection image Y")
    axes[1].set_title("Native intensity statistics")
    axes[2].set_title("RAW texture (not normalized)")
    axes[3].set(title="CORE: cyan\nMARGIN: orange\nINVALID: gray", xticks=[])
    axes[0].set_ylim(end, offset)
    fig.savefig(output / "raw_stack_mask.png", dpi=130)
    plt.close(fig)


def plot_track_geometry(pool, output):
    """Dots are observations; dashed curve segments are bounded interpolation."""
    fig, ax = plt.subplots(figsize=(10, 13), constrained_layout=True)
    colors = {"selected": "green", "rescued": "goldenrod", "suppressed": "red"}
    labels = set()
    for track in pool:
        geometry = track["curve_geometry"]
        if not geometry:
            continue
        decision = track.get("sequence_decision", "suppressed")
        color = colors.get(decision, "gray")
        positions = np.array([np.nan if y is None else y for y in geometry["positions"]])
        observed = np.array(geometry["support_mask"])
        ax.plot(np.arange(len(positions)), positions, "--", lw=.6, alpha=.65, c=color)
        ax.scatter(np.flatnonzero(observed), positions[observed], s=4, c=color,
                   label=decision if decision not in labels else None)
        labels.add(decision)
    ax.invert_yaxis()
    ax.set(xlabel="Strip index", ylabel="Detection image Y", title="Full usable track geometry; dots = real observations")
    if labels:
        ax.legend()
    fig.savefig(output / "track_geometry.png", dpi=140)
    plt.close(fig)


def plot_strip_candidates(candidates: list, statistics: list, profiles: np.ndarray,
                          offset: int, output: Path) -> None:
    """Show independent strip curves, adaptive thresholds and candidate classes."""
    fig, axes = plt.subplots(len(profiles), 1, figsize=(16, max(5, 1.8 * len(profiles))),
                              sharex=True, squeeze=False, constrained_layout=True)
    y = np.arange(profiles.shape[1]) + offset
    for i, ax in enumerate(axes[:, 0]):
        ax.plot(y, profiles[i], lw=0.6, label="Strip response")
        ax.plot(y, statistics[i]["strong_threshold"], lw=0.6, ls="--", color="orange", label="Local prominence cutoff (not height)")
        for kind, color in (("strong", "green"), ("weak", "goldenrod")):
            selected = [p for p in candidates[i] if p.strength_type == kind and not p.reason]
            ax.scatter([p.y for p in selected], [p.height for p in selected], s=9, color=color, label=kind)
        ax.set_ylabel(f"Strip {i + 1}")
    axes[0, 0].legend(loc="upper right", ncol=4, fontsize=7)
    axes[-1, 0].set_xlabel("Original image Y (px)")
    fig.savefig(output / "strip_peak_candidates.png", dpi=120)
    plt.close(fig)


def plot_tracks(tracks: list, rejected: list, local_period: object, initial: float | None,
                output: Path) -> None:
    """Show observed strip paths and smoothed local-period estimates."""
    fig, ax = plt.subplots(figsize=(10, 12), constrained_layout=True)
    for group, color, label in ((rejected, "red", "rejected"),
                                ([t for t in tracks if t.source == "detected" and t.sequence_decision != "rescued"], "green", "selected"),
                                ([t for t in tracks if t.source == "recovered" or t.sequence_decision == "rescued"], "goldenrod", "rescued")):
        for i, track in enumerate(group):
            points = sorted(track.strip_positions.items())
            ax.plot([p[0] for p in points], [p[1] for p in points], ".-", lw=0.6,
                    markersize=2, color=color, alpha=0.7, label=label if i == 0 else None)
    ax.invert_yaxis()
    ax.set(xlabel="Strip index", ylabel="Original image Y", title="Observed paper-edge tracks")
    if tracks or rejected:
        ax.legend()
    fig.savefig(output / "edge_tracks.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
    if local_period is not None and len(local_period.y):
        ax.plot(local_period.y, local_period.values, label="Smoothed local period")
        ax.fill_between(local_period.y, local_period.values - local_period.mad,
                        local_period.values + local_period.mad, alpha=0.15, label="Local gap MAD")
    if initial is not None:
        ax.axhline(initial, ls="--", color="red", label="Initial global period")
    ax.set(xlabel="Original image Y", ylabel="Period (px)", title="Period prior (never directly determines count)")
    if initial is not None:
        ax.legend()
    fig.savefig(output / "local_period.png", dpi=140)
    plt.close(fig)
