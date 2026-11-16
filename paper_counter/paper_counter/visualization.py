from __future__ import annotations

"""Diagnostic plots and machine-readable intermediate signals."""

import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import CounterConfig
from .coordinates import build_render_metadata
from .models import PaperCountResult
from .preprocessing import display_preview


def write_png(path: Path, image: np.ndarray) -> None:
    """Write PNG through a Unicode-safe path."""
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise OSError(f"Cannot encode {path}")
    buffer.tofile(path)


def display_uint8(array: np.ndarray) -> np.ndarray:
    """Convert diagnostic intensity to display range without touching detection."""
    if array.dtype == np.uint8:
        return array
    return np.clip(array * (255 / 65535 if array.dtype == np.uint16 else 1), 0, 255).astype(np.uint8)


def save_overlay(result: PaperCountResult, directory: Path, config: CounterConfig) -> dict:
    """Draw accepted/repaired/rejected edges on a bounded-size ROI preview."""
    roi = result.artifacts["roi"]
    canvas = display_uint8(display_preview(roi, config.display_max_height, config.display_max_width))
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    elif canvas.shape[2] == 4:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
    plain = canvas.copy() if result.debug_info.get('algorithm')=='narrow_regions' else None
    offset = result.debug_info["roi"]["y1"]
    scale_y = canvas.shape[0] / roi.shape[0]
    if result.debug_info.get('algorithm')=='narrow_regions':
        scale_x=canvas.shape[1]/roi.shape[1]
        for box in result.region_boxes:
            x1,y1,x2,y2=box['detection_xyxy']
            color=(0,255,0) if box['number']%2 else (255,200,0)
            if box.get('thickness_inferred'):color=(0,180,255)
            a=(int(round(x1*scale_x)),max(0,int(round(y1*scale_y))))
            b=(min(canvas.shape[1]-1,int(round(x2*scale_x))-1),min(canvas.shape[0]-1,int(round(y2*scale_y))-1))
            cv2.rectangle(canvas,a,b,color,1)
            cv2.putText(canvas,str(box['number']),(3,min(canvas.shape[0]-2,a[1]+10)),cv2.FONT_HERSHEY_SIMPLEX,.3,color,1)
    elif result.debug_info.get("algorithm") == "multi_strip_tracks":
        strips = result.debug_info["strip_bounds_roi_x"]
        scale_x = canvas.shape[1] / roi.shape[1]
        red_tracks = ([t for t in result.rejected_tracks if t.sequence_decision == "suppressed"]
                      if config.vertical_sequence_enabled else result.rejected_tracks)
        for tracks, color in ((red_tracks, (0, 0, 255)),
                ([t for t in result.paper_edge_tracks if t.source == "detected" and t.sequence_decision != "rescued"], (0, 255, 0)),
                ([t for t in result.paper_edge_tracks if t.source == "recovered" or t.sequence_decision == "rescued"], (0, 255, 255))):
            for track in tracks:
                points = np.array([[(strips[s][0] + strips[s][1]) * 0.5 * scale_x,
                                    (y - offset) * scale_y] for s, y in sorted(track.strip_positions.items())], dtype=np.int32)
                if len(points) > 1:
                    cv2.polylines(canvas, [points], False, color, 1)
                elif len(points):
                    if config.strip_sequence_enabled:
                        s, y = next(iter(track.strip_positions.items()))
                        a, b = strips[s]
                        row = int(round((y-offset)*scale_y))
                        cv2.line(canvas, (int(a*scale_x), row), (int(b*scale_x), row), color, 1)
                    else:
                        cv2.circle(canvas, tuple(points[0]), 1, color, -1)
        raw_mask = result.debug_info.get("raw_stack_mask", {})
        for name, color in (("CORE", (255, 255, 0)), ("MARGIN", (0, 165, 255))):
            for limits in raw_mask.get("intervals", {}).get(name, []):
                for boundary in limits:
                    y = int(round((boundary - offset) * scale_y))
                    if 0 <= y < canvas.shape[0]:
                        cv2.line(canvas, (0, y), (min(12, canvas.shape[1] - 1), y), color, 2)
                        cv2.line(canvas, (max(0, canvas.shape[1] - 13), y), (canvas.shape[1] - 1, y), color, 2)
    else:
        raise ValueError('Only tracks and regions results can be rendered')
    # Put metadata outside the image, so narrow line-scan input stays legible.
    title_height = 116
    narrow=result.debug_info.get('algorithm')=='narrow_regions'
    compare_width=2*canvas.shape[1] if plain is not None else canvas.shape[1]
    header = np.full((title_height, max(225,compare_width) if narrow else max(760, canvas.shape[1]), 3), 28, dtype=np.uint8)
    period = f"{result.period:.2f}px" if result.period is not None else "unavailable"
    local = result.debug_info.get("median_local_period")
    local_label = f"{local:.2f}px" if local is not None else "unavailable"
    lines = [f"Count: {result.count}  Initial global period: {period}  Median local: {local_label}",
             f"Gradient polarity: {result.gradient_polarity}  Confidence: {result.confidence:.3f}",
             "Green: selected | Yellow: rescued | Red: sequence suppressed" if config.vertical_sequence_enabled else "Green: detected | Yellow: recovered | Red: rejected",
             "Y: detection image | Edge ticks: cyan CORE, orange MARGIN"]
    if result.debug_info.get('algorithm')=='narrow_regions':
        lines=[f'Complete units: {result.count} | {roi.shape[1]}px',
               f"Unit: {result.debug_info['region_detection']['structure']}",
               'Left: boxes | Right: mirrored original', 'Orange: thickness inferred']
    for i, line in enumerate(lines):
        cv2.putText(header, line, (10, 23 + i * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.36 if narrow else 0.48, (235, 235, 235), 1, cv2.LINE_AA)
    body = np.full((canvas.shape[0], header.shape[1], 3), 28, dtype=np.uint8)
    body[:, :canvas.shape[1]] = canvas
    if plain is not None:
        # Reflect only the comparison panel: adjacent edges show the same
        # source columns, while detection coordinates and vertical alignment stay intact.
        body[:,canvas.shape[1]:2*canvas.shape[1]]=plain[:,::-1]
    overlay = np.vstack((header, body))
    write_png(directory / "result_overlay.png", overlay)
    metadata=build_render_metadata(result.debug_info, canvas.shape[1], canvas.shape[0],
                                  overlay.shape[1], overlay.shape[0], title_height)
    if plain is not None:
        metadata['comparison_panel']=dict(offset_x=canvas.shape[1],offset_y=title_height,
            width=canvas.shape[1],height=canvas.shape[0],horizontal_mirror=True,gap_px=0)
    return metadata


def save_debug(result: PaperCountResult, output: str | Path, config: CounterConfig) -> None:
    from .exports import save_recipe_metadata
    save_recipe_metadata(result,output)
    if result.debug_info.get('algorithm')=='narrow_regions':
        from .region_output import save_regions
        return save_regions(result,output,config,full=True)
    """Save plots, stage previews, complete candidate records and numeric profiles."""
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    render_metadata = save_overlay(result, directory, config)
    from .detection_coordinates import export_detection_coordinates
    coordinates=export_detection_coordinates(result,directory,render_metadata)
    result.debug_info["render_metadata"] = render_metadata
    with (directory / "render_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(render_metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
    for name, preview in result.artifacts["previews"].items():
        write_png(directory / f"{name}.png", display_uint8(preview))
    data = result.artifacts["polarities"]
    selected = data[result.gradient_polarity]
    profile = selected["global"]
    offset = result.debug_info["roi"]["y1"]
    y = np.arange(len(profile)) + offset
    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
    ax.plot(y, profile, lw=0.8, label="Global median profile")
    for peaks, color, label, marker in ((result.detected_peaks, "green", "Detected", "o"),
                                      (result.repaired_peaks, "goldenrod", "Repaired", "^"),
                                      (result.rejected_peaks, "red", "Rejected", "x")):
        if peaks:
            xs = [p.y for p in peaks]
            ax.scatter(xs, [profile[x - offset] for x in xs], s=30, c=color, marker=marker, label=label)
    threshold = result.debug_info.get("prominence_threshold")
    mode = f"prominence >= {threshold:.3g}" if threshold is not None else "Track centers; per-strip local thresholds (see strip_peak_candidates.png)"
    ax.set(xlabel="Original image Y (px)", ylabel="Normalized signed response",
           title=f"{result.gradient_polarity} | Count {result.count} | {mode}")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(directory / "profile.png", dpi=160)
    # 编号点是轨迹中心在全局曲线上的投影，并非该曲线的寻峰结果。
    numbered,number_ax=plt.subplots(figsize=(17,6),constrained_layout=True)
    number_ax.plot(y,profile,lw=.8,color='steelblue',label='Global median response')
    for track in coordinates['tracks']:
        value=track['primary_global_response']
        if value is None:continue
        center=track['center_detection_y'];number=track['number']
        color='goldenrod' if track['source']=='recovered' or track['decision']=='rescued' else 'green'
        number_ax.scatter([center],[value],c=color,s=18)
        number_ax.annotate(str(number),(center,value),xytext=(0,8 if number%2 else -14),
            textcoords='offset points',ha='center',fontsize=6)
    number_ax.set(xlabel='Detection image Y (px)',ylabel='Normalized gradient response',
        title='Numbered final track centers projected onto global response (NOT global peak detections)')
    number_ax.grid(alpha=.2);number_ax.legend()
    numbered.savefig(directory/'profile_numbered.png',dpi=160)
    plt.close(numbered)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for polarity, d in data.items():
        corr = d["assessment"]["acf"]
        ax.plot(np.arange(len(corr)), corr, label=f"{polarity}: score={d['assessment']['score']:.3f}")
    ax.axvspan(config.min_period, config.max_period, alpha=0.07, color="green", label="Search range")
    if result.period is not None:
        ax.axvline(result.period, color="red", ls="--", label=f"Chosen period: {result.period:.1f}")
    ax.set(xlabel="Lag (px)", ylabel="ACF / ACF(0)", title="Autocorrelation: both gradient polarities")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(directory / "autocorrelation.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(config.num_strips, 2, figsize=(16, max(5, 1.7 * config.num_strips)),
                              sharex=True, squeeze=False, constrained_layout=True)
    for col, (polarity, d) in enumerate(data.items()):
        for row, signal in enumerate(d["profiles"]):
            axes[row, col].plot(y, signal, lw=0.6)
            axes[row, col].set_ylabel(f"Strip {row + 1}")
            axes[row, col].grid(alpha=0.2)
        axes[0, col].set_title(polarity)
        axes[-1, col].set_xlabel("Original image Y (px)")
    fig.savefig(directory / "strip_profiles.png", dpi=140)
    plt.close(fig)
    arrays = {"original_y": y}
    for polarity, d in data.items():
        for key in ("raw_profiles", "profiles", "global"):
            arrays[f"{polarity}_{key}"] = d[key]
        arrays[f"{polarity}_acf"] = d["assessment"]["acf"]
    if result.debug_info.get("algorithm") == "multi_strip_tracks":
        from .track_visualization import plot_strip_candidates, plot_tracks, plot_vertical_sequence, plot_regularity, plot_hybrid_sequence
        if result.artifacts["local_statistics"]:
            plot_strip_candidates(result.artifacts["strip_candidates"], result.artifacts["local_statistics"],
                                  selected["profiles"], offset, directory)
            for key in result.artifacts["local_statistics"][0]:
                arrays["local_" + key] = np.stack([s[key] for s in result.artifacts["local_statistics"]])
        else:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No reliable period / no strip candidates", ha="center")
            fig.savefig(directory / "strip_peak_candidates.png")
            plt.close(fig)
        plot_tracks(result.paper_edge_tracks, result.rejected_tracks, result.artifacts["local_period"], result.period, directory)
        plot_vertical_sequence(result.debug_info.get("vertical_sequence", {}), directory)
        plot_regularity(result.debug_info.get("regularity", {}), directory)
        hybrid_sequence = dict(result.debug_info.get("vertical_sequence", {}))
        hybrid_sequence["unresolved_regular_gaps"] = result.debug_info.get("unresolved_regular_gaps", [])
        plot_hybrid_sequence(hybrid_sequence, result.debug_info.get("regularity", {}), directory)
        if "raw_stack_mask" in result.artifacts:
            from .track_visualization import plot_raw_stack_mask, plot_track_geometry
            mask = result.artifacts["raw_stack_mask"]
            plot_raw_stack_mask(result.artifacts["raw_gray_preview"], mask, directory)
            plot_track_geometry(result.debug_info.get("track_candidate_pool", []), directory)
            arrays["raw_mask_states"] = mask.states
            arrays["raw_mask_y"] = np.arange(len(mask.states)) + mask.offset
            arrays.update(mask.statistics)
        identity = result.artifacts.get("track_identity", {})
        if identity.get("profiles"):
            from .track_visualization import plot_track_identity_profiles
            identity_debug = result.debug_info.get("track_identity", {})
            plot_track_identity_profiles(identity, identity_debug, directory)
            track_ids = list(identity["profiles"])
            arrays["identity_track_ids"] = np.asarray(track_ids, dtype=str)
            arrays["identity_relative_y"] = np.asarray(identity["relative_y"])
            for name in ("raw", "normalized", "negative", "positive"):
                arrays[f"identity_{name}_profiles"] = np.stack(
                    [getattr(identity["profiles"][track_id], name) for track_id in track_ids])
                arrays[f"identity_template_{name}"] = getattr(identity["template"], name)
        energy = result.artifacts.get("active_energy")
        if energy is not None:
            arrays["active_row_energy"] = energy
            arrays["active_energy_original_y"] = np.arange(len(energy)) + result.artifacts["active_energy_y_offset"]
        if result.debug_info.get('strip_consensus'):
            from .track_visualization import plot_strip_sequences
            plot_strip_sequences(result, directory)
            arrays['strip_local_quality'] = result.artifacts['strip_quality']
            arrays['strip_original_intensity_8bit'] = result.artifacts['strip_raw_intensity_profiles']
            for polarity, gradient in result.artifacts['strip_raw_gradient_profiles'].items():
                arrays[polarity+'_unnormalized_original_gradient'] = gradient
    composite=result.artifacts.get('composite_model')
    if composite is not None:
        composite.plot_evidence(result.paper_edge_tracks,directory)
        arrays['composite_raw_views']=composite.views
    np.savez_compressed(directory / "profiles.npz", **arrays)
    with (directory / "debug_info.json").open("w", encoding="utf-8") as stream:
        json.dump(result.debug_info, stream, ensure_ascii=False, indent=2, allow_nan=False)

