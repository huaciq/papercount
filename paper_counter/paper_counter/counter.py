from __future__ import annotations

"""Dispatch the two supported detectors: tracks and regions."""
from dataclasses import replace
from typing import Any
import numpy as np
from .config import CounterConfig
from .models import PaperCountResult


class PaperCounter:
    """Select the configured algorithm without an implicit legacy fallback."""

    def __init__(self, config: CounterConfig | dict[str, Any] | None = None):
        self.config = config if isinstance(config, CounterConfig) else CounterConfig(**(config or {}))

    def process(self, image: np.ndarray) -> PaperCountResult:
        """Process an image and preserve its detection-to-input geometry."""
        cfg = self.config
        if cfg.algorithm not in {"tracks", "regions"}:
            raise ValueError("Only tracks and regions are supported")
        if cfg.auto_geometry_enabled or cfg.algorithm == "regions":
            from .paper_geometry import prepare_geometry
            if cfg.auto_geometry_enabled:
                prepared, geometry = prepare_geometry(
                    image, cfg.geometry_target_spacing,
                    bounded_angle_analysis=cfg.algorithm == "regions",
                    auto_quarter_turn=cfg.algorithm == "regions",
                    center_width=cfg.narrow_width if cfg.algorithm == "regions" else None,
                    preserve_extent=(cfg.algorithm == "tracks" and cfg.tracks_geometry_preserve_extent))
            else:
                prepared, geometry = image, dict(
                    enabled=False, applied=False, input_shape=list(image.shape),
                    input_to_detection=np.eye(3).tolist(), detection_to_input=np.eye(3).tolist())
            if cfg.algorithm == "regions":
                from .geometry_alignment import center_window
                from .region_counter import process_regions
                if not geometry.get("center_window"):
                    prepared, geometry = center_window(prepared, cfg.narrow_width, geometry)
                result = process_regions(prepared, cfg)
            else:
                result = PaperCounter(replace(cfg, auto_geometry_enabled=False)).process(prepared)
            result.debug_info["geometry_preprocessing"] = geometry
            result.debug_info["input_original_shape"] = list(image.shape)
            result.debug_info["config"] = cfg.to_dict()
            result.debug_info["coordinate_system"] = "prepared detection image; use geometry affine for input coordinates"
            return result
        from .track_counter import process_tracks
        return process_tracks(image, cfg)
