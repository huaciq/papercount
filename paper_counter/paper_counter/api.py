"""Installable, file-or-array API for the existing paper detection pipeline."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import CounterConfig
from .models import PaperCountResult
from .preprocessing import read_image
from .recipe import Recipe,load_config_or_recipe
from .engine import DetectionEngine
from .contracts import PaperInstance


@dataclass
class DetectionResult:
    """Compact result. All detection coordinates include ROI offsets."""

    count: int
    confidence: float
    status: str
    warnings: list[str]
    period: float | None
    preset: str
    original_shape: tuple[int, ...]
    detection_shape: tuple[int, ...]
    rotation_ccw_degrees: int
    tracks: list[dict[str, Any]]
    raw_result: PaperCountResult = field(repr=False)
    boxes: list[dict[str, Any]] = field(default_factory=list)
    pipeline_id: str = ''
    recipe_id: str = ''
    count_unit: str = ''
    resolved_recipe: dict[str,Any] = field(default_factory=dict)
    instances: list[PaperInstance] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible summary, excluding internal image and signal arrays."""
        return deepcopy({
            "count": self.count, "confidence": self.confidence,
            "confidence_interpretation": "uncalibrated consistency score; not a probability",
            "status": self.status, "warnings": self.warnings, "period": self.period,
            "preset": self.preset, "original_shape": list(self.original_shape),
            "detection_shape": list(self.detection_shape),
            "rotation_ccw_degrees": self.rotation_ccw_degrees,
            "coordinate_system": "detection image after requested rotation and optional geometry preprocessing; ROI offsets included; x2/y2 exclusive",
            "geometry_preprocessing": self.raw_result.debug_info.get('geometry_preprocessing'),
            "tracks": self.tracks,
            "boxes": self.boxes,
            "pipeline_id": self.pipeline_id,"recipe_id": self.recipe_id,"count_unit": self.count_unit,
            "resolved_recipe": self.resolved_recipe,
            "instances": [instance.to_dict() for instance in self.instances],
        })


class PaperDetector:
    """Reuse one configured detector for multiple images, without implicit writes.

    The default selects the latest explicit v4 configuration. ``e3_3`` matches
    the existing CLI default. A custom config replaces the preset completely.
    """

    def __init__(self, *, preset: str | None = None,
                 config: CounterConfig | str | Path | None = None,
                 recipe: Recipe | str | Path | None = None):
        self._explicit_recipe=recipe is not None
        if recipe is not None:
            if config is not None or preset is not None:
                raise ValueError('Choose recipe or preset/config, not both')
            self.recipe=Recipe.load(recipe)
            self.config=self.recipe.to_counter_config()
            self.preset=self.recipe.recipe_id
        elif config is None:
            preset='paper_layers_v4' if preset is None else preset
            filenames = {"tracks": "default.yaml", "regions": "regions.yaml",
                         "paper_layers_v4": "paper_layers_v4.yaml", "e3_3": "default.yaml",
                         "tracks_preprocessed": "tracks_preprocessed.yaml", "corrugated": "corrugated.yaml"}
            if preset not in filenames:
                raise ValueError(f"Unknown preset: {preset!r}; choose tracks, tracks_preprocessed, regions, corrugated, paper_layers_v4 or e3_3")
            self.config = CounterConfig.from_yaml(Path(__file__).parent / "presets" / filenames[preset])
            self.preset = preset
            self.recipe=Recipe.from_legacy(self.config,recipe_id=f'preset_{preset}')
        elif isinstance(config, CounterConfig):
            self.config = deepcopy(config)
            self.preset = "custom"
            self.recipe=Recipe.from_legacy(self.config)
        elif isinstance(config, (str, Path)):
            self.recipe=load_config_or_recipe(config)
            self.config=self.recipe.to_counter_config()
            self._explicit_recipe=self.recipe.compatibility_config is None
            self.preset = "custom"
        else:
            raise TypeError("config must be CounterConfig, a YAML path, or None")

    def detect(self, image: str | Path | np.ndarray, *, rotate_ccw: bool = False,
               output_dir: str | Path | None = None) -> DetectionResult:
        """Detect a file or uint8/uint16 gray/BGR/opaque BGRA array.

        Paper layers must be ordered along detection-image Y. Set rotate_ccw
        for horizontal captures. Saving overlays/coordinates is opt-in.
        """
        if not isinstance(rotate_ccw, bool):
            raise TypeError("rotate_ccw must be bool")
        if isinstance(image, (str, Path)):
            array = read_image(image)
        elif isinstance(image, np.ndarray):
            array = image
        else:
            raise TypeError("image must be a file path or numpy.ndarray")
        if array.ndim not in (2, 3) or array.size == 0:
            raise ValueError("image must be a nonempty gray/BGR/BGRA array")
        original_shape = tuple(array.shape)
        # Isolate the caller's image from the existing pipeline and result artifacts.
        detection_image = np.rot90(array, 1).copy() if rotate_ccw else array.copy()
        if self._explicit_recipe:
            if self.config.to_dict()!=self.recipe.to_counter_config().to_dict():
                raise ValueError('Edit the recipe parameters instead of the compatibility config')
            active_recipe=self.recipe
        else:
            active_recipe=Recipe.from_legacy(self.config,recipe_id=self.recipe.recipe_id)
        raw = DetectionEngine(active_recipe).process(detection_image)
        raw.debug_info.update(
            input_original_shape=list(original_shape),
            rotation_ccw_degrees=90 if rotate_ccw else 0,
            coordinate_system="detection image after requested rotation and optional geometry preprocessing; x2/y2 exclusive",
        )
        if isinstance(image, (str, Path)):
            raw.debug_info["input_image"] = str(Path(image).resolve())
        from .instances import exported_tracks
        tracks=exported_tracks(raw,original_shape,90 if rotate_ccw else 0)
        result = DetectionResult(
            count=int(raw.count), confidence=float(raw.confidence),
            status=raw.debug_info["status"], warnings=list(raw.debug_info.get("warnings", [])),
            period=None if raw.period is None else float(raw.period), preset=self.preset,
            original_shape=original_shape, detection_shape=tuple(raw.debug_info.get('image_shape', detection_image.shape)),
            rotation_ccw_degrees=90 if rotate_ccw else 0, tracks=tracks, raw_result=raw,
        )
        if raw.debug_info.get('algorithm')=='narrow_regions':
            from .region_output import box_coordinates
            result.boxes=box_coordinates(raw)
        from .instances import counted_instances
        result.pipeline_id=active_recipe.pipeline
        result.recipe_id=active_recipe.recipe_id
        result.count_unit=active_recipe.count_unit
        result.resolved_recipe=active_recipe.to_dict()
        result.instances=counted_instances(result.count,result.count_unit,result.boxes,result.tracks)
        if output_dir is not None:
            from .compact_output import save_compact
            from .visualization import save_debug
            exporter=save_debug if active_recipe.runtime.export=='full_debug' else save_compact
            exporter(raw, output_dir, self.config)
        return result


def detect_papers(image: str | Path | np.ndarray, *, rotate_ccw: bool = False,
                  preset: str | None = None,
                  config: CounterConfig | str | Path | None = None,
                  recipe: Recipe | str | Path | None = None,
                  output_dir: str | Path | None = None) -> DetectionResult:
    """One-call entry point; use PaperDetector for repeated calls."""
    return PaperDetector(preset=preset, config=config,recipe=recipe).detect(
        image, rotate_ccw=rotate_ccw, output_dir=output_dir)
