"""Explicit coordinate transforms for detector overlays and rotated inputs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class OverlayTransform:
    """Affine mapping between detection-image and saved-overlay pixels.

    ``detection_offset_*`` is the origin of the rendered ROI in detection
    coordinates. ``overlay_offset_*`` is the top-left corner of the image body
    in the saved overlay.  Keeping both makes padding and title bars explicit.
    """

    title_height_px: float
    scale_x: float
    scale_y: float
    detection_offset_x: float = 0.0
    detection_offset_y: float = 0.0
    overlay_offset_x: float = 0.0
    overlay_offset_y: float = 0.0

    def detection_to_overlay(self, x: float, y: float) -> tuple[float, float]:
        return (
            self.overlay_offset_x + (float(x) - self.detection_offset_x) * self.scale_x,
            self.overlay_offset_y + (float(y) - self.detection_offset_y) * self.scale_y,
        )

    def overlay_to_detection(self, x: float, y: float) -> tuple[float, float]:
        if self.scale_x <= 0 or self.scale_y <= 0:
            raise ValueError("Overlay scales must be positive")
        return (
            self.detection_offset_x + (float(x) - self.overlay_offset_x) / self.scale_x,
            self.detection_offset_y + (float(y) - self.overlay_offset_y) / self.scale_y,
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def preview_geometry(width: int, height: int, max_width: int, max_height: int) -> tuple[int, int, float, float]:
    """Mirror ``display_preview`` rounding and return exact realized scales."""
    if min(width, height, max_width, max_height) <= 0:
        raise ValueError("Preview dimensions must be positive")
    ratio = min(1.0, max_height / height, max_width / width)
    preview_width = width if ratio == 1.0 else max(1, round(width * ratio))
    preview_height = height if ratio == 1.0 else max(1, round(height * ratio))
    return preview_width, preview_height, preview_width / width, preview_height / height


def build_render_metadata(debug: dict[str, Any], canvas_width: int, canvas_height: int,
                          overlay_width: int, overlay_height: int,
                          title_height: int, body_offset_x: int = 0) -> dict[str, Any]:
    roi = debug["roi"]
    roi_width = int(roi["x2"] - roi["x1"])
    roi_height = int(roi["y2"] - roi["y1"])
    transform = OverlayTransform(
        title_height_px=float(title_height),
        scale_x=float(canvas_width / roi_width),
        scale_y=float(canvas_height / roi_height),
        detection_offset_x=float(roi["x1"]),
        detection_offset_y=float(roi["y1"]),
        overlay_offset_x=float(body_offset_x),
        overlay_offset_y=float(title_height),
    )
    image_shape = debug.get("image_shape", [roi["y2"], roi["x2"]])
    original_shape = debug.get("input_original_shape", image_shape)
    return {
        "mapping_status": "exact",
        "mapping_source": "render_time",
        "overlay_width": int(overlay_width),
        "overlay_height": int(overlay_height),
        "detection_image_width": int(image_shape[1]),
        "detection_image_height": int(image_shape[0]),
        "title_height_px": int(title_height),
        "preview_width": int(canvas_width),
        "preview_height": int(canvas_height),
        "preview_scale_x": transform.scale_x,
        "preview_scale_y": transform.scale_y,
        "detection_offset_x": transform.detection_offset_x,
        "detection_offset_y": transform.detection_offset_y,
        "overlay_offset_x": transform.overlay_offset_x,
        "overlay_offset_y": transform.overlay_offset_y,
        "rotation_ccw_degrees": int(debug.get("rotation_ccw_degrees", 0)),
        "original_width": int(original_shape[1]),
        "original_height": int(original_shape[0]),
        "stack_crop": debug.get("active_stack_roi", roi),
        "geometry_preprocessing": debug.get("geometry_preprocessing"),
        "roi": roi,
        "transform": transform.to_dict(),
    }


def reconstruct_render_metadata(debug: dict[str, Any], overlay_width: int,
                                overlay_height: int) -> dict[str, Any]:
    """Recover a legacy mapping from saved dimensions and effective config.

    The title height is derived as ``overlay_height - expected_preview_height``;
    it is never assumed from a historical constant.
    """
    try:
        roi = debug["roi"]
        cfg = debug["config"]
        roi_width = int(roi["x2"] - roi["x1"])
        roi_height = int(roi["y2"] - roi["y1"])
        width, height, _, _ = preview_geometry(
            roi_width, roi_height, int(cfg["display_max_width"]), int(cfg["display_max_height"])
        )
        title_height = int(overlay_height - height)
        expected_overlay_width = max(760, width)
        if title_height < 0 or overlay_width != expected_overlay_width:
            raise ValueError(
                f"saved overlay {overlay_width}x{overlay_height} is inconsistent with "
                f"expected body {width}x{height} and width {expected_overlay_width}"
            )
        metadata = build_render_metadata(
            debug, width, height, overlay_width, overlay_height, title_height
        )
        metadata["mapping_status"] = "reconstructed"
        metadata["mapping_source"] = "overlay_dimensions+debug_info+preview_geometry"
        return metadata
    except (KeyError, TypeError, ValueError, IndexError) as error:
        return {
            "mapping_status": "ambiguous",
            "mapping_source": "legacy_reconstruction_failed",
            "reason": str(error),
            "overlay_width": int(overlay_width),
            "overlay_height": int(overlay_height),
        }


def transform_from_metadata(metadata: dict[str, Any]) -> OverlayTransform:
    if metadata.get("mapping_status") == "ambiguous":
        raise ValueError(f"Ambiguous overlay mapping: {metadata.get('reason', 'unknown reason')}")
    values = metadata.get("transform", metadata)
    return OverlayTransform(**{name: float(values[name]) for name in OverlayTransform.__dataclass_fields__})


def detection_to_original(x: float, y: float, rotation_ccw_degrees: int,
                          original_width: int, original_height: int,
                          geometry: dict | None = None) -> tuple[float, float]:
    if geometry is not None:
        matrix = geometry['detection_to_input']
        x, y = (matrix[0][0]*x+matrix[0][1]*y+matrix[0][2],
                matrix[1][0]*x+matrix[1][1]*y+matrix[1][2])
    if rotation_ccw_degrees == 0:
        return float(x), float(y)
    if rotation_ccw_degrees == 90:
        return float(original_width - 1 - y), float(x)
    raise ValueError("Only 0 and 90 degree counter-clockwise mappings are supported")


def original_to_detection(x: float, y: float, rotation_ccw_degrees: int,
                          original_width: int, original_height: int,
                          geometry: dict | None = None) -> tuple[float, float]:
    if rotation_ccw_degrees == 90:
        x, y = float(y), float(original_width - 1 - x)
    elif rotation_ccw_degrees != 0:
        raise ValueError("Only 0 and 90 degree counter-clockwise mappings are supported")
    if geometry is not None:
        matrix = geometry.get('input_to_detection', [[1,0,0],[0,1,0],[0,0,1]])
        x, y = (matrix[0][0]*x+matrix[0][1]*y+matrix[0][2],
                matrix[1][0]*x+matrix[1][1]*y+matrix[1][2])
    return float(x), float(y)


__all__ = [
    "OverlayTransform", "preview_geometry", "build_render_metadata",
    "reconstruct_render_metadata", "transform_from_metadata",
    "detection_to_original", "original_to_detection",
]
