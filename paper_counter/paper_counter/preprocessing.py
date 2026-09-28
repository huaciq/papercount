from __future__ import annotations

"""Full-resolution ROI, illumination correction and signed gradients."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    """Read Unicode paths on Windows without resizing or losing bit depth."""
    image = cv2.imdecode(np.fromfile(Path(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def crop_roi(image: np.ndarray, roi: Optional[dict[str, int]]) -> tuple[np.ndarray, dict[str, int]]:
    """Return a view; reject out-of-bounds ROI instead of silently clipping."""
    if image is None or image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("Image must be a nonempty gray/BGR/BGRA array")
    h, w = image.shape[:2]
    bounds = dict(roi) if roi is not None else {"x1": 0, "x2": w, "y1": 0, "y2": h}
    x1, x2, y1, y2 = (bounds[k] for k in ("x1", "x2", "y1", "y2"))
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError(f"ROI {bounds} is outside image width={w}, height={h}")
    if y2 - y1 < 3:
        raise ValueError("ROI must have at least 3 rows")
    return image[y1:y2, x1:x2], bounds


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert supported integer images to grayscale; reject transparent inputs."""
    if image.dtype not in (np.uint8, np.uint16):
        raise ValueError("Supported image depths: uint8 and uint16")
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        if np.any(image[:, :, 3] != np.iinfo(image.dtype).max):
            raise ValueError("Image has transparency; composite onto a known background first")
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("Expected gray, BGR or BGRA input")


def normalize_illumination(gray: np.ndarray, sigma: float, epsilon: float) -> np.ndarray:
    """Remove slowly varying multiplicative illumination in float32."""
    gray_float = gray.astype(np.float32)
    background = cv2.GaussianBlur(gray_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
    np.maximum(background, epsilon, out=background)
    return cv2.divide(gray_float, background, scale=128.0)


def signed_gradient(normalized: np.ndarray) -> np.ndarray:
    """Preserve Scharr Y sign: opposite sides of a band remain separate."""
    return cv2.Scharr(normalized, cv2.CV_32F, 0, 1)


def signed_gradient_levels(normalized: np.ndarray, config) -> tuple[list[tuple[float, np.ndarray, str]], np.ndarray]:
    """Return configurable signed gradients plus an untouched evidence level.

    Scale zero is independent of the estimated paper period.  The untouched
    native gradient is always returned separately so enhanced responses remain
    proposals until supported by an original-resolution observation.
    """
    native = signed_gradient(normalized)
    if not config.directional_smoothing_enabled and not config.multiscale_gradient_enabled:
        return [(0.0, native, "native")], native
    base = normalized
    if config.directional_smoothing_enabled:
        base = cv2.GaussianBlur(base, (0, 0), sigmaX=config.directional_sigma_x,
                                sigmaY=config.directional_sigma_y)
    scales = config.gradient_scales if config.multiscale_gradient_enabled else [0.0]
    levels = []
    for scale in scales:
        source = base if scale <= config.epsilon else cv2.GaussianBlur(
            base, (0, 0), sigmaX=max(config.directional_sigma_x, config.epsilon),
            sigmaY=float(scale))
        label = "directional" if config.directional_smoothing_enabled else "native"
        if scale > config.epsilon:
            label += f"+scale_{float(scale):g}"
        levels.append((float(scale), signed_gradient(source), label))
    return levels, native


def display_preview(array: np.ndarray, max_height: int, max_width: int) -> np.ndarray:
    """Resize exclusively for debug display, never for detection."""
    ratio = min(1.0, max_height / array.shape[0], max_width / array.shape[1])
    if ratio == 1.0:
        return array.copy()
    return cv2.resize(array, (max(1, round(array.shape[1] * ratio)), max(1, round(array.shape[0] * ratio))), interpolation=cv2.INTER_AREA)

