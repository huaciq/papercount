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
    # 假设照明比纸边变化更慢：I_norm = 128 * I / max(G_sigma * I, epsilon)。
    # 归一化便于提出候选；纸身验证仍需读取未经此处理的原始灰度。
    background = cv2.GaussianBlur(gray_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
    np.maximum(background, epsilon, out=background)
    return cv2.divide(gray_float, background, scale=128.0)


def signed_gradient(normalized: np.ndarray) -> np.ndarray:
    """Preserve Scharr Y sign: opposite sides of a band remain separate."""
    # 图像 y 向下增大：暗→亮为正，亮→暗为负；取绝对值会混合纸身两侧。
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
        if scale <= config.epsilon:
            source = base
        elif config.directional_smoothing_enabled:
            # Horizontal suppression was already applied to ``base``.  Each
            # scale adds only a vertical integration scale, keeping the
            # directional and scale factors independently attributable.
            source = cv2.GaussianBlur(base, (0, 0), sigmaX=config.epsilon,
                                      sigmaY=float(scale))
        else:
            # The multiscale-only ablation is isotropic by construction.  It
            # must not silently inherit directional_sigma_x.
            source = cv2.GaussianBlur(base, (0, 0), sigmaX=float(scale),
                                      sigmaY=float(scale))
        label = "directional" if config.directional_smoothing_enabled else "native"
        if scale > config.epsilon:
            label += f"+scale_{float(scale):g}"
        levels.append((float(scale), signed_gradient(source), label))
    return levels, native


def preprocess_tracks_signal(gray: np.ndarray, config) -> tuple[np.ndarray, dict]:
    """Prepare a tracks signal for stacks whose layers should be horizontal.

    The original grayscale image remains the source for RAW masks, paper-body
    evidence, identity checks and coordinate reporting.  This function only
    creates the signal image used by the gradient frontend.  That separation is
    important: a contrast enhancement may propose a boundary, but it cannot by
    itself create RAW evidence for a counted sheet.

    The optional pass does two conservative things:

    * smooths mostly along x while retaining y transitions, which suppresses
      vertical print/texture and rewards horizontal layer edges;
    * applies a wide horizontal black-hat only when explicitly enabled, making
      dark, long paper gaps more visible without using a period or count prior.
    """
    if not getattr(config, "tracks_preprocessing_enabled", False):
        return gray, {"enabled": False, "source": "original_gray"}
    if gray.ndim != 2 or gray.size == 0:
        raise ValueError("tracks preprocessing expects a nonempty grayscale image")

    source = gray.astype(np.float32)
    max_value = float(np.iinfo(gray.dtype).max) if np.issubdtype(gray.dtype, np.integer) else float(np.max(source))
    if max_value <= 0:
        return source, {"enabled": True, "source": "original_gray", "reason": "zero_image",
                        "raw_evidence_source": "original_gray"}
    unit = np.clip(source / max_value, 0.0, 1.0)

    sigma_x = float(getattr(config, "tracks_horizontal_smoothing_sigma_x", 0.0))
    sigma_y = float(getattr(config, "tracks_horizontal_smoothing_sigma_y", 0.0))
    if sigma_x > 0 or sigma_y > 0:
        unit = cv2.GaussianBlur(unit, (0, 0), sigmaX=max(sigma_x, 1e-3),
                                sigmaY=max(sigma_y, 1e-3), borderType=cv2.BORDER_REPLICATE)

    details = {
        "enabled": True,
        "source": "original_gray",
        "horizontal_smoothing": {"sigma_x": sigma_x, "sigma_y": sigma_y},
        "gap_enhancement": {"enabled": False},
        "raw_evidence_source": "original_gray",
    }
    if getattr(config, "tracks_gap_enhancement_enabled", False):
        height = int(max(1, round(getattr(config, "tracks_gap_kernel_height", 3))))
        if height % 2 == 0:
            height += 1
        width = int(max(3, round(unit.shape[1] * getattr(config, "tracks_gap_kernel_width_ratio", .08))))
        if width % 2 == 0:
            width += 1
        # Black-hat = closing - image; a long dark trough therefore receives
        # a response while isolated dark print is mostly rejected by the x span.
        encoded = np.clip(unit * 255.0, 0, 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, height))
        blackhat = cv2.morphologyEx(encoded, cv2.MORPH_BLACKHAT, kernel).astype(np.float32) / 255.0
        gain = float(getattr(config, "tracks_gap_gain", .75))
        dynamic = max(float(np.percentile(unit, 99.5) - np.percentile(unit, .5)), 1e-3)
        unit = np.clip(unit - gain * blackhat * dynamic, 0.0, 1.0)
        details["gap_enhancement"] = {"enabled": True, "kernel_width": width,
            "kernel_height": height, "gain": gain, "blackhat_p99": float(np.percentile(blackhat, 99))}

    # Keep a floating signal so uint16 input is not quantized a second time.
    return (unit * max_value).astype(np.float32), details


def display_preview(array: np.ndarray, max_height: int, max_width: int) -> np.ndarray:
    """Resize exclusively for debug display, never for detection."""
    ratio = min(1.0, max_height / array.shape[0], max_width / array.shape[1])
    if ratio == 1.0:
        return array.copy()
    return cv2.resize(array, (max(1, round(array.shape[1] * ratio)), max(1, round(array.shape[0] * ratio))), interpolation=cv2.INTER_AREA)

