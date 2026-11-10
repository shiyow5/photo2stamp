"""End-to-end photo -> flat stamp illustration pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .palette import assign_colors, get_palette
from .segment import (
    extract_accents,
    extract_sky,
    majority_filter,
    merge_small_regions,
    quantize,
    smooth,
    to_lab,
)
from .vectorize import SHAPES, apply_shape_mask, labels_to_shapes, render_png, shapes_to_svg


TONE_MAPPINGS = ("area", "rank")


@dataclass(frozen=True)
class StampConfig:
    max_size: int = 800  # long side in px after resize (0 = keep original size)
    colors: int = 8  # number of k-means clusters before palette mapping
    smooth: int = 2  # edge-preserving smoothing passes
    majority: int = 17  # majority-filter window (px); larger = blockier shapes
    min_region: float = 0.003  # regions smaller than this fraction of the image are merged
    simplify: float = 0.005  # polygon simplification tolerance as a fraction of the diagonal
    snap_deg: float = 12.0  # edges this close to vertical are made exactly vertical (0 = off)
    palette: str = "teal"  # preset name or "original"
    detect_sky: bool = True
    accent_chroma: float = 28.0  # clusters more saturated than this become accent colors
    shape: str = "rect"  # "rect" or "circle"
    tone_mapping: str = "area"  # "area" (equalize by pixel area) or "rank" (one tone per cluster)
    seed: int = 0

    def __post_init__(self) -> None:
        if self.colors < 2:
            raise ValueError("colors must be >= 2")
        if self.tone_mapping not in TONE_MAPPINGS:
            raise ValueError(f"tone_mapping must be one of {TONE_MAPPINGS}")
        if self.shape not in SHAPES:
            raise ValueError(f"shape must be one of {SHAPES}")
        if min(self.max_size, self.smooth, self.majority, self.snap_deg) < 0:
            raise ValueError("max_size, smooth, majority and snap_deg must be >= 0")
        if not 0 <= self.min_region < 1 or not 0 <= self.simplify < 1:
            raise ValueError("min_region and simplify must be in [0, 1)")
        get_palette(self.palette)  # raises on unknown name


@dataclass(frozen=True)
class StampResult:
    image: np.ndarray  # BGR, or BGRA when shape != "rect"
    svg: str
    labels: np.ndarray
    flat_colors: np.ndarray  # (N,3) BGR colors actually used


def to_bgr_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize gray / 1-channel / BGRA / 16-bit / float input to a new uint8 BGR image."""
    img = np.asarray(image)
    if img.size == 0 or img.ndim not in (2, 3):
        raise ValueError("image is empty or not a 2-D/3-D array")
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.integer):
            img = (img.astype(np.float64) / np.iinfo(img.dtype).max * 255).round()
        else:  # float images are assumed to be in 0..1
            img = img.astype(np.float64) * 255
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2 or img.shape[2] == 1:
        return cv2.cvtColor(img.reshape(img.shape[:2]), cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        alpha = img[..., 3:].astype(np.float32) / 255.0
        white = np.full_like(img[..., :3], 255, dtype=np.float32)
        return (img[..., :3] * alpha + white * (1 - alpha)).round().astype(np.uint8)
    return img[..., :3].copy()


def resize_long_side(image: np.ndarray, max_size: int) -> np.ndarray:
    """Shrink so the long side is at most max_size. Small images are never enlarged."""
    h, w = image.shape[:2]
    if max_size <= 0 or max(h, w) <= max_size:
        return image.copy()
    scale = max_size / max(h, w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _unify_by_color(
    labels: np.ndarray, colors: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clusters that map to the same palette color become a single label.

    Returns (new_labels, unique_colors, old_to_new) where old_to_new maps old label ids.
    """
    unique, inverse = np.unique(colors, axis=0, return_inverse=True)
    old_to_new = inverse.reshape(-1).astype(np.int32)
    return old_to_new[labels], unique, old_to_new


def stylize(image: np.ndarray, config: StampConfig | None = None) -> StampResult:
    cfg = config or StampConfig()
    img = resize_long_side(to_bgr_uint8(image), cfg.max_size)
    h, w = img.shape[:2]

    smoothed = smooth(img, cfg.smooth)
    min_area = int(cfg.min_region * h * w)
    labels, centers = quantize(smoothed, cfg.colors, cfg.seed)
    window = cfg.majority | 1 if cfg.majority > 1 else 1
    labels = majority_filter(labels, centers.shape[0], size=window, iterations=2)
    n_base = centers.shape[0]
    labels, centers = extract_accents(
        to_lab(smoothed), labels, centers, cfg.accent_chroma, max(1, min_area // 4)
    )
    accent_ids = range(n_base, centers.shape[0])

    sky = frozenset()
    if cfg.detect_sky:
        labels, centers, sky_label = extract_sky(labels, centers)
        sky = frozenset() if sky_label is None else frozenset({sky_label})

    weights = None
    if cfg.tone_mapping == "area":
        weights = np.bincount(labels.ravel(), minlength=centers.shape[0])
    colors = assign_colors(centers, get_palette(cfg.palette), sky, cfg.accent_chroma, weights)
    labels, colors, old_to_new = _unify_by_color(labels, colors)
    keep = frozenset(int(old_to_new[i]) for i in accent_ids)
    labels = merge_small_regions(labels, min_area, keep=keep)

    epsilon = cfg.simplify * float(np.hypot(h, w))
    shapes = labels_to_shapes(labels, colors, epsilon, cfg.snap_deg)
    flat = render_png(shapes, (h, w), base=colors[labels])
    out = flat if cfg.shape == "rect" else apply_shape_mask(flat, cfg.shape)
    used = colors[np.unique(labels)]
    return StampResult(out, shapes_to_svg(shapes, (h, w), cfg.shape), labels, used)
