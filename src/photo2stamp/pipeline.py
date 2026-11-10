"""End-to-end photo -> flat stamp illustration pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .palette import assign_colors, get_palette, hex_to_bgr
from .segment import (
    edge_density,
    extract_accents,
    extract_far,
    extract_sky,
    majority_filter,
    merge_small_regions,
    quantize,
    smooth,
    texture_map,
    to_lab,
    upscale_labels,
)
from .vectorize import SHAPES, frame_stamp, labels_to_shapes, render_png, shapes_to_svg


TONE_MAPPINGS = ("area", "rank")
WORK_SIZE_CALM, WORK_SIZE_BUSY = 400, 260
PAPER = "#F6F1E7"  # paper color for the gap inside the stamp ring (natural palette)  # auto work size for calm / busy photos
_DENSITY_CALM, _DENSITY_BUSY = 0.07, 0.25  # edge density of a calm seascape / a street of trees


@dataclass(frozen=True)
class StampConfig:
    max_size: int = 800  # long side in px after resize (0 = keep original size)
    work_size: int = 0  # long side of the image used for segmentation; 0 = auto from detail
    colors: int = 8  # k-means clusters before palette mapping
    smooth: int = 1  # edge-preserving smoothing passes (on the work image)
    majority: int = 3  # majority-filter window in work-image px; larger = blockier shapes
    min_region: float = 0.004  # regions smaller than this fraction of the image are merged
    simplify: float = 0.006  # polygon simplification tolerance as a fraction of the diagonal
    snap_deg: float = 12.0  # edges this close to vertical are made exactly vertical (0 = off)
    palette: str = "natural"  # "natural" (photo hues, stylized), a preset name, or "original"
    detect_sky: bool = True
    detect_far: bool = True  # paint hazy distant scenery under the sky as one silhouette color
    far_delta: float = 20.0  # max Lab distance from the sky color for a region to count as far
    accent_chroma: float = 28.0  # clusters more saturated than this become accent colors
    shape: str = "rect"  # "rect", "circle" or "rounded" (circle/rounded are cropped square)
    ring: float = 0.035  # stamp ring width as a fraction of the stamp size (0 = no ring)
    ring_gap: float = 0.02  # paper-colored gap between the ring and the art
    tone_mapping: str = "area"  # "area" (equalize by pixel area) or "rank" (one tone per cluster)
    seed: int = 0

    def __post_init__(self) -> None:
        if self.colors < 2:
            raise ValueError("colors must be >= 2")
        if self.tone_mapping not in TONE_MAPPINGS:
            raise ValueError(f"tone_mapping must be one of {TONE_MAPPINGS}")
        if self.shape not in SHAPES:
            raise ValueError(f"shape must be one of {SHAPES}")
        if min(self.max_size, self.work_size, self.smooth, self.majority, self.snap_deg) < 0:
            raise ValueError("max_size, work_size, smooth, majority and snap_deg must be >= 0")
        if self.ring < 0 or self.ring_gap < 0 or self.ring + self.ring_gap >= 0.5:
            raise ValueError("ring and ring_gap must be >= 0 and ring + ring_gap < 0.5")
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


def auto_work_size(density: float) -> int:
    """Pick the segmentation resolution from the photo's edge density.

    Busy photos (leaves, carvings, night lights) are segmented on a smaller image so
    that only big shapes survive; calm photos keep more resolution. The anchor values
    come from the example photos: a calm seascape has density ~0.07, a street of
    trees ~0.25.
    """
    return int(round(np.interp(density, [_DENSITY_CALM, _DENSITY_BUSY], [WORK_SIZE_CALM, WORK_SIZE_BUSY])))



def crop_square(image: np.ndarray, focus_gain: float = 1.2) -> np.ndarray:
    """Crop to a square (stamps are round or square), centered on the subject.

    The window slides along the long side and goes where the edge density is highest,
    since the subject is usually the most detailed part. It stays centered unless some
    other window has `focus_gain` times more detail.
    """
    h, w = image.shape[:2]
    s = min(h, w)
    if h == w:
        return image.copy()
    long_side = max(h, w)
    scale = min(1.0, 200 / long_side)
    small = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small[..., :3], cv2.COLOR_BGR2GRAY)
    edges = (cv2.Canny(gray, 60, 150) > 0).astype(np.float64)
    profile = edges.sum(axis=0 if w > h else 1)
    win = max(1, min(len(profile), round(s * scale)))
    sums = np.convolve(profile, np.ones(win), mode="valid")
    center = (len(sums) - 1) // 2
    best = int(sums.argmax())
    if sums[best] > focus_gain * sums[center]:
        offset = min(long_side - s, round(best / scale))
    else:
        offset = (long_side - s) // 2
    if w > h:
        return image[:, offset : offset + s].copy()
    return image[offset : offset + s, :].copy()


def _segment(
    small: np.ndarray, texture: np.ndarray, cfg: StampConfig, n_colors: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster, clean and classify the small working image. Returns (labels, BGR colors)."""
    sh, sw = small.shape[:2]
    min_area = int(cfg.min_region * sh * sw)
    smoothed = smooth(small, cfg.smooth)
    lab = to_lab(smoothed)
    labels, centers = quantize(smoothed, n_colors, cfg.seed)
    window = cfg.majority | 1 if cfg.majority > 1 else 1
    labels = majority_filter(labels, centers.shape[0], size=window, iterations=2)
    n_base = centers.shape[0]
    labels, centers = extract_accents(lab, labels, centers, cfg.accent_chroma, max(1, min_area // 4))
    accent_ids = range(n_base, centers.shape[0])

    sky: frozenset[int] = frozenset()
    far: frozenset[int] = frozenset()
    if cfg.detect_sky:
        labels, centers, sky_label = extract_sky(labels, centers, texture=texture, lab=lab)
        sky = frozenset() if sky_label is None else frozenset({sky_label})
        if sky_label is not None and cfg.detect_far:
            labels, centers, far_label = extract_far(labels, centers, sky_label, cfg.far_delta, lab=lab)
            far = frozenset() if far_label is None else frozenset({far_label})

    weights = np.bincount(labels.ravel(), minlength=centers.shape[0]) if cfg.tone_mapping == "area" else None
    colors = assign_colors(centers, get_palette(cfg.palette), sky, cfg.accent_chroma, weights, far_labels=far)
    labels, colors, old_to_new = _unify_by_color(labels, colors)
    keep = frozenset(int(old_to_new[i]) for i in accent_ids)
    return merge_small_regions(labels, min_area, keep=keep), colors


def _frame_colors(colors: np.ndarray, palette_name: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """(ring, paper): the darkest and lightest colors of the palette (or of the result)."""
    palette = get_palette(palette_name)
    if palette is not None and palette.tones:
        return hex_to_bgr(palette.tones[0]), hex_to_bgr(palette.tones[-1])
    lum = colors.astype(np.float32) @ np.array([0.114, 0.587, 0.299], np.float32)
    dark, light = colors[int(lum.argmin())], colors[int(lum.argmax())]
    if palette is not None:  # natural: a fixed paper color keeps every stamp consistent
        return tuple(int(c) for c in dark), hex_to_bgr(PAPER)
    return tuple(int(c) for c in dark), tuple(int(c) for c in light)


def stylize(image: np.ndarray, config: StampConfig | None = None) -> StampResult:
    cfg = config or StampConfig()
    img = to_bgr_uint8(image)
    if cfg.shape != "rect":
        img = crop_square(img)
    img = resize_long_side(img, cfg.max_size)
    h, w = img.shape[:2]

    density = edge_density(img)
    work = cfg.work_size or auto_work_size(density)
    small = resize_long_side(img, work)
    sh, sw = small.shape[:2]
    texture = cv2.resize(texture_map(img), (sw, sh), interpolation=cv2.INTER_AREA)
    small_labels, colors = _segment(small, texture, cfg, cfg.colors)

    labels = upscale_labels(small_labels, (h, w), len(colors)) if (sh, sw) != (h, w) else small_labels
    shapes = labels_to_shapes(labels, colors, cfg.simplify * float(np.hypot(h, w)), cfg.snap_deg)
    flat = render_png(shapes, (h, w), base=colors[labels])
    used = colors[np.unique(labels)]
    if cfg.shape == "rect":
        return StampResult(flat, shapes_to_svg(shapes, (h, w)), labels, used)

    ring_color, paper = _frame_colors(used, cfg.palette)
    out = frame_stamp(flat, cfg.shape, ring_color, cfg.ring, cfg.ring_gap, paper)
    svg = shapes_to_svg(shapes, (h, w), cfg.shape, ring_color, cfg.ring, cfg.ring_gap, paper)
    return StampResult(out, svg, labels, used)
