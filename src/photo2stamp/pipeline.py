"""End-to-end photo -> flat stamp illustration pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .details import details_svg, find_details, paint_details, select_details
from .palette import NATURAL, assign_colors, get_palette, hex_to_bgr
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
    tint_shaded_foliage,
    to_lab,
    upscale_labels,
)
from .lines import clean_lines, dog_lines, draw_lines, light_spots, lines_svg
from .vectorize import SHAPES, frame_stamp, labels_to_shapes, render_png, shapes_to_svg
from .whitebalance import white_balance


TONE_MAPPINGS = ("area", "rank")
WORK_SIZE_CALM, WORK_SIZE_BUSY = 500, 400
# auto work size for calm / busy photos (above)
PAPER = "#F6F1E7"  # paper color for the gap inside the stamp ring (natural palette)
INK = "#26221F"  # outline color on light fills
LIGHT_INK = "#F3DFA2"  # night mode: lamp-colored outlines on dark fills
NIGHT_LUMA = 45  # photos with a median luma below this are drawn in night mode
NIGHT_L_RANGE = (10.0, 62.0)  # night mode keeps fills dark
NIGHT_MODES = ("auto", "on", "off")
BRIGHT_SKY_L = 55.0  # a sky this light (Lab L) is daytime: its clouds get no outlines
SKYLINE_MARGIN = 0.04  # fraction of the height below the lowest sky pixel still counted as sky
CLOUD_MIN_LUMA = 110.0  # soft edges are dropped only where the photo is this light around them
SOFT_EDGE_GRAD = 150.0  # Sobel magnitude (0..255 image) below which an edge is soft (cloud rim)
NIGHT_BRIGHT_LUMA = 120  # pixels brighter than this are lit (windows, lamps, a lit dish)
NIGHT_MAX_BLOB = 0.05  # night: no lit area covers more than this fraction (measured 0.005 vs 0.13)
SVG_BASE_EPSILON = 0.75  # px: the SVG base layer follows the label map this closely
LINE_SIZE = 600  # long side of the image the outlines are detected on
SMOOTHERS = ("l0", "meanshift")
CROPS = ("before", "after")  # circle/rounded: crop the photo first, or cut the finished art
_DENSITY_CALM, _DENSITY_BUSY = 0.07, 0.25  # edge density of a calm seascape / a street of trees


@dataclass(frozen=True)
class StampConfig:
    max_size: int = 800  # long side in px after resize (0 = keep original size)
    white_balance: bool = True  # remove the color cast of the light first (Gray-Edge)
    work_size: int = 0  # long side of the image used for segmentation; 0 = auto from detail
    colors: int = 8  # k-means clusters before palette mapping
    chroma_weight: float = 2.0  # weight of hue/chroma vs lightness in k-means (hue makes things recognizable)
    smoother: str = "l0"  # "l0" (L0 gradient minimization, Xu et al. 2011) or "meanshift"
    l0_lambda: float = 0.04  # L0 smoothing strength; larger = flatter regions
    smooth: int = 1  # mean-shift smoothing passes (smoother="meanshift")
    lines: bool = True  # draw DoG outlines over the flat regions
    details: bool = True  # natural palette: small colored details (carvings, flowers) as spots
    night: str = "auto"  # "auto" (by photo brightness), "on" or "off": dark fills + light outlines
    line_sigma: float = 2.0  # outline scale (Gaussian sigma at 600 px); larger = only big structures
    line_threshold: float = 0.012  # DoG response needed for a line; larger = fewer lines
    line_min_length: float = 0.07  # drop line fragments shorter than this fraction of the long side
    line_width: float = 1.5  # outline width in px at 600 px
    majority: int = 3  # majority-filter window in work-image px; larger = blockier shapes
    min_region: float = 0.002  # regions smaller than this fraction of the image are merged
    simplify: float = 0.006  # polygon simplification tolerance as a fraction of the diagonal
    snap_deg: float = 12.0  # edges this close to vertical are made exactly vertical (0 = off)
    palette: str = "natural"  # "natural" (photo hues, stylized), a preset name, or "original"
    detect_sky: bool = True
    detect_far: bool = True  # paint hazy distant scenery under the sky as one silhouette color
    far_delta: float = 20.0  # max Lab distance from the sky color for a region to count as far
    accent_chroma: float = 28.0  # clusters more saturated than this become accent colors
    shape: str = "rect"  # "rect", "circle" or "rounded" (circle/rounded are cropped square)
    crop: str = "before"  # "before": stylize the square crop / "after": cut it out of the full art
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
        if self.night not in NIGHT_MODES:
            raise ValueError(f"night must be one of {NIGHT_MODES}")
        if self.crop not in CROPS:
            raise ValueError(f"crop must be one of {CROPS}")
        if self.crop == "after" and self.shape == "rect":
            raise ValueError('crop="after" needs shape "circle" or "rounded"')
        if self.smoother not in SMOOTHERS:
            raise ValueError(f"smoother must be one of {SMOOTHERS}")
        if min(self.l0_lambda, self.line_sigma, self.line_threshold, self.line_min_length, self.line_width) < 0:
            raise ValueError("l0_lambda and line_* settings must be >= 0")
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



def _color_saliency(image: np.ndarray) -> np.ndarray:
    """Frequency-tuned saliency (Achanta, Hemami, Estrada & Susstrunk, CVPR 2009): how far
    each slightly blurred pixel is from the image's mean Lab color. A plain colorful
    subject (a flower, a dish) stands out even when the background has more edges."""
    lab = cv2.cvtColor(image[..., :3].astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    blur = cv2.GaussianBlur(lab, (0, 0), 2.0)
    return np.linalg.norm(blur - lab.reshape(-1, 3).mean(axis=0), axis=2).astype(np.float64)


def square_window(image: np.ndarray, focus_gain: float = 1.2) -> tuple[int, int, int]:
    """(y, x, side) of the square a stamp is cut from, centered on the subject.

    The window slides along the long side and goes where the subject most likely is.
    Two cues vote: edge density (detail) and color saliency (a plain colorful subject).
    It stays centered unless a cue finds a window scoring `focus_gain` times more than
    the center; then the cue with the larger ratio decides.
    """
    h, w = image.shape[:2]
    s = min(h, w)
    if h == w:
        return 0, 0, s
    long_side = max(h, w)
    scale = min(1.0, 200 / long_side)
    small = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small[..., :3], cv2.COLOR_BGR2GRAY)
    edges = (cv2.Canny(gray, 60, 150) > 0).astype(np.float64)
    axis = 0 if w > h else 1
    edge_profile = edges.sum(axis=axis)
    sal_profile = _color_saliency(small).sum(axis=axis)
    win = max(1, min(len(edge_profile), round(s * scale)))
    offset = (long_side - s) // 2
    gain = focus_gain
    # each cue votes on its own (adding them dilutes a clear peak): the one that points
    # away from the center most decisively wins
    for profile in (edge_profile, sal_profile):
        sums = np.convolve(profile, np.ones(win), mode="valid")
        center = sums[(len(sums) - 1) // 2]
        best = int(sums.argmax())
        if center > 0 and sums[best] > gain * center:
            gain = sums[best] / center
            offset = min(long_side - s, round(best / scale))
        elif center <= 0 < sums[best]:
            offset = min(long_side - s, round(best / scale))
            break
    return (0, offset, s) if w > h else (offset, 0, s)


def crop_square(image: np.ndarray, focus_gain: float = 1.2) -> np.ndarray:
    """Crop to a square (stamps are round or square), centered on the subject."""
    y, x, s = square_window(image, focus_gain)
    return image[y : y + s, x : x + s].copy()


def _smooth(small: np.ndarray, cfg: StampConfig) -> np.ndarray:
    # l0Smooth fails on images only a few px wide; mean-shift handles those
    if cfg.smoother == "l0" and hasattr(cv2, "ximgproc") and min(small.shape[:2]) >= 8:
        return cv2.ximgproc.l0Smooth(small, None, cfg.l0_lambda, 2.0)
    return smooth(small, cfg.smooth)


def is_night(image: np.ndarray) -> bool:
    """A photo is a night shot when most of it is dark (median luma below NIGHT_LUMA) and
    its light comes from small scattered points (windows, lamps). A dish on a black
    table is also mostly dark, but its bright part is one large area."""
    gray = cv2.cvtColor(resize_long_side(image[..., :3], 400), cv2.COLOR_BGR2GRAY)
    if float(np.median(gray)) >= NIGHT_LUMA:
        return False
    count, _, stats, _ = cv2.connectedComponentsWithStats((gray > NIGHT_BRIGHT_LUMA).astype(np.uint8))
    largest = stats[1:, cv2.CC_STAT_AREA].max() / gray.size if count > 1 else 0.0
    return largest < NIGHT_MAX_BLOB


def _segment(
    smoothed: np.ndarray, texture: np.ndarray, cfg: StampConfig, n_colors: int, night: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster, clean and classify the smoothed working image.

    Returns (labels, BGR colors, mask of a bright (daytime) sky; all False if none)."""
    sh, sw = smoothed.shape[:2]
    min_area = int(cfg.min_region * sh * sw)
    lab = to_lab(smoothed)
    labels, centers = quantize(smoothed, n_colors, cfg.seed, cfg.chroma_weight)
    window = cfg.majority | 1 if cfg.majority > 1 else 1
    labels = majority_filter(labels, centers.shape[0], size=window, iterations=2)
    n_base = centers.shape[0]
    labels, centers = extract_accents(lab, labels, centers, cfg.accent_chroma, max(1, min_area // 4))
    accent_pixels = labels >= n_base

    sky: frozenset[int] = frozenset()
    far: frozenset[int] = frozenset()
    if cfg.detect_sky:
        labels, centers, sky_label = extract_sky(labels, centers, texture=texture, lab=lab)
        sky = frozenset() if sky_label is None else frozenset({sky_label})
        palette = get_palette(cfg.palette)
        # the far silhouette exists to give fixed palettes a distant-skyline color; with
        # photo hues (natural/original) it only adds patches (e.g. on a distant sea)
        uses_fixed_palette = palette is not None and bool(palette.tones)
        if sky_label is not None and cfg.detect_far and uses_fixed_palette:
            labels, centers, far_label = extract_far(labels, centers, sky_label, cfg.far_delta, lab=lab)
            far = frozenset() if far_label is None else frozenset({far_label})

    if cfg.palette == NATURAL and not night:
        centers = tint_shaded_foliage(labels, centers)
    weights = np.bincount(labels.ravel(), minlength=centers.shape[0]) if cfg.tone_mapping == "area" else None
    l_range = NIGHT_L_RANGE if night else None
    colors = assign_colors(
        centers, get_palette(cfg.palette), sky, cfg.accent_chroma, weights, far_labels=far, l_range=l_range
    )
    bright_sky = bool(sky) and float(centers[next(iter(sky)), 0]) >= BRIGHT_SKY_L
    sky_mask = np.isin(labels, list(sky)) if bright_sky else np.zeros(labels.shape, bool)
    labels, colors, _ = _unify_by_color(labels, colors)
    return merge_small_regions(labels, min_area, protect=accent_pixels), colors, sky_mask


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
    if cfg.shape != "rect" and cfg.crop == "before":
        img = crop_square(img)
    img = resize_long_side(img, cfg.max_size)
    if cfg.white_balance:  # after resizing: a 12 MP photo needs no full-size float copies
        img = white_balance(img)
    h, w = img.shape[:2]

    density = edge_density(img)
    work = cfg.work_size or auto_work_size(density)
    small = resize_long_side(img, work)
    sh, sw = small.shape[:2]
    texture = cv2.resize(texture_map(img), (sw, sh), interpolation=cv2.INTER_AREA)
    smoothed = _smooth(small, cfg)
    night = cfg.night == "on" or (cfg.night == "auto" and is_night(img))
    small_labels, colors, small_sky = _segment(smoothed, texture, cfg, cfg.colors, night)

    labels = upscale_labels(small_labels, (h, w), len(colors)) if (sh, sw) != (h, w) else small_labels
    shapes = labels_to_shapes(labels, colors, cfg.simplify * float(np.hypot(h, w)), cfg.snap_deg)
    base = labels_to_shapes(labels, colors, SVG_BASE_EPSILON)  # near-exact, under the smooth shapes in the SVG
    flat = render_png(shapes, (h, w), base=colors[labels])
    used = colors[np.unique(labels)]

    overlay = ""
    if cfg.details and cfg.palette == NATURAL and not night:
        spots, spot_colors = find_details(resize_long_side(img, LINE_SIZE), (h, w))
        spots, spot_colors = select_details(spots, spot_colors, flat)
        if len(spot_colors):
            flat = paint_details(flat, spots, spot_colors)
            overlay = details_svg(spots, spot_colors)
    if cfg.lines and night:
        # outlines on a night photo become a maze; the lit windows are what you recognise
        mask = light_spots(resize_long_side(img, LINE_SIZE), (h, w))
        ink = hex_to_bgr(LIGHT_INK)
        overlay += lines_svg(flat, mask, ink, "dark")
        flat = draw_lines(flat, mask, ink, "dark")
    elif cfg.lines:
        mask = _outline_mask(img, (h, w), cfg) & ~_soft_sky_edges(img, small_sky, (h, w))
        ink, on = (hex_to_bgr(LIGHT_INK), "dark") if night else (hex_to_bgr(INK), "light")
        overlay += lines_svg(flat, mask, ink, on)
        flat = draw_lines(flat, mask, ink, on)
    if cfg.shape == "rect":
        return StampResult(flat, shapes_to_svg(shapes, (h, w), overlay=overlay, base=base), labels, used)

    y, x, side = (0, 0, 0)
    if cfg.crop == "after":  # cut the stamp out of the finished full-photo art
        y, x, side = square_window(img)
        flat, labels = flat[y : y + side, x : x + side].copy(), labels[y : y + side, x : x + side].copy()
        used = colors[np.unique(labels)]
        shapes = [sh for sh in shapes if _overlaps(sh.points, x, y, side)]  # keep the SVG small
        base = [sh for sh in base if _overlaps(sh.points, x, y, side)]
        h = w = side
    ring_color, paper = _frame_colors(used, cfg.palette)
    out = frame_stamp(flat, cfg.shape, ring_color, cfg.ring, cfg.ring_gap, paper)
    svg = shapes_to_svg(
        shapes, (h, w), cfg.shape, ring_color, cfg.ring, cfg.ring_gap, paper, overlay=overlay, offset=(x, y),
        base=base,
    )
    return StampResult(out, svg, labels, used)


def _overlaps(points: np.ndarray, x: int, y: int, side: int) -> bool:
    """Whether a polygon's bounding box touches the square window at (x, y)."""
    (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)
    return bool(x1 >= x and x0 < x + side and y1 >= y and y0 < y + side)


def _soft_sky_edges(img: np.ndarray, sky: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Mask of soft edges (cloud rims) above the skyline of a daytime sky.

    Above the skyline (the lowest sky pixel) mostly clouds and things
    standing against the sky are found. Cloud rims are soft: the brightness changes
    gently, while a spire or a roof edge is sharp. Lines on soft rims are dropped."""
    h, w = size
    if not sky.any():
        return np.zeros((h, w), bool)
    sh = sky.shape[0]
    # a wide cloud cuts a deep notch into the sky above it, so one skyline (the lowest
    # sky pixel) is used for the whole width; sharp edges above it keep their lines
    skyline = int(np.nonzero(sky.any(axis=1))[0].max()) + int(SKYLINE_MARGIN * sh)  # lines sit beside it
    above = np.broadcast_to(np.arange(sh)[:, None] <= skyline, sky.shape)
    above = cv2.resize(above.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    gray = cv2.cvtColor(resize_long_side(img, LINE_SIZE), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    grad = cv2.dilate(grad, np.ones((7, 7), np.uint8))  # a line sits beside its edge
    bright = cv2.GaussianBlur(gray, (0, 0), 4.0) > CLOUD_MIN_LUMA  # clouds are light; a hazy ridge is not
    soft = cv2.resize(((grad < SOFT_EDGE_GRAD) & bright).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return above & soft.astype(bool)


def _stretch_contrast(image: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    """Stretch the 2nd..98th percentile to the full range so that low-contrast (night)
    photos still have edges strong enough for smoothing and DoG to keep."""
    lo, hi = np.percentile(image, [low, high])
    if hi - lo < 1:
        return image.copy()
    return np.clip((image.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def _outline_mask(img: np.ndarray, size: tuple[int, int], cfg: StampConfig) -> np.ndarray:
    """DoG outlines found on a LINE_SIZE copy of the photo (smoothed the same way as the
    regions), enlarged smoothly to the output size. Lines need more resolution than the
    color regions: at the small work size windows and building edges disappear."""
    h, w = size
    base = _smooth(_stretch_contrast(resize_long_side(img, LINE_SIZE)), cfg)
    sh, sw = base.shape[:2]
    small = dog_lines(base, max(0.5, cfg.line_sigma * max(sh, sw) / 600.0), cfg.line_threshold)
    small = clean_lines(small, int(cfg.line_min_length * max(sh, sw)), 1)
    big = cv2.resize(small.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.35
    width = max(1, int(round(cfg.line_width * max(h, w) / 600.0)))
    return clean_lines(big, 1, width)
