"""Small colored details drawn as flat spots over the color regions.

Painted carvings, flowers in a bed or fruit on a stall are smaller than the regions
k-means can keep at the work size, so their colors are averaged into grey. They are
found again on the finer LINE_SIZE copy of the photo: pixels whose color differs from
the local average color (a/b distance) and that are colorful themselves. Each small
connected spot is painted in its own natural-palette color.

A stamp holds only a few such details. Grass, leaves and gravel also produce colored
specks, many and small, so specks of a few pixels are dropped, spots whose color barely
differs from the fill under them are dropped, and only the DETAIL_BUDGET largest and
most contrasting spots are drawn.
"""

from __future__ import annotations

import cv2
import numpy as np

from .palette import bgr_to_hex, detail_colors
from .segment import to_lab

DETAIL_DIFF = 10.0  # a/b distance from the local average color that makes a pixel stand out
DETAIL_CHROMA = 12.0  # ... and chroma it needs itself (a grey stone chip is not a detail)
DETAIL_LOCAL_SIGMA = 8.0  # px at the detail resolution: size of the "local average" window
DETAIL_SPECK_AREA = 12  # px at the detail resolution: smaller specks are pixel noise
DETAIL_MIN_AREA = 40  # px: specks between the two count as texture (grass, leaf glints), not detail
DETAIL_MAX_AREA = 0.004  # fraction of the frame: larger patches belong to the regions
DETAIL_MAX_SPOTS = 400  # more spots than this is texture (foliage, gravel), not detail
DETAIL_BUDGET = 60  # at most this many spots are drawn: the largest and most contrasting
DETAIL_MIN_CONTRAST = 15.0  # Lab distance to the fill around a spot below which it is only noise


def find_details(image: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return (spot label map at `size` with 0 = none, (K,3) BGR colors for labels 1..K).

    `image` is the photo at detail resolution (BGR uint8).
    """
    h, w = size
    smooth = cv2.bilateralFilter(image[..., :3], 5, 30, 5)
    lab = to_lab(smooth)
    ab = lab[..., 1:]
    local = cv2.GaussianBlur(ab, (0, 0), DETAIL_LOCAL_SIGMA)
    diff = np.linalg.norm(ab - local, axis=2)
    chroma = np.hypot(ab[..., 0], ab[..., 1])
    mask = ((diff > DETAIL_DIFF) & (chroma > DETAIL_CHROMA)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count, comp, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA]
    small_enough = area <= DETAIL_MAX_AREA * mask.size
    specks = (area >= DETAIL_SPECK_AREA) & small_enough
    specks[0] = False
    keep = specks & (area >= DETAIL_MIN_AREA)
    ids = np.flatnonzero(keep)
    # texture is judged on all specks: a lawn makes hundreds, most of them tiny
    if len(ids) == 0 or int(specks.sum()) > DETAIL_MAX_SPOTS:
        return np.zeros((h, w), np.int32), np.empty((0, 3), np.uint8)
    remap = np.zeros(count, np.int32)
    remap[ids] = np.arange(1, len(ids) + 1)
    spots = remap[comp]
    flat = spots.ravel()
    n = len(ids) + 1
    counts = np.bincount(flat, minlength=n)[1:, None]
    means = np.stack([np.bincount(flat, weights=lab[..., c].ravel(), minlength=n)[1:] for c in range(3)], axis=1)
    means = means / counts
    big = cv2.resize(spots.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    return big, detail_colors(means.astype(np.float32))


def select_details(
    spots: np.ndarray, colors: np.ndarray, flat: np.ndarray, budget: int = DETAIL_BUDGET
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the spots that stand out from the fill around them, at most `budget` of them.

    Each spot is scored by sqrt(area) x its Lab distance to the fill in a thin ring
    around it, so large spots in a clearly different color win.
    """
    n = len(colors)
    if n == 0:
        return spots, colors
    ids = spots.ravel()
    area = np.bincount(ids, minlength=n + 1)[1:]
    ring = cv2.dilate(spots.astype(np.float32), np.ones((5, 5), np.uint8)).astype(np.int32)
    ring[spots > 0] = 0
    ring_ids = ring.ravel()
    ring_count = np.maximum(np.bincount(ring_ids, minlength=n + 1)[1:], 1)
    lab = to_lab(flat[..., :3]).reshape(-1, 3).astype(np.float64)
    fill = np.stack([np.bincount(ring_ids, weights=lab[:, c], minlength=n + 1)[1:] for c in range(3)], axis=1)
    contrast = np.linalg.norm(to_lab(colors[None])[0].astype(np.float64) - fill / ring_count[:, None], axis=1)
    score = np.where((contrast >= DETAIL_MIN_CONTRAST) & (area > 0), np.sqrt(area) * contrast, -1.0)
    order = np.argsort(-score, kind="stable")[:budget]
    keep = np.sort(order[score[order] > 0])
    remap = np.zeros(n + 1, np.int32)
    remap[keep + 1] = np.arange(1, len(keep) + 1)
    return remap[spots], colors[keep]


def paint_details(flat: np.ndarray, spots: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Return a copy of `flat` with the spots painted in their colors."""
    out = flat[..., :3].copy()
    if len(colors):
        lut = np.vstack([np.zeros((1, 3), np.uint8), colors])
        on = spots > 0
        out[on] = lut[spots[on]]
    return out


def _label_boxes(spots: np.ndarray, n: int) -> list[tuple[int, int, int, int] | None]:
    """Bounding box (y0, y1, x0, x1) of each label 1..n in one pass (index 0 unused)."""
    ys, xs = np.nonzero(spots)
    lab = spots[ys, xs]
    boxes: list[tuple[int, int, int, int] | None] = [None] * (n + 1)
    if len(lab) == 0:
        return boxes
    order = np.argsort(lab, kind="stable")
    lab, ys, xs = lab[order], ys[order], xs[order]
    starts = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
    ends = np.r_[starts[1:], len(lab)]
    for s0, e0 in zip(starts, ends):
        boxes[int(lab[s0])] = (int(ys[s0:e0].min()), int(ys[s0:e0].max()) + 1, int(xs[s0:e0].min()), int(xs[s0:e0].max()) + 1)
    return boxes


def details_svg(spots: np.ndarray, colors: np.ndarray) -> str:
    """SVG group with one path per detail color."""
    parts = ['<g id="details">']
    boxes = _label_boxes(spots, len(colors))
    for k, color in enumerate(colors, start=1):
        if boxes[k] is None:
            continue
        y0, y1, x0, x1 = boxes[k]
        sub = (spots[y0:y1, x0:x1] == k).astype(np.uint8)
        contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(int(x0), int(y0)))
        d = " ".join("M" + " L".join(f"{x + 0.5:g} {y + 0.5:g}" for x, y in c.reshape(-1, 2)) + " Z" for c in contours)
        if d:
            hx = bgr_to_hex(color)
            parts.append(f'<path fill="{hx}" stroke="{hx}" stroke-width="1" stroke-linejoin="round" d="{d}"/>')
    parts.append("</g>")
    return "".join(parts)
