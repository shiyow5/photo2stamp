"""Small colored details drawn as flat spots over the color regions.

Painted carvings, flowers in a bed or fruit on a stall are smaller than the regions
k-means can keep at the work size, so their colors are averaged into grey. They are
found again on the finer LINE_SIZE copy of the photo: pixels whose color differs from
the local average color (a/b distance) and that are colorful themselves. Each small
connected spot is painted in its own natural-palette color.
"""

from __future__ import annotations

import cv2
import numpy as np

from .palette import bgr_to_hex, detail_colors
from .segment import to_lab

DETAIL_DIFF = 10.0  # a/b distance from the local average color that makes a pixel stand out
DETAIL_CHROMA = 12.0  # ... and chroma it needs itself (a grey stone chip is not a detail)
DETAIL_LOCAL_SIGMA = 8.0  # px at the detail resolution: size of the "local average" window
DETAIL_MIN_AREA = 12  # px at the detail resolution: smaller spots are noise
DETAIL_MAX_AREA = 0.004  # fraction of the frame: larger patches belong to the regions
DETAIL_MAX_SPOTS = 400  # more spots than this is texture (foliage, gravel), not detail


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
    keep = (area >= DETAIL_MIN_AREA) & (area <= DETAIL_MAX_AREA * mask.size)
    keep[0] = False
    ids = np.flatnonzero(keep)
    if len(ids) == 0 or len(ids) > DETAIL_MAX_SPOTS:
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
