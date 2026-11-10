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
    sums = np.zeros((len(ids) + 1, 3), np.float64)
    np.add.at(sums, flat, lab.reshape(-1, 3))
    means = sums[1:] / np.bincount(flat, minlength=len(ids) + 1)[1:, None]
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


def details_svg(spots: np.ndarray, colors: np.ndarray) -> str:
    """SVG group with one path per detail color."""
    parts = ['<g id="details">']
    for k, color in enumerate(colors, start=1):
        contours, _ = cv2.findContours((spots == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        d = " ".join("M" + " L".join(f"{x + 0.5:g} {y + 0.5:g}" for x, y in c.reshape(-1, 2)) + " Z" for c in contours)
        if d:
            hx = bgr_to_hex(color)
            parts.append(f'<path fill="{hx}" stroke="{hx}" stroke-width="1" stroke-linejoin="round" d="{d}"/>')
    parts.append("</g>")
    return "".join(parts)
