"""Outline layer drawn over the flat color regions.

Flat color alone often hides what the subject is; outlines carry most of the
structure (Winnemoeller, Olsen & Gooch, "Real-Time Video Abstraction", SIGGRAPH 2006;
Winnemoeller et al., "XDoG", Computers & Graphics 2012). We use a plain
difference-of-Gaussians with tau = 1: classic XDoG (tau < 1) also fills flat dark
areas with ink, which turned night scenes into black blobs.
"""

from __future__ import annotations

import cv2
import numpy as np

from .palette import bgr_to_hex

DARK_FILL_LUMA = 60  # no lines on fills darker than this (0..255): ink would not show


def dog_lines(image: np.ndarray, sigma: float = 2.0, threshold: float = 0.012) -> np.ndarray:
    """Boolean mask of edge lines: the dark side of every luminance step."""
    gray = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    dog = cv2.GaussianBlur(gray, (0, 0), sigma) - cv2.GaussianBlur(gray, (0, 0), sigma * 1.6)
    lines = dog < -threshold
    margin = int(np.ceil(3 * sigma))  # blur borders produce false lines along the image edge
    if margin:
        lines[:margin] = lines[-margin:] = False
        lines[:, :margin] = lines[:, -margin:] = False
    return lines


def clean_lines(mask: np.ndarray, min_length: int, width: int) -> np.ndarray:
    """Drop line fragments shorter than min_length px and thicken the rest to `width` px."""
    count, comp, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    extent = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = extent >= max(1, min_length)
    keep[0] = False
    out = keep[comp].astype(np.uint8)
    if width > 1:
        out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width)))
    return out.astype(bool)


def _luma(image: np.ndarray) -> np.ndarray:
    return image[..., :3].astype(np.float32) @ np.array([0.114, 0.587, 0.299], np.float32)


LIGHT_FILL_LUMA = 200  # light (night-mode) lines are not drawn on fills brighter than this
LIGHT_GAP = 0.015  # night lights closer than this (fraction of the long side) to a placed dot are left out


def _visible(fill: np.ndarray, on: str) -> np.ndarray:
    """Where a line of this kind shows: dark ink on light fills, or light ink on dark fills."""
    if on == "dark":
        return _luma(fill) < LIGHT_FILL_LUMA
    return _luma(fill) >= DARK_FILL_LUMA


def draw_lines(
    fill: np.ndarray, mask: np.ndarray, ink: tuple[int, int, int], on: str = "light"
) -> np.ndarray:
    """Return a copy of `fill` with ink painted on the line mask where it is visible.

    on="light": dark ink, skipped over dark fills. on="dark": light ink (night mode),
    skipped over very bright fills."""
    out = fill[..., :3].copy()
    out[mask & _visible(fill, on)] = ink
    return out


def _trace(mask: np.ndarray, color: tuple[int, int, int]) -> list[str]:
    """Trace a mask as one SVG path. findContours follows pixel centres, so the fill alone
    is half a pixel too thin on every side; a same-colour 1 px stroke restores the width,
    and round caps let 1-2 point contours (thin lines, dots) render instead of vanishing."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    subpaths = []
    for c in contours:
        pts = c.reshape(-1, 2)
        subpaths.append("M" + " L".join(f"{x + 0.5:g} {y + 0.5:g}" for x, y in pts) + " Z")
    if not subpaths:
        return []
    hex_color = bgr_to_hex(color)
    return [
        f'<path fill="{hex_color}" fill-rule="evenodd" stroke="{hex_color}" stroke-width="1" '
        f'stroke-linejoin="round" stroke-linecap="round" d="{" ".join(subpaths)}"/>'
    ]


def lines_svg(fill: np.ndarray, mask: np.ndarray, ink: tuple[int, int, int], on: str = "light") -> str:
    """SVG group with the visible part of the line mask traced as filled ink paths."""
    return '<g id="lines">' + "".join(_trace(mask & _visible(fill, on), ink)) + "</g>"


def light_spots(
    image: np.ndarray,
    size: tuple[int, int],
    min_luma: float = 150.0,
    percentile: float = 92.0,
    dot: float = 0.006,
    gap: float = LIGHT_GAP,
) -> np.ndarray:
    """Night mode: mask of lit windows and lamps, drawn as round dots.

    A pixel is a light when it is among the brightest `percentile` of the photo and
    brighter than `min_luma`. Each light becomes one dot of diameter `dot` x long side.
    A lit tower has hundreds of windows, so lights are placed largest first and one
    closer than `gap` x long side to a placed dot is left out: dense parts stay denser
    than lone lamps, but never turn into a field of specks.
    """
    gray = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
    bright = gray >= max(min_luma, float(np.percentile(gray, percentile)))
    h, w = size
    out = np.zeros((h, w), np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(bright.astype(np.uint8), connectivity=8)
    sy, sx = h / gray.shape[0], w / gray.shape[1]
    radius = max(1, int(round(dot * max(h, w) / 2)))
    centers = centroids[1:] * (sx, sy)
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA], kind="stable")
    min_dist = gap * max(h, w)
    cell = max(min_dist, 1.0)  # grid of placed dots: only the 3x3 cells around need checking
    grid: dict[tuple[int, int], list[np.ndarray]] = {}
    for i in order:
        p = centers[i]
        gx, gy = int(p[0] // cell), int(p[1] // cell)
        near = (q for dx in (-1, 0, 1) for dy in (-1, 0, 1) for q in grid.get((gx + dx, gy + dy), ()))
        if any(np.hypot(*(q - p)) < min_dist for q in near):
            continue
        grid.setdefault((gx, gy), []).append(p)
        cv2.circle(out, (int(round(p[0])), int(round(p[1]))), radius, 1, -1, lineType=cv2.LINE_8)
    return out.astype(bool)
