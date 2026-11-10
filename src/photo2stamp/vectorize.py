"""Label map -> simplified polygons -> PNG / SVG."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .palette import bgr_to_hex

SHAPES = ("rect", "circle")


@dataclass(frozen=True)
class Shape:
    points: np.ndarray = field(compare=False)  # (N, 2) int32, x/y pixel coordinates
    color: tuple[int, int, int]  # BGR
    area: float


def snap_vertical(points: np.ndarray, max_deg: float) -> np.ndarray:
    """Make polygon edges that are within max_deg of vertical exactly vertical.

    Buildings read as buildings when their walls are straight; this is the main
    trick that turns blobby regions into architectural shapes. Returns a new array.
    """
    out = points.astype(np.float64).copy()
    if max_deg <= 0 or len(out) < 3:
        return points.copy()
    tan = np.tan(np.radians(max_deg))
    n = len(out)
    for i in range(n):
        j = (i + 1) % n
        dx, dy = out[j] - out[i]
        if dy != 0 and abs(dx) <= tan * abs(dy):
            x = (out[i, 0] + out[j, 0]) / 2
            out[i, 0] = out[j, 0] = x
    return np.round(out).astype(points.dtype)


def labels_to_shapes(
    labels: np.ndarray, colors: np.ndarray, epsilon: float, snap_deg: float = 0.0
) -> list[Shape]:
    """Trace every connected region, simplify it with Douglas-Peucker and snap walls vertical.

    Shapes are sorted by area (largest first) so they can be painted back to front.
    """
    shapes: list[Shape] = []
    for lab in np.unique(labels):
        mask = (labels == lab).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        color = tuple(int(c) for c in colors[lab])
        for contour in contours:
            approx = cv2.approxPolyDP(contour, epsilon, True) if epsilon > 0 else contour
            if len(approx) < 3:
                continue
            points = snap_vertical(approx.reshape(-1, 2).astype(np.int32), snap_deg)
            shapes.append(Shape(points, color, float(cv2.contourArea(contour))))
    return sorted(shapes, key=lambda s: s.area, reverse=True)


def _background(shapes: list[Shape]) -> tuple[int, int, int]:
    return shapes[0].color if shapes else (255, 255, 255)


def render_png(
    shapes: list[Shape], size: tuple[int, int], supersample: int = 2, base: np.ndarray | None = None
) -> np.ndarray:
    """Paint shapes back to front. Supersampling gives smooth (anti-aliased) edges.

    `base` (HxWx3 BGR) is painted first so that slivers between simplified polygons
    show the right local color instead of the background.
    """
    h, w = size[:2]
    s = max(1, supersample)
    if base is not None:
        canvas = cv2.resize(base, (w * s, h * s), interpolation=cv2.INTER_NEAREST)
    else:
        canvas = np.empty((h * s, w * s, 3), np.uint8)
        canvas[:] = _background(shapes)
    for shape in shapes:
        pts = (shape.points * s + s // 2).reshape(-1, 1, 2)
        cv2.fillPoly(canvas, [pts], shape.color, lineType=cv2.LINE_8)
    if s == 1:
        return canvas
    return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)


def shapes_to_svg(shapes: list[Shape], size: tuple[int, int], shape: str = "rect") -> str:
    """Serialize shapes as an SVG document (one <path> per region)."""
    h, w = size[:2]
    bg = bgr_to_hex(_background(shapes))
    head = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">']
    clip = ""
    if shape == "circle":
        r = min(w, h) / 2
        head.append(f'<defs><clipPath id="stamp"><circle cx="{w / 2}" cy="{h / 2}" r="{r}"/></clipPath></defs>')
        clip = ' clip-path="url(#stamp)"'
    body = [f"<g{clip}>", f'<rect width="{w}" height="{h}" fill="{bg}"/>']
    for s in shapes:
        d = "M" + " L".join(f"{x + 0.5:g} {y + 0.5:g}" for x, y in s.points) + " Z"
        c = bgr_to_hex(s.color)
        body.append(f'<path d="{d}" fill="{c}" stroke="{c}" stroke-width="1" stroke-linejoin="round"/>')
    body.append("</g></svg>")
    return "\n".join(head + body) + "\n"


def apply_shape_mask(image: np.ndarray, shape: str) -> np.ndarray:
    """Return a BGRA copy whose alpha channel cuts the image into the given stamp shape."""
    if shape not in SHAPES:
        raise ValueError(f"unknown shape {shape!r} (choose from: {', '.join(SHAPES)})")
    h, w = image.shape[:2]
    alpha = np.full((h, w), 255, np.uint8)
    if shape == "circle":
        alpha[:] = 0
        cv2.circle(alpha, (w // 2, h // 2), min(w, h) // 2, 255, thickness=-1, lineType=cv2.LINE_AA)
    return np.dstack([image[..., :3], alpha])
