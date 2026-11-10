"""Label map -> simplified polygons -> PNG / SVG."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .palette import bgr_to_hex

SHAPES = ("rect", "circle", "rounded")


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
        chain = cv2.CHAIN_APPROX_NONE if epsilon > 0 else cv2.CHAIN_APPROX_SIMPLE
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, chain)
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


def _svg_outline(shape: str, w: int, h: int, inset: float, attrs: str) -> str:
    """SVG element for the stamp outline shrunk by `inset` px."""
    s = min(w, h)
    if shape == "circle":
        return f'<circle cx="{w / 2:g}" cy="{h / 2:g}" r="{s / 2 - inset:g}" {attrs}/>'
    x, y, side = (w - s) / 2 + inset, (h - s) / 2 + inset, s - 2 * inset
    rx = max(0.0, ROUNDED_RADIUS * s - inset)
    return f'<rect x="{x:g}" y="{y:g}" width="{side:g}" height="{side:g}" rx="{rx:g}" {attrs}/>'


def _shape_path(shape: Shape) -> str:
    d = "M" + " L".join(f"{x + 0.5:g} {y + 0.5:g}" for x, y in shape.points) + " Z"
    c = bgr_to_hex(shape.color)
    return f'<path d="{d}" fill="{c}" stroke="{c}" stroke-width="1" stroke-linejoin="round"/>'


def shapes_to_svg(
    shapes: list[Shape],
    size: tuple[int, int],
    shape: str = "rect",
    ring_color: tuple[int, int, int] | None = None,
    ring: float = 0.0,
    gap: float = 0.0,
    paper: tuple[int, int, int] = (255, 255, 255),
    overlay: str = "",
    offset: tuple[int, int] = (0, 0),
    base: list[Shape] | None = None,
) -> str:
    """Serialize shapes as an SVG document (one <path> per region), with an optional stamp frame.

    `overlay` (e.g. the outline group) is drawn over the regions, inside the stamp clip.
    `base` (shapes traced without simplification) is drawn first, like the label raster
    under the PNG: simplified neighbours leave thin gaps, and the gaps then show the right
    local color instead of the background.
    `offset` (x, y) shifts the art so that a window of `size` starting there is shown
    (used to cut a stamp out of art drawn for the whole photo).

    `ring` and `gap` are fractions of the stamp size, as in frame_stamp.
    """
    h, w = size[:2]
    bg = bgr_to_hex(_background(shapes))
    head = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">']
    clip = ""
    if shape != "rect":
        head.append(f'<defs><clipPath id="stamp">{_svg_outline(shape, w, h, 0, "")}</clipPath></defs>')
        clip = ' clip-path="url(#stamp)"'
    body = [f"<g{clip}>", f'<rect width="{w}" height="{h}" fill="{bg}"/>']
    if offset != (0, 0):
        body.append(f'<g transform="translate({-offset[0]} {-offset[1]})">')
    if base:
        body.append('<g id="base">')
        body.extend(_shape_path(s) for s in base)
        body.append("</g>")
    body.append('<g id="shapes">')
    body.extend(_shape_path(s) for s in shapes)
    body.append("</g>")
    if overlay:
        body.append(overlay)
    if offset != (0, 0):
        body.append("</g>")
    body.append("</g>")
    if shape != "rect" and ring_color is not None and ring > 0:
        side = min(w, h)
        r_px, g_px = ring * side, gap * side
        if g_px > 0:
            attrs = f'fill="none" stroke="{bgr_to_hex(paper)}" stroke-width="{g_px:g}"'
            body.append(_svg_outline(shape, w, h, r_px + g_px / 2, attrs))
        attrs = f'fill="none" stroke="{bgr_to_hex(ring_color)}" stroke-width="{r_px:g}"'
        body.append(_svg_outline(shape, w, h, r_px / 2, attrs))
    body.append("</svg>")
    return "\n".join(head + body) + "\n"


ROUNDED_RADIUS = 0.18  # corner radius of the "rounded" stamp, as a fraction of its size
_SS = 4  # supersampling factor for anti-aliased frame masks
_SS_MAX_PIXELS = 16_000_000  # cap on the supersampled mask size (memory)


def _coverage(h: int, w: int, shape: str, inset: float) -> np.ndarray:
    """Anti-aliased coverage (0..1) of the stamp outline shrunk by `inset` px."""
    k = max(1, min(_SS, int((_SS_MAX_PIXELS / max(1, h * w)) ** 0.5)))
    H, W = h * k, w * k
    s = min(w, h)
    mask = np.zeros((H, W), np.uint8)
    if shape == "rect":
        mask[:] = 1
    elif shape == "circle":
        r = int(round((s / 2 - inset) * k))
        if r > 0:
            cv2.circle(mask, (W // 2, H // 2), r, 1, thickness=-1)
    else:
        x0, y0 = int(round(((w - s) / 2 + inset) * k)), int(round(((h - s) / 2 + inset) * k))
        x1, y1 = W - x0 - 1, H - y0 - 1
        r = int(round(max(0.0, ROUNDED_RADIUS * s - inset) * k))
        if x1 > x0 and y1 > y0:
            cv2.rectangle(mask, (x0 + r, y0), (x1 - r, y1), 1, -1)
            cv2.rectangle(mask, (x0, y0 + r), (x1, y1 - r), 1, -1)
            for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
                cv2.circle(mask, (cx, cy), r, 1, -1)
    return cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)


def frame_stamp(
    image: np.ndarray,
    shape: str,
    ring_color: tuple[int, int, int],
    ring: float = 0.035,
    gap: float = 0.02,
    paper: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Cut the art into a stamp shape with a solid ring and a paper-colored gap.

    ring / gap are fractions of the stamp size (the shorter image side).
    Returns a new BGRA image; outside the stamp is transparent.
    """
    if shape not in SHAPES:
        raise ValueError(f"unknown shape {shape!r} (choose from: {', '.join(SHAPES)})")
    h, w = image.shape[:2]
    s = min(h, w)
    outer = _coverage(h, w, shape, 0)
    inner = _coverage(h, w, shape, ring * s)
    art = _coverage(h, w, shape, (ring + gap) * s)
    rgb = image[..., :3].astype(np.float32)
    weights = (art, inner - art, outer - inner)
    layers = (rgb, np.array(paper, np.float32), np.array(ring_color, np.float32))
    color = sum(wgt[..., None] * layer for wgt, layer in zip(weights, layers))
    color = color / np.maximum(outer, 1e-6)[..., None]
    alpha = np.round(outer * 255).astype(np.uint8)
    return np.dstack([np.clip(np.round(color), 0, 255).astype(np.uint8), alpha])
