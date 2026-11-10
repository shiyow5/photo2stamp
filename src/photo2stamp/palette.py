"""Color palettes and the mapping from color clusters to flat palette colors."""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")

BGR = tuple[int, int, int]


@dataclass(frozen=True)
class Palette:
    """A flat-illustration palette.

    tones:   neutral colors ordered dark -> light; clusters are mapped by lightness rank.
    sky:     color used for the detected sky region.
    accents: vivid colors used for strongly saturated clusters (signs, lights, plants).
    far:     color for hazy distant scenery right under the sky (skyline silhouette).
    """

    name: str
    tones: tuple[str, ...]
    sky: str
    accents: tuple[str, ...]
    far: str


PRESETS: dict[str, Palette] = {
    p.name: p
    for p in (
        Palette(
            name="teal",
            tones=("#173F4A", "#22525E", "#3A6875", "#5F8590", "#93ABB3", "#E8DCBB", "#F3F1EA"),
            sky="#8DCAE6",
            accents=("#D9472F", "#EB8F35", "#F2C24B", "#3E7D6B"),
            far="#2F6684",
        ),
        Palette(
            name="warm",
            tones=("#3B2A26", "#6B3F2F", "#A65E3F", "#D99A6C", "#F2D3A7", "#FBF1E1"),
            sky="#F4D8B8",
            accents=("#2F6F73", "#E0A526", "#C2412D"),
            far="#C98F6B",
        ),
        Palette(
            name="mono",
            tones=("#1A1A1A", "#4D4D4D", "#8C8C8C", "#C8C8C8", "#F2F2F2"),
            sky="#DADADA",
            accents=("#D7263D",),
            far="#A8A8A8",
        ),
    )
}

ORIGINAL = "original"
NATURAL = "natural"
# A marker palette: colors are derived from the photo's own hues (see natural_colors).
NATURAL_PALETTE = Palette(name=NATURAL, tones=(), sky="", accents=(), far="")

# Hue anchors (Lab hue angle, degrees): red, orange, yellow, green, teal, blue, sky blue, purple.
# Measured skies in the examples sit at 245..272 deg; 280 rendered them lavender.
_HUE_ANCHORS = np.radians([30, 60, 88, 135, 190, 230, 262, 320])
_NEUTRAL_CHROMA = 10.0  # below this a color is treated as grey
_MUTED, _VIVID = 30.0, 55.0  # output chroma for muted / vivid colors
_L_RANGE = (34.0, 95.0)  # output lightness range: light enough for ink outlines to show
_L_STEP = 6.0  # lightness is snapped to this step so near-equal tones merge


def hex_to_bgr(value: str) -> BGR:
    match = _HEX_RE.match(value.strip())
    if not match:
        raise ValueError(f"invalid hex color: {value!r}")
    h = match.group(1)
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def bgr_to_hex(color: BGR) -> str:
    b, g, r = (int(c) for c in color)
    return f"#{r:02x}{g:02x}{b:02x}"


def get_palette(name: str) -> Palette | None:
    """Return a preset palette, or None for 'original' (keep the cluster colors)."""
    if name == ORIGINAL:
        return None
    if name == NATURAL:
        return NATURAL_PALETTE
    if name not in PRESETS:
        choices = ", ".join([NATURAL, *PRESETS, ORIGINAL])
        raise ValueError(f"unknown palette {name!r} (choose from: {choices})")
    return PRESETS[name]


def lab_to_bgr(lab: np.ndarray) -> np.ndarray:
    """Convert (N,3) float Lab (L in 0..100) to (N,3) uint8 BGR."""
    pixels = np.asarray(lab, np.float32).reshape(-1, 1, 3)
    bgr = cv2.cvtColor(pixels, cv2.COLOR_Lab2BGR)
    return np.clip(np.round(bgr * 255), 0, 255).astype(np.uint8).reshape(-1, 3)


def bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    """Convert (N,3) uint8 BGR to (N,3) float Lab."""
    pixels = (np.asarray(bgr, np.float32) / 255.0).reshape(-1, 1, 3)
    return cv2.cvtColor(pixels, cv2.COLOR_BGR2Lab).reshape(-1, 3)


def _hue(lab: np.ndarray) -> np.ndarray:
    return np.arctan2(lab[..., 2], lab[..., 1])


def _nearest_accent(center: np.ndarray, accents_lab: np.ndarray) -> int:
    diff = np.abs(_hue(accents_lab) - _hue(center))
    diff = np.minimum(diff, 2 * np.pi - diff)
    return int(np.argmin(diff))


def assign_colors(
    centers_lab: np.ndarray,
    palette: Palette | None,
    sky_labels: frozenset[int] = frozenset(),
    accent_chroma: float = 35.0,
    weights: np.ndarray | None = None,
    far_labels: frozenset[int] = frozenset(),
    l_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Map each cluster center (Lab) to a flat BGR color. Returns a new (K,3) uint8 array.

    Neutral clusters are sorted by lightness. Without `weights` each cluster takes the
    next tone by rank. With `weights` (pixel counts) tones are assigned by the cluster's
    position in the cumulative area, like histogram equalization, so a dark photo still
    uses the whole tone range.
    """
    centers = np.asarray(centers_lab, np.float32)
    if palette is None:
        return lab_to_bgr(centers)
    if palette.name == NATURAL:
        return natural_colors(centers, weights, l_range)

    k = centers.shape[0]
    tones = np.array([hex_to_bgr(c) for c in palette.tones], np.uint8)
    accents = np.array([hex_to_bgr(c) for c in palette.accents], np.uint8)
    accents_lab = bgr_to_lab(accents) if len(accents) else np.empty((0, 3), np.float32)
    chroma = np.hypot(centers[:, 1], centers[:, 2])

    out = np.zeros((k, 3), np.uint8)
    neutral: list[int] = []
    for i in range(k):
        if i in sky_labels:
            out[i] = hex_to_bgr(palette.sky)
        elif i in far_labels:
            out[i] = hex_to_bgr(palette.far)
        elif len(accents) and chroma[i] >= accent_chroma:
            out[i] = accents[_nearest_accent(centers[i], accents_lab)]
        else:
            neutral.append(i)

    if neutral:
        order = sorted(neutral, key=lambda i: float(centers[i, 0]))
        for idx, tone in zip(order, _tone_indices(order, len(tones), weights)):
            out[idx] = tones[tone]
    return out


def _tone_indices(order: list[int], n_tones: int, weights: np.ndarray | None) -> list[int]:
    n = len(order)
    if weights is None or float(np.sum(np.asarray(weights)[order])) <= 0:
        if n == 1:
            return [n_tones // 2]
        return [round(rank * (n_tones - 1) / (n - 1)) for rank in range(n)]
    w = np.asarray(weights, np.float64)[order]
    mids = (np.cumsum(w) - w / 2) / w.sum()
    return [min(n_tones - 1, int(m * n_tones)) for m in mids]


def _equalized_lightness(L: np.ndarray, weights: np.ndarray | None, l_range: tuple[float, float]) -> np.ndarray:
    lo, hi = l_range
    if weights is None or len(L) < 2:
        span = float(L.max() - L.min())
        return np.full_like(L, (lo + hi) / 2) if span < 1e-6 else lo + (L - L.min()) / span * (hi - lo)
    w = np.asarray(weights, np.float64)
    total = max(float(w.sum()), 1e-9)
    below = np.array([w[L < v].sum() + w[L == v].sum() / 2 for v in L]) / total
    return (lo + below * (hi - lo)).astype(np.float32)


def natural_colors(
    centers_lab: np.ndarray, weights: np.ndarray | None = None, l_range: tuple[float, float] | None = None
) -> np.ndarray:
    """Stylize cluster colors while keeping what makes things recognizable: their hue.

    Sky stays blue and trees stay green, but every hue is snapped to one of a few
    anchors, saturation is set to one of two levels (muted / vivid), and lightness is
    spread over a fixed range (half photo lightness, half area-equalized) in clear steps.
    Greys stay grey. Returns a new (K,3) uint8 BGR array.
    """
    c = np.asarray(centers_lab, np.float32)
    L, a, b = c[:, 0], c[:, 1], c[:, 2]
    chroma = np.hypot(a, b)
    hue = np.arctan2(b, a)
    diff = np.abs((hue[:, None] - _HUE_ANCHORS[None, :] + np.pi) % (2 * np.pi) - np.pi)
    snapped = _HUE_ANCHORS[diff.argmin(axis=1)]
    out_c = np.where(chroma < _NEUTRAL_CHROMA, 3.0, np.where(chroma >= 35.0, _VIVID, _MUTED))
    out_h = np.where(chroma < _NEUTRAL_CHROMA, np.radians(70), snapped)  # greys get a warm paper tint

    lo, hi = l_range or _L_RANGE
    stretched = lo + (L - L.min()) / max(float(L.max() - L.min()), 1e-6) * (hi - lo) if len(L) > 1 else L
    out_l = 0.5 * stretched + 0.5 * _equalized_lightness(L, weights, (lo, hi))
    out_l = np.clip(np.round(out_l / _L_STEP) * _L_STEP, lo, hi)
    out_c = out_c * np.clip((L - 10.0) / 40.0, 0.15, 1.0)  # colors that were dark in the photo stay muted
    lab = np.stack([out_l, out_c * np.cos(out_h), out_c * np.sin(out_h)], axis=1)
    return lab_to_bgr(lab)
