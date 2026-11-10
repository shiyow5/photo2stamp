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
# Dull colors use olive (115) instead of yellow: a dull yellow in a photo is nearly always
# foliage (conifers measure ~94 deg), while real yellow (signs, lamps) is saturated.
_DULL_HUE_ANCHORS = np.radians([30, 60, 115, 135, 190, 230, 262, 320])
_DULL_CHROMA = 20.0  # a hard switch on purpose: the olive/yellow split needs one rule
_VIVID_INPUT = 35.0  # photo chroma above which a color counts as vivid (signs, lamps, flowers)
# Output chroma follows the photo's chroma (x gain, capped): a faint tint in a photo is
# mostly the light (blue skylight in shadows, overcast sky, glass) and must stay faint.
# The first _TINT_FLOOR of chroma is treated as noise (white balance, haze) and removed.
_CHROMA_GAIN = 3.5
_TINT_FLOOR = 6.0
_CHROMA_CAP = 64.0
_CHROMA_STEP = 12.0  # output chroma is snapped to this step so near-equal colors merge
# Below this lightness hues are unreliable (lighting, noise): chroma fades to 30%.
_SHADOW_L = (5.0, 25.0)
_NIGHT_SHADOW_L = (10.0, 50.0)  # night: fade longer and deeper (no maroon night sky)
_L_RANGE = (34.0, 95.0)  # output lightness range: light enough for ink outlines to show
_COLOR_L_MAX = 78.0  # max lightness of chromatic colors
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


def _max_chroma(L: np.ndarray, C: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Largest chroma <= C (per color) that stays inside sRGB at lightness L."""
    out = C.astype(np.float32).copy()
    for _ in range(25):
        lab = np.stack([L, out * np.cos(h), out * np.sin(h)], axis=1).astype(np.float32)
        rgb = cv2.cvtColor(lab.reshape(-1, 1, 3), cv2.COLOR_Lab2BGR)  # clips to 0..1
        back = cv2.cvtColor(rgb, cv2.COLOR_BGR2Lab).reshape(-1, 3)
        outside = np.linalg.norm(back - lab, axis=1) > 1.0
        if not outside.any():
            break
        out = np.where(outside, out * 0.93, out)
    else:  # near black / white no chroma fits: fall back to grey rather than a clipped hue
        out = np.where(outside, 0.0, out)
    return out


# Lightness a vivid color may give up to keep its chroma; one unit of L costs one unit of C.
_GAMUT_L_DROPS = (0.0, 4.0, 8.0, 12.0)


def _fit_gamut(
    L: np.ndarray, C: np.ndarray, h: np.ndarray, may_darken: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Bring each color inside sRGB without a hue shift (clipping would turn light blue
    into cyan). Colors in `may_darken` (vivid in the photo) may get slightly darker when
    that keeps much more chroma; others (e.g. a pale sky) keep their lightness."""
    cand_l = np.stack([np.where(may_darken, np.maximum(L - d, 0.0), L) for d in _GAMUT_L_DROPS])
    cand_c = np.stack([_max_chroma(lv, C, h) for lv in cand_l])
    score = cand_c - np.array(_GAMUT_L_DROPS)[:, None]
    best = score.argmax(axis=0)
    idx = np.arange(len(L))
    return cand_l[best, idx], cand_c[best, idx]


def _snap_hue(hue: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Nearest anchor angle (radians) for each hue."""
    diff = np.abs((hue[:, None] - anchors[None, :] + np.pi) % (2 * np.pi) - np.pi)
    return anchors[diff.argmin(axis=1)]


def natural_colors(
    centers_lab: np.ndarray, weights: np.ndarray | None = None, l_range: tuple[float, float] | None = None
) -> np.ndarray:
    """Stylize cluster colors while keeping what makes things recognizable: their hue.

    Sky stays blue and trees stay green, but every hue is snapped to one of a few
    anchors, saturation follows the photo's (boosted, in steps; faint tints stay grey)
    and lightness is spread over a fixed range (half photo lightness, half
    area-equalized) in clear steps. Colors are fitted into sRGB without hue shifts.
    Returns a new (K,3) uint8 BGR array.
    """
    c = np.asarray(centers_lab, np.float32)
    L, a, b = c[:, 0], c[:, 1], c[:, 2]
    chroma = np.hypot(a, b)
    hue = np.arctan2(b, a)
    snapped = np.where(chroma < _DULL_CHROMA, _snap_hue(hue, _DULL_HUE_ANCHORS), _snap_hue(hue, _HUE_ANCHORS))
    night = l_range is not None and l_range[1] < 80
    dark_lo, dark_hi = _NIGHT_SHADOW_L if night else _SHADOW_L
    shadow = np.clip((L - dark_lo) / (dark_hi - dark_lo), 0.15 if night else 0.3, 1.0)
    out_c = np.minimum(np.maximum(chroma - _TINT_FLOOR, 0.0) * _CHROMA_GAIN * shadow, _CHROMA_CAP)
    out_c = np.round(out_c / _CHROMA_STEP) * _CHROMA_STEP
    grey = out_c == 0  # chroma at or below about _TINT_FLOOR + 2.5
    out_c = np.where(grey, 3.0, out_c)
    out_h = np.where(grey, np.radians(70), snapped)  # greys get a warm paper tint

    lo, hi = l_range or _L_RANGE
    stretched = lo + (L - L.min()) / max(float(L.max() - L.min()), 1e-6) * (hi - lo) if len(L) > 1 else L
    out_l = 0.5 * stretched + 0.5 * _equalized_lightness(L, weights, (lo, hi))
    out_l = np.clip(np.round(out_l / _L_STEP) * _L_STEP, lo, hi)
    # very light colors fall outside sRGB and get washed out: only greys may go near white
    out_l = np.where(grey, out_l, np.minimum(out_l, _COLOR_L_MAX))
    out_l, out_c = _fit_gamut(out_l, out_c, out_h, may_darken=chroma >= _VIVID_INPUT)
    lab = np.stack([out_l, out_c * np.cos(out_h), out_c * np.sin(out_h)], axis=1)
    return lab_to_bgr(lab)
