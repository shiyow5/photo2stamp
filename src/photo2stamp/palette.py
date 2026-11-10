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
    """

    name: str
    tones: tuple[str, ...]
    sky: str
    accents: tuple[str, ...]


PRESETS: dict[str, Palette] = {
    p.name: p
    for p in (
        Palette(
            name="teal",
            tones=("#173F4A", "#22525E", "#3A6875", "#5F8590", "#93ABB3", "#E8DCBB", "#F3F1EA"),
            sky="#8DCAE6",
            accents=("#D9472F", "#F2C24B", "#3E7D6B"),
        ),
        Palette(
            name="warm",
            tones=("#3B2A26", "#6B3F2F", "#A65E3F", "#D99A6C", "#F2D3A7", "#FBF1E1"),
            sky="#F4D8B8",
            accents=("#2F6F73", "#E0A526", "#C2412D"),
        ),
        Palette(
            name="mono",
            tones=("#1A1A1A", "#4D4D4D", "#8C8C8C", "#C8C8C8", "#F2F2F2"),
            sky="#DADADA",
            accents=("#D7263D",),
        ),
    )
}

ORIGINAL = "original"


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
    if name not in PRESETS:
        choices = ", ".join([*PRESETS, ORIGINAL])
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
) -> np.ndarray:
    """Map each cluster center (Lab) to a flat BGR color. Returns a new (K,3) uint8 array."""
    centers = np.asarray(centers_lab, np.float32)
    if palette is None:
        return lab_to_bgr(centers)

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
        elif len(accents) and chroma[i] >= accent_chroma:
            out[i] = accents[_nearest_accent(centers[i], accents_lab)]
        else:
            neutral.append(i)

    if neutral:
        order = sorted(neutral, key=lambda i: float(centers[i, 0]))
        n, t = len(order), len(tones)
        for rank, idx in enumerate(order):
            tone = round(rank * (t - 1) / (n - 1)) if n > 1 else t // 2
            out[idx] = tones[tone]
    return out
