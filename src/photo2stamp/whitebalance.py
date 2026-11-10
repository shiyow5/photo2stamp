"""Remove the color cast of the light (white balance) before colors are chosen.

Indoor photos under warm lamps, and shade or overcast scenes, tint every surface: a
white plate measures salmon, a grey wall blue. The natural palette boosts chroma, so
such a cast becomes a wrong color.

The light is estimated from near-grey surfaces only, in the spirit of grey-pixel color
constancy (Yang, Gao & Li, "Efficient Illuminant Estimation for Color Constancy Using
Grey Pixels", CVPR 2015): pixels whose chroma is low are assumed to be grey objects,
and their average tint is the light. Gray-world and Gray-Edge (van de Weijer et al.
2007) were tried first; their averages include colored objects, so a yellow-windowed
city turned cyan and a sunset beach olive. A scene with hardly any grey (a sunset) is
left alone, and the correction is capped. It is a von Kries channel scaling in linear
RGB.
"""

from __future__ import annotations

import cv2
import numpy as np

GREY_CHROMA = 22.0  # pixels below this Lab chroma count as grey objects (tinted by the light)
GREY_FALLOFF = 8.0  # chroma scale of the weight that prefers the most colorless pixels
GREY_L = (25.0, 97.0)  # too dark: noisy hue; brighter: clipped highlights
MIN_GREY_FRACTION = 0.08  # fewer grey pixels than this: the light cannot be measured
GREY_REF_L = 70.0  # lightness at which the light's tint is converted to channel gains
BLUE_HUES = (200.0, 300.0)  # Lab hues of sky; in the upper part they never count as tinted greys
SKY_PART = 0.5  # upper fraction of the frame where bluish pixels are taken for sky
TOP_BAND = 0.25  # fraction of the height checked for a warm sky
WARM_SKY = (55.0, 21.0)  # median L and b of a sunset sky (a warm-lit indoor wall measures b ~19)
MAX_SHIFT = 25.0  # largest cast (Lab chroma of the light color) removed in full
_LUMA_BGR = np.array([0.0722, 0.7152, 0.2126])  # linear-light luminance weights


def _to_linear(img: np.ndarray) -> np.ndarray:
    c = img.astype(np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    c = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(np.round(c * 255.0), 0, 255).astype(np.uint8)


def _grey_cast(image: np.ndarray) -> tuple[float, float] | None:
    """Average (a, b) tint of the near-grey pixels, or None when too few are grey."""
    small = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA) if image.size > 3 * 128 * 128 else image
    lab = cv2.cvtColor(small.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab).reshape(-1, 3)
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    chroma, hue = np.hypot(a, b), np.degrees(np.arctan2(b, a)) % 360
    upper = (np.arange(len(L)) // small.shape[1]) < small.shape[0] * SKY_PART
    # bluish greys in the upper part are sky (a real color); lower down they are shade
    bluish = upper & (hue > BLUE_HUES[0]) & (hue < BLUE_HUES[1]) & (chroma > 6)
    grey = (chroma < GREY_CHROMA) & (L > GREY_L[0]) & (L < GREY_L[1]) & ~bluish
    if grey.mean() < MIN_GREY_FRACTION:
        return None
    # bright, nearly colorless pixels tell the light best; a light orange cat (chroma ~20)
    # barely counts next to a white blanket
    w = L[grey] * np.exp(-chroma[grey] / GREY_FALLOFF)
    return float(np.average(a[grey], weights=w)), float(np.average(b[grey], weights=w))


def _warm_sky(image: np.ndarray) -> bool:
    """A bright warm band at the top of the frame: a sunset or golden hour sky."""
    h = image.shape[0]
    top = image[: max(1, int(h * TOP_BAND))]
    lab = cv2.cvtColor(top.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab).reshape(-1, 3)
    return bool(np.median(lab[:, 0]) > WARM_SKY[0] and np.median(lab[:, 2]) > WARM_SKY[1])


def estimate_illuminant(image: np.ndarray) -> np.ndarray:
    """Light color as linear BGR, normalized to mean 1 (no cast: all ones)."""
    cast = _grey_cast(image[..., :3])
    if cast is None:
        return np.ones(3)
    lab = np.array([[[GREY_REF_L, cast[0], cast[1]]]], np.float32)
    srgb = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR), 1e-3, 1.0)
    light = _to_linear(np.round(srgb * 255)).reshape(3).astype(np.float64)
    return light / light.mean()


def _cast_chroma(light: np.ndarray) -> float:
    """Lab chroma of the light color (how strong the cast is)."""
    srgb = _to_srgb((light / light.max()).reshape(1, 1, 3).astype(np.float32))
    lab = cv2.cvtColor(srgb.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)[0, 0]
    return float(np.hypot(lab[1], lab[2]))


def white_balance(image: np.ndarray, strength: float = 1.0, max_shift: float = MAX_SHIFT) -> np.ndarray:
    """Return a copy of `image` (BGR uint8) with the light's color cast removed.

    Casts up to `max_shift` (Lab chroma) are removed in full; stronger ones only by
    that amount, so deliberate warm light (sunsets) keeps its mood.
    """
    img = image[..., :3]
    light = estimate_illuminant(img)
    if np.allclose(light, 1.0, atol=1e-3) or strength <= 0:
        return img.copy()
    if light[0] < light[2] and _warm_sky(img):  # warm light under a warm sky is the mood
        return img.copy()
    cast = _cast_chroma(light)
    t = strength * (1.0 if cast <= max_shift else max_shift / cast)
    gains = (1.0 / light) ** t
    gains = gains / float(_LUMA_BGR @ gains)  # keep brightness
    return _to_srgb(_to_linear(img) * gains.astype(np.float32))
