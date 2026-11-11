"""Small colored details (painted carvings, flowers in a bed) survive as color spots."""

import cv2
import numpy as np

from photo2stamp import StampConfig, stylize
from photo2stamp.segment import to_lab


def _carved_wall(seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((300, 400, 3), 150, np.uint8)  # grey stone
    colors = [(170, 120, 200), (110, 170, 120), (200, 160, 110)]  # muted pink, green, blue (BGR)
    for i, y in enumerate(range(40, 260, 30)):
        for j, x in enumerate(range(30, 370, 34)):
            img[y : y + 9, x : x + 12] = colors[(i + j) % 3]
    return np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)


def _hues(img, min_chroma=12):
    lab = to_lab(img).reshape(-1, 3)
    c = np.hypot(lab[:, 1], lab[:, 2])
    h = np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360
    return h[c > min_chroma]


def test_small_colored_details_keep_their_hues():
    out = stylize(_carved_wall(), StampConfig(max_size=0, lines=False)).image
    h = _hues(out)
    pink = ((h > 290) | (h < 20)).sum()  # purplish pink snaps to the 320 deg anchor
    green = ((h > 110) & (h < 170)).sum()
    blue = ((h > 220) & (h < 290)).sum()
    assert min(pink, green, blue) > 200


def test_details_can_be_turned_off():
    out = stylize(_carved_wall(), StampConfig(max_size=0, lines=False, details=False)).image
    assert len(_hues(out)) < 0.02 * out.shape[0] * out.shape[1]


def test_plain_scene_gets_no_detail_spots(city_image):
    a = stylize(city_image, StampConfig(max_size=0, lines=False)).image
    b = stylize(city_image, StampConfig(max_size=0, lines=False, details=False)).image
    assert np.mean(np.any(a != b, axis=2)) < 0.01


def test_detail_spots_are_in_the_svg():
    svg = stylize(_carved_wall(), StampConfig(max_size=0)).svg
    assert '<g id="details">' in svg
