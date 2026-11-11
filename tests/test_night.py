"""Night mode: dark photos keep dark fills and get light (lamp-colored) outlines."""

import numpy as np
import pytest

from photo2stamp import StampConfig, stylize
from photo2stamp.lines import draw_lines
from photo2stamp.palette import hex_to_bgr, natural_colors
from photo2stamp.pipeline import LIGHT_INK, is_night


def _night_city():
    rng = np.random.default_rng(0)
    img = np.full((240, 320, 3), 18, np.uint8)  # night sky
    for x in range(20, 300, 50):
        img[100:, x : x + 30] = 45  # dark towers
        img[110:230:12, x + 5 : x + 25 : 8] = (90, 200, 240)  # lit windows
    return np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)


def test_is_night_detects_dark_photos():
    assert is_night(_night_city())
    assert not is_night(np.full((50, 50, 3), 150, np.uint8))


def test_natural_colors_respects_lightness_range():
    lab = np.array([[5, 0, 0], [50, 0, 0], [95, 0, 0]], np.float32)
    out = natural_colors(lab, l_range=(10.0, 60.0)).astype(float)
    luma = out @ np.array([0.114, 0.587, 0.299])
    assert luma.max() < 170  # L 60 is roughly luma 150


def test_draw_lines_on_dark_only():
    fill = np.zeros((10, 20, 3), np.uint8)
    fill[:, 10:] = 230
    out = draw_lines(fill, np.ones((10, 20), bool), ink=(200, 220, 240), on="dark")
    assert tuple(out[5, 2]) == (200, 220, 240)
    assert tuple(out[5, 15]) == (230, 230, 230)


def test_night_photo_gets_light_outlines():
    out = stylize(_night_city(), StampConfig(max_size=0)).image
    assert (out.reshape(-1, 3) == np.array(hex_to_bgr(LIGHT_INK), np.uint8)).all(axis=1).any()


def test_night_mode_can_be_forced_off():
    out = stylize(_night_city(), StampConfig(max_size=0, night="off")).image
    assert not (out.reshape(-1, 3) == np.array(hex_to_bgr(LIGHT_INK), np.uint8)).all(axis=1).any()


def test_night_mode_validation():
    with pytest.raises(ValueError):
        StampConfig(night="sometimes")


def test_light_spots_marks_bright_points_only():
    from photo2stamp.lines import light_spots

    img = np.full((100, 100, 3), 20, np.uint8)
    img[40:44, 40:44] = 230  # a lit window
    img[0:100, 90:100] = 60  # dim wall, not a light
    spots = light_spots(img, size=(100, 100))
    assert spots[39:46, 39:46].any()
    assert not spots[50, 95] and not spots[10, 10]


def test_dark_table_with_a_bright_dish_is_not_night():
    # measured: ramen bowls on a black table, median luma 22 but one large bright bowl
    import cv2

    img = np.full((240, 320, 3), 20, np.uint8)
    cv2.circle(img, (200, 140), 70, (150, 190, 220), -1)
    assert not is_night(img)


def test_day_scene_gets_no_lamp_dots():
    # sun glints in the shade of a tree look like dusk lamps; day photos get no lamp dots
    rng = np.random.default_rng(0)
    img = np.full((200, 300, 3), 35, np.uint8)
    img[:90] = (200, 170, 150)  # bright sky (not night overall)
    for x in range(20, 280, 40):
        img[150:154, x : x + 4] = 250  # bright points in the dark half
    img = np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
    assert not is_night(img)
    out = stylize(img, StampConfig(max_size=0)).image
    lamp = np.array(hex_to_bgr(LIGHT_INK), float)
    river = out[120:190].astype(float)
    assert (np.abs(river - lamp).sum(axis=2) < 30).sum() == 0


def test_dense_windows_become_spaced_dots():
    # a tower of lit windows: one dot per window would be hundreds of specks
    import cv2

    from photo2stamp.lines import LIGHT_GAP, light_spots

    img = np.full((400, 400, 3), 15, np.uint8)
    windows = 0
    for y in range(20, 380, 3):
        for x in range(100, 300, 3):
            img[y : y + 2, x : x + 2] = 230
            windows += 1
    spots = light_spots(img, size=(400, 400))
    count, _, _, centers = cv2.connectedComponentsWithStats(spots.astype(np.uint8))
    pts = centers[1:]
    assert 20 < len(pts) < windows / 3
    d = np.hypot(*(pts[:, None] - pts[None]).transpose(2, 0, 1))
    np.fill_diagonal(d, np.inf)
    assert d.min() >= LIGHT_GAP * 400 - 1


def test_the_larger_light_wins_a_crowded_spot():
    from photo2stamp.lines import light_spots

    img = np.full((200, 200, 3), 10, np.uint8)
    img[100:102, 100:102] = 240  # small lamp
    img[100:108, 104:112] = 240  # big lit window right next to it
    spots = light_spots(img, size=(200, 200), gap=0.05)  # 10 px: the two are too close
    ys, xs = np.nonzero(spots)
    assert abs(xs.mean() - 107.5) < 2  # the dot sits on the big window


def test_gap_zero_keeps_every_light():
    import cv2

    from photo2stamp.lines import light_spots

    img = np.full((200, 200, 3), 10, np.uint8)
    for x in range(20, 180, 8):
        img[100:102, x : x + 2] = 240
    spots = light_spots(img, size=(200, 200), dot=0.001, gap=0.0)
    assert cv2.connectedComponents(spots.astype(np.uint8))[0] - 1 == 20
