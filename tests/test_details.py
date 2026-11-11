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


def _speckle(img, step, size):
    colors = [(170, 120, 200), (110, 170, 120), (200, 160, 110)]
    for i, y in enumerate(range(10, img.shape[0] - size, step)):
        for j, x in enumerate(range(10, img.shape[1] - size, step)):
            img[y : y + size, x : x + size] = colors[(i + j) % 3]
    return img


def _noisy(img):
    rng = np.random.default_rng(1)
    return np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)


def _speckled_wall(step, size, shape=(400, 600)):
    return _noisy(_speckle(np.full((*shape, 3), 150, np.uint8), step, size))


def test_tiny_specks_are_not_details():
    # grass, gravel and leaf glints: specks of a few pixels are texture, not carvings
    out = stylize(_speckled_wall(40, 4), StampConfig(max_size=0, lines=False)).image
    assert len(_hues(out)) < 0.002 * out.shape[0] * out.shape[1]


def test_detail_spots_are_capped():
    # 260 colored spots: only the strongest DETAIL_BUDGET are drawn, the rest stay in the fill
    from photo2stamp.details import DETAIL_BUDGET

    out = stylize(_speckled_wall(30, 9), StampConfig(max_size=0, lines=False)).image
    lab = to_lab(out)
    colored = (np.hypot(lab[..., 1], lab[..., 2]) > 12).astype(np.uint8)
    count = cv2.connectedComponents(colored)[0] - 1
    assert 0 < count <= DETAIL_BUDGET


def test_spots_that_match_their_fill_are_dropped():
    from photo2stamp.details import select_details

    flat = np.full((60, 60, 3), (60, 140, 90), np.uint8)  # green fill
    spots = np.zeros((60, 60), np.int32)
    spots[10:20, 10:20] = 1  # the same green as the fill: invisible, only noise
    spots[40:50, 40:50] = 2  # red on green: a real detail
    colors = np.array([(62, 138, 92), (50, 60, 200)], np.uint8)
    kept, kept_colors = select_details(spots, colors, flat)
    assert len(kept_colors) == 1 and tuple(kept_colors[0]) == (50, 60, 200)
    assert kept[45, 45] == 1 and kept[15, 15] == 0


def test_larger_and_stronger_spots_win_the_budget():
    from photo2stamp.details import select_details

    flat = np.full((100, 100, 3), 150, np.uint8)
    spots = np.zeros((100, 100), np.int32)
    spots[5:25, 5:25] = 1  # big
    spots[60:68, 60:68] = 2  # small
    colors = np.array([(50, 60, 200), (50, 60, 200)], np.uint8)
    kept, kept_colors = select_details(spots, colors, flat, budget=1)
    assert len(kept_colors) == 1 and kept[10, 10] == 1 and kept[64, 64] == 0


def test_a_field_of_specks_gets_no_details_at_all():
    # 500+ specks (a lawn, autumn leaves) are texture even though few are large
    img = _speckle(np.full((500, 600, 3), 150, np.uint8), 20, 4)
    img = _noisy(_speckle(img, 90, 10))  # a few larger spots among the specks
    a = stylize(img, StampConfig(max_size=0, lines=False)).image
    b = stylize(img, StampConfig(max_size=0, lines=False, details=False)).image
    assert np.mean(np.any(a != b, axis=2)) < 0.002


def test_a_spot_enclosed_by_another_is_judged_by_the_fill_not_by_black():
    # its ring lies on the other spot: no fill around it is known, so it is not drawn
    from photo2stamp.details import select_details

    flat = np.full((60, 60, 3), (60, 140, 90), np.uint8)
    spots = np.zeros((60, 60), np.int32)
    spots[20:40, 20:40] = 2  # a red spot ...
    spots[28:32, 28:32] = 1  # ... enclosing a spot of the fill's own green
    colors = np.array([(62, 138, 92), (50, 60, 200)], np.uint8)
    kept, kept_colors = select_details(spots, colors, flat)
    assert [tuple(c) for c in kept_colors] == [(50, 60, 200)]


def test_neighbouring_spots_each_keep_their_ring():
    from photo2stamp.details import select_details

    flat = np.full((40, 60, 3), 150, np.uint8)
    spots = np.zeros((40, 60), np.int32)
    spots[10:30, 10:29] = 1
    spots[10:30, 30:50] = 2  # 1 px apart
    colors = np.array([(50, 60, 200), (60, 160, 60)], np.uint8)
    _, kept_colors = select_details(spots, colors, flat)
    assert len(kept_colors) == 2


def test_select_details_handles_empty_input_and_bad_budgets():
    import pytest

    from photo2stamp.details import select_details

    flat = np.full((10, 10, 3), 150, np.uint8)
    spots = np.zeros((10, 10), np.int32)
    out, cols = select_details(spots, np.empty((0, 3), np.uint8), flat)
    assert len(cols) == 0 and not out.any()
    spots[2:6, 2:6] = 1
    with pytest.raises(ValueError):
        select_details(spots, np.array([(50, 60, 200)], np.uint8), flat, budget=-1)
