"""Outline layer: DoG edge lines (Winnemoeller et al. 2006/2012 style) for recognizability."""

import numpy as np

from photo2stamp import StampConfig, stylize
from photo2stamp.lines import clean_lines, dog_lines, draw_lines


def _step(h=80, w=80, dark=40, light=200):
    img = np.full((h, w, 3), light, np.uint8)
    img[:, : w // 2] = dark
    return img


def test_dog_lines_marks_the_dark_side_of_an_edge():
    lines = dog_lines(_step(), sigma=1.5, threshold=0.01)
    assert lines[:, 34:40].any()  # just left of the edge (dark side)
    assert not lines[:, 44:].any()  # light side stays clean


def test_dog_lines_leave_flat_dark_areas_empty():
    # regression: classic XDoG (tau < 1) fills flat dark regions with ink
    flat = np.full((60, 60, 3), 30, np.uint8)
    assert not dog_lines(flat, sigma=2.0, threshold=0.01).any()


def test_clean_lines_drops_short_specks_and_thickens():
    mask = np.zeros((50, 50), bool)
    mask[10, 5:45] = True  # long line
    mask[30, 30] = True  # speck
    out = clean_lines(mask, min_length=10, width=3)
    assert out[10, 25] and out[11, 25]  # kept and thickened
    assert not out[30, 30]
    assert mask.sum() == 41  # input untouched


def test_draw_lines_skips_dark_fills():
    # ink on a dark fill is invisible and white lines looked like halos: draw nothing there
    fill = np.zeros((10, 20, 3), np.uint8)
    fill[:, 10:] = 230
    mask = np.ones((10, 20), bool)
    out = draw_lines(fill, mask, ink=(30, 30, 30))
    assert tuple(out[5, 2]) == (0, 0, 0)  # dark fill unchanged
    assert tuple(out[5, 15]) == (30, 30, 30)


def test_dog_lines_ignore_the_image_border():
    img = np.full((60, 60, 3), 200, np.uint8)
    img[:, 30:] = 60
    lines = dog_lines(img, sigma=2.0, threshold=0.01)
    assert not lines[:3].any() and not lines[-3:].any()


def test_stylize_draws_outlines_by_default(city_image):
    with_lines = stylize(city_image, StampConfig(max_size=0))
    without = stylize(city_image, StampConfig(max_size=0, lines=False))
    assert not np.array_equal(with_lines.image, without.image)
    assert 'id="lines"' in with_lines.svg
    assert 'id="lines"' not in without.svg


def test_lines_svg_keeps_thin_lines_and_dots_with_a_stroke():
    from photo2stamp.lines import lines_svg

    fill = np.full((40, 60, 3), 230, np.uint8)
    mask = np.zeros((40, 60), bool)
    mask[10, 5:45] = True  # 1 px line
    mask[30, 30] = True  # single-pixel dot
    svg = lines_svg(fill, mask, ink=(30, 30, 30))
    assert svg.count("M") == 2  # both kept
    assert 'stroke-width="1"' in svg and 'stroke-linecap="round"' in svg
