"""The 'natural' palette keeps each cluster's hue family but snaps it to a small,
consistent set of hues, boosts saturation and spreads lightness into clear steps."""

import numpy as np

from photo2stamp.palette import assign_colors, get_palette, natural_colors


def _lab(*rows):
    return np.array(rows, np.float32)


def test_natural_is_a_known_palette_name():
    assert get_palette("natural") is None or get_palette("natural") is not None  # no ValueError


def test_natural_keeps_hue_family():
    sky, tree, lamp = (70, -5, -30), (45, -25, 25), (65, 20, 50)
    blue, green, orange = natural_colors(_lab(sky, tree, lamp))
    assert blue[0] == max(blue)  # B channel dominates
    assert green[1] == max(green)  # G channel dominates
    assert orange[2] == max(orange) and orange[0] == min(orange)


def test_natural_boosts_saturation_of_dull_colors():
    dull_green = _lab((50, -8, 8))
    b, g, r = natural_colors(dull_green)[0].astype(int)
    assert g - max(b, r) > 15


def test_natural_keeps_greys_neutral():
    b, g, r = natural_colors(_lab((50, 1, -1)))[0].astype(int)
    assert max(b, g, r) - min(b, g, r) < 20


def test_natural_lightness_order_is_preserved():
    out = natural_colors(_lab((20, 0, 0), (50, 0, 0), (80, 0, 0)))
    lum = out.astype(float) @ np.array([0.114, 0.587, 0.299])
    assert lum[0] < lum[1] < lum[2]


def test_natural_snaps_similar_hues_to_the_same_color():
    a, b = natural_colors(_lab((60, -20, 22), (60, -24, 20)))
    assert tuple(a) == tuple(b)


def test_assign_colors_with_natural_uses_natural_colors_for_sky():
    centers = _lab((70, -5, -30), (40, 0, 0))
    out = assign_colors(centers, get_palette("natural"), sky_labels=frozenset({0}))
    assert out[0][0] == max(out[0])  # sky stays blue


def test_natural_circle_stamp_uses_fixed_paper_color(city_image):
    from photo2stamp import StampConfig, stylize
    from photo2stamp.palette import hex_to_bgr
    from photo2stamp.pipeline import PAPER

    out = stylize(city_image, StampConfig(shape="circle", max_size=0, ring=0.06, ring_gap=0.04)).image
    s = out.shape[0]
    gap_px = out[s // 2, int(s * 0.08), :3]  # inside the ring, in the paper gap
    assert tuple(int(v) for v in gap_px) == hex_to_bgr(PAPER)


def test_natural_sky_hue_is_blue_not_purple():
    # measured sky of examples/free/castle.jpg (Lab hue about -88 deg)
    b, g, r = natural_colors(_lab((70, 1.3, -41.7)))[0].astype(int)
    assert b > r + 40 and g > r and r < 150  # lavender (r ~170) is not sky blue


def test_natural_dark_colors_are_desaturated():
    # a brownish night sky must not turn maroon
    b, g, r = natural_colors(_lab((12, 8, 10), (60, 0, 0)))[0].astype(int)
    assert max(b, g, r) - min(b, g, r) < 25
