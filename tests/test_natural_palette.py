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
    assert g > max(b, r)
    _, chroma, _ = _out_lch([(50, -8, 8)])
    assert chroma[0] > np.hypot(8, 8)


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


def test_natural_dark_colors_are_desaturated_in_night_range():
    # a brownish night sky must not turn maroon
    b, g, r = natural_colors(_lab((12, 8, 10), (60, 0, 0)), l_range=(10.0, 62.0))[0].astype(int)
    assert max(b, g, r) - min(b, g, r) < 25


def test_natural_daytime_shadow_foliage_is_not_tinted():
    # measured dark leaf cluster of examples/free/street.jpg: its teal tint is skylight,
    # boosting it painted the canopy blue
    _, chroma, _ = _out_lch([(15.6, -8.5, -1.1), (61, -4, 2.2)])
    assert chroma[0] <= 12


def test_natural_vivid_colors_are_strong():
    lab_out = None
    from photo2stamp.palette import bgr_to_lab

    out = natural_colors(_lab((60, 30, 50), (40, 0, 0)))
    lab_out = bgr_to_lab(out)
    assert np.hypot(*lab_out[0, 1:]) >= 55


def _lch(L, C, h_deg):
    h = np.radians(h_deg)
    return (L, C * np.cos(h), C * np.sin(h))


def _out_lch(lab_rows):
    from photo2stamp.palette import bgr_to_lab

    lab = bgr_to_lab(natural_colors(_lab(*lab_rows)))
    return lab[:, 0], np.hypot(lab[:, 1], lab[:, 2]), np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360


def test_natural_faint_tint_stays_faint():
    # measured dark glass of examples/photo.jpg: a lighting tint, not an object color
    _, chroma, _ = _out_lch([_lch(25.5, 10.3, 251), (60, 0, 0)])
    assert chroma[0] <= 25


def test_natural_chroma_order_is_preserved():
    _, chroma, _ = _out_lch([_lch(50, 10, 135), _lch(50, 20, 135), _lch(50, 30, 135)])
    assert chroma[0] < chroma[1] < chroma[2]


def test_natural_dull_yellow_reads_as_vegetation():
    # measured conifer forest of examples/free/castle.jpg (hue 94 deg, looks dark green)
    _, _, hue = _out_lch([_lch(40.8, 16.4, 94), (75, 0, 0)])
    assert 100 <= hue[0] <= 135


def test_natural_bright_yellow_stays_yellow():
    # measured yellow sign of examples/photo.jpg
    _, chroma, hue = _out_lch([_lch(71.2, 44, 98), (30, 0, 0)])
    assert hue[0] < 100 and chroma[0] >= 55


def test_natural_deep_shadow_tint_is_muted():
    # measured shadowed leaves of examples/free/street.jpg: blue cast from the sky
    _, chroma, _ = _out_lch([_lch(10, 9.2, 237), (61, 0, 0)])
    assert chroma[0] <= 15


def test_natural_light_sky_keeps_its_hue_in_gamut():
    # measured sky of examples/free/street.jpg: chroma 44 at L 76 is outside sRGB and
    # used to clip to cyan (hue 237)
    _, _, hue = _out_lch([_lch(72, 29.3, 262), (10, 0, 0)])
    assert abs(hue[0] - 262) <= 8


def test_natural_light_sky_stays_light():
    # measured sky of examples/free/castle.jpg: must not darken into a deep azure
    lightness, chroma, _ = _out_lch([_lch(74.5, 26.1, 263), (40, 0, 0)])
    assert lightness[0] >= 72 and chroma[0] <= 45


def test_natural_barely_tinted_stone_stays_grey():
    # measured pier of examples/free/lighthouse.jpg (circle crop): beige stone, not olive
    _, chroma, _ = _out_lch([_lch(57.6, 7.2, 113), (36, 0, 0)])
    assert chroma[0] <= 6


def test_natural_vivid_color_may_darken_to_keep_chroma():
    # orange at the top of the lightness range only holds chroma ~46 in sRGB
    lightness, chroma, hue = _out_lch([(60, 30, 50), (40, 0, 0)])
    assert lightness[0] < 78 and chroma[0] >= 55 and abs(hue[0] - 60) <= 6


def test_max_chroma_falls_back_to_grey_where_nothing_fits():
    from photo2stamp.palette import _max_chroma

    out = _max_chroma(np.array([99.5, 60.0], np.float32), np.array([60.0, 10.0], np.float32), np.radians([60.0, 60.0]))
    assert out[0] == 0 and out[1] == 10


def test_natural_moderate_colors_stay_colorful():
    # measured sea of examples/free/lighthouse.jpg: a real color, not a lighting tint
    _, chroma, _ = _out_lch([_lch(36.7, 11.4, 226), (70, 0, 0)])
    assert chroma[0] >= 20


def test_shadowed_foliage_next_to_leaves_turns_green():
    # dark bluish patches (leaves in shade, lit by the sky) inside a green canopy
    import cv2

    from photo2stamp import StampConfig, stylize

    rng = np.random.default_rng(0)
    img = np.full((240, 320, 3), (60, 150, 80), np.uint8)  # sunlit green leaves (BGR)
    for cx, cy in ((80, 80), (220, 100), (150, 180)):
        cv2.circle(img, (cx, cy), 35, (50, 42, 30), -1)  # shade: dark, slightly blue
    img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    out = stylize(img, StampConfig(max_size=0, lines=False, details=False, white_balance=False)).image
    b, g, r = out[180, 150].astype(int)
    assert g > b + 8 and g > r + 8
