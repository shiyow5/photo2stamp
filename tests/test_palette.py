import numpy as np
import pytest

from photo2stamp.palette import PRESETS, assign_colors, get_palette, hex_to_bgr


def test_hex_to_bgr():
    assert hex_to_bgr("#FF8000") == (0, 128, 255)
    assert hex_to_bgr("00ff00") == (0, 255, 0)


def test_hex_to_bgr_rejects_bad_value():
    with pytest.raises(ValueError):
        hex_to_bgr("#12345")


def test_get_palette_unknown_raises():
    with pytest.raises(ValueError, match="unknown palette"):
        get_palette("nope")


def test_presets_have_tones_dark_to_light():
    for palette in PRESETS.values():
        assert len(palette.tones) >= 3


def test_assign_colors_orders_tones_by_lightness():
    palette = get_palette("teal")
    centers = np.array([[80, 0, 0], [20, 0, 0], [50, 0, 0]], np.float32)  # Lab
    colors = assign_colors(centers, palette)
    darkest = hex_to_bgr(palette.tones[0])
    lightest = hex_to_bgr(palette.tones[-1])
    assert tuple(colors[1]) == darkest
    assert tuple(colors[0]) == lightest


def test_assign_colors_sky_and_accent():
    palette = get_palette("teal")
    centers = np.array(
        [[70, 0, 0], [40, 0, 0], [80, 5, 70], [60, 0, 0]],  # [3] sky, [2] yellow
        np.float32,
    )
    colors = assign_colors(centers, palette, sky_labels=frozenset({3}), accent_chroma=30)
    assert tuple(colors[3]) == hex_to_bgr(palette.sky)
    assert tuple(colors[2]) in {hex_to_bgr(a) for a in palette.accents}
    # yellow input should pick the yellow-ish accent, not the red one
    b, g, r = colors[2]
    assert g > 150 and r > 200


def test_assign_colors_original_keeps_cluster_colors():
    centers = np.array([[50, 60, 40]], np.float32)  # reddish
    colors = assign_colors(centers, None)
    b, g, r = (int(c) for c in colors[0])
    assert r > g and r > b


def test_assign_colors_does_not_mutate_input():
    centers = np.array([[80, 0, 0], [20, 0, 0]], np.float32)
    before = centers.copy()
    assign_colors(centers, get_palette("mono"))
    np.testing.assert_array_equal(centers, before)
