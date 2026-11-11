"""Color cast removal: the light is estimated from near-grey surfaces."""

import numpy as np

from photo2stamp.whitebalance import estimate_illuminant, white_balance


def _scene(seed=0):
    """Neutral textured scene: grey blocks of different lightness plus noise."""
    rng = np.random.default_rng(seed)
    img = np.zeros((120, 160, 3), np.float32)
    for i, v in enumerate([60, 120, 180, 230]):
        img[:, i * 40 : (i + 1) * 40] = v
    img[40:80, 20:140] = 90
    img += rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _cast(img, bgr_gain):
    return np.clip(img.astype(np.float32) * np.array(bgr_gain, np.float32), 0, 255).astype(np.uint8)


def _mean_chroma(img):
    import cv2

    lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    return float(np.hypot(lab[..., 1], lab[..., 2]).mean())


def test_neutral_scene_is_left_alone():
    img = _scene()
    out = white_balance(img)
    assert np.abs(out.astype(int) - img.astype(int)).max() <= 3


def test_warm_cast_is_removed():
    warm = _cast(_scene(), (0.75, 0.92, 1.0))  # tungsten-like: less blue
    assert _mean_chroma(warm) > 12
    assert _mean_chroma(white_balance(warm)) < 7  # what is left is drawn as grey (tint floor 6)


def test_blue_cast_is_removed():
    cool = _cast(_scene(), (1.0, 0.95, 0.82))  # shade-like: less red
    assert _mean_chroma(white_balance(cool)) < 6


def test_strong_cast_is_only_partly_removed():
    # a sunset is orange on purpose: the correction is capped
    sunset = _cast(_scene(), (0.35, 0.7, 1.0))
    out = white_balance(sunset)
    assert _mean_chroma(out) > 0.6 * _mean_chroma(sunset)


def test_scene_of_saturated_color_is_not_neutralized():
    # sunset beach: sky, sea and sand are all orange; there is no grey to measure
    rng = np.random.default_rng(1)
    img = np.clip(np.array([60, 150, 240], np.float32) * rng.uniform(0.4, 1.0, (120, 160, 1)), 0, 255).astype(np.uint8)
    assert np.abs(white_balance(img).astype(int) - img.astype(int)).max() <= 3


def test_colored_details_do_not_bias_the_estimate():
    # grey city with many small yellow lit windows (their edges fooled Gray-Edge)
    img = _scene()
    img[10:110:6, 5:155:6] = (40, 210, 240)
    assert _mean_chroma(white_balance(img)) <= _mean_chroma(img) + 1


def test_large_colored_area_does_not_fool_the_estimate():
    # half the frame is a flat yellow (rice); edges are neutral
    img = _scene()
    img[:, :80] = (40, 200, 230)
    est = estimate_illuminant(img)
    assert np.ptp(est) / est.mean() < 0.15


def test_input_is_not_modified_and_dtype_kept():
    img = _cast(_scene(), (0.75, 0.92, 1.0))
    before = img.copy()
    out = white_balance(img)
    assert np.array_equal(img, before) and out.dtype == np.uint8 and out.shape == img.shape


def test_flat_image_is_returned_unchanged():
    img = np.full((10, 10, 3), (40, 120, 200), np.uint8)
    assert np.array_equal(white_balance(img), img)


def test_stylize_removes_a_warm_cast(city_image):
    from photo2stamp import StampConfig, stylize

    warm = _cast(city_image, (0.72, 0.9, 1.0))
    road = (slice(215, 235), slice(40, 140))  # grey road in the fixture
    on = stylize(warm, StampConfig(max_size=0, lines=False)).image[road]
    off = stylize(warm, StampConfig(max_size=0, lines=False, white_balance=False)).image[road]
    assert _mean_chroma(on) < 8 < _mean_chroma(off)


def test_cli_can_turn_white_balance_off():
    from photo2stamp.cli import _config, build_parser

    assert _config(build_parser().parse_args(["x.jpg", "--no-white-balance"])).white_balance is False


def _lab_image(top_lab, bottom_lab, seed=2):
    import cv2

    rng = np.random.default_rng(seed)
    lab = np.zeros((120, 160, 3), np.float32)
    lab[:60] = top_lab
    lab[60:] = bottom_lab
    lab[..., 0] += rng.normal(0, 4, lab.shape[:2])
    lab[60:, ::8, 0] -= 30  # some edges below
    bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return np.clip(bgr * 255, 0, 255).astype(np.uint8)


def test_blue_sky_is_not_taken_for_a_cast():
    img = _lab_image((75, -4, -18), (55, 0, 0))  # pale blue sky over grey ground
    out = white_balance(img)
    assert _mean_chroma(out[:60]) > 14


def test_sunset_sky_keeps_its_warmth():
    img = _lab_image((78, 10, 24), (45, 4, 10))  # measured sunset sky of a beach photo: b ~24
    assert np.abs(white_balance(img).astype(int) - img.astype(int)).max() <= 3
