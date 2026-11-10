"""--crop after: stylize the whole photo, then cut the stamp out of the finished art."""

import numpy as np
import pytest

from photo2stamp.cli import main
from photo2stamp.pipeline import StampConfig, square_window, stylize


def test_crop_mode_is_validated():
    with pytest.raises(ValueError):
        StampConfig(crop="middle")


def test_square_window_matches_crop_square(city_image):
    from photo2stamp.pipeline import crop_square

    y, x, s = square_window(city_image)
    assert np.array_equal(city_image[y : y + s, x : x + s], crop_square(city_image))


def test_crop_after_is_a_cutout_of_the_rect_art(city_image):
    rect = stylize(city_image, StampConfig(max_size=0)).image
    cut = stylize(city_image, StampConfig(max_size=0, shape="circle", crop="after", ring=0, ring_gap=0)).image
    y, x, s = square_window(city_image)
    assert cut.shape[:2] == (s, s)
    inside = cut[..., 3] == 255
    assert inside.mean() > 0.7
    assert np.array_equal(cut[..., :3][inside], rect[y : y + s, x : x + s][inside])


def test_crop_after_svg_is_square_and_shifted(city_image):
    svg = stylize(city_image, StampConfig(max_size=0, shape="rounded", crop="after")).svg
    y, x, s = square_window(city_image)
    assert f'viewBox="0 0 {s} {s}"' in svg
    assert f'translate({-x} {-y})' in svg


def test_crop_before_stays_the_default(city_image):
    assert StampConfig().crop == "before"
    a = stylize(city_image, StampConfig(max_size=0, shape="circle")).image
    b = stylize(city_image, StampConfig(max_size=0, shape="circle", crop="before")).image
    assert np.array_equal(a, b)


def test_cli_crop_after(tmp_path, city_image):
    import cv2

    src = tmp_path / "c.png"
    cv2.imwrite(str(src), city_image)
    assert main([str(src), "-o", str(tmp_path / "o"), "--shape", "circle", "--crop", "after", "--svg"]) == 0
    assert (tmp_path / "o" / "c.png").exists() and (tmp_path / "o" / "c.svg").exists()


def test_crop_after_needs_a_stamp_shape():
    with pytest.raises(ValueError):
        StampConfig(shape="rect", crop="after")


@pytest.mark.parametrize("transpose", [False, True])
def test_crop_after_matches_rect_when_resized(city_image, transpose):
    import cv2

    img = np.ascontiguousarray(city_image.transpose(1, 0, 2)) if transpose else city_image  # landscape / portrait
    rect = stylize(img, StampConfig(max_size=200)).image
    cut = stylize(img, StampConfig(max_size=200, shape="rounded", crop="after", ring=0, ring_gap=0)).image
    small = cv2.resize(img, (rect.shape[1], rect.shape[0]), interpolation=cv2.INTER_AREA)
    y, x, s = square_window(small)
    inside = cut[..., 3] == 255
    assert np.array_equal(cut[..., :3][inside], rect[y : y + s, x : x + s][inside])


def test_crop_after_svg_drops_shapes_outside_the_window(city_image):
    wide = np.ascontiguousarray(np.concatenate([city_image] * 3, axis=1))  # 240 x 540
    full = stylize(wide, StampConfig(max_size=0)).svg
    cut = stylize(wide, StampConfig(max_size=0, shape="circle", crop="after")).svg
    assert cut.count("<path") < full.count("<path")
