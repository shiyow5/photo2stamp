"""Tests for the abstraction features: detail-adaptive work size, texture-based sky,
smooth label upscaling and stamp frames."""

import numpy as np
import pytest

from photo2stamp.pipeline import StampConfig, auto_work_size, crop_square, stylize
from photo2stamp.segment import edge_density, extract_sky, texture_map, upscale_labels
from photo2stamp.vectorize import frame_stamp


def _busy(h=200, w=200):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_edge_density_busy_is_higher_than_flat():
    flat = np.full((200, 200, 3), 128, np.uint8)
    assert edge_density(_busy()) > edge_density(flat) == 0.0


def test_auto_work_size_shrinks_for_detailed_photos():
    assert auto_work_size(0.05) > auto_work_size(0.15) > auto_work_size(0.30)
    assert 64 <= auto_work_size(1.0) and auto_work_size(0.0) <= 400


def test_texture_map_shape_and_range():
    t = texture_map(_busy(120, 80))
    assert t.shape == (120, 80)
    assert 0.0 <= t.min() and t.max() <= 1.0


def test_upscale_labels_keeps_regions_and_smooths_edges():
    labels = np.zeros((10, 10), np.int32)
    labels[np.triu_indices(10)] = 1  # diagonal boundary (staircase at low res)
    big = upscale_labels(labels, (100, 100), n_labels=2)
    assert big.shape == (100, 100)
    assert big[90, 5] == 0 and big[5, 90] == 1
    # a staircase upscaled with nearest has 10px steps; smooth upscaling does not
    steps = np.abs(np.diff((big == 1).argmax(axis=1)[20:80]))  # away from the image border
    assert steps.max() <= 3


def test_extract_sky_uses_texture_for_dark_or_warm_sky():
    labels = np.zeros((100, 100), np.int32)
    labels[50:] = 1
    centers = np.array([[20, 5, 20], [40, 0, 0]], np.float32)  # dark, warm "sunset/night" sky
    texture = np.zeros((100, 100), np.float32)
    texture[50:] = 0.3
    new_labels, _, sky = extract_sky(labels, centers, texture=texture)
    assert sky is not None and new_labels[10, 10] == sky
    assert new_labels[80, 50] == 1


def test_extract_sky_rejects_textured_top_region():
    labels = np.zeros((100, 100), np.int32)
    labels[50:] = 1
    centers = np.array([[80, 0, -20], [40, 0, 0]], np.float32)
    texture = np.full((100, 100), 0.3, np.float32)  # e.g. tree canopy at the top
    _, _, sky = extract_sky(labels, centers, texture=texture)
    assert sky is None


def test_crop_square_centers():
    img = np.zeros((100, 60, 3), np.uint8)
    img[20:80] = 255
    sq = crop_square(img)
    assert sq.shape == (60, 60, 3) and (sq == 255).all()


@pytest.mark.parametrize("shape", ["circle", "rounded"])
def test_frame_stamp_draws_ring_and_transparent_outside(shape):
    img = np.full((200, 200, 3), (10, 200, 10), np.uint8)
    out = frame_stamp(img, shape, ring_color=(0, 0, 255), ring=0.04, gap=0.02, paper=(255, 255, 255))
    assert out.shape == (200, 200, 4)
    assert out[0, 0, 3] == 0  # corner is outside
    assert tuple(out[100, 100, :3]) == (10, 200, 10)  # center keeps the art
    assert tuple(out[100, 3, :3]) == (0, 0, 255)  # ring at the left edge
    assert tuple(out[100, 11, :3]) == (255, 255, 255)  # paper gap inside the ring


def test_stylize_badge_is_square_with_alpha(city_image):
    result = stylize(city_image, StampConfig(shape="circle", max_size=0))
    h, w = result.image.shape[:2]
    assert h == w and result.image.shape[2] == 4
    assert "<circle" in result.svg


def test_smaller_work_size_gives_fewer_regions():
    busy = np.clip(
        np.full((300, 300, 3), 120, np.int16) + np.random.default_rng(1).integers(-60, 60, (300, 300, 3)), 0, 255
    ).astype(np.uint8)
    coarse = stylize(busy, StampConfig(max_size=0, work_size=80))
    fine = stylize(busy, StampConfig(max_size=0, work_size=300))
    assert coarse.svg.count("<path") < fine.svg.count("<path")


def test_extract_sky_keeps_off_color_objects_out_of_the_sky():
    # a white tower shares a cluster with the grey sky; pixel colors tell them apart
    labels = np.zeros((100, 100), np.int32)
    labels[70:] = 1
    centers = np.array([[70, 0, -5], [30, 0, 0]], np.float32)
    lab = np.zeros((100, 100, 3), np.float32)
    lab[:70] = (65, 0, -6)
    lab[30:70, 80:86] = (95, 0, 0)  # white tower
    lab[70:] = (30, 0, 0)
    texture = np.zeros((100, 100), np.float32)
    new_labels, new_centers, sky = extract_sky(labels, centers, texture=texture, lab=lab)
    assert new_labels[10, 10] == sky
    tower = new_labels[50, 82]
    assert tower != sky
    assert new_centers[tower][0] > 90  # the tower keeps its own (white) color


def test_crop_square_follows_the_detailed_side():
    img = np.full((100, 300, 3), 128, np.uint8)
    rng = np.random.default_rng(0)
    img[:, 220:280] = rng.integers(0, 255, (100, 60, 3))  # the subject is on the right
    sq = crop_square(img)
    assert sq.shape == (100, 100, 3)
    assert sq[:, -30:].std() > 30  # the busy part is inside the crop


def test_colors_below_two_are_rejected():
    for bad in (0, 1):
        with pytest.raises(ValueError):
            StampConfig(colors=bad)


@pytest.mark.parametrize(
    "image, config",
    [
        (np.full((1, 1, 3), 128, np.uint8), StampConfig()),
        (np.full((64, 48, 3), 128, np.uint8), StampConfig(work_size=1)),
        (np.full((1, 50, 3), 128, np.uint8), StampConfig(shape="circle")),
    ],
)
def test_stylize_survives_one_pixel_work_images(image, config):
    assert stylize(image, config).image.ndim == 3


def test_quantize_single_pixel_returns_lab_centers():
    from photo2stamp.segment import quantize

    labels, centers = quantize(np.full((1, 1, 3), 128, np.uint8), k=6)
    assert centers.shape == (1, 3) and labels.shape == (1, 1)


def test_ring_plus_gap_must_leave_room_for_the_art():
    with pytest.raises(ValueError):
        StampConfig(ring=0.3, ring_gap=0.25)
