import numpy as np

from photo2stamp.segment import (
    extract_sky,
    majority_filter,
    merge_small_regions,
    quantize,
    smooth,
)


def test_smooth_keeps_shape_and_reduces_noise(city_image):
    out = smooth(city_image, strength=2)
    assert out.shape == city_image.shape
    assert out.dtype == np.uint8
    patch = (slice(150, 190), slice(110, 150))  # inside light building
    assert out[patch].std() < city_image[patch].std()


def test_smooth_zero_strength_returns_copy(city_image):
    out = smooth(city_image, strength=0)
    np.testing.assert_array_equal(out, city_image)
    assert out is not city_image


def test_quantize_returns_k_clusters(city_image):
    labels, centers = quantize(city_image, k=5, seed=0)
    assert labels.shape == city_image.shape[:2]
    assert centers.shape == (5, 3)
    assert labels.min() >= 0 and labels.max() < 5
    # sky and dark building must land in different clusters
    assert labels[10, 90] != labels[100, 40]


def test_quantize_is_deterministic(city_image):
    a, _ = quantize(city_image, k=4, seed=1)
    b, _ = quantize(city_image, k=4, seed=1)
    np.testing.assert_array_equal(a, b)


def test_majority_filter_removes_isolated_pixels():
    labels = np.zeros((30, 30), np.int32)
    labels[15, 15] = 1
    out = majority_filter(labels, n_labels=2, size=5, iterations=1)
    assert out[15, 15] == 0
    assert labels[15, 15] == 1  # input untouched


def test_merge_small_regions_absorbs_specks():
    labels = np.zeros((40, 40), np.int32)
    labels[:, 20:] = 1
    labels[5:7, 5:7] = 2  # 4 px speck inside region 0
    out = merge_small_regions(labels, min_area=10)
    assert out[5, 5] == 0
    assert (out[:, 20:] == 1).all()


def test_extract_sky_finds_bright_top_region(city_image):
    labels, centers = quantize(city_image, k=5, seed=0)
    new_labels, new_centers, sky = extract_sky(labels, centers)
    assert sky is not None
    assert new_labels[5, 90] == sky
    assert new_labels[150, 40] != sky  # dark building is not sky
    assert new_centers.shape[0] == centers.shape[0] + 1


def test_extract_sky_none_when_top_is_dark():
    labels = np.zeros((20, 20), np.int32)
    centers = np.array([[15, 0, 0]], np.float32)
    new_labels, new_centers, sky = extract_sky(labels, centers)
    assert sky is None
    np.testing.assert_array_equal(new_labels, labels)


def test_extract_accents_adds_cluster_for_small_vivid_patch():
    from photo2stamp.segment import extract_accents, to_lab

    img = np.full((100, 100, 3), 120, np.uint8)
    img[40:55, 40:60] = (30, 200, 240)  # small yellow sign (BGR)
    labels = np.zeros((100, 100), np.int32)
    centers = np.array([[50, 0, 0]], np.float32)
    new_labels, new_centers = extract_accents(to_lab(img), labels, centers, chroma=35, min_area=50)
    assert new_centers.shape[0] == 2
    assert new_labels[47, 50] == 1
    assert new_labels[5, 5] == 0
    assert np.hypot(*new_centers[1, 1:]) > 35
    assert labels.max() == 0  # input untouched


def test_extract_accents_ignores_tiny_specks():
    from photo2stamp.segment import extract_accents, to_lab

    img = np.full((60, 60, 3), 120, np.uint8)
    img[10:13, 10:13] = (0, 0, 255)
    labels = np.zeros((60, 60), np.int32)
    centers = np.array([[50, 0, 0]], np.float32)
    _, new_centers = extract_accents(to_lab(img), labels, centers, chroma=35, min_area=50)
    assert new_centers.shape[0] == 1
