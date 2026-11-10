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


def test_merge_small_regions_keeps_protected_labels():
    labels = np.zeros((40, 40), np.int32)
    labels[5:7, 5:7] = 2  # tiny accent
    labels[20:22, 20:22] = 3  # tiny non-accent
    out = merge_small_regions(labels, min_area=10, keep=frozenset({2}))
    assert out[5, 5] == 2
    assert out[20, 20] == 0


def test_extract_accents_bridges_text_holes():
    from photo2stamp.segment import extract_accents, to_lab

    img = np.full((100, 100, 3), 120, np.uint8)
    img[40:60, 30:70] = (40, 40, 220)  # red billboard
    img[40:60, 38:40] = 230  # white text strokes split it
    img[40:60, 50:52] = 230
    img[40:60, 60:62] = 230
    labels = np.zeros((100, 100), np.int32)
    centers = np.array([[50, 0, 0]], np.float32)
    new_labels, new_centers = extract_accents(to_lab(img), labels, centers, chroma=35, min_area=300)
    assert new_centers.shape[0] == 2  # one accent cluster, not three fragments
    assert new_labels[50, 50] == 1  # text stroke inside the sign is filled


def test_extract_far_marks_hazy_region_touching_sky():
    from photo2stamp.segment import extract_far

    labels = np.zeros((100, 100), np.int32)  # 0 = sky
    labels[40:60, :] = 1  # hazy skyline right under the sky
    labels[60:, :] = 2  # dark foreground
    centers = np.array([[80, -5, -15], [76, -3, -8], [25, 0, 0]], np.float32)
    new_labels, new_centers, far = extract_far(labels, centers, sky_label=0)
    assert far == 3
    assert new_labels[50, 50] == far
    assert new_labels[80, 50] == 2
    assert new_centers.shape[0] == 4


def test_extract_far_ignores_regions_unlike_sky_or_low_in_frame():
    from photo2stamp.segment import extract_far

    labels = np.zeros((100, 100), np.int32)
    labels[40:60, :50] = 1  # dark building touching the sky
    labels[40:, 50:] = 2  # sky-colored but reaching the bottom (a white wall)
    labels[60:, :50] = 3
    centers = np.array([[80, -5, -15], [30, 0, 0], [78, -4, -12], [20, 0, 0]], np.float32)
    new_labels, _, far = extract_far(labels, centers, sky_label=0)
    assert far is None
    np.testing.assert_array_equal(new_labels, labels)


def test_assign_colors_far_label_gets_far_color():
    from photo2stamp.palette import assign_colors, get_palette, hex_to_bgr

    palette = get_palette("teal")
    centers = np.array([[80, 0, 0], [40, 0, 0], [70, 0, 0]], np.float32)
    colors = assign_colors(centers, palette, sky_labels=frozenset({0}), far_labels=frozenset({2}))
    assert tuple(colors[2]) == hex_to_bgr(palette.far)


def test_extract_far_uses_pixel_colors_when_clusters_merge_haze_and_wall():
    from photo2stamp.segment import extract_far

    # one cluster (1) covers both a hazy skyline band and a wall reaching the ground,
    # but their actual pixel colors differ: only the haze is close to the sky color.
    labels = np.zeros((100, 100), np.int32)
    labels[40:, :] = 1
    centers = np.array([[80, -5, -15], [70, -2, -5]], np.float32)
    lab = np.zeros((100, 100, 3), np.float32)
    lab[:40] = (80, -5, -15)
    lab[40:55] = (78, -4, -12)  # haze
    lab[55:] = (60, 0, 2)  # wall
    new_labels, _, far = extract_far(labels, centers, sky_label=0, lab=lab)
    assert far is not None
    assert new_labels[45, 50] == far
    assert new_labels[80, 50] == 1


def test_extract_far_compares_with_nearby_sky_not_mean_sky():
    from photo2stamp.segment import extract_far

    # sky is a vertical gradient: deep blue at the top, pale near the horizon
    labels = np.zeros((100, 100), np.int32)
    labels[50:, :] = 1
    centers = np.array([[70, 0, -30], [60, 0, 0]], np.float32)
    lab = np.zeros((100, 100, 3), np.float32)
    rows = np.linspace(0, 1, 50)[:, None]
    lab[:50, :, 0] = 55 + 30 * rows
    lab[:50, :, 2] = -45 + 40 * rows  # -45 at top -> -5 at the horizon
    lab[50:65] = (84, 0, -4)  # haze: close to the horizon sky, far from the mean sky
    lab[65:] = (30, 0, 0)
    new_labels, _, far = extract_far(labels, centers, sky_label=0, lab=lab)
    assert far is not None and new_labels[55, 50] == far


def test_extract_far_ignores_clouds_inside_the_sky():
    from photo2stamp.segment import extract_far

    labels = np.zeros((100, 100), np.int32)
    labels[20:30, 30:60] = 1  # cloud floating in the sky
    labels[70:, :] = 2  # ground
    centers = np.array([[75, 0, -20], [80, 0, -12], [30, 0, 0]], np.float32)
    _, _, far = extract_far(labels, centers, sky_label=0)
    assert far is None


def test_extract_accents_skips_patches_touching_the_top_edge():
    from photo2stamp.segment import extract_accents, to_lab

    img = np.full((100, 100, 3), 120, np.uint8)
    img[0:12, 40:60] = (230, 140, 40)  # deep-blue sky corner, saturated
    labels = np.zeros((100, 100), np.int32)
    centers = np.array([[50, 0, 0]], np.float32)
    _, new_centers = extract_accents(to_lab(img), labels, centers, chroma=28, min_area=50)
    assert new_centers.shape[0] == 1


def test_quantize_chroma_weight_separates_muted_green_from_grey():
    from photo2stamp.segment import to_lab

    img = np.zeros((60, 120, 3), np.uint8)
    img[:, :40] = (100, 108, 100)  # grey road
    img[:, 40:80] = (88, 120, 88)  # muted green tree, same lightness
    img[:, 80:] = (230, 230, 230)  # white wall
    plain, _ = quantize(img, k=2, seed=0)
    weighted, centers = quantize(img, k=2, seed=0, chroma_weight=4.0)
    assert plain[30, 20] == plain[30, 60]  # lightness wins: green merged with grey
    assert weighted[30, 20] != weighted[30, 60]  # chroma weighting splits them
    green = centers[weighted[30, 60]]
    lab = to_lab(img[:1, 40:41]).reshape(3)
    assert abs(green[1] - lab[1]) < 3  # centers are returned in plain Lab units


def _paella():
    """Yellow rice in a dark pan with thin red pepper strips (the hues must not mix)."""
    import cv2

    rng = np.random.default_rng(0)
    img = np.full((240, 320, 3), 225, np.uint8)
    cv2.circle(img, (160, 120), 110, (40, 40, 40), -1)
    cv2.circle(img, (160, 120), 100, (30, 200, 235), -1)  # saffron rice (BGR)
    for i in range(6):
        a = i * np.pi / 3
        p0 = (int(160 + 20 * np.cos(a)), int(120 + 20 * np.sin(a)))
        p1 = (int(160 + 85 * np.cos(a)), int(120 + 85 * np.sin(a)))
        cv2.line(img, p0, p1, (40, 40, 210), 7)  # red pepper
    return np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)


def test_vivid_patches_of_different_hues_stay_apart():
    from photo2stamp import StampConfig, stylize
    from photo2stamp.segment import to_lab

    out = stylize(_paella(), StampConfig(max_size=0, lines=False)).image
    lab = to_lab(out).reshape(-1, 3)
    hue = np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    red = (chroma > 30) & ((hue < 45) | (hue > 340))
    yellow = (chroma > 30) & (hue > 70) & (hue < 110)
    assert red.mean() > 0.01 and yellow.mean() > 0.2
