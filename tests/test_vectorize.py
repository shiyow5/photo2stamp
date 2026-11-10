import xml.etree.ElementTree as ET

import numpy as np
import pytest

from photo2stamp.vectorize import (
    frame_stamp,
    labels_to_shapes,
    render_png,
    shapes_to_svg,
)

COLORS = np.array([[255, 0, 0], [0, 0, 255]], np.uint8)  # BGR: blue, red


def _two_regions() -> np.ndarray:
    labels = np.zeros((50, 60), np.int32)
    labels[10:40, 15:45] = 1
    return labels


def test_labels_to_shapes_sorted_by_area_desc():
    shapes = labels_to_shapes(_two_regions(), COLORS, epsilon=1.0)
    assert len(shapes) == 2
    assert shapes[0].area >= shapes[1].area
    assert shapes[0].color == (255, 0, 0)
    assert shapes[1].color == (0, 0, 255)


def test_render_png_reproduces_regions():
    labels = _two_regions()
    shapes = labels_to_shapes(labels, COLORS, epsilon=1.0)
    img = render_png(shapes, labels.shape)
    assert img.shape == (50, 60, 3)
    assert tuple(img[25, 30]) == (0, 0, 255)
    assert tuple(img[2, 2]) == (255, 0, 0)


def test_shapes_to_svg_is_valid_xml():
    labels = _two_regions()
    shapes = labels_to_shapes(labels, COLORS, epsilon=1.0)
    svg = shapes_to_svg(shapes, labels.shape)
    root = ET.fromstring(svg)
    assert root.attrib["viewBox"] == "0 0 60 50"
    paths = [el for el in root.iter() if el.tag.endswith("path")]
    assert len(paths) == 2
    assert "#ff0000" in svg  # red region (BGR 0,0,255)


def test_shapes_to_svg_circle_adds_clip():
    labels = _two_regions()
    shapes = labels_to_shapes(labels, COLORS, epsilon=1.0)
    svg = shapes_to_svg(shapes, labels.shape, shape="circle")
    assert "clipPath" in svg


def test_frame_stamp_rect_is_opaque():
    out = frame_stamp(np.full((10, 10, 3), 128, np.uint8), "rect", (0, 0, 0), ring=0, gap=0)
    assert (out[..., 3] == 255).all()


def test_frame_stamp_unknown_shape_raises():
    with pytest.raises(ValueError):
        frame_stamp(np.zeros((4, 4, 3), np.uint8), "star", (0, 0, 0))


def test_snap_vertical_straightens_near_vertical_edges():
    from photo2stamp.vectorize import snap_vertical

    pts = np.array([[10, 0], [12, 50], [40, 50], [40, 0]], np.int32)  # left edge ~2.3 deg off
    out = snap_vertical(pts, max_deg=10)
    assert out[0, 0] == out[1, 0]
    assert pts[0, 0] == 10 and pts[1, 0] == 12  # input untouched


def test_snap_vertical_keeps_diagonals():
    from photo2stamp.vectorize import snap_vertical

    pts = np.array([[0, 0], [30, 40], [0, 40]], np.int32)
    np.testing.assert_array_equal(snap_vertical(pts, max_deg=10), pts)


def test_svg_has_an_exact_base_layer_under_the_smooth_shapes():
    labels = _two_regions()
    shapes = labels_to_shapes(labels, COLORS, epsilon=3.0)
    base = labels_to_shapes(labels, COLORS, epsilon=0)
    svg = shapes_to_svg(shapes, labels.shape, base=base)
    assert svg.index('<g id="base">') < svg.index('<g id="shapes">')
    ET.fromstring(svg)  # well-formed


def test_stylize_svg_contains_the_base_layer(city_image):
    from photo2stamp import StampConfig, stylize

    assert '<g id="base">' in stylize(city_image, StampConfig(max_size=0)).svg
