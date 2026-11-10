"""Integration (pipeline) and end-to-end (CLI subprocess) tests."""

import subprocess
import sys

import cv2
import numpy as np
import pytest

from photo2stamp.cli import main
from photo2stamp.palette import get_palette, hex_to_bgr
from photo2stamp.pipeline import StampConfig, stylize


def test_stylize_output_uses_only_palette_colors(city_image):
    config = StampConfig(colors=6, palette="teal", max_size=0)
    result = stylize(city_image, config)
    palette = get_palette("teal")
    allowed = {hex_to_bgr(c) for c in (*palette.tones, palette.sky, palette.far, *palette.accents)}
    used = {tuple(int(v) for v in c) for c in result.flat_colors}
    assert used <= allowed
    assert result.image.shape[:2] == city_image.shape[:2]
    assert result.svg.startswith("<svg")


def test_stylize_paints_sky_and_accent(city_image):
    result = stylize(city_image, StampConfig(colors=6, max_size=0, palette="teal"))
    palette = get_palette("teal")
    assert tuple(result.image[5, 90]) == hex_to_bgr(palette.sky)
    sign = tuple(result.image[135, 35])
    assert sign in {hex_to_bgr(a) for a in palette.accents}


def test_stylize_resizes_to_max_size(city_image):
    result = stylize(city_image, StampConfig(max_size=120))
    assert max(result.image.shape[:2]) == 120


def test_stylize_circle_shape_gives_alpha(city_image):
    result = stylize(city_image, StampConfig(shape="circle", max_size=0))
    assert result.image.shape[2] == 4


def test_config_validation():
    with pytest.raises(ValueError):
        StampConfig(colors=1)
    with pytest.raises(ValueError):
        StampConfig(shape="hexagon")
    with pytest.raises(ValueError):
        StampConfig(palette="unknown")


def test_cli_writes_png_and_svg(tmp_path, city_image):
    src = tmp_path / "photo.png"
    cv2.imwrite(str(src), city_image)
    out_dir = tmp_path / "out"
    code = main([str(src), "-o", str(out_dir), "--colors", "5", "--svg"])
    assert code == 0
    png = cv2.imread(str(out_dir / "photo.png"), cv2.IMREAD_UNCHANGED)
    assert png is not None
    assert (out_dir / "photo.svg").read_text().startswith("<svg")


def test_cli_missing_file_returns_error(tmp_path, capsys):
    code = main([str(tmp_path / "missing.jpg"), "-o", str(tmp_path)])
    assert code == 1
    assert "cannot read" in capsys.readouterr().err


def test_cli_e2e_subprocess(tmp_path, city_image):
    src = tmp_path / "e2e.jpg"
    cv2.imwrite(str(src), city_image)
    proc = subprocess.run(
        [sys.executable, "-m", "photo2stamp", str(src), "-o", str(tmp_path), "--shape", "circle"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    out = cv2.imread(str(tmp_path / "e2e.png"), cv2.IMREAD_UNCHANGED)
    assert out.shape[2] == 4
    assert np.any(out[..., 3] == 0)


def test_stylize_tiny_image_with_more_colors_than_pixels():
    result = stylize(np.zeros((2, 2, 3), np.uint8), StampConfig(max_size=0))
    assert result.image.shape[:2] == (2, 2)


def test_stylize_does_not_upscale_small_images():
    result = stylize(np.zeros((40, 30, 3), np.uint8), StampConfig(max_size=800))
    assert result.image.shape[:2] == (40, 30)


@pytest.mark.parametrize(
    "image",
    [
        np.full((60, 60), 128, np.uint8),
        np.full((60, 60, 1), 128, np.uint8),
        np.full((60, 60, 3), 30000, np.uint16),
    ],
)
def test_stylize_accepts_gray_and_uint16(image):
    assert stylize(image, StampConfig(max_size=0)).image.shape[:2] == (60, 60)


def test_stylize_composites_rgba_on_white():
    img = np.zeros((60, 60, 4), np.uint8)  # fully transparent black
    result = stylize(img, StampConfig(max_size=0, palette="original", detect_sky=False))
    assert result.image[30, 30].mean() > 200


def test_stylize_empty_image_raises():
    with pytest.raises(ValueError, match="empty"):
        stylize(np.zeros((0, 0, 3), np.uint8))


def test_cli_duplicate_stems_do_not_overwrite(tmp_path, city_image):
    (tmp_path / "sub").mkdir()
    cv2.imwrite(str(tmp_path / "a.png"), city_image)
    cv2.imwrite(str(tmp_path / "sub" / "a.jpg"), city_image)
    out = tmp_path / "out"
    assert main([str(tmp_path / "a.png"), str(tmp_path / "sub" / "a.jpg"), "-o", str(out)]) == 0
    assert (out / "a.png").exists() and (out / "a_1.png").exists()


def test_cli_continues_after_processing_error(tmp_path, city_image, monkeypatch, capsys):
    import photo2stamp.cli as cli

    calls = {"n": 0}
    real = cli.stylize

    def flaky(image, config):
        calls["n"] += 1
        if calls["n"] == 1:
            raise cv2.error("boom")
        return real(image, config)

    monkeypatch.setattr(cli, "stylize", flaky)
    for name in ("x.png", "y.png"):
        cv2.imwrite(str(tmp_path / name), city_image)
    code = main([str(tmp_path / "x.png"), str(tmp_path / "y.png"), "-o", str(tmp_path / "o")])
    assert code == 1
    assert "x.png" in capsys.readouterr().err
    assert (tmp_path / "o" / "y.png").exists()


def test_natural_palette_skips_far_silhouette(monkeypatch, city_image):
    import photo2stamp.pipeline as pl

    called = []
    monkeypatch.setattr(pl, "extract_far", lambda *a, **k: called.append(1) or (a[0], a[1], None))
    stylize(city_image, StampConfig(max_size=0, palette="natural"))
    assert not called
    stylize(city_image, StampConfig(max_size=0, palette="teal"))
    assert called


def test_cli_version(capsys):
    import pytest as _pytest

    from photo2stamp import __version__

    with _pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_cli_line_and_smoother_options_reach_config():
    from photo2stamp.cli import _config, build_parser

    args = build_parser().parse_args(
        ["x.jpg", "--smoother", "meanshift", "--line-width", "3", "--line-threshold", "0.02",
         "--line-min-length", "0.1", "--line-sigma", "2.5"]
    )
    cfg = _config(args)
    assert (cfg.smoother, cfg.line_width, cfg.line_threshold, cfg.line_min_length, cfg.line_sigma) == (
        "meanshift", 3.0, 0.02, 0.1, 2.5
    )
