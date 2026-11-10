"""Command line interface: photo2stamp INPUT... -o OUT_DIR"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from . import __version__
from .palette import NATURAL, ORIGINAL, PRESETS
from .pipeline import CROPS, NIGHT_MODES, SMOOTHERS, TONE_MAPPINGS, StampConfig, stylize
from .vectorize import SHAPES


def build_parser() -> argparse.ArgumentParser:
    d = StampConfig()
    p = argparse.ArgumentParser(prog="photo2stamp", description="Turn photos into flat stamp-style illustrations.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("inputs", nargs="+", type=Path, help="input image files")
    p.add_argument("-o", "--out-dir", type=Path, default=Path("out"), help="output directory (default: out)")
    p.add_argument("--palette", default=d.palette, choices=[NATURAL, *PRESETS, ORIGINAL])
    p.add_argument("--colors", type=int, default=d.colors, help="color clusters (default: %(default)s)")
    p.add_argument("--smoother", default=d.smoother, choices=SMOOTHERS,
                   help="l0 = L0 gradient minimization (default) / meanshift")
    p.add_argument("--l0-lambda", type=float, default=d.l0_lambda,
                   help="L0 smoothing strength; larger = flatter (default: %(default)s)")
    p.add_argument("--smooth", type=int, default=d.smooth,
                   help="mean-shift passes, with --smoother meanshift (default: %(default)s)")
    p.add_argument("--majority", type=int, default=d.majority, help="shape blockiness window px (default: %(default)s)")
    p.add_argument("--min-region", type=float, default=d.min_region, help="min region area fraction")
    p.add_argument("--simplify", type=float, default=d.simplify, help="polygon tolerance (fraction of diagonal)")
    p.add_argument("--snap-deg", type=float, default=d.snap_deg, help="snap near-vertical edges (0 = off)")
    p.add_argument("--work-size", type=int, default=d.work_size,
                   help="segmentation resolution (long side); 0 = auto from detail (default)")
    p.add_argument("--ring", type=float, default=d.ring, help="stamp ring width (fraction of size)")
    p.add_argument("--ring-gap", type=float, default=d.ring_gap, help="gap between ring and art")
    p.add_argument("--max-size", type=int, default=d.max_size, help="resize long side; 0 keeps size")
    p.add_argument("--accent-chroma", type=float, default=d.accent_chroma, help="saturation threshold for accents")
    p.add_argument("--night", default=d.night, choices=NIGHT_MODES,
                   help="night mode: dark fills with light outlines (default: auto by brightness)")
    p.add_argument("--no-white-balance", action="store_true",
                   help="keep the photo's color cast (by default the light's color is removed)")
    p.add_argument("--no-lines", action="store_true", help="no outlines, flat color only")
    p.add_argument("--line-width", type=float, default=d.line_width,
                   help="outline width in px at 600 px (default: %(default)s)")
    p.add_argument("--line-threshold", type=float, default=d.line_threshold,
                   help="edge strength needed for a line; larger = fewer lines (default: %(default)s)")
    p.add_argument("--line-min-length", type=float, default=d.line_min_length,
                   help="drop lines shorter than this fraction of the long side (default: %(default)s)")
    p.add_argument("--line-sigma", type=float, default=d.line_sigma,
                   help="outline scale; larger = only big structures (default: %(default)s)")
    p.add_argument("--no-sky", action="store_true", help="disable sky detection")
    p.add_argument("--no-far", action="store_true", help="disable distant-skyline silhouette")
    p.add_argument("--far-delta", type=float, default=d.far_delta, help="max color distance to sky for far")
    p.add_argument("--shape", default=d.shape, choices=SHAPES, help="stamp outline")
    p.add_argument(
        "--crop", default=d.crop, choices=CROPS,
        help="with --shape circle/rounded: before = stylize the square crop (default) / "
             "after = cut it out of the full-photo art (colors and night mode follow the whole photo)",
    )
    p.add_argument(
        "--tone-mapping", default=d.tone_mapping, choices=TONE_MAPPINGS,
        help="area: spread tones by pixel area (default) / rank: one tone per cluster",
    )
    p.add_argument("--svg", action="store_true", help="also write an SVG next to the PNG")
    p.add_argument("--seed", type=int, default=d.seed)
    return p


def _config(args: argparse.Namespace) -> StampConfig:
    return StampConfig(
        max_size=args.max_size,
        colors=args.colors,
        smoother=args.smoother,
        l0_lambda=args.l0_lambda,
        smooth=args.smooth,
        line_width=args.line_width,
        line_threshold=args.line_threshold,
        line_min_length=args.line_min_length,
        line_sigma=args.line_sigma,
        majority=args.majority,
        min_region=args.min_region,
        simplify=args.simplify,
        snap_deg=args.snap_deg,
        work_size=args.work_size,
        ring=args.ring,
        ring_gap=args.ring_gap,
        palette=args.palette,
        detect_sky=not args.no_sky,
        night=args.night,
        lines=not args.no_lines,
        white_balance=not args.no_white_balance,
        detect_far=not args.no_far,
        far_delta=args.far_delta,
        accent_chroma=args.accent_chroma,
        shape=args.shape,
        crop=args.crop,
        tone_mapping=args.tone_mapping,
        seed=args.seed,
    )


def _unique_stem(stem: str, used: set[str]) -> str:
    """Avoid overwriting outputs when two inputs share a file name (a.png, sub/a.jpg)."""
    candidate, n = stem, 0
    while candidate in used:
        n += 1
        candidate = f"{stem}_{n}"
    return candidate


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _config(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    status = 0
    used_stems: set[str] = set()
    for path in args.inputs:
        # IMREAD_COLOR also applies the EXIF orientation of phone photos
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"error: cannot read image: {path}", file=sys.stderr)
            status = 1
            continue
        try:
            result = stylize(image, config)
        except (cv2.error, ValueError) as exc:
            print(f"error: {path}: {exc}", file=sys.stderr)
            status = 1
            continue
        stem = _unique_stem(path.stem, used_stems)
        used_stems.add(stem)
        png = args.out_dir / f"{stem}.png"
        if not cv2.imwrite(str(png), result.image):
            print(f"error: cannot write: {png}", file=sys.stderr)
            status = 1
            continue
        written = [png]
        if args.svg:
            svg = args.out_dir / f"{stem}.svg"
            svg.write_text(result.svg, encoding="utf-8")
            written.append(svg)
        print(" ".join(str(p) for p in written))
    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
