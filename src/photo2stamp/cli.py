"""Command line interface: photo2stamp INPUT... -o OUT_DIR"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from .palette import ORIGINAL, PRESETS
from .pipeline import TONE_MAPPINGS, StampConfig, stylize
from .vectorize import SHAPES


def build_parser() -> argparse.ArgumentParser:
    d = StampConfig()
    p = argparse.ArgumentParser(prog="photo2stamp", description="Turn photos into flat stamp-style illustrations.")
    p.add_argument("inputs", nargs="+", type=Path, help="input image files")
    p.add_argument("-o", "--out-dir", type=Path, default=Path("out"), help="output directory (default: out)")
    p.add_argument("--palette", default=d.palette, choices=[*PRESETS, ORIGINAL])
    p.add_argument("--colors", type=int, default=d.colors, help="color clusters (default: %(default)s)")
    p.add_argument("--smooth", type=int, default=d.smooth, help="smoothing passes (default: %(default)s)")
    p.add_argument("--majority", type=int, default=d.majority, help="shape blockiness window px (default: %(default)s)")
    p.add_argument("--min-region", type=float, default=d.min_region, help="min region area fraction")
    p.add_argument("--simplify", type=float, default=d.simplify, help="polygon tolerance (fraction of diagonal)")
    p.add_argument("--snap-deg", type=float, default=d.snap_deg, help="snap near-vertical edges (0 = off)")
    p.add_argument("--max-size", type=int, default=d.max_size, help="resize long side; 0 keeps size")
    p.add_argument("--accent-chroma", type=float, default=d.accent_chroma, help="saturation threshold for accents")
    p.add_argument("--no-sky", action="store_true", help="disable sky detection")
    p.add_argument("--no-far", action="store_true", help="disable distant-skyline silhouette")
    p.add_argument("--far-delta", type=float, default=d.far_delta, help="max color distance to sky for far")
    p.add_argument("--shape", default=d.shape, choices=SHAPES, help="stamp outline")
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
        smooth=args.smooth,
        majority=args.majority,
        min_region=args.min_region,
        simplify=args.simplify,
        snap_deg=args.snap_deg,
        palette=args.palette,
        detect_sky=not args.no_sky,
        detect_far=not args.no_far,
        far_delta=args.far_delta,
        accent_chroma=args.accent_chroma,
        shape=args.shape,
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
