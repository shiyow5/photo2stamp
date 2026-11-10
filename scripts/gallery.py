"""Make a before/after grid for several photos with one StampConfig.

    uv run python scripts/gallery.py -o examples/gallery.jpg examples/photo.jpg examples/free/*.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from photo2stamp import StampConfig, stylize

BOX_W, BOX_H, GAP = 360, 270, 8


def _fit(image: np.ndarray) -> np.ndarray:
    """Scale into a BOX_W x BOX_H white box, keeping the aspect ratio."""
    h, w = image.shape[:2]
    s = min(BOX_W / w, BOX_H / h)
    small = cv2.resize(image[..., :3], (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    box = np.full((BOX_H, BOX_W, 3), 255, np.uint8)
    y, x = (BOX_H - small.shape[0]) // 2, (BOX_W - small.shape[1]) // 2
    box[y : y + small.shape[0], x : x + small.shape[1]] = small
    return box


def _pair(path: Path, config: StampConfig) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"skip: cannot read {path}", file=sys.stderr)
        return None
    gap = np.full((BOX_H, GAP, 3), 255, np.uint8)
    art = stylize(image, config).image
    if art.shape[2] == 4:  # composite transparent stamps onto white
        alpha = art[..., 3:].astype(np.float32) / 255
        art = (art[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    return np.hstack([_fit(image), gap, _fit(art)])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("-o", "--output", type=Path, default=Path("gallery.jpg"))
    p.add_argument("--columns", type=int, default=2)
    p.add_argument("--shape", default="rect")
    p.add_argument("--palette", default="teal")
    args = p.parse_args()

    config = StampConfig(shape=args.shape, palette=args.palette)
    pairs = [t for t in (_pair(path, config) for path in args.inputs) if t is not None]
    if not pairs:
        return 1
    blank = np.full_like(pairs[0], 255)
    pairs += [blank] * (-len(pairs) % args.columns)
    vgap = np.full((GAP * 2, pairs[0].shape[1], 3), 255, np.uint8)
    cols = []
    for c in range(args.columns):
        cells = pairs[c :: args.columns]
        stacked = [x for cell in cells for x in (cell, vgap)][:-1]
        cols.append(np.vstack(stacked))
    hgap = np.full((cols[0].shape[0], GAP * 3, 3), 255, np.uint8)
    grid = np.hstack([x for col in cols for x in (col, hgap)][:-1])
    cv2.imwrite(str(args.output), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
