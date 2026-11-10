"""Render several StampConfig variants side by side for visual comparison.

    uv run python scripts/experiment.py examples/photo.jpg -o sheet.png \
        "rank:tone_mapping=rank" "area:tone_mapping=area" "blocky:majority=21,min_region=0.006"

Each variant is NAME:key=value,key=value (StampConfig fields). Use --reference to append
a target image at the end of the row.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import cv2
import numpy as np

from photo2stamp import StampConfig, stylize

_FIELDS = {f.name: f.type for f in dataclasses.fields(StampConfig)}


def parse_variant(spec: str) -> tuple[str, StampConfig]:
    name, _, body = spec.partition(":")
    kwargs: dict[str, object] = {}
    for item in filter(None, body.split(",")):
        key, _, raw = item.partition("=")
        if key not in _FIELDS:
            raise ValueError(f"unknown field {key!r}")
        default = getattr(StampConfig(), key)
        if isinstance(default, bool):
            kwargs[key] = raw.lower() in ("1", "true", "yes")
        else:
            kwargs[key] = type(default)(raw)
    return name, StampConfig(**kwargs)


def _tile(image: np.ndarray, label: str, height: int) -> np.ndarray:
    if image.shape[2] == 4:
        alpha = image[..., 3:].astype(np.float32) / 255
        image = (image[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    w = int(image.shape[1] * height / image.shape[0])
    tile = cv2.resize(image, (w, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (w, 28), (255, 255, 255), -1)
    cv2.putText(tile, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return tile


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("image", type=Path)
    p.add_argument("variants", nargs="+")
    p.add_argument("-o", "--output", type=Path, default=Path("experiment.png"))
    p.add_argument("--reference", type=Path, help="target image appended at the end")
    p.add_argument("--height", type=int, default=600)
    args = p.parse_args(argv)

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        print(f"error: cannot read image: {args.image}", file=sys.stderr)
        return 1
    tiles = [_tile(image, "input", args.height)]
    for spec in args.variants:
        name, config = parse_variant(spec)
        tiles.append(_tile(stylize(image, config).image, name, args.height))
    if args.reference:
        ref = cv2.imread(str(args.reference), cv2.IMREAD_COLOR)
        if ref is not None:
            tiles.append(_tile(ref, "reference", args.height))
    cv2.imwrite(str(args.output), np.hstack(tiles))
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
