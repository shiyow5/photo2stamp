"""Draw a synthetic 'photo' of a city street for testing (no license concerns).

    uv run python scripts/make_synthetic.py -o examples/synthetic_city.png

The scene has a gradient sky, a hazy distant skyline close to the sky color, near
buildings with window grids, a road, a red and a yellow sign, a tree, plus blur and
sensor noise so it behaves like a photo rather than flat art.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

W, H = 600, 800


def _sky(img: np.ndarray) -> None:
    top, bottom = np.array([215, 170, 120]), np.array([225, 205, 185])  # BGR
    t = np.linspace(0, 1, H)[:, None, None]
    img[:] = (top * (1 - t) + bottom * t).astype(np.uint8)


def _skyline(img: np.ndarray, rng: np.random.Generator) -> None:
    x = 0
    while x < W:
        bw, bh = int(rng.integers(18, 50)), int(rng.integers(30, 110))
        shade = int(rng.integers(0, 18))
        cv2.rectangle(img, (x, 330 - bh), (x + bw, 340), (200 - shade, 185 - shade, 170 - shade), -1)
        x += bw


def _building(img, x0, y0, x1, y1, color, window, rng) -> None:
    cv2.rectangle(img, (x0, y0), (x1, H), color, -1)
    side = tuple(int(c * 0.7) for c in color)
    cv2.rectangle(img, (x1, y0 + 10), (x1 + (x1 - x0) // 4, H), side, -1)
    for y in range(y0 + 12, H - 10, 22):
        for x in range(x0 + 8, x1 - 8, 16):
            lit = rng.random() < 0.15
            cv2.rectangle(img, (x, y), (x + 8, y + 12), (120, 200, 240) if lit else window, -1)


def make(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.uint8)
    _sky(img)
    _skyline(img, rng)
    img[340:] = (120, 125, 130)  # ground / mid-distance blocks below the skyline
    pts = np.array([[230, H], [370, H], [330, 420], [290, 420]], np.int32)
    cv2.fillPoly(img, [pts], (80, 80, 85))  # road
    _building(img, 0, 120, 170, 0, (95, 85, 75), (60, 55, 50), rng)
    _building(img, 400, 200, 540, 0, (170, 165, 160), (110, 105, 100), rng)
    cv2.rectangle(img, (40, 430), (130, 500), (40, 40, 210), -1)  # red sign
    cv2.rectangle(img, (420, 520), (500, 560), (40, 200, 235), -1)  # yellow sign
    cv2.circle(img, (330, 560), 45, (60, 120, 60), -1)  # tree
    img = cv2.GaussianBlur(img, (3, 3), 0)
    noise = rng.normal(0, 7, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output", type=Path, default=Path("examples/synthetic_city.png"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    cv2.imwrite(str(args.output), make(args.seed))
    print(args.output)


if __name__ == "__main__":
    main()
