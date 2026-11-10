"""Smoothing, color clustering and label-map cleanup."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

_KMEANS_SAMPLE = 30_000


def smooth(image: np.ndarray, strength: int = 2) -> np.ndarray:
    """Edge-preserving smoothing: repeated bilateral filter + mean-shift. Returns a new image."""
    if strength <= 0:
        return image.copy()
    out = image
    for _ in range(strength):
        out = cv2.bilateralFilter(out, d=9, sigmaColor=40, sigmaSpace=9)
    return cv2.pyrMeanShiftFiltering(out, sp=6 + 4 * strength, sr=18 + 6 * strength)


def to_lab(image: np.ndarray) -> np.ndarray:
    """uint8 BGR image -> float32 Lab (L in 0..100)."""
    return cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def quantize(image: np.ndarray, k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """K-means in Lab space. Returns (labels HxW int32, centers Kx3 float32 Lab)."""
    lab = to_lab(image).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    n = lab.shape[0]
    k = max(1, min(k, n))  # k-means needs at least k samples
    sample = lab[rng.choice(n, size=min(n, _KMEANS_SAMPLE), replace=False)]

    cv2.setRNGSeed(seed)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, _, centers = cv2.kmeans(sample, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.float32)

    # one cluster at a time keeps memory at O(N) instead of O(N*K*3)
    best = np.full(n, np.inf, np.float32)
    labels = np.zeros(n, np.int32)
    for i, center in enumerate(centers):
        d = ((lab - center) ** 2).sum(axis=1)
        closer = d < best
        best[closer] = d[closer]
        labels[closer] = i
    return labels.reshape(image.shape[:2]), centers


def majority_filter(labels: np.ndarray, n_labels: int, size: int = 7, iterations: int = 2) -> np.ndarray:
    """Replace each pixel by the most common label in a size x size window."""
    out = labels.copy()
    if size <= 1:
        return out
    for _ in range(iterations):
        votes = np.stack(
            [cv2.blur((out == i).astype(np.float32), (size, size)) for i in range(n_labels)],
            axis=0,
        )
        out = votes.argmax(axis=0).astype(np.int32)
    return out


def merge_small_regions(
    labels: np.ndarray, min_area: int, keep: frozenset[int] = frozenset()
) -> np.ndarray:
    """Absorb connected regions smaller than min_area into their nearest large neighbour.

    Labels in `keep` (e.g. accent colors such as signs) are never absorbed.
    """
    if min_area <= 1:
        return labels.copy()
    small = np.zeros(labels.shape, bool)
    for lab in np.unique(labels):
        if int(lab) in keep:
            continue
        mask = (labels == lab).astype(np.uint8)
        count, comp, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        tiny = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] < min_area) + 1
        if tiny.size:
            small |= np.isin(comp, tiny)
    if not small.any() or small.all():
        return labels.copy()
    _, (rows, cols) = ndimage.distance_transform_edt(small, return_indices=True)
    return labels[rows, cols]


def extract_accents(
    lab: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    chroma: float,
    min_area: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Give small but vivid regions (signs, lights) their own cluster.

    K-means tends to swallow small saturated patches into a large neutral cluster.
    Every connected patch with chroma >= `chroma` and area >= `min_area` becomes a new
    label whose center is the patch's mean Lab color. Returns new arrays.
    """
    vivid = (np.hypot(lab[..., 1], lab[..., 2]) >= chroma).astype(np.uint8)
    vivid = cv2.morphologyEx(vivid, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # close gaps left by text and frames so a sign becomes one solid patch
    vivid = cv2.morphologyEx(vivid, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, comp, stats, _ = cv2.connectedComponentsWithStats(vivid, connectivity=8)
    new_labels = labels.copy()
    extra: list[np.ndarray] = []
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < max(1, min_area):
            continue
        if stats[i, cv2.CC_STAT_TOP] == 0:  # vivid patches at the top edge are sky, not signs
            continue
        mask = comp == i
        new_labels[mask] = centers.shape[0] + len(extra)
        extra.append(lab[mask].mean(axis=0))
    if not extra:
        return new_labels, centers.copy()
    return new_labels, np.vstack([centers, np.array(extra, np.float32)]).astype(np.float32)


def extract_far(
    labels: np.ndarray,
    centers: np.ndarray,
    sky_label: int,
    max_delta: float = 20.0,
    max_bottom: float = 0.75,
    lab: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Relabel hazy distant scenery (aerial perspective) as one 'far' region.

    Distant scenery seen through haze takes on the sky's color. A pixel is a candidate
    when its Lab color (from `lab`, or its cluster center if `lab` is None) is within
    `max_delta` of the mean sky color. Connected candidate patches that touch the sky
    and end above `max_bottom` of the image height become far; a sky-colored wall that
    reaches the ground is not distant scenery.
    Returns (new_labels, new_centers, far_label) or copies and None if nothing matched.
    """
    h = labels.shape[0]
    sky = labels == sky_label
    if not sky.any():
        return labels.copy(), centers.copy(), None
    pixels = centers[labels] if lab is None else lab
    # compare each pixel with the nearest sky pixel: skies are graded, and haze matches
    # the pale sky at the horizon, not the deep blue overhead
    _, (rows, cols) = ndimage.distance_transform_edt(~sky, return_indices=True)
    local_sky = pixels[rows, cols]
    close = (np.linalg.norm(pixels - local_sky, axis=2) <= max_delta) & ~sky
    # far scenery lies below the sky; patches with sky underneath them are clouds
    has_sky = sky.any(axis=0)
    horizon = np.where(has_sky, h - 1 - np.argmax(sky[::-1], axis=0), h)
    close &= np.arange(h)[:, None] > horizon[None, :]
    close = cv2.morphologyEx(close.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ring = cv2.dilate(sky.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~sky

    count, comp, stats, _ = cv2.connectedComponentsWithStats(close, connectivity=4)
    far = np.zeros(labels.shape, bool)
    for i in np.unique(comp[ring & (comp > 0)]):
        bottom = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]
        if bottom <= max_bottom * h:
            far |= comp == i
    if not far.any():
        return labels.copy(), centers.copy(), None
    far_label = centers.shape[0]
    new_labels = np.where(far, far_label, labels).astype(np.int32)
    new_centers = np.vstack([centers, pixels[far].mean(axis=0, keepdims=True)]).astype(np.float32)
    return new_labels, new_centers, far_label


def extract_sky(
    labels: np.ndarray,
    centers: np.ndarray,
    min_lightness: float = 50.0,
    max_yellow: float = 12.0,
    min_fraction: float = 0.005,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Detect bright, non-warm regions touching the top edge and relabel them as sky.

    Returns (new_labels, new_centers, sky_label). sky_label is None if nothing was found,
    in which case copies of the inputs are returned.
    """
    h, w = labels.shape
    sky = np.zeros(labels.shape, bool)
    used: list[int] = []
    for lab in np.unique(labels[0]):
        L, _, b = centers[lab]
        if L < min_lightness or b > max_yellow:
            continue
        mask = (labels == lab).astype(np.uint8)
        _, comp, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        top_ids = np.unique(comp[0][comp[0] > 0])
        keep = [i for i in top_ids if stats[i, cv2.CC_STAT_AREA] >= min_fraction * h * w]
        if keep:
            sky |= np.isin(comp, keep)
            used.append(int(lab))

    if not used:
        return labels.copy(), centers.copy(), None
    sky_label = centers.shape[0]
    new_labels = np.where(sky, sky_label, labels).astype(np.int32)
    new_centers = np.vstack([centers, centers[used].mean(axis=0, keepdims=True)]).astype(np.float32)
    return new_labels, new_centers, sky_label
