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


def _edges(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY)
    return (cv2.Canny(gray, 60, 150) > 0).astype(np.float32)


def edge_density(image: np.ndarray, size: int = 400) -> float:
    """Fraction of edge pixels at a fixed scale: a cheap 'how detailed is this photo'."""
    h, w = image.shape[:2]
    s = min(1.0, size / max(h, w))
    small = cv2.resize(image, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    return float(_edges(small).mean())


def texture_map(image: np.ndarray, window: float = 0.03) -> np.ndarray:
    """Local edge density in [0, 1]; low values mean smooth areas such as the sky."""
    h, w = image.shape[:2]
    k = max(3, int(round(max(h, w) * window)) | 1)
    return np.clip(cv2.blur(_edges(image), (k, k)), 0.0, 1.0)


def upscale_labels(labels: np.ndarray, size: tuple[int, int], n_labels: int) -> np.ndarray:
    """Enlarge a label map with smooth, curved boundaries instead of nearest-neighbour stairs.

    Each label's mask is enlarged with linear interpolation and blurred; every pixel then
    takes the label with the highest value.
    """
    h, w = size[:2]
    sh, sw = labels.shape
    sigma = max(0.5, 0.5 * max(h / sh, w / sw))
    best = np.full((h, w), -1.0, np.float32)
    out = np.zeros((h, w), np.int32)
    for i in range(n_labels):
        mask = (labels == i).astype(np.float32)
        if not mask.any():
            continue
        big = cv2.GaussianBlur(cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR), (0, 0), sigma)
        higher = big > best
        best[higher] = big[higher]
        out[higher] = i
    return out


def to_lab(image: np.ndarray) -> np.ndarray:
    """uint8 BGR image -> float32 Lab (L in 0..100)."""
    return cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def quantize(
    image: np.ndarray, k: int, seed: int = 0, chroma_weight: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """K-means in Lab space. Returns (labels HxW int32, centers Kx3 float32 Lab).

    chroma_weight > 1 scales a/b before clustering, so hue differences (green tree vs
    grey road of the same lightness) count more than lightness differences. Centers are
    returned in plain Lab units.
    """
    scale = np.array([1.0, chroma_weight, chroma_weight], np.float32)
    lab = to_lab(image).reshape(-1, 3) * scale
    rng = np.random.default_rng(seed)
    n = lab.shape[0]
    k = max(1, min(k, n))  # k-means needs at least k samples
    if n < 2 or k == 1:  # cv2.kmeans returns malformed centers for a single sample
        return np.zeros(image.shape[:2], np.int32), (lab.mean(axis=0, keepdims=True) / scale).astype(np.float32)
    sample = lab[rng.choice(n, size=min(n, _KMEANS_SAMPLE), replace=False)]

    cv2.setRNGSeed(seed)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, _, centers = cv2.kmeans(sample, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.float32).reshape(-1, 3)

    # one cluster at a time keeps memory at O(N) instead of O(N*K*3)
    best = np.full(n, np.inf, np.float32)
    labels = np.zeros(n, np.int32)
    for i, center in enumerate(centers):
        d = ((lab - center) ** 2).sum(axis=1)
        closer = d < best
        best[closer] = d[closer]
        labels[closer] = i
    return labels.reshape(image.shape[:2]), (centers / scale).astype(np.float32)


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


ACCENT_HUE_SECTOR = 45.0  # vivid pixels are grouped per hue sector of this width (deg)
ACCENT_MAX_AREA = 0.12  # a vivid patch larger than this fraction is a main region, not an accent


def extract_accents(
    lab: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    chroma: float,
    min_area: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Give small but vivid regions (signs, lights, peppers on rice) their own cluster.

    K-means tends to swallow small saturated patches into a large cluster. Vivid pixels
    (chroma >= `chroma`) are grouped per hue sector, so red strips lying on yellow rice
    are not averaged into mustard. Every connected patch with area in [`min_area`,
    ACCENT_MAX_AREA of the frame] becomes a new label whose center is the patch's mean
    Lab color. Larger vivid areas are left to k-means. Returns new arrays.
    """
    c = np.hypot(lab[..., 1], lab[..., 2])
    hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360
    sector = (hue // ACCENT_HUE_SECTOR).astype(np.int32)
    max_area = ACCENT_MAX_AREA * labels.size
    new_labels = labels.copy()
    extra: list[np.ndarray] = []
    for sec in np.unique(sector[c >= chroma]):
        vivid = ((c >= chroma) & (sector == sec)).astype(np.uint8)
        vivid = cv2.morphologyEx(vivid, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # close gaps left by text and frames so a sign becomes one solid patch
        vivid = cv2.morphologyEx(vivid, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        count, comp, stats, _ = cv2.connectedComponentsWithStats(vivid, connectivity=8)
        for i in range(1, count):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < max(1, min_area) or area > max_area:
                continue
            if stats[i, cv2.CC_STAT_TOP] == 0:  # vivid patches at the top edge are sky, not signs
                continue
            mask = (comp == i) & (new_labels < centers.shape[0])  # first sector to claim a pixel keeps it
            if not mask.any():
                continue
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
    texture: np.ndarray | None = None,
    max_texture: float = 0.05,
    lab: np.ndarray | None = None,
    max_sky_delta: float = 18.0,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Detect the sky: regions touching the top edge, relabelled as one sky label.

    With a `texture` map (see texture_map) a region is sky when it is smooth, whatever
    its color, so sunset, overcast and night skies are found too. Without it the old
    color rule is used: bright and not warm.
    Returns (new_labels, new_centers, sky_label). sky_label is None if nothing was found,
    in which case copies of the inputs are returned.
    """
    h, w = labels.shape
    sky = np.zeros(labels.shape, bool)
    used: list[int] = []
    for lbl in np.unique(labels[0]):
        L, _, b = centers[lbl]
        if texture is None and (L < min_lightness or b > max_yellow):
            continue
        mask = (labels == lbl).astype(np.uint8)
        _, comp, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        top_ids = np.unique(comp[0][comp[0] > 0])
        keep = [
            i
            for i in top_ids
            if stats[i, cv2.CC_STAT_AREA] >= min_fraction * h * w
            and (texture is None or float(texture[comp == i].mean()) <= max_texture)
        ]
        if keep:
            sky |= np.isin(comp, keep)
            used.append(int(lbl))

    if not used:
        return labels.copy(), centers.copy(), None
    sky_label = centers.shape[0]
    new_labels = np.where(sky, sky_label, labels).astype(np.int32)
    new_centers = np.vstack([centers, centers[used].mean(axis=0, keepdims=True)]).astype(np.float32)
    if lab is not None:
        new_labels, new_centers = _split_off_sky_outliers(new_labels, new_centers, sky_label, lab, max_sky_delta)
    return new_labels, new_centers, sky_label


def _split_off_sky_outliers(
    labels: np.ndarray, centers: np.ndarray, sky_label: int, lab: np.ndarray, max_delta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Objects that share a cluster with the sky (a white tower against a grey sky) are
    given back their own label, with their own mean color, instead of being painted as sky."""
    sky = labels == sky_label
    ref = np.median(lab[sky], axis=0)
    off = (np.linalg.norm(lab - ref, axis=2) > max_delta) & sky
    off = cv2.morphologyEx(off.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    if not off.any():
        return labels, centers
    new_labels = np.where(off, centers.shape[0], labels).astype(np.int32)
    return new_labels, np.vstack([centers, lab[off].mean(axis=0, keepdims=True)]).astype(np.float32)
