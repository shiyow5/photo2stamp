import numpy as np
import pytest


@pytest.fixture
def city_image() -> np.ndarray:
    """Synthetic BGR 'photo': sky, two buildings, a road, a yellow sign, plus noise."""
    rng = np.random.default_rng(0)
    h, w = 240, 180
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (200, 185, 170)  # hazy sky (BGR, bluish grey)
    img[60:, 10:70] = (60, 55, 50)  # dark building
    img[90:, 100:160] = (150, 150, 155)  # light building
    img[200:, :] = (90, 90, 90)  # road
    img[120:150, 20:50] = (40, 200, 230)  # yellow sign
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
