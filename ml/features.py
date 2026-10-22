from __future__ import annotations

import cv2
import numpy as np

try:
    from skimage.feature import local_binary_pattern
except ImportError:  # keep the classical pipeline usable with minimal installs
    def local_binary_pattern(gray, P=8, R=1, method="uniform"):
        center = gray.astype(np.int16)
        padded = np.pad(center, 1, mode="edge")
        values = np.zeros_like(center, dtype=np.uint8)
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        for bit, (dy, dx) in enumerate(offsets):
            values |= ((padded[1 + dy:1 + dy + gray.shape[0], 1 + dx:1 + dx + gray.shape[1]] >= center) << bit).astype(np.uint8)
        return values


def extract_features(image_path: str, size: tuple[int, int] = (128, 128)) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    pieces = []
    for space in (rgb, hsv, lab):
        for channel in cv2.split(space):
            hist = cv2.calcHist([channel], [0], None, [16], [0, 256]).flatten()
            pieces.append(hist / max(hist.sum(), 1.0))
        pieces.append(np.array([space.mean(axis=(0, 1)), space.std(axis=(0, 1))], dtype=np.float32).flatten() / 255.0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    pieces.append(hist.astype(np.float32) / max(hist.sum(), 1.0))
    return np.concatenate(pieces).astype(np.float32)
