from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

ImageLike = Union[str, Path, np.ndarray]


def preprocess_image(image: ImageLike, size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Load and resize an image to a fixed size for feature extraction.

    This function is intended to be the single preprocessing entry point used by
    training, evaluation, and app inference to guarantee identical behavior.

    Args:
        image: Image path or already-loaded ndarray (BGR or grayscale).
        size: Target output size as (width, height).

    Returns:
        A uint8 BGR image of shape (height, width, 3).

    Raises:
        ValueError: If the image cannot be loaded or is invalid.
        TypeError: If the input type is not supported.
    """
    if not (isinstance(size, tuple) and len(size) == 2 and size[0] > 0 and size[1] > 0):
        raise ValueError(f"size must be a 2-tuple of positive ints, got: {size!r}")

    img: np.ndarray | None

    if isinstance(image, (str, Path)):
        image_path = Path(image)
        if not image_path.exists() or not image_path.is_file():
            raise ValueError(f"Image path does not exist or is not a file: {image_path}")
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    elif isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("Input image array is empty.")
        img = image
    else:
        raise TypeError(
            "image must be a path (str/Path) or a NumPy ndarray, "
            f"got {type(image).__name__}"
        )

    if img is None:
        raise ValueError("Failed to decode image. The file may be corrupted or unsupported.")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def extract_hsv_histogram(
    bgr_image: np.ndarray,
    h_bins: int = 16,
    s_bins: int = 16,
    v_bins: int = 16,
) -> np.ndarray:
    """Compute and L1-normalize a 3D HSV color histogram."""
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [h_bins, s_bins, v_bins], [0, 180, 0, 256, 0, 256])
    hist = hist.flatten().astype(np.float32)
    norm = np.sum(hist)
    if norm > 0:
        hist /= norm
    return hist


def extract_lbp_histogram(
    bgr_image: np.ndarray,
    radius: int = 1,
    n_points: int = 8,
    method: str = "uniform",
) -> np.ndarray:
    """Compute normalized Local Binary Pattern histogram on grayscale image."""
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=n_points, R=radius, method=method)

    if method == "uniform":
        n_bins = n_points + 2
    else:
        n_bins = int(lbp.max() + 1)

    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float32)
    norm = np.sum(hist)
    if norm > 0:
        hist /= norm
    return hist


def extract_hog_descriptor(
    bgr_image: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2),
) -> np.ndarray:
    """Compute HOG descriptor on grayscale image and return float32 vector."""
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    descriptor = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )
    return descriptor.astype(np.float32)


def extract_feature_vector(image: ImageLike, size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Build a single fused feature vector from HSV, LBP, and HOG features.

    All pipelines should call this function (or at least preprocess_image first)
    so preprocessing is consistent across train/eval/inference.
    """
    processed = preprocess_image(image, size=size)
    color_feat = extract_hsv_histogram(processed)
    texture_feat = extract_lbp_histogram(processed)
    shape_feat = extract_hog_descriptor(processed)
    return np.concatenate([color_feat, texture_feat, shape_feat], axis=0).astype(np.float32)
