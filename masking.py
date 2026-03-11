import cv2
import numpy as np

# ─────────────────────────────────────────────
#  Low-level helpers
# ─────────────────────────────────────────────

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image values to [0, 1] (float32)."""
    image = image.astype(np.float32)
    return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize any float/int image to uint8 [0, 255]."""
    norm = normalize_image(image)
    return (norm * 255).astype(np.uint8)

# ─────────────────────────────────────────────
#  1. Static threshold (legacy)
# ─────────────────────────────────────────────

def create_mask(image: np.ndarray, threshold: float = 0.6) -> np.ndarray:
    """
    Create a binary mask with a static threshold (legacy).
    Prefer `create_adaptive_mask` for better accuracy.
    """
    normalized = normalize_image(image)
    mask = normalized > threshold
    return mask.astype(np.uint8)

# ─────────────────────────────────────────────
#  2. Adaptive threshold
# ─────────────────────────────────────────────

def create_adaptive_mask(image: np.ndarray, method: str = "otsu") -> np.ndarray:
    """
    Create a binary mask using Otsu's or Triangle auto-thresholding.

    Parameters
    ----------
    image  : Grayscale or single-channel float/uint8 array.
    method : 'otsu'     → cv2.THRESH_OTSU
             'triangle' → cv2.THRESH_TRIANGLE

    Returns
    -------
    mask : uint8 ndarray, values 0 or 255.
    """
    gray_u8 = normalize_to_uint8(image)

    method = method.lower()
    if method == "otsu":
        flags = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    elif method == "triangle":
        flags = cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'otsu' or 'triangle'.")

    _, mask = cv2.threshold(gray_u8, 0, 255, flags)
    return mask

# ─────────────────────────────────────────────
#  3. Morphological refinement
# ─────────────────────────────────────────────

def morphological_refine(
    mask: np.ndarray,
    close_ksize: int = 7,
    erode_ksize: int = 3,
    close_iterations: int = 2,
    erode_iterations: int = 1,
) -> np.ndarray:
    """Clean up a binary mask with morphological operations."""
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_ksize, erode_ksize))

    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=close_iterations)
    refined = cv2.erode(closed, kernel_erode, iterations=erode_iterations)

    return refined

# ─────────────────────────────────────────────
#  4. Red overlay
# ─────────────────────────────────────────────

def apply_red_overlay(original_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Composite a semi-transparent red highlight over suspicious regions."""
    if original_bgr.dtype != np.uint8:
        original_bgr = np.clip(original_bgr, 0, 255).astype(np.uint8)

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_bool = mask > 0

    red_layer = np.zeros_like(original_bgr)
    red_layer[:, :] = (0, 0, 255)

    overlay = original_bgr.copy()
    overlay[mask_bool] = cv2.addWeighted(red_layer, alpha, original_bgr, 1.0 - alpha, 0)[mask_bool]

    return overlay

# ─────────────────────────────────────────────
#  5. Composite maps
# ─────────────────────────────────────────────

def combine_maps(*maps: np.ndarray, weights: list[float] | None = None) -> np.ndarray:
    """Weighted average of multiple forensic maps (ELA, FFT, Noise …)."""
    if not maps:
        raise ValueError("At least one map is required.")

    if weights is None:
        weights = [1.0 / len(maps)] * len(maps)

    if len(weights) != len(maps):
        raise ValueError("Length of `weights` must match the number of maps.")

    combined = np.zeros_like(maps[0], dtype=np.float32)
    for mp, w in zip(maps, weights):
        combined += w * normalize_image(mp)

    return combined

# ─────────────────────────────────────────────
#  6. Scoring
# ─────────────────────────────────────────────

def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply mask to highlight suspicious regions (element-wise multiply)."""
    m = mask / 255.0 if mask.max() > 1 else mask.astype(np.float32)
    return np.multiply(image, m)

def calculate_score(masked_image: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Calculate statistics restricted to the masked region."""
    if mask is not None:
        m = mask / 255 if mask.max() > 1 else mask
        masked_pixels = masked_image[m > 0]
    else:
        masked_pixels = masked_image.flatten()

    if masked_pixels.size == 0:
        return {"mean": 0.0, "max": 0, "min": 0, "variance": 0.0}

    return {
        "mean": float(np.mean(masked_pixels)),
        "max": int(np.max(masked_pixels)),
        "min": int(np.min(masked_pixels)),
        "variance": float(np.var(masked_pixels)),
    }