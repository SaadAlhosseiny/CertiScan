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
#  1. Static threshold (kept for back-compat)
# ─────────────────────────────────────────────

def create_mask(image: np.ndarray, threshold: float = 0.6) -> np.ndarray:
    """
    Create a binary mask with a *static* threshold (legacy).
    Prefer `create_adaptive_mask` for better accuracy.

    Returns uint8 mask: 0 or 1.
    """
    normalized = normalize_image(image)
    mask = normalized > threshold
    return mask.astype(np.uint8)


# ─────────────────────────────────────────────
#  2. Adaptive threshold  ← NEW
# ─────────────────────────────────────────────

def create_adaptive_mask(
    image: np.ndarray,
    method: str = "otsu",
) -> np.ndarray:
    """
    Create a binary mask using Otsu's or Triangle auto-thresholding.

    Parameters
    ----------
    image  : Grayscale or single-channel float/uint8 array.
    method : 'otsu'     → cv2.THRESH_OTSU   (bimodal histograms)
             'triangle' → cv2.THRESH_TRIANGLE (skewed histograms / ELA maps)

    Returns
    -------
    mask : uint8 ndarray, values 0 or 255.
    """
    # Convert to uint8 for OpenCV
    gray_u8 = normalize_to_uint8(image)

    method = method.lower()
    if method == "otsu":
        flags = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    elif method == "triangle":
        flags = cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'otsu' or 'triangle'.")

    _, mask = cv2.threshold(gray_u8, 0, 255, flags)
    return mask  # uint8, 0 or 255


# ─────────────────────────────────────────────
#  3. Morphological refinement  ← NEW
# ─────────────────────────────────────────────

def morphological_refine(
    mask: np.ndarray,
    close_ksize: int = 7,
    erode_ksize: int = 3,
    close_iterations: int = 2,
    erode_iterations: int = 1,
) -> np.ndarray:
    """
    Clean up a binary mask with morphological operations:

    Step A – Closing  (Dilation → Erosion):
        Bridges small gaps inside forged regions (text edits, brush strokes).

    Step B – Erosion:
        Removes isolated 'salt-and-pepper' noise pixels that are
        too small to represent real manipulation.

    Parameters
    ----------
    mask              : uint8 binary mask (0 / 255).
    close_ksize       : Kernel size for closing  (odd int, default 7).
    erode_ksize       : Kernel size for erosion  (odd int, default 3).
    close_iterations  : Closing iterations  (default 2).
    erode_iterations  : Erosion iterations  (default 1).

    Returns
    -------
    refined_mask : uint8 ndarray (0 / 255).
    """
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)
    )
    kernel_erode = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_ksize, erode_ksize)
    )

    # Step A: Morphological Closing
    closed = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel_close, iterations=close_iterations
    )

    # Step B: Erosion to kill isolated noise
    refined = cv2.erode(closed, kernel_erode, iterations=erode_iterations)

    return refined


# ─────────────────────────────────────────────
#  4. Red overlay  ← NEW
# ─────────────────────────────────────────────

def apply_red_overlay(
    original_bgr: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Composite a semi-transparent red highlight over suspicious regions.

    Parameters
    ----------
    original_bgr : Original image in BGR colour space (uint8).
    mask         : Binary mask (0 / 255) – suspicious pixels are 255.
    alpha        : Opacity of the red overlay [0 = invisible, 1 = solid].

    Returns
    -------
    overlay_bgr : Composited image (uint8, BGR), same size as original.
    """
    if original_bgr.dtype != np.uint8:
        original_bgr = np.clip(original_bgr, 0, 255).astype(np.uint8)

    # Ensure mask is single-channel uint8
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_bool = mask > 0

    # Build a solid-red canvas
    red_layer = np.zeros_like(original_bgr)
    red_layer[:, :] = (0, 0, 255)  # BGR red

    # Blend: output = alpha * red + (1 - alpha) * original  (only in mask)
    overlay = original_bgr.copy()
    overlay[mask_bool] = cv2.addWeighted(
        red_layer, alpha, original_bgr, 1.0 - alpha, 0
    )[mask_bool]

    return overlay


# ─────────────────────────────────────────────
#  5. Composite mask from multiple maps  ← NEW
# ─────────────────────────────────────────────

def combine_maps(
    *maps: np.ndarray,
    weights: list[float] | None = None,
) -> np.ndarray:
    """
    Weighted average of multiple forensic maps (ELA, FFT, Noise …).
    All maps are normalised to [0, 1] before blending.

    Parameters
    ----------
    *maps   : Variable number of 2-D float/uint8 arrays (same shape).
    weights : Optional list of floats summing to 1.0.
               Defaults to equal weights.

    Returns
    -------
    combined : float32 ndarray in [0, 1].
    """
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
#  6. Scoring (kept + improved)
# ─────────────────────────────────────────────

def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply mask to highlight suspicious regions (element-wise multiply)."""
    # Normalise mask to [0, 1] if it uses 0/255 encoding
    m = mask / 255.0 if mask.max() > 1 else mask.astype(np.float32)
    return np.multiply(image, m)


def calculate_score(
    masked_image: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict:
    """
    Calculate statistics restricted to the masked region.

    Returns dict with keys: mean, max, min, variance.
    """
    if mask is not None:
        m = mask / 255 if mask.max() > 1 else mask
        masked_pixels = masked_image[m > 0]
    else:
        masked_pixels = masked_image.flatten()

    if masked_pixels.size == 0:
        return {"mean": 0.0, "max": 0, "min": 0, "variance": 0.0}

    return {
        "mean":     float(np.mean(masked_pixels)),
        "max":      int(np.max(masked_pixels)),
        "min":      int(np.min(masked_pixels)),
        "variance": float(np.var(masked_pixels)),
    }
