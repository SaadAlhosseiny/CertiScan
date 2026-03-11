import numpy as np

def normalize_image(image):
    """
    Normalize image values between 0 and 1
    """
    image = image.astype(np.float32)
    return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

def create_mask(image, threshold=0.6):
    """
    Create a binary mask based on threshold
    """
    normalized = normalize_image(image)
    mask = normalized > threshold
    return mask.astype(np.uint8)

def apply_mask(image, mask):
    """
    Apply mask to highlight suspicious regions
    """
    masked = np.multiply(image, mask)
    return masked

def calculate_score(masked_image, mask=None):
    """
    Calculate statistics based only on masked region
    Returns mean, max, min, variance
    """
    if mask is not None:
        masked_pixels = masked_image[mask > 0]
    else:
        masked_pixels = masked_image.flatten()

    if masked_pixels.size == 0:
        return {"mean": 0.0, "max": 0, "min": 0, "variance": 0.0}

    return {
        "mean": float(np.mean(masked_pixels)),
        "max": int(np.max(masked_pixels)),
        "min": int(np.min(masked_pixels)),
        "variance": float(np.var(masked_pixels))
    }



