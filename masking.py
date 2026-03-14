import cv2
import numpy as np

def create_adaptive_mask(image, method="triangle"):
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 255)).astype(np.uint8)
    flag = cv2.THRESH_TRIANGLE if method == "triangle" else cv2.THRESH_OTSU
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + flag)
    return mask

def morphological_refine(mask, close_ksize=7, erode_ksize=3, close_iterations=2, erode_iterations=1):
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_ksize, erode_ksize))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=close_iterations)
    refined = cv2.erode(closed, k_erode, iterations=erode_iterations)
    return refined

def apply_red_overlay(original, mask, alpha=0.45):
    try:
        h, w = original.shape[:2]
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if len(original.shape) == 2:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        overlay = original.copy()
        overlay[mask_resized > 0] = [0, 0, 255]
        return cv2.addWeighted(overlay, alpha, original, 1 - alpha, 0)
    except:
        return original

def combine_maps(ela, fft, noise, weights=None):
    if weights is None: weights = [0.3, 0.5, 0.2]
    def to_gray(img):
        if len(img.shape) == 3: return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    ela_g = cv2.resize(to_gray(ela), (512, 512))
    fft_g = cv2.resize(to_gray(fft), (512, 512))
    noise_g = cv2.resize(to_gray(noise), (512, 512))
    combined = (ela_g.astype(np.float32) * weights[0] + 
                fft_g.astype(np.float32) * weights[1] + 
                noise_g.astype(np.float32) * weights[2])
    return np.clip(combined, 0, 255).astype(np.uint8)