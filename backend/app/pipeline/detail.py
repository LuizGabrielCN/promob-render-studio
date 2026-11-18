"""
Edge-Preserving Detail Enhancement & Micro-Contrast Boosting.
Combines optional bilateral filtering (JPEG artifact removal) with unsharp masking.
"""
import cv2
import numpy as np


def apply_detail_enhancement(
    img: np.ndarray,
    sharpness: float = 0.40,
    denoise_strength: float = 0.10,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Boosts micro-textures (wood grain, marble veins) while suppressing JPEG artifacts.

    Args:
        img: np.ndarray BGR or BGRA, uint8 or float32.
        sharpness: float in [0.0, 2.0]. Unsharp mask gain (default 0.40).
        denoise_strength: float in [0.0, 1.0]. Bilateral smoothing blend (default 0.10).
        sigma: float. Gaussian blur standard deviation (default 1.0).

    Returns:
        Sharpened image preserving geometric straight lines.
    """
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("Input image cannot be None or empty.")
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"Expected 3 or 4 channels, got shape {img.shape}")

    sharp = float(np.clip(sharpness, 0.0, 2.0))
    denoise = float(np.clip(denoise_strength, 0.0, 1.0))

    if sharp == 0.0 and denoise == 0.0:
        return img.copy()

    has_alpha = (img.shape[2] == 4)
    bgr = img[:, :, :3]
    is_uint8 = (img.dtype == np.uint8)

    res = bgr.copy()

    # 1. Edge-preserving bilateral filter (cleans JPEG ringing on walls)
    if denoise > 0.0:
        if is_uint8:
            sigma_c = int(10 + denoise * 40)
            sigma_s = int(3 + denoise * 4)
            denoised = cv2.bilateralFilter(res, d=5, sigmaColor=sigma_c, sigmaSpace=sigma_s)
            res = cv2.addWeighted(res, 1.0 - denoise, denoised, denoise, 0)
        else:
            sigma_c = 0.04 + denoise * 0.16
            sigma_s = 3.0 + denoise * 4.0
            denoised = cv2.bilateralFilter(res, d=5, sigmaColor=sigma_c, sigmaSpace=sigma_s)
            res = (1.0 - denoise) * res + denoise * denoised

    # 2. Unsharp Masking for micro-contrast
    if sharp > 0.0:
        if is_uint8:
            blurred = cv2.GaussianBlur(res, (0, 0), sigmaX=sigma, sigmaY=sigma)
            res = cv2.addWeighted(res, 1.0 + sharp, blurred, -sharp, 0)
        else:
            blurred = cv2.GaussianBlur(res, (0, 0), sigmaX=sigma, sigmaY=sigma)
            high_freq = res - blurred
            res = np.clip(res + sharp * high_freq, 0.0, 1.0)

    if has_alpha:
        return np.concatenate([res, img[:, :, 3:4]], axis=2)
    return res
