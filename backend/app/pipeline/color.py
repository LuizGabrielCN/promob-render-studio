"""
Photometric White Balance, Color Temperature, and Exposure Calibration.
Vectorized OpenCV/NumPy implementation with Rec. 709 luminance preservation.
"""
from typing import Union
import cv2
import numpy as np


def apply_white_balance(
    img: np.ndarray,
    temperature: float = 0.0,
    tint: float = 0.0,
    exposure: float = 0.0,
    preserve_luminance: bool = True,
) -> np.ndarray:
    """
    Applies chromatic white balance, Kelvin color temperature shift, tint, and exposure.

    Args:
        img: np.ndarray of shape (H, W, 3) BGR or (H, W, 4) BGRA, dtype uint8 or float32.
        temperature: float in [-1.0, 1.0]. Default 0.0. Warm shift (+R, -B) if > 0.
        tint: float in [-1.0, 1.0]. Default 0.0. Magenta shift (-G) if > 0.
        exposure: float in [-2.0, 2.0]. Default 0.0. EV stops multiplier.
        preserve_luminance: bool. If True, decouples chromatic gains from scene brightness.

    Returns:
        Enhanced image with identical dtype and shape as input.
    """
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("Input image cannot be None or empty.")
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"Expected 3-channel BGR or 4-channel BGRA image, got shape {img.shape}")

    # Clamp parameters to valid ranges
    temp = float(np.clip(temperature, -1.0, 1.0))
    t = float(np.clip(tint, -1.0, 1.0))
    exp = float(np.clip(exposure, -2.0, 2.0))

    # Fast identity bypass
    if temp == 0.0 and t == 0.0 and exp == 0.0:
        return img.copy()

    has_alpha = (img.shape[2] == 4)
    bgr = img[:, :, :3]
    is_uint8 = (img.dtype == np.uint8)

    # Compute raw gains
    r_gain = 1.0 + temp * 0.28
    b_gain = 1.0 - temp * 0.28
    g_gain = 1.0 - t * 0.12

    raw_gains = np.array([b_gain, g_gain, r_gain], dtype=np.float32)  # BGR order

    if preserve_luminance:
        # Rec. 709 luminance weights: 0.0722*B + 0.7152*G + 0.2126*R
        lum_weights = np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)
        norm_factor = float(np.sum(raw_gains * lum_weights))
        gains = (raw_gains / norm_factor) * (2.0 ** exp)
    else:
        gains = raw_gains * (2.0 ** exp)

    if is_uint8:
        # High-performance 3-channel 256-entry Look-Up Table
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        idx = np.arange(256, dtype=np.float32)
        lut[:, 0, 0] = np.clip(np.round(idx * gains[0]), 0, 255).astype(np.uint8)
        lut[:, 0, 1] = np.clip(np.round(idx * gains[1]), 0, 255).astype(np.uint8)
        lut[:, 0, 2] = np.clip(np.round(idx * gains[2]), 0, 255).astype(np.uint8)
        out_bgr = cv2.LUT(bgr, lut)
    else:
        out_bgr = np.clip(bgr.astype(np.float32) * gains, 0.0, 1.0)

    if has_alpha:
        return np.concatenate([out_bgr, img[:, :, 3:4]], axis=2)
    return out_bgr


# Required alias
apply_color_correction = apply_white_balance
