"""
Deterministic photographic finish for the generative ("Realismo IA") branch.

Only global, content-independent camera cues are added: a soft filmic shoulder in L*, slightly lifted
blacks, a gentle bloom around the brightest areas, a light vignette and luminance film grain. Nothing here
can invent an object, change a material or move geometry: a*/b* are never edited (bloom is added in L*
only) and the mean L* of the frame is restored after the tone stage.

Grain must be applied AFTER super-resolution (Real-ESRGAN wipes it out), so the two stages are separate:
    apply_photo_finish(img, amount, grain=False)   # before LED repaint / upscaling
    apply_photo_finish(img, amount, tone=False)    # after upscaling (grain only)
"""
from typing import Optional

import cv2
import numpy as np

DEFAULT_GRAIN_SEED = 1234


def _split(img: np.ndarray):
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("Input image cannot be None or empty.")
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"Expected 3 or 4 channels, got shape {img.shape}")
    is_u8 = img.dtype == np.uint8
    bgr = img[:, :, :3]
    bgr_u8 = bgr if is_u8 else np.clip(np.round(bgr * 255.0), 0, 255).astype(np.uint8)
    return bgr_u8, is_u8


def _merge(img: np.ndarray, out_u8: np.ndarray, is_u8: bool) -> np.ndarray:
    out = out_u8 if is_u8 else out_u8.astype(np.float32) * (1.0 / 255.0)
    if img.shape[2] == 4:
        full = np.empty(img.shape, dtype=img.dtype)
        full[:, :, :3] = out
        full[:, :, 3] = img[:, :, 3]
        return full
    return out


def _filmic_l(l01: np.ndarray, amount: float) -> np.ndarray:
    """Soft shoulder above the knee + slight black lift, blended by `amount` (L in [0, 1])."""
    knee = 0.72
    span = 1.0 - knee
    shoulder = np.where(l01 > knee, knee + span * np.tanh((l01 - knee) / span), l01)
    curved = 0.015 + shoulder * (1.0 - 0.015)
    return l01 + amount * (curved - l01)


def _tone_stage(bgr_u8: np.ndarray, amount: float) -> np.ndarray:
    h, w = bgr_u8.shape[:2]
    s = min(h, w)
    lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2Lab).astype(np.float32)
    l01 = lab[:, :, 0] / 255.0
    mean_in = float(l01.mean())

    l_new = _filmic_l(l01, amount)

    # Bloom: blurred excess above the 90th percentile, screened onto L (strength 0.20*amount -> 10% at 0.5)
    thr = float(np.percentile(l01, 90.0))
    if thr < 0.999:
        hi = np.clip((l01 - thr) / max(1.0 - thr, 1e-3), 0.0, 1.0)
        sigma = max(2.0, s * 0.02)
        glow = cv2.GaussianBlur(hi * l01, (0, 0), sigma)
        k = 0.20 * amount
        l_new = 1.0 - (1.0 - l_new) * (1.0 - k * glow)

    # Vignette: up to 14%*amount darker in the corners (7% at the default 0.5)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - (w - 1) / 2.0) / (w / 2.0)) ** 2 + ((yy - (h - 1) / 2.0) / (h / 2.0)) ** 2
    vig = 1.0 - 0.14 * amount * np.clip(r2 / 2.0, 0.0, 1.0) ** 1.5
    l_new = l_new * vig

    # Exposure is not the finish's job: restore the frame's mean L*
    l_new = l_new + (mean_in - float(l_new.mean()))
    lab[:, :, 0] = np.clip(l_new * 255.0, 0.0, 255.0)
    return cv2.cvtColor(np.round(lab).astype(np.uint8), cv2.COLOR_Lab2BGR)


def _grain_stage(bgr_u8: np.ndarray, amount: float, seed: int) -> np.ndarray:
    h, w = bgr_u8.shape[:2]
    rng = np.random.default_rng(int(seed))
    # Work in row bands: full-frame float RGB temporaries cost hundreds of MB after 4x upscaling.
    # Carry the unfiltered overlap so the noise remains continuous across band boundaries.
    radius = 2
    kernel = cv2.getGaussianKernel(2 * radius + 1, 0.6)
    noise_std = float(np.sum(kernel * kernel))  # standard deviation after the separable 2D filter
    pending = np.empty((0, w), np.float32)
    noise_start = noise_end = 0
    out = bgr_u8.copy()
    for y0 in range(0, h, 256):
        y1 = min(y0 + 256, h)
        lo, hi = max(0, y0 - radius), min(h, y1 + radius)
        pending = np.concatenate((pending[lo - noise_start:],
                                  rng.standard_normal((hi - noise_end, w), dtype=np.float32)))
        noise_start, noise_end = lo, hi
        noise = cv2.GaussianBlur(pending, (5, 5), 0.6)[y0 - lo:y1 - lo] / noise_std
        band = bgr_u8[y0:y1]
        l01 = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        # Subtle grain; dark lacquer gets at most 8% of the mid-tone amplitude.
        hl = np.clip((l01 - 0.6) / 0.35, 0.0, 1.0)
        sh = np.clip((l01 - 0.12) / 0.43, 0.0, 1.0)
        amp = 255.0 * 0.0045 * amount * (1.0 - 0.5 * hl) * (0.08 + 0.92 * sh)
        values = band.astype(np.float32) + (amp * noise)[:, :, None]
        np.rint(values, out=values)
        np.clip(values, 0, 255, out=values)
        out[y0:y1] = values.astype(np.uint8)
    return out


def apply_photo_finish(
    img: np.ndarray,
    amount: float,
    tone: bool = True,
    grain: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Photographic finish with strength `amount` in [0, 1] (0 = identity). Deterministic: the grain uses a
    fixed seed (`DEFAULT_GRAIN_SEED` unless given). Keeps shape, dtype and alpha.
    """
    a = float(amount)
    if not np.isfinite(a):
        raise ValueError(f"amount must be finite, got {amount}")
    a = float(np.clip(a, 0.0, 1.0))
    bgr_u8, is_u8 = _split(img)
    if a <= 0.0 or not (tone or grain):
        return img.copy()
    out = bgr_u8
    if tone:
        out = _tone_stage(out, a)
    if grain:
        out = _grain_stage(out, a, DEFAULT_GRAIN_SEED if seed is None else seed)
    return _merge(img, out, is_u8)
