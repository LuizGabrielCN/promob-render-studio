"""
Photometric PBR Lighting: Soft Global Illumination (GI) Bloom & Selective Specular Highlights.
Pure-OpenCV SIMD + 4x downsampled Gaussian diffusion with warm architectural bounce tint
and highlight-safe screen modulation for real-time (<10ms) preview and full render stages.
"""
import cv2
import numpy as np

_IDX = np.arange(256, dtype=np.float32)

# Smooth emitter curve above 50% luminance to avoid hard thresholding/banding
_EMIT_LUT = np.clip(
    np.round(255.0 * (np.clip((_IDX / 255.0 - 0.50) * 2.0, 0.0, 1.0) ** 1.5)),
    0,
    255,
).astype(np.uint8)

# Mid-to-high bell gate peaking near Y~0.68, zeroing shadows (<=0.35) and peak whites (1.0)
_SPEC_GATE_LUT = np.clip(
    np.round(255.0 * np.clip((_IDX / 255.0 - 0.35) / 0.35, 0.0, 1.0) * (1.0 - _IDX / 255.0)),
    0,
    255,
).astype(np.uint8)

# Warm architectural bounce LUT in BGR order: B=0.90, G=0.98, R=1.10
_WARM_TINT_LUT = np.empty((256, 1, 3), dtype=np.uint8)
_WARM_TINT_LUT[:, 0, 0] = np.clip(np.round(_IDX * 0.90), 0, 255).astype(np.uint8)
_WARM_TINT_LUT[:, 0, 1] = np.clip(np.round(_IDX * 0.98), 0, 255).astype(np.uint8)
_WARM_TINT_LUT[:, 0, 2] = np.clip(np.round(_IDX * 1.10), 0, 255).astype(np.uint8)


def apply_pbr_lighting(
    img: np.ndarray,
    strength: float = 0.25,
) -> np.ndarray:
    """
    Applies Soft GI Bloom (indirect warm architectural light bounce) and selective PBR
    specular highlight enhancement without clipping bright whites: distant shadows and
    darks remain unlifted while surfaces adjacent to bright emitters receive soft warm GI bounce.

    Args:
        img: np.ndarray of shape (H, W, 3) BGR or (H, W, 4) BGRA, uint8 or float32.
        strength: float in [0.0, 1.0]. Lighting and specular bloom strength (default 0.25).

    Returns:
        Image with soft GI bloom, selective specular highlights, and preserved alpha channel.
    """
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("Input image cannot be None or empty.")
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"Expected 3 or 4 channels, got shape {img.shape}")
    if img.dtype not in (np.uint8, np.float32):
        raise ValueError(f"Unsupported image dtype {img.dtype}; expected uint8 or float32.")

    strength_f = float(strength)
    if not np.isfinite(strength_f):
        raise ValueError(f"strength must be finite, got {strength}")
    s = float(np.clip(strength_f, 0.0, 1.0))
    if s == 0.0:
        return img.copy()

    has_alpha = (img.shape[2] == 4)
    bgr = img[:, :, :3]
    is_uint8 = (img.dtype == np.uint8)

    if is_uint8:
        bgr_u8 = bgr
        bgr_f = None
    else:
        b_min = float(bgr.min())
        b_max = float(bgr.max())
        if 0.0 <= b_min and b_max <= 1.0:
            bgr_f = bgr
        else:
            bgr_f = np.nan_to_num(bgr, nan=0.0, posinf=1.0, neginf=0.0)
            np.clip(bgr_f, 0.0, 1.0, out=bgr_f)
        bgr_u8 = cv2.convertScaleAbs(bgr_f, alpha=255.0)

    gray = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Soft GI Bloom (4x downsampled emitter LUT + Gaussian blur + highlight-safe Screen add)
    if h >= 8 and w >= 8:
        small_gray = cv2.resize(
            gray,
            (max(1, w // 4), max(1, h // 4)),
            interpolation=cv2.INTER_AREA,
        )
        emit_small = cv2.LUT(small_gray, _EMIT_LUT)
        gi_small = cv2.GaussianBlur(emit_small, (15, 15), sigmaX=4.5, sigmaY=4.5)
        gi_full = cv2.resize(gi_small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        emit_full = cv2.LUT(gray, _EMIT_LUT)
        gi_full = cv2.GaussianBlur(emit_full, (5, 5), sigmaX=1.5, sigmaY=1.5)

    gi_add = cv2.multiply(gi_full, cv2.bitwise_not(gray), scale=(0.35 * s) / 255.0)

    # 2. Selective Specular Highlight Enhancement (local peak delta * bell gate)
    local_mean = cv2.boxFilter(gray, -1, (5, 5))
    sd = cv2.subtract(gray, local_mean)
    spec = cv2.multiply(sd, cv2.LUT(gray, _SPEC_GATE_LUT), scale=(2.2 * s) / 255.0)

    # 3. Warm Architectural Bounce Blend
    boost = cv2.add(gi_add, spec)
    boost_3 = cv2.LUT(cv2.cvtColor(boost, cv2.COLOR_GRAY2BGR), _WARM_TINT_LUT)

    if is_uint8:
        out_bgr = cv2.add(bgr_u8, boost_3)
    else:
        out_bgr = boost_3.astype(np.float32)
        out_bgr *= np.float32(1.0 / 255.0)
        cv2.add(bgr_f, out_bgr, dst=out_bgr)
        np.clip(out_bgr, 0.0, 1.0, out=out_bgr)

    if has_alpha:
        out = np.empty(img.shape, dtype=img.dtype)
        out[:, :, :3] = out_bgr
        out[:, :, 3] = img[:, :, 3]
        return out
    return out_bgr
