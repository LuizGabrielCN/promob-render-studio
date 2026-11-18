"""
Multi-Scale Screen-Space Ambient Occlusion (SSAO) via Pure-OpenCV SIMD & Gaussian Pyramids.
Combines fine crevices (radius=3), medium joint occlusion (radius=11 via 2x pyrDown),
and diffuse corner/depth shading (radius=35 via 4x Gaussian pyramid) with highlight protection.
"""
import cv2
import numpy as np

# Precomputed 8-bit LUTs eliminate per-call float32 elementwise overhead on full-res frames.
_IDX = np.arange(256, dtype=np.float32)
_AO_FINE_W_LUT = np.clip(np.round(np.minimum(255.0, _IDX * 4.0) * 0.55), 0, 255).astype(np.uint8)
_AO_MED_W_LUT = np.clip(np.round(np.minimum(255.0, _IDX * 3.5) * 0.30), 0, 255).astype(np.uint8)
_AO_DEPTH_W_LUT = np.clip(np.round(np.minimum(255.0, _IDX * 3.0) * 0.28), 0, 255).astype(np.uint8)
_HL_PROT_LUT = np.clip(np.round(255.0 * (1.0 - (_IDX / 255.0) ** 3)), 0, 255).astype(np.uint8)

_K_FINE_DEFAULT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
_K_MED_DEFAULT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))


def apply_ambient_occlusion(
    img: np.ndarray,
    strength: float = 0.15,
    radius: int = 3,
) -> np.ndarray:
    """
    Emulates multi-scale contact shadows and architectural depth shading in cabinet joints,
    kickboards, door seams, and room/niche corners.

    Args:
        img: np.ndarray of shape (H, W, 3) BGR or (H, W, 4) BGRA, uint8 or float32.
        strength: float in [0.0, 1.0]. Crease deepening strength (default 0.15).
        radius: int. Fine structuring element radius in pixels (default 3, kernel 7x7).

    Returns:
        Image with multi-scale darkened contact creases/corners and preserved highlights.
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
    ao_str = float(np.clip(strength_f, 0.0, 1.0))
    if ao_str == 0.0:
        return img.copy()

    has_alpha = (img.shape[2] == 4)
    bgr = img[:, :, :3]
    is_uint8 = (img.dtype == np.uint8)

    if is_uint8:
        bgr_f = None
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        b_min = float(bgr.min())
        b_max = float(bgr.max())
        if 0.0 <= b_min and b_max <= 1.0:
            bgr_f = bgr
        else:
            bgr_f = np.nan_to_num(bgr, nan=0.0, posinf=1.0, neginf=0.0)
            np.clip(bgr_f, 0.0, 1.0, out=bgr_f)
        gray_u8_bgr = cv2.convertScaleAbs(bgr_f, alpha=255.0)
        gray = cv2.cvtColor(gray_u8_bgr, cv2.COLOR_BGR2GRAY)

    # Exact uniform-surface invariance check
    g_min, g_max, _, _ = cv2.minMaxLoc(gray)
    if g_min == g_max:
        if is_uint8 or bgr_f is bgr:
            return img.copy()
        if has_alpha:
            out = np.empty(img.shape, dtype=img.dtype)
            out[:, :, :3] = bgr_f
            out[:, :, 3] = img[:, :, 3]
            return out
        return bgr_f

    h, w = gray.shape

    # Scale 1: Fine crevices (default radius=3 -> 7x7 ellipse)
    r = max(1, int(radius))
    k_fine = _K_FINE_DEFAULT if r == 3 else cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    ao_tot = cv2.LUT(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_fine), _AO_FINE_W_LUT)

    # Scale 2: Medium joint occlusion (effective radius~11 via 2x pyrDown + 11x11 ellipse)
    if h >= 4 and w >= 4:
        g2 = cv2.pyrDown(gray)
        ao2 = cv2.LUT(cv2.morphologyEx(g2, cv2.MORPH_BLACKHAT, _K_MED_DEFAULT), _AO_MED_W_LUT)

        # Scale 3: Diffuse corner / architectural depth shading (effective radius~35 via 4x Gaussian pyramid)
        if h >= 8 and w >= 8:
            g4 = cv2.pyrDown(g2)
            g4_blur = cv2.GaussianBlur(g4, (19, 19), sigmaX=6.5, sigmaY=6.5)
            depth_cavity = cv2.subtract(g4_blur, g4)
            ao3 = cv2.LUT(depth_cavity, _AO_DEPTH_W_LUT)
            ao2 = cv2.add(ao2, cv2.resize(ao3, (g2.shape[1], g2.shape[0]), interpolation=cv2.INTER_LINEAR))

        ao_tot = cv2.add(ao_tot, cv2.resize(ao2, (w, h), interpolation=cv2.INTER_LINEAR))

    if cv2.countNonZero(ao_tot) == 0:
        if is_uint8 or bgr_f is bgr:
            return img.copy()
        if has_alpha:
            out = np.empty(img.shape, dtype=img.dtype)
            out[:, :, :3] = bgr_f
            out[:, :, 3] = img[:, :, 3]
            return out
        return bgr_f

    # Highlight protection: (1 - Y^3) prevents darkening lamps/specular reflections
    hl_prot = cv2.LUT(gray, _HL_PROT_LUT)
    ao_prot = cv2.multiply(ao_tot, hl_prot, scale=ao_str / 255.0)
    dark_u8 = cv2.bitwise_not(ao_prot)
    dark_3 = cv2.cvtColor(dark_u8, cv2.COLOR_GRAY2BGR)

    if is_uint8:
        out_bgr = cv2.multiply(bgr, dark_3, scale=1.0 / 255.0)
    else:
        out_bgr = cv2.multiply(bgr_f, dark_3, scale=1.0 / 255.0, dtype=cv2.CV_32F)
        np.clip(out_bgr, 0.0, 1.0, out=out_bgr)

    if has_alpha:
        out = np.empty(img.shape, dtype=img.dtype)
        out[:, :, :3] = out_bgr
        out[:, :, 3] = img[:, :, 3]
        return out
    return out_bgr
