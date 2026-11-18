"""
Scene Depth Estimation & Physically Plausible 2D Illumination Light-Map Synthesis.

Produces `SceneLightingAnalysis` (depth map, detected ceiling spots / wall sconces / pendants,
daylight windows, unlit fireboxes, protection masks, and a 2D Illumination Guide Map) using
`depth-anything/Depth-Anything-V2-Small-hf` (`local_files_only=True`) with strict ephemeral
VRAM cleanup (`HardwareGuardrails.release_memory()`) and an offline heuristic fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.hardware import HardwareGuardrails
from app.pipeline import led, objects

logger = logging.getLogger(__name__)

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_ENV_VAR = "PROMOB_DEPTH_ESTIMATION"
OBJECT_ENV_VAR = "PROMOB_OBJECT_DETECTION"
MAX_DEPTH_LONG_SIDE = 768

# Lighting palette in BGR order (0..1 floats during synthesis)
_AMBIENT_BASE_BGR = np.array([108.0, 114.0, 122.0], dtype=np.float32) / 255.0
_SPOT_WARM_BGR = np.array([145.0, 205.0, 255.0], dtype=np.float32) / 255.0
_SCONCE_WARM_BGR = np.array([155.0, 210.0, 255.0], dtype=np.float32) / 255.0
_FIRE_CORE_BGR = np.array([55.0, 145.0, 255.0], dtype=np.float32) / 255.0
_FIRE_SPILL_BGR = np.array([85.0, 172.0, 255.0], dtype=np.float32) / 255.0
_LED_WARM_BGR = np.array([165.0, 212.0, 255.0], dtype=np.float32) / 255.0
_DAYLIGHT_BGR = np.array([255.0, 242.0, 224.0], dtype=np.float32) / 255.0


@dataclass
class SceneLightingAnalysis:
    depth_map: np.ndarray              # float32 [H, W] normalized 0..1 (far..near, inverse depth)
    illumination_map_bgr: np.ndarray   # uint8 [H, W, 3] synthesized warm/daylight 2D lighting target
    spot_mask: np.ndarray              # uint8 [H, W] 0/255 detected ceiling spots / luminaires
    fireplace_boxes: list[tuple[int, int, int, int]]  # (x1, y1, x2, y2) unlit firebox/hearth regions
    window_mask: np.ndarray            # uint8 [H, W] 0/255 daylight openings
    protection_mask: np.ndarray        # float32 [H, W] 0..1 protected objects + TV/screens + OCR text/logos
    text_logo_mask: np.ndarray         # float32 [H, W] 0..1 strict OCR/TV/sign mask (highest preservation weight)


def _prepare_bgr_u8(img: np.ndarray) -> np.ndarray:
    """Validates BGR/BGRA uint8/float32 input and returns sanitized uint8 BGR."""
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("Input image cannot be None or empty.")
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"Expected 3-channel BGR or 4-channel BGRA image, got shape {getattr(img, 'shape', None)}")
    if img.dtype not in (np.uint8, np.float32):
        raise ValueError(f"Unsupported image dtype {img.dtype}; expected uint8 or float32.")

    bgr = img[:, :, :3]
    if bgr.dtype == np.uint8:
        return bgr
    sanitized = np.nan_to_num(bgr, nan=0.0, posinf=1.0, neginf=0.0)
    sanitized = np.clip(sanitized, 0.0, 1.0)
    return cv2.convertScaleAbs(sanitized, alpha=255.0)


def _heuristic_depth_map(bgr_u8: np.ndarray) -> np.ndarray:
    """Deterministic CPU fallback inverse depth map (0=far, 1=near) in float32."""
    h, w = bgr_u8.shape[:2]
    if int(bgr_u8.min()) == int(bgr_u8.max()):
        return np.full((h, w), 0.5, dtype=np.float32)

    y_grad = np.linspace(0.25, 0.85, h, dtype=np.float32)[:, None]
    y_field = np.broadcast_to(y_grad, (h, w))

    gray = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    sigma = max(1.0, min(h, w) * 0.03)
    luma_smooth = cv2.GaussianBlur(gray, (0, 0), sigma)
    depth = 0.85 * y_field + 0.15 * luma_smooth
    return np.clip(depth, 0.0, 1.0).astype(np.float32)


def _load_depth_model():
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID, local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID, local_files_only=True)
    model = model.to(device).eval()
    return processor, model, device, torch


def estimate_depth_map(bgr_u8: np.ndarray, allow_model: bool = True) -> Tuple[np.ndarray, str]:
    """
    Estimates normalized inverse depth [0, 1] (0=far, 1=near) using Depth-Anything-V2-Small-hf.
    Always releases the model and CUDA cache in `finally` before returning.
    """
    if not allow_model or os.environ.get(DEPTH_ENV_VAR, "1").strip() == "0":
        return _heuristic_depth_map(bgr_u8), "heuristic"

    processor = model = inputs = outputs = post = None
    attempted = False
    try:
        from PIL import Image

        h, w = bgr_u8.shape[:2]
        attempted = True
        processor, model, device, torch = _load_depth_model()

        max_dim = max(h, w)
        if max_dim > MAX_DEPTH_LONG_SIDE:
            scale = MAX_DEPTH_LONG_SIDE / float(max_dim)
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            work_bgr = cv2.resize(bgr_u8, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            work_bgr = bgr_u8

        pil_rgb = Image.fromarray(cv2.cvtColor(work_bgr, cv2.COLOR_BGR2RGB))
        inputs = processor(images=pil_rgb, return_tensors="pt")
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)

        post = processor.post_process_depth_estimation(outputs, target_sizes=[(h, w)])
        raw = post[0]["predicted_depth"].detach().float().cpu().numpy()
        if raw.shape != (h, w):
            raw = cv2.resize(raw.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)

        finite = np.isfinite(raw)
        if not finite.any():
            return _heuristic_depth_map(bgr_u8), "heuristic"
        p_lo, p_hi = np.percentile(raw[finite], [0.5, 99.5])
        if not np.isfinite(p_lo) or not np.isfinite(p_hi) or (p_hi - p_lo) < 1e-6:
            return np.full((h, w), 0.5, dtype=np.float32), "depth_anything_v2"
        depth = np.clip((raw - p_lo) / (p_hi - p_lo), 0.0, 1.0).astype(np.float32)
        return depth, "depth_anything_v2"
    except Exception as exc:
        logger.info("Depth-Anything-V2 indisponível (%s); usando profundidade heurística.", exc)
        return _heuristic_depth_map(bgr_u8), "heuristic"
    finally:
        post = None
        outputs = None
        inputs = None
        model = None
        processor = None
        if attempted:
            HardwareGuardrails.release_memory()


def detect_window_mask(
    analysis_bgr: np.ndarray,
    depth_map: np.ndarray,
    exclude_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Detects exterior daylight windows/openings as uint8 {0, 255}, excluding TVs/screens/text/LEDs.
    """
    h, w = analysis_bgr.shape[:2]
    s = min(h, w)
    if h < 4 or w < 4:
        return np.zeros((h, w), dtype=np.uint8)

    lab = cv2.cvtColor(analysis_bgr, cv2.COLOR_BGR2LAB)
    L_lab = lab[:, :, 0].astype(np.float32)
    B_ch = analysis_bgr[:, :, 0].astype(np.float32)
    G_ch = analysis_bgr[:, :, 1].astype(np.float32)
    R_ch = analysis_bgr[:, :, 2].astype(np.float32)

    foliage = (G_ch - R_ch >= 8.0) & (G_ch >= 80.0) & (L_lab >= 95.0)
    sky = (B_ch - R_ch >= 10.0) & (L_lab >= 150.0)
    top_ceiling_cutoff = int(0.12 * h)
    y_idx = np.arange(h, dtype=np.int32)[:, None]

    # Cue 1: bright far exterior daylight opening (L >= 222 avoids interior white/grey walls) or exterior foliage/sky
    cue1 = (((L_lab >= 222.0) & (depth_map <= 0.35)) | foliage | sky)
    cue1 &= (y_idx >= top_ceiling_cutoff) | foliage | sky

    # Cue 2: recessed bright window pane relative to surrounding frame depth
    ring_sigma = max(6.0, 0.06 * s)
    local_ring_depth = cv2.GaussianBlur(depth_map, (0, 0), ring_sigma)
    cue2 = (L_lab >= 218.0) & (depth_map <= (local_ring_depth - 0.035)) & (y_idx >= int(0.12 * h))

    raw = (cue1 | cue2).astype(np.uint8) * 255
    if exclude_mask is not None and exclude_mask.shape[:2] == (h, w):
        raw[exclude_mask > 0] = 0

    if not raw.any():
        return raw

    k_close = max(3, int(round(0.015 * s)) | 1)
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((k_close, k_close), dtype=np.uint8))
    if exclude_mask is not None and exclude_mask.shape[:2] == (h, w):
        closed[exclude_mask > 0] = 0

    min_area = max(64, int(0.008 * h * w))
    max_area = int(0.30 * h * w)
    out = np.zeros((h, w), dtype=np.uint8)
    n_lbl, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, 8)
    for i in range(1, n_lbl):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        cy = float(centroids[i, 1])
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if cy <= 0.15 * h or aspect > 6.0:
            continue
        out[labels == i] = 255
    return out


def _detect_fireplace_boxes_heuristic(
    analysis_bgr: np.ndarray,
    depth_map: np.ndarray,
) -> List[Tuple[int, int, int, int]]:
    """Fallback heuristic detector for unlit rectangular firebox niches."""
    h, w = analysis_bgr.shape[:2]
    if h < 24 or w < 24:
        return []

    lab = cv2.cvtColor(analysis_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    a_ch = lab[:, :, 1]
    b_ch = lab[:, :, 2]

    warm_tan = ((a_ch > 125.0) & (b_ch > 132.0) & (L < 165.0)).astype(np.uint8) * 255
    warm_tan[: int(0.35 * h), :] = 0
    warm_tan[int(0.92 * h) :, :] = 0
    if not warm_tan.any():
        return []

    k = max(3, int(round(0.01 * min(h, w))) | 1)
    closed = cv2.morphologyEx(warm_tan, cv2.MORPH_CLOSE, np.ones((k, k), dtype=np.uint8))
    n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    boxes: List[Tuple[int, int, int, int]] = []
    min_area = 0.008 * h * w
    max_area = 0.15 * h * w
    for i in range(1, n_lbl):
        area = float(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        box_area = float(max(1, bw * bh))
        if not (min_area <= box_area <= max_area):
            continue
        aspect = bw / float(max(1, bh))
        solidity = area / box_area
        if not (0.7 <= aspect <= 3.0 and solidity >= 0.70):
            continue

        warm_frac = float(((a_ch[y : y + bh, x : x + bw] > 125.0) & (b_ch[y : y + bh, x : x + bw] > 132.0)).mean())
        if warm_frac < 0.12:
            continue

        pad = max(6, int(round(0.25 * max(bw, bh))))
        ox0, oy0 = max(0, x - pad), max(0, y - pad)
        ox1, oy1 = min(w, x + bw + pad), min(h, y + bh + pad)
        inner_L = float(L[y : y + bh, x : x + bw].mean())
        outer_L_sum = float(L[oy0:oy1, ox0:ox1].sum()) - float(L[y : y + bh, x : x + bw].sum())
        outer_n = max(1, (oy1 - oy0) * (ox1 - ox0) - bw * bh)
        ring_L = outer_L_sum / outer_n

        inner_D = float(depth_map[y : y + bh, x : x + bw].mean())
        outer_D_sum = float(depth_map[oy0:oy1, ox0:ox1].sum()) - float(depth_map[y : y + bh, x : x + bw].sum())
        ring_D = outer_D_sum / outer_n

        if inner_L < ring_L - 12.0 or inner_D < ring_D - 0.015:
            boxes.append((x, y, x + bw, y + bh))
    return boxes


def _detect_spot_components(
    analysis_bgr: np.ndarray,
    window_mask: Optional[np.ndarray] = None,
    led_mask: Optional[np.ndarray] = None,
    exclude_mask: Optional[np.ndarray] = None,
    depth_map: Optional[np.ndarray] = None,
    luminaire_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Detects ceiling recessed spots, vertical wall sconces, and hanging pendant bulbs.
    Rejects dark-surround mirror reflections (`s06`) and exterior daylight/TV glare.
    """
    h, w = analysis_bgr.shape[:2]
    s = min(h, w)
    if h < 4 or w < 4:
        return np.zeros((h, w), dtype=np.uint8), []

    lab = cv2.cvtColor(analysis_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(analysis_bgr, cv2.COLOR_BGR2HSV)
    L = lab[:, :, 0].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)

    L_bg = cv2.GaussianBlur(L, (0, 0), max(3.0, 0.022 * s))
    excess = L - L_bg

    cand = (((V >= 215.0) & (excess >= 16.0)) | ((V >= 238.0) & (excess >= 10.0))).astype(np.uint8)
    if window_mask is not None and window_mask.shape[:2] == (h, w):
        cand[window_mask > 0] = 0
    if led_mask is not None and led_mask.shape[:2] == (h, w):
        cand[led_mask > 0] = 0
    if exclude_mask is not None and exclude_mask.shape[:2] == (h, w):
        cand[exclude_mask > 0.5] = 0

    if not cand.any():
        return np.zeros((h, w), dtype=np.uint8), []

    gy_s = None
    if depth_map is not None and depth_map.shape[:2] == (h, w):
        D_s = cv2.GaussianBlur(depth_map, (0, 0), max(1.0, 0.004 * s))
        gy_s = cv2.Sobel(D_s, cv2.CV_32F, 0, 1, ksize=3) * (s / 8.0)

    min_area = max(4, int((0.0015 * s) ** 2))
    max_area = max(min_area + 1, int(0.004 * h * w))
    r_ring = max(4, int(round(0.012 * s)))
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_ring + 1, 2 * r_ring + 1))

    out_mask = np.zeros((h, w), dtype=np.uint8)
    components: List[Dict[str, Any]] = []
    n_lbl, labels, stats, centroids = cv2.connectedComponentsWithStats(cand, 8)
    for i in range(1, n_lbl):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = float(centroids[i, 0]), float(centroids[i, 1])

        px0, py0 = max(0, x - r_ring), max(0, y - r_ring)
        px1, py1 = min(w, x + bw + r_ring), min(h, y + bh + r_ring)
        comp_roi = (labels[py0:py1, px0:px1] == i).astype(np.uint8)
        dil_roi = cv2.dilate(comp_roi, ring_kernel)
        ring_roi = (dil_roi > 0) & (comp_roi == 0)
        if not ring_roi.any():
            continue

        ring_mean_V = float(V[py0:py1, px0:px1][ring_roi].mean())
        ring_mean_S = float(S[py0:py1, px0:px1][ring_roi].mean())
        comp_excess = float(excess[py0:py1, px0:px1][comp_roi > 0].mean())

        aspect = max(bw, bh) / float(max(1, min(bw, bh)))
        solidity = area / float(max(1, bw * bh))

        is_sconce = (
            0.15 * h <= cy <= 0.65 * h
            and bh >= 2.0 * bw
            and bw <= max(8.0, 0.03 * w)
            and bh <= 0.25 * h
            and ring_mean_V >= 120.0
            and comp_excess >= 13.0
        )

        is_ceiling_or_pendant = (
            cy < 0.48 * h
            and aspect <= 5.5
            and solidity >= 0.45
            and ring_mean_V >= 125.0
            and ring_mean_S < 85.0
        )

        if not (is_sconce or is_ceiling_or_pendant):
            continue

        kind = "sconce_pendant" if is_sconce else "ceiling_spot"
        if kind == "ceiling_spot" and cy >= 0.22 * h:
            in_luminaire = False
            if luminaire_boxes:
                for lx0, ly0, lx1, ly1 in luminaire_boxes:
                    if lx0 <= cx <= lx1 and ly0 <= cy <= ly1:
                        in_luminaire = True
                        break
            above_is_wall = False
            if gy_s is not None:
                ay0 = max(0, y - max(6, bh))
                ay1 = max(ay0 + 1, y)
                ax0, ax1 = max(0, x - 2), min(w, x + bw + 2)
                mean_above_gy = float(gy_s[ay0:ay1, ax0:ax1].mean())
                above_is_wall = abs(mean_above_gy) <= 0.50
            if in_luminaire or (above_is_wall and cy >= 0.24 * h):
                kind = "sconce_pendant"

        out_mask[labels == i] = 255
        components.append(
            {
                "kind": kind,
                "cx": cx,
                "cy": cy,
                "bbox": (x, y, x + bw, y + bh),
            }
        )
    return out_mask, components


def detect_spot_mask(
    analysis_bgr: np.ndarray,
    window_mask: Optional[np.ndarray] = None,
    led_mask: Optional[np.ndarray] = None,
    exclude_mask: Optional[np.ndarray] = None,
    depth_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Public detector returning uint8 {0, 255} mask of ceiling spots, wall sconces, and pendant bulbs."""
    mask, _ = _detect_spot_components(
        analysis_bgr,
        window_mask=window_mask,
        led_mask=led_mask,
        exclude_mask=exclude_mask,
        depth_map=depth_map,
    )
    return mask


def _screen_add(accum: np.ndarray, add: np.ndarray) -> np.ndarray:
    add_c = np.clip(add, 0.0, 1.0)
    return 1.0 - (1.0 - accum) * (1.0 - add_c)


def synthesize_illumination_map(
    analysis_bgr: np.ndarray,
    depth_map: np.ndarray,
    spot_components: List[Dict[str, Any]],
    fireplace_boxes: List[Tuple[int, int, int, int]],
    window_mask: np.ndarray,
    led_mask: np.ndarray,
) -> np.ndarray:
    """
    Synthesizes a smooth, physically grounded uint8 [H, W, 3] BGR illumination target map
    from depth-gated ceiling spot cones, sconce/pendant halos, fireplace glow, LED spill, and window daylight.
    """
    h, w = analysis_bgr.shape[:2]
    s = min(h, w)

    D_s = cv2.GaussianBlur(depth_map.astype(np.float32), (0, 0), max(1.0, 0.004 * s))
    gy_s = cv2.Sobel(D_s, cv2.CV_32F, 0, 1, ksize=3) * (s / 8.0)

    # Subtle depth-modulated ambient interior base
    ambient_scale = np.clip(0.90 + 0.15 * D_s, 0.85, 1.08)[:, :, None]
    light_accum = (_AMBIENT_BASE_BGR[None, None, :] * ambient_scale).astype(np.float32)

    # 1. Ceiling spots & wall sconces / pendants
    for comp in spot_components:
        cx, cy = float(comp["cx"]), float(comp["cy"])
        x0, y0, x1, y1 = comp["bbox"]
        r_spot = max(2.0, 0.5 * max(x1 - x0, y1 - y0))
        d_spot = float(np.median(D_s[y0:y1, x0:x1]))

        if comp["kind"] == "ceiling_spot":
            # Search downward ROI for adjacent vertical wall/cabinet
            sy0 = min(h, y1 + 1)
            sy1 = min(h, y1 + max(4, int(round(0.26 * h))))
            sx0 = max(0, int(round(cx - 0.07 * w)))
            sx1 = min(w, int(round(cx + 0.07 * w)))
            has_wall = False
            d_wall = d_spot
            x_cone = cx
            if sy1 > sy0 and sx1 > sx0:
                roi_gy = gy_s[sy0:sy1, sx0:sx1]
                roi_d = D_s[sy0:sy1, sx0:sx1]
                wall_cand = (np.abs(roi_gy) <= 0.75) & (np.abs(roi_d - d_spot) <= 0.18)
                if int(wall_cand.sum()) >= max(8, int(0.08 * wall_cand.size)):
                    has_wall = True
                    d_wall = float(np.median(roi_d[wall_cand]))
                    _, xs_rel = np.nonzero(wall_cand)
                    x_wall = float(np.median(xs_rel)) + sx0
                    x_cone = 0.5 * cx + 0.5 * x_wall

            if has_wall:
                cy0 = max(0, int(round(cy)))
                cy1 = min(h, y1 + max(8, int(round(0.42 * h))))
                cx0 = max(0, int(round(x_cone - 0.16 * w)))
                cx1 = min(w, int(round(x_cone + 0.16 * w)))
                if cy1 > cy0 and cx1 > cx0:
                    ys = np.arange(cy0, cy1, dtype=np.float32)[:, None]
                    xs = np.arange(cx0, cx1, dtype=np.float32)[None, :]
                    dy = np.maximum(0.0, ys - cy)
                    half_w = np.minimum(0.14 * w, r_spot * 1.4 + 0.38 * np.sqrt(dy * s * 0.22))
                    horiz = np.exp(-0.5 * ((xs - x_cone) / np.maximum(half_w, 1.0)) ** 2)
                    vert = np.exp(-dy / max(0.24 * h, 4.0)) * np.clip(dy / max(r_spot * 1.5, 1.0), 0.0, 1.0)
                    roi_d_cone = D_s[cy0:cy1, cx0:cx1]
                    roi_gy_cone = gy_s[cy0:cy1, cx0:cx1]
                    depth_gate = np.exp(-0.5 * ((roi_d_cone - d_wall) / 0.11) ** 2)
                    surf_gate = np.exp(-0.5 * (np.maximum(0.0, -roi_gy_cone - 0.55) / 0.50) ** 2)
                    cone = 0.68 * horiz * vert * depth_gate * surf_gate
                    light_accum[cy0:cy1, cx0:cx1] = _screen_add(
                        light_accum[cy0:cy1, cx0:cx1],
                        cone[:, :, None] * _SPOT_WARM_BGR[None, None, :],
                    )
            else:
                # Open-room spot: gentle floor warm fill only, no floating mid-air cone
                fy0 = min(h, int(round(0.60 * h)))
                fx0 = max(0, int(round(cx - 0.18 * w)))
                fx1 = min(w, int(round(cx + 0.18 * w)))
                if h > fy0 and fx1 > fx0:
                    ys = np.arange(fy0, h, dtype=np.float32)[:, None]
                    xs = np.arange(fx0, fx1, dtype=np.float32)[None, :]
                    floor_gauss = np.exp(-0.5 * ((xs - cx) / max(8.0, 0.09 * w)) ** 2)
                    floor_gate = ((gy_s[fy0:h, fx0:fx1] > 0.25) | (ys > 0.65 * h)).astype(np.float32)
                    fill = 0.10 * floor_gauss * floor_gate
                    light_accum[fy0:h, fx0:fx1] = _screen_add(
                        light_accum[fy0:h, fx0:fx1],
                        fill[:, :, None] * _SPOT_WARM_BGR[None, None, :],
                    )

            # Tight fixture halo at spot center
            halo_sigma = max(1.5, 1.8 * r_spot)
            hr = int(np.ceil(3.0 * halo_sigma))
            hy0, hy1 = max(0, int(round(cy)) - hr), min(h, int(round(cy)) + hr + 1)
            hx0, hx1 = max(0, int(round(cx)) - hr), min(w, int(round(cx)) + hr + 1)
            if hy1 > hy0 and hx1 > hx0:
                ys = np.arange(hy0, hy1, dtype=np.float32)[:, None]
                xs = np.arange(hx0, hx1, dtype=np.float32)[None, :]
                halo = 0.55 * np.exp(-0.5 * (((xs - cx) ** 2 + (ys - cy) ** 2) / (halo_sigma ** 2)))
                light_accum[hy0:hy1, hx0:hx1] = _screen_add(
                    light_accum[hy0:hy1, hx0:hx1],
                    halo[:, :, None] * _SPOT_WARM_BGR[None, None, :],
                )
        else:
            # Sconce or hanging pendant: local radial/elliptical glow gated to adjacent wall depth
            pad_r = max(4, int(round(4.0 * r_spot)))
            ry0, ry1 = max(0, y0 - pad_r), min(h, y1 + pad_r)
            rx0, rx1 = max(0, x0 - pad_r), min(w, x1 + pad_r)
            ring_d_roi = D_s[ry0:ry1, rx0:rx1]
            ring_gy_roi = gy_s[ry0:ry1, rx0:rx1]
            wall_ring = np.abs(ring_gy_roi) <= 0.75
            d_adj = float(np.median(ring_d_roi[wall_ring])) if wall_ring.any() else d_spot

            sig_x = max(4.0, 0.065 * s)
            sig_y = max(5.0, 0.095 * s)
            gx0 = max(0, int(round(cx - 3.0 * sig_x)))
            gx1 = min(w, int(round(cx + 3.0 * sig_x)))
            gy0 = max(0, int(round(cy - 3.0 * sig_y)))
            gy1 = min(h, int(round(cy + 3.0 * sig_y)))
            if gy1 > gy0 and gx1 > gx0:
                ys = np.arange(gy0, gy1, dtype=np.float32)[:, None]
                xs = np.arange(gx0, gx1, dtype=np.float32)[None, :]
                lobe = np.exp(-0.5 * (((xs - cx) / sig_x) ** 2 + ((ys - cy) / sig_y) ** 2))
                depth_gate = np.exp(-0.5 * ((D_s[gy0:gy1, gx0:gx1] - d_adj) / 0.16) ** 2)
                glow = 0.48 * lobe * depth_gate
                light_accum[gy0:gy1, gx0:gx1] = _screen_add(
                    light_accum[gy0:gy1, gx0:gx1],
                    glow[:, :, None] * _SCONCE_WARM_BGR[None, None, :],
                )

    # 2. Fireplaces: firebox warm core + hearth/floor bounce
    for (x1, y1, x2, y2) in fireplace_boxes:
        fw, fh = max(1, x2 - x1), max(1, y2 - y1)
        fx = 0.5 * (x1 + x2)
        d_fire = float(np.median(D_s[y1:y2, x1:x2]))

        # Firebox interior radial gradient
        fy_core = y1 + 0.62 * fh
        ys_in = np.arange(y1, y2, dtype=np.float32)[:, None]
        xs_in = np.arange(x1, x2, dtype=np.float32)[None, :]
        core = np.exp(
            -0.5 * (((xs_in - fx) / max(1.0, 0.42 * fw)) ** 2 + ((ys_in - fy_core) / max(1.0, 0.45 * fh)) ** 2)
        )
        # Feather box boundary smoothly
        dx_edge = np.minimum(xs_in - x1 + 0.5, x2 - 0.5 - xs_in)
        dy_edge = np.minimum(ys_in - y1 + 0.5, y2 - 0.5 - ys_in)
        feather = max(1.0, 0.08 * min(fw, fh))
        box_alpha = np.clip(np.minimum(dx_edge, dy_edge) / feather, 0.0, 1.0)
        core_add = (0.85 * core * box_alpha)[:, :, None] * _FIRE_CORE_BGR[None, None, :]
        light_accum[y1:y2, x1:x2] = _screen_add(light_accum[y1:y2, x1:x2], core_add)

        # Hearth & floor bounce downward/outward
        sig_hx = max(4.0, 0.68 * fw)
        sig_hy = max(4.0, 0.52 * fh)
        hy_center = y2 + 0.10 * fh
        by0 = max(0, int(round(y1 + 0.25 * fh)))
        by1 = min(h, int(round(hy_center + 3.0 * sig_hy)))
        bx0 = max(0, int(round(fx - 3.0 * sig_hx)))
        bx1 = min(w, int(round(fx + 3.0 * sig_hx)))
        if by1 > by0 and bx1 > bx0:
            ys_b = np.arange(by0, by1, dtype=np.float32)[:, None]
            xs_b = np.arange(bx0, bx1, dtype=np.float32)[None, :]
            lobe = np.exp(-0.5 * (((xs_b - fx) / sig_hx) ** 2 + ((ys_b - hy_center) / sig_hy) ** 2))
            depth_gate = np.clip((D_s[by0:by1, bx0:bx1] - (d_fire - 0.10)) / 0.08, 0.0, 1.0)
            bounce = (0.42 * lobe * depth_gate)[:, :, None] * _FIRE_SPILL_BGR[None, None, :]
            light_accum[by0:by1, bx0:bx1] = _screen_add(light_accum[by0:by1, bx0:bx1], bounce)

    # 3. Concealed linear LEDs: depth-gated directional warm spill
    if led_mask is not None and led_mask.any():
        bin_led = (led_mask > 0).astype(np.uint8)
        n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(bin_led, 8)
        near_sig = max(2.0, 0.006 * s)
        far_sig = max(8.0, 0.030 * s)
        pad = int(np.ceil(3.0 * far_sig))
        for i in range(1, n_lbl):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            cw = int(stats[i, cv2.CC_STAT_WIDTH])
            ch = int(stats[i, cv2.CC_STAT_HEIGHT])
            comp_full = labels == i
            d_strip = float(np.median(D_s[comp_full]))

            rx0, ry0 = max(0, x - pad), max(0, y - pad)
            rx1, ry1 = min(w, x + cw + pad), min(h, y + ch + pad)
            comp_roi = comp_full[ry0:ry1, rx0:rx1].astype(np.float32)
            near = cv2.GaussianBlur(comp_roi, (0, 0), near_sig)
            far = cv2.GaussianBlur(comp_roi, (0, 0), far_sig)
            if cw > ch:
                shift = int(round(0.70 * far_sig))
                if 0 < shift < (ry1 - ry0):
                    far = np.roll(far, shift, axis=0)
                    far[:shift, :] = 0.0
            near /= max(float(near.max()), 1e-6)
            far /= max(float(far.max()), 1e-6)
            depth_gate = np.exp(-0.5 * ((D_s[ry0:ry1, rx0:rx1] - d_strip) / 0.10) ** 2)
            spill = 0.58 * (0.50 * near + 0.50 * far) * depth_gate
            light_accum[ry0:ry1, rx0:rx1] = _screen_add(
                light_accum[ry0:ry1, rx0:rx1],
                spill[:, :, None] * _LED_WARM_BGR[None, None, :],
            )

    # 4. Daylight windows: inward-biased soft daylight fill
    if window_mask is not None and window_mask.any():
        bin_win = (window_mask > 0).astype(np.uint8)
        n_lbl, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_win, 8)
        sig_short = max(3.0, 0.03 * s)
        sig_long = max(10.0, 0.14 * s)
        for i in range(1, n_lbl):
            comp = labels == i
            d_win = float(np.median(D_s[comp]))
            cx_w, cy_w = float(centroids[i, 0]), float(centroids[i, 1])
            dx_in = (0.5 * w - cx_w)
            dy_in = (0.5 * h - cy_w)
            norm = max(1.0, float(np.hypot(dx_in, dy_in)))
            shift_x = int(round((dx_in / norm) * 0.05 * w))
            shift_y = int(round((dy_in / norm) * 0.03 * h))

            comp_f = comp.astype(np.float32)
            halo_s = cv2.GaussianBlur(comp_f, (0, 0), sig_short)
            halo_l = cv2.GaussianBlur(comp_f, (0, 0), sig_long)
            if shift_x != 0 or shift_y != 0:
                M = np.float32([[1.0, 0.0, float(shift_x)], [0.0, 1.0, float(shift_y)]])
                halo_l = cv2.warpAffine(halo_l, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            halo_s /= max(float(halo_s.max()), 1e-6)
            halo_l /= max(float(halo_l.max()), 1e-6)
            depth_gate = np.clip((D_s - (d_win - 0.14)) / 0.14, 0.0, 1.0)
            fill = 0.36 * (0.45 * halo_s + 0.55 * halo_l) * depth_gate
            light_accum = _screen_add(light_accum, fill[:, :, None] * _DAYLIGHT_BGR[None, None, :])

    return np.clip(np.round(light_accum * 255.0), 0, 255).astype(np.uint8)


def analyze_scene_lighting(
    image_bgr: np.ndarray,
    *,
    depth_map: Optional[np.ndarray] = None,
    led_mask: Optional[np.ndarray] = None,
    protection_mask: Optional[np.ndarray] = None,
    text_logo_mask: Optional[np.ndarray] = None,
    fireplace_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    allow_depth_model: bool = True,
    allow_object_detection: bool = True,
) -> SceneLightingAnalysis:
    """
    Executes M1 pre-analysis to produce a complete `SceneLightingAnalysis` bundle.
    Loads and releases Depth-Anything-V2-Small and GroundingDINO+SAM ephemerally.
    """
    bgr_u8 = _prepare_bgr_u8(image_bgr)
    h, w = bgr_u8.shape[:2]

    # 1. LED extraction (handle both 3-tuple and legacy 2-tuple returns)
    if led_mask is None:
        split_res = led.split_led_markers(bgr_u8)
        if len(split_res) >= 2:
            analysis_bgr = split_res[0]
            led_mask_u8 = (split_res[1] > 0).astype(np.uint8) * 255
        else:
            analysis_bgr = bgr_u8
            led_mask_u8 = np.zeros((h, w), dtype=np.uint8)
    else:
        if led_mask.shape[:2] != (h, w):
            led_mask_resized = cv2.resize((led_mask > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            led_mask_resized = led_mask
        led_mask_u8 = (led_mask_resized > 0).astype(np.uint8) * 255
        analysis_bgr = led.remove_led_strips(bgr_u8, led_mask_u8) if led_mask_u8.any() else bgr_u8

    # 2. Depth estimation (always releases model + CUDA memory in finally)
    if depth_map is None:
        depth_f32, _ = estimate_depth_map(analysis_bgr, allow_model=allow_depth_model)
    else:
        depth_f32 = np.asarray(depth_map, dtype=np.float32)
        if depth_f32.shape != (h, w):
            depth_f32 = cv2.resize(depth_f32, (w, h), interpolation=cv2.INTER_LINEAR)
        depth_f32 = np.clip(np.nan_to_num(depth_f32, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)

    # 3. Objects, TVs, OCR text/logos, and fireplace boxes
    luminaire_boxes: List[Tuple[int, int, int, int]] = []
    need_det = (protection_mask is None) or (text_logo_mask is None) or (fireplace_boxes is None)
    det = None
    if need_det and allow_object_detection and os.environ.get(OBJECT_ENV_VAR, "1").strip() != "0":
        try:
            det = objects.detect_objects(analysis_bgr)
        except Exception as exc:
            logger.warning("Object detection failed in lighting_map (%s); using fallbacks.", exc)
            det = None
        finally:
            HardwareGuardrails.release_memory()

    if protection_mask is None:
        if det is not None and getattr(det, "mask", None) is not None:
            prot_f32 = np.asarray(det.mask, dtype=np.float32)
        else:
            prot_f32 = np.zeros((h, w), dtype=np.float32)
    else:
        prot_f32 = np.asarray(protection_mask, dtype=np.float32)
    if prot_f32.shape != (h, w):
        prot_f32 = cv2.resize(prot_f32, (w, h), interpolation=cv2.INTER_LINEAR)

    if text_logo_mask is None:
        det_txt = getattr(det, "text_logo_mask", None) if det is not None else None
        if det_txt is not None:
            text_f32 = np.asarray(det_txt, dtype=np.float32)
        else:
            text_f32 = np.zeros((h, w), dtype=np.float32)
    else:
        text_f32 = np.asarray(text_logo_mask, dtype=np.float32)
    if text_f32.shape != (h, w):
        text_f32 = cv2.resize(text_f32, (w, h), interpolation=cv2.INTER_LINEAR)

    text_f32 = np.clip(np.nan_to_num(text_f32, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    prot_f32 = np.clip(
        np.maximum(np.nan_to_num(prot_f32, nan=0.0, posinf=1.0, neginf=0.0), text_f32),
        0.0,
        1.0,
    ).astype(np.float32)

    if det is not None:
        luminaire_boxes = list(getattr(det, "luminaire_boxes", []) or [])

    if fireplace_boxes is None:
        det_fp = getattr(det, "fireplace_boxes", None) if det is not None else None
        if det_fp is not None:
            raw_fp = det_fp
        else:
            raw_fp = _detect_fireplace_boxes_heuristic(analysis_bgr, depth_f32)
    else:
        raw_fp = fireplace_boxes

    valid_fp: List[Tuple[int, int, int, int]] = []
    for box in raw_fp:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(w, x1))
        y1 = max(0, min(h, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 > x1 and y2 > y1:
            valid_fp.append((x1, y1, x2, y2))

    # 4. Exclude mask for windows and spots (never misclassify TVs/logos/LEDs as windows or spots)
    win_exclude = (text_f32 > 0.3) | (prot_f32 > 0.6) | (led_mask_u8 > 0)
    window_mask = detect_window_mask(analysis_bgr, depth_f32, exclude_mask=win_exclude)

    spot_exclude = (text_f32 > 0.3) | (prot_f32 > 0.6)
    spot_mask, spot_components = _detect_spot_components(
        analysis_bgr,
        window_mask=window_mask,
        led_mask=led_mask_u8,
        exclude_mask=spot_exclude.astype(np.float32),
        depth_map=depth_f32,
        luminaire_boxes=luminaire_boxes,
    )

    # 5. Synthesize 2D Illumination Guide Map
    illum_bgr = synthesize_illumination_map(
        analysis_bgr=analysis_bgr,
        depth_map=depth_f32,
        spot_components=spot_components,
        fireplace_boxes=valid_fp,
        window_mask=window_mask,
        led_mask=led_mask_u8,
    )

    return SceneLightingAnalysis(
        depth_map=depth_f32,
        illumination_map_bgr=illum_bgr,
        spot_mask=spot_mask,
        fireplace_boxes=valid_fp,
        window_mask=window_mask,
        protection_mask=prot_f32,
        text_logo_mask=text_f32,
    )
