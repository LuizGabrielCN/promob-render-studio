"""Opaque-box programmatic R1 (Relighting) and R2 (Preservation) metrics for E2E tests.

Uses strictly cv2, numpy, skimage, and PIL (zero imports of backend.app.pipeline.*).
Automatically downsamples 4x (or higher-res) outputs via cv2.INTER_AREA to match the
input reference resolution before computing spatial/perceptual metrics.
"""

from __future__ import annotations

import io
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from skimage.metrics import structural_similarity

ImageInput = Union[bytes, Image.Image, np.ndarray]
NormalizedBox = Tuple[float, float, float, float]  # (y0, y1, x0, x1) in [0.0, 1.0]
RoiSpec = Union[NormalizedBox, Tuple[slice, slice]]

_RAPIDOCR_ENGINE: Any = None
_RAPIDOCR_CHECKED: bool = False


R1_THRESHOLDS: Dict[str, float] = {
    "d_r_minus_b_min": 14.0,
    "db_mean_min": 4.5,
    "dL_low_std_min": 2.2,
    "dL_low_spread_min": 8.0,
    "spot_dL_spread_min": 3.0,
    "spot_d_r_minus_b_min": 15.0,
    "fp_L_rng_ratio_min": 1.6,
    "fp_dchr_min": 10.0,
    "fp_spill_dL_min": 2.0,
    "fp_spill_d_r_minus_b_min": 15.0,
    "led_yellow_rem_max": 0.05,
    "led_center_cv_max": 0.045,
    "led_gap_frac_max": 0.04,
    "led_spill_dL_min": 1.5,
    "led_spill_db_min": 1.5,
}

R2_THRESHOLDS: Dict[str, float] = {
    "sobel_corr_min": 0.72,
    "canny_f1_d2_min": 0.72,
    "global_ssim_L_min": 0.80,
    "dE_ab_norm_max": 6.5,
    "roi_zncc_min": 0.72,
    "roi_edge_f1_d1_min": 0.75,
    "roi_ssim_L_min": 0.70,
}


def decode_bgr(image: ImageInput) -> np.ndarray:
    """Decodes bytes / PIL.Image / BGR ndarray into a contiguous uint8 BGR ndarray."""
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype == np.uint16:
            arr = (arr >> 8).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.ndim == 3 and arr.shape[2] == 1:
            return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
        if arr.ndim == 3 and arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        return np.ascontiguousarray(arr, dtype=np.uint8)

    if isinstance(image, Image.Image):
        if image.mode.startswith("I;16") or image.mode == "I":
            arr_i = np.asarray(image)
            arr_u8 = (np.clip(arr_i, 0, 65535).astype(np.uint16) >> 8).astype(np.uint8)
            if arr_u8.ndim == 2:
                return cv2.cvtColor(arr_u8, cv2.COLOR_GRAY2BGR)
            if arr_u8.ndim == 3 and arr_u8.shape[2] == 1:
                return cv2.cvtColor(arr_u8[:, :, 0], cv2.COLOR_GRAY2BGR)
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if isinstance(image, (bytes, bytearray)):
        buf = np.frombuffer(image, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is not None and bgr.size > 0:
            return bgr
        pil_img = Image.open(io.BytesIO(image))
        return decode_bgr(pil_img)

    raise TypeError(f"Unsupported image type: {type(image)!r}")


def align_to_reference(orig: ImageInput, out: ImageInput) -> Tuple[np.ndarray, np.ndarray]:
    """Decodes orig and out, downsampling 4x/larger outputs via cv2.INTER_AREA to match orig."""
    orig_bgr = decode_bgr(orig)
    out_bgr = decode_bgr(out)
    oh, ow = orig_bgr.shape[:2]
    gh, gw = out_bgr.shape[:2]
    if (gh, gw) != (oh, ow):
        interp = cv2.INTER_AREA if (gh >= oh and gw >= ow) else cv2.INTER_CUBIC
        out_bgr = cv2.resize(out_bgr, (ow, oh), interpolation=interp)
    return orig_bgr, out_bgr


def bgr_to_cie_lab(bgr: np.ndarray) -> np.ndarray:
    """Converts uint8 BGR to CIE L*a*b* (L* in [0, 100], a*/b* centered at 0)."""
    rgb_f = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb2lab(rgb_f).astype(np.float32)


def roi_to_slices(roi: RoiSpec, height: int, width: int, inset_px: int = 0) -> Tuple[slice, slice]:
    """Resolves a normalized (y0, y1, x0, x1) box or (slice, slice) into clamped pixel slices."""
    if len(roi) == 2 and isinstance(roi[0], slice) and isinstance(roi[1], slice):
        ys, xs = roi
        y0 = max(0, (ys.start or 0) + inset_px)
        y1 = min(height, (ys.stop if ys.stop is not None else height) - inset_px)
        x0 = max(0, (xs.start or 0) + inset_px)
        x1 = min(width, (xs.stop if xs.stop is not None else width) - inset_px)
    elif len(roi) == 4:
        ny0, ny1, nx0, nx1 = roi  # type: ignore[misc]
        y0 = max(0, int(round(ny0 * height)) + inset_px)
        y1 = min(height, int(round(ny1 * height)) - inset_px)
        x0 = max(0, int(round(nx0 * width)) + inset_px)
        x1 = min(width, int(round(nx1 * width)) - inset_px)
    else:
        raise ValueError(f"Invalid ROI specification: {roi!r}")

    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"Empty ROI after clamping ({y0}:{y1}, {x0}:{x1}) for {width}x{height}")
    return slice(y0, y1), slice(x0, x1)


def _build_valid_mask(height: int, width: int, exclude_rois: Optional[Sequence[RoiSpec]]) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    if exclude_rois:
        for roi in exclude_rois:
            ys, xs = roi_to_slices(roi, height, width)
            mask[ys, xs] = False
    return mask


def compute_relighting_metrics(
    orig: ImageInput,
    out: ImageInput,
    exclude_rois: Optional[Sequence[RoiSpec]] = None,
) -> Dict[str, float]:
    """Computes R1 global warmth shift and spatial low-frequency illumination dynamics."""
    orig_bgr, out_bgr = align_to_reference(orig, out)
    h, w = orig_bgr.shape[:2]
    mask = _build_valid_mask(h, w, exclude_rois)
    if not np.any(mask):
        return {
            "d_r_minus_b": 0.0,
            "da_mean": 0.0,
            "db_mean": 0.0,
            "dL_low_mean": 0.0,
            "dL_low_std": 0.0,
            "dL_low_spread": 0.0,
        }

    lab_in = bgr_to_cie_lab(orig_bgr)
    lab_out = bgr_to_cie_lab(out_bgr)

    rb_in = orig_bgr[:, :, 2].astype(np.float32) - orig_bgr[:, :, 0].astype(np.float32)
    rb_out = out_bgr[:, :, 2].astype(np.float32) - out_bgr[:, :, 0].astype(np.float32)
    d_r_minus_b = float(np.mean(rb_out[mask]) - np.mean(rb_in[mask]))

    da_mean = float(np.mean(lab_out[:, :, 1][mask] - lab_in[:, :, 1][mask]))
    db_mean = float(np.mean(lab_out[:, :, 2][mask] - lab_in[:, :, 2][mask]))

    sigma = max(1.0, 0.06 * float(max(h, w)))
    mask_f = mask.astype(np.float32)
    weight = np.maximum(cv2.GaussianBlur(mask_f, (0, 0), sigma), 1e-6)
    l_low_in = cv2.GaussianBlur(lab_in[:, :, 0] * mask_f, (0, 0), sigma) / weight
    l_low_out = cv2.GaussianBlur(lab_out[:, :, 0] * mask_f, (0, 0), sigma) / weight
    dl_signed = (l_low_out - l_low_in)[mask]

    return {
        "d_r_minus_b": d_r_minus_b,
        "da_mean": da_mean,
        "db_mean": db_mean,
        "dL_low_mean": float(np.mean(np.abs(dl_signed))),
        "dL_low_std": float(np.std(dl_signed)),
        "dL_low_spread": float(np.percentile(dl_signed, 95) - np.percentile(dl_signed, 5)),
    }


def compute_spot_cone_metrics(
    orig: ImageInput,
    out: ImageInput,
    spot_roi: RoiSpec = (0.04, 0.38, 0.30, 0.52),
) -> Dict[str, float]:
    """Computes R1 spot-cone / wall-wash spatial illumination spread and warmth shift."""
    orig_bgr, out_bgr = align_to_reference(orig, out)
    h, w = orig_bgr.shape[:2]
    ys, xs = roi_to_slices(spot_roi, h, w)

    lab_in = bgr_to_cie_lab(orig_bgr)
    lab_out = bgr_to_cie_lab(out_bgr)

    sigma = max(1.0, 0.06 * float(max(h, w)))
    l_low_in = cv2.GaussianBlur(lab_in[:, :, 0], (0, 0), sigma)
    l_low_out = cv2.GaussianBlur(lab_out[:, :, 0], (0, 0), sigma)
    dl_spot = (l_low_out - l_low_in)[ys, xs]

    roi_in_bgr = orig_bgr[ys, xs]
    roi_out_bgr = out_bgr[ys, xs]
    rb_in = float(np.mean(roi_in_bgr[:, :, 2].astype(np.float32) - roi_in_bgr[:, :, 0].astype(np.float32)))
    rb_out = float(np.mean(roi_out_bgr[:, :, 2].astype(np.float32) - roi_out_bgr[:, :, 0].astype(np.float32)))

    return {
        "spot_dL_mean": float(np.mean(lab_out[ys, xs, 0] - lab_in[ys, xs, 0])),
        "spot_dL_spread": float(np.percentile(dl_spot, 95) - np.percentile(dl_spot, 5)),
        "spot_dL_p90": float(np.percentile(dl_spot, 90)),
        "spot_d_r_minus_b": rb_out - rb_in,
        "spot_db_mean": float(np.mean(lab_out[ys, xs, 2] - lab_in[ys, xs, 2])),
    }


def compute_fireplace_metrics(
    orig: ImageInput,
    out: ImageInput,
    firebox_roi: RoiSpec = (0.68, 0.80, 0.355, 0.470),
    spill_roi: RoiSpec = (0.81, 0.94, 0.30, 0.52),
) -> Dict[str, float]:
    """Computes R1 firebox flame dynamic range/chroma and warm hearth/floor spill metrics."""
    orig_bgr, out_bgr = align_to_reference(orig, out)
    h, w = orig_bgr.shape[:2]
    fy, fx = roi_to_slices(firebox_roi, h, w)
    sy, sx = roi_to_slices(spill_roi, h, w)

    lab_in = bgr_to_cie_lab(orig_bgr)
    lab_out = bgr_to_cie_lab(out_bgr)

    fp_in_l = lab_in[fy, fx, 0]
    fp_out_l = lab_out[fy, fx, 0]
    fp_l_rng_in = float(np.percentile(fp_in_l, 95) - np.percentile(fp_in_l, 5))
    fp_l_rng_out = float(np.percentile(fp_out_l, 95) - np.percentile(fp_out_l, 5))
    fp_l_rng_ratio = fp_l_rng_out / max(fp_l_rng_in, 1e-3)

    fp_chr_in = float(np.mean(np.hypot(lab_in[fy, fx, 1], lab_in[fy, fx, 2])))
    fp_chr_out = float(np.mean(np.hypot(lab_out[fy, fx, 1], lab_out[fy, fx, 2])))

    fp_rb_in = float(np.mean(orig_bgr[fy, fx, 2].astype(np.float32) - orig_bgr[fy, fx, 0].astype(np.float32)))
    fp_rb_out = float(np.mean(out_bgr[fy, fx, 2].astype(np.float32) - out_bgr[fy, fx, 0].astype(np.float32)))

    spill_dL = float(np.mean(lab_out[sy, sx, 0] - lab_in[sy, sx, 0]))
    spill_rb_in = float(np.mean(orig_bgr[sy, sx, 2].astype(np.float32) - orig_bgr[sy, sx, 0].astype(np.float32)))
    spill_rb_out = float(np.mean(out_bgr[sy, sx, 2].astype(np.float32) - out_bgr[sy, sx, 0].astype(np.float32)))
    spill_chr_in = float(np.mean(np.hypot(lab_in[sy, sx, 1], lab_in[sy, sx, 2])))
    spill_chr_out = float(np.mean(np.hypot(lab_out[sy, sx, 1], lab_out[sy, sx, 2])))

    return {
        "fp_L_rng_in": fp_l_rng_in,
        "fp_L_rng_out": fp_l_rng_out,
        "fp_L_rng_ratio": fp_l_rng_ratio,
        "fp_chroma_in": fp_chr_in,
        "fp_chroma_out": fp_chr_out,
        "fp_dchr": fp_chr_out - fp_chr_in,
        "fp_d_r_minus_b": fp_rb_out - fp_rb_in,
        "spill_dL": spill_dL,
        "spill_d_r_minus_b": spill_rb_out - spill_rb_in,
        "spill_dchr": spill_chr_out - spill_chr_in,
    }


def _filter_thin_linear_strokes(bin_mask: np.ndarray) -> np.ndarray:
    """Keeps only thin elongated/L-shaped linear strokes, rejecting blobs, wood grain, and foliage."""
    h, w = bin_mask.shape[:2]
    max_dim = float(max(h, w))
    min_span = max(24.0, min(64.0, 0.04 * max_dim))
    max_half_w = max(8.0, 0.012 * max_dim)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask.astype(np.uint8), connectivity=8)
    filtered = np.zeros((h, w), dtype=np.uint8)
    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < 30:
            continue
        comp = (labels == lbl).astype(np.uint8)
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
        if float(dist.max()) > max_half_w * 1.35:
            continue
        if float(np.percentile(dist[comp > 0], 90)) > max_half_w * 1.1:
            continue
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        major = max(float(rw), float(rh))
        minor = max(min(float(rw), float(rh)), 1.0)
        aspect = major / minor
        perim = float(cv2.arcLength(cnt, True))
        circularity = (4.0 * np.pi * float(area)) / max(perim * perim, 1e-6)
        hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
        solidity = float(area) / max(hull_area, 1.0)
        bw = float(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = float(stats[lbl, cv2.CC_STAT_HEIGHT])
        span = max(major, bw, bh)
        is_thin_shape = (
            aspect >= 3.2
            or circularity <= 0.30
            or solidity <= 0.65
            or (circularity <= 0.56 and solidity <= 0.86)
        )
        if span >= min_span and is_thin_shape:
            filtered[labels == lbl] = 1
    return filtered


def _extract_led_guide_mask(bgr: np.ndarray) -> np.ndarray:
    """Detects vivid Promob yellow LED guide strokes using HSV + thin-stroke geometry."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    strict_yellow = (
        (hsv[:, :, 0] >= 17)
        & (hsv[:, :, 0] <= 38)
        & (hsv[:, :, 1] >= 155)
        & (hsv[:, :, 2] >= 185)
    ).astype(np.uint8)
    return _filter_thin_linear_strokes(strict_yellow)


def _local_median_surround(
    arr_f32: np.ndarray,
    ksize: int = 31,
    scale: float = 1.0,
    offset: float = 128.0,
) -> np.ndarray:
    """Computes a resolution-aware local-surround median baseline (erasing thin linear strokes)."""
    h, w = arr_f32.shape[:2]
    max_dim = max(h, w)
    target_max = 384
    if max_dim > target_max:
        ratio = float(target_max) / float(max_dim)
        sw = max(1, int(round(w * ratio)))
        sh = max(1, int(round(h * ratio)))
        max_strip_w_proxy = max(16.0, 0.024 * float(max_dim)) * ratio
        ksize_eff = max(int(ksize), int(np.ceil(max_strip_w_proxy / 0.40))) | 1
        proxy_f32 = cv2.resize(arr_f32, (sw, sh), interpolation=cv2.INTER_AREA)
        u8 = np.ascontiguousarray(np.clip(np.rint(proxy_f32 * scale + offset), 0, 255), dtype=np.uint8)
        med_proxy = cv2.medianBlur(u8, ksize_eff).astype(np.float32)
        med = cv2.resize(med_proxy, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        ksize_eff = max(int(ksize), 45) | 1
        u8 = np.ascontiguousarray(np.clip(np.rint(arr_f32 * scale + offset), 0, 255), dtype=np.uint8)
        med = cv2.medianBlur(u8, ksize_eff).astype(np.float32)
    return (med - offset) / scale


def _extract_hallucinated_yellow_mask(
    orig_bgr: np.ndarray,
    out_bgr: np.ndarray,
    lab_in: np.ndarray,
    lab_out: np.ndarray,
) -> np.ndarray:
    """Detects newly appearing yellow LED streaks in out on zero-LED scenes without tripping on warmed foliage or wood."""
    hsv_in = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2HSV)
    hsv_out = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2HSV)
    strict_yellow_out = (
        (hsv_out[:, :, 0] >= 17)
        & (hsv_out[:, :, 0] <= 38)
        & (hsv_out[:, :, 1] >= 155)
        & (hsv_out[:, :, 2] >= 185)
    )
    was_warm_surface = (
        (hsv_in[:, :, 0] >= 10)
        & (hsv_in[:, :, 0] <= 55)
        & (hsv_in[:, :, 1] >= 80)
    )
    dl = lab_out[:, :, 0] - lab_in[:, :, 0]
    db = lab_out[:, :, 2] - lab_in[:, :, 2]
    dl_loc = dl - _local_median_surround(dl, ksize=31, scale=1.0, offset=128.0)
    db_loc = db - _local_median_surround(db, ksize=31, scale=0.5, offset=128.0)
    # Over already-warm wood, curtains, or foliage, genuine hallucinated LEDs exhibit extreme
    # yellow saturation AND a sharp local thin-stroke contrast jump over their immediate surround.
    newly_emissive_local = (
        (hsv_out[:, :, 1] >= 225)
        & (lab_out[:, :, 0] >= 78.0)
        & (
            ((dl_loc >= 8.0) & (db_loc >= 10.0) & ((dl >= 10.0) | (db >= 20.0)))
            | ((dl_loc >= -8.0) & (db_loc >= 24.0) & (db >= 24.0))
        )
    )
    # On neutral/desaturated surfaces, reject global warm tinting and smooth spot cones
    # lacking strong local-surround contrast.
    sat_jump = hsv_out[:, :, 1].astype(np.int16) - hsv_in[:, :, 1].astype(np.int16)
    has_local_contrast = ((dl_loc >= 4.0) & (db_loc >= 15.0)) | (db_loc >= 20.0)
    neutral_new_yellow = ~was_warm_surface & (
        ((dl_loc >= 6.0) & (db_loc >= 15.0))
        | (db_loc >= 20.0)
        | (((hsv_out[:, :, 0] >= 19) | (sat_jump >= 55)) & has_local_contrast)
    )
    cand = strict_yellow_out & (neutral_new_yellow | newly_emissive_local)
    return _filter_thin_linear_strokes(cand.astype(np.uint8))


def _detect_spurious_warm_streaks(
    lab_in: np.ndarray,
    lab_out: np.ndarray,
    ignore_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Detects newly hallucinated bright warm thin linear LED streaks in out via local-surround contrast."""
    dl = lab_out[:, :, 0] - lab_in[:, :, 0]
    db = lab_out[:, :, 2] - lab_in[:, :, 2]
    dl_loc = dl - _local_median_surround(dl, ksize=31, scale=1.0, offset=128.0)
    db_loc = db - _local_median_surround(db, ksize=31, scale=0.5, offset=128.0)
    b_out_loc = lab_out[:, :, 2] - _local_median_surround(lab_out[:, :, 2], ksize=31, scale=1.0, offset=128.0)

    branch_emissive = (
        (lab_out[:, :, 0] >= 79.0)
        & (
            (
                (lab_out[:, :, 2] >= 18.0)
                & (dl >= 14.0)
                & (db >= 14.0)
                & (dl_loc >= 10.0)
                & (db_loc >= 10.0)
                & (b_out_loc >= 10.0)
            )
            | (
                (lab_out[:, :, 2] >= 24.0)
                & (dl >= 18.0)
                & (dl_loc >= 14.0)
                & (db >= 8.0)
                & (db_loc >= 8.0)
                & (b_out_loc >= 8.0)
            )
        )
    )
    branch_bright_ceiling = (
        (lab_out[:, :, 0] >= 82.0)
        & (lab_out[:, :, 2] >= 28.0)
        & (dl >= -8.0)
        & (db >= 26.0)
        & (db_loc >= 16.0)
        & (b_out_loc >= 16.0)
    )
    bright_warm_delta = branch_emissive | branch_bright_ceiling
    if ignore_mask is not None:
        bright_warm_delta = bright_warm_delta & ~ignore_mask
    return _filter_thin_linear_strokes(bright_warm_delta.astype(np.uint8))


def compute_led_continuity_metrics(orig: ImageInput, out: ImageInput) -> Dict[str, float]:
    """Computes R1/R2 concealed LED yellow-removal, axial smoothness, warm spill, and hallucination checks."""
    orig_bgr, out_bgr = align_to_reference(orig, out)

    guide_mask = _extract_led_guide_mask(orig_bgr)

    lab_in = bgr_to_cie_lab(orig_bgr)
    lab_out = bgr_to_cie_lab(out_bgr)

    yellow_px = int(guide_mask.sum())
    if yellow_px == 0:
        out_yellow_mask = _extract_hallucinated_yellow_mask(orig_bgr, out_bgr, lab_in, lab_out)
        spurious_streaks = _detect_spurious_warm_streaks(lab_in, lab_out, ignore_mask=None)
        return {
            "yellow_px": 0.0,
            "center_px": 0.0,
            "yellow_rem": 0.0,
            "center_L": 0.0,
            "center_cv": 0.0,
            "gap_frac": 0.0,
            "spill_dL": 0.0,
            "spill_db": 0.0,
            "spill_d_r_minus_b": 0.0,
            "out_yellow_px": float(out_yellow_mask.sum()),
            "out_spurious_streak_px": float(spurious_streaks.sum()),
        }

    out_yellow_mask = _extract_led_guide_mask(out_bgr)

    eroded = cv2.erode(guide_mask, np.ones((3, 3), np.uint8))
    center_mask = (eroded > 0) if int(eroded.sum()) >= 20 else (guide_mask > 0)
    center_px = int(center_mask.sum())

    dil_outer = cv2.dilate(guide_mask, np.ones((17, 17), np.uint8)) > 0
    dil_inner = cv2.dilate(guide_mask, np.ones((3, 3), np.uint8)) > 0
    spill_mask = dil_outer & ~dil_inner
    if not spill_mask.any():
        spill_mask = (cv2.dilate(guide_mask, np.ones((9, 9), np.uint8)) > 0) & ~(guide_mask > 0)

    hsv_out = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2HSV)
    yellow_out = (
        (hsv_out[:, :, 0] >= 18)
        & (hsv_out[:, :, 0] <= 40)
        & (hsv_out[:, :, 1] > 140)
    )
    yellow_rem = float(np.mean(yellow_out[guide_mask > 0]))

    l_out = lab_out[:, :, 0]
    center_vals = l_out[center_mask]
    center_mean = float(np.mean(center_vals))
    center_cv = float(np.std(center_vals) / max(center_mean, 1e-6))

    local_max_9x9 = cv2.dilate(l_out, np.ones((9, 9), np.uint8))
    gap_frac = float(np.mean((local_max_9x9[center_mask] - center_vals) > 8.0))

    if spill_mask.any():
        spill_dL = float(np.mean(lab_out[:, :, 0][spill_mask] - lab_in[:, :, 0][spill_mask]))
        spill_db = float(np.mean(lab_out[:, :, 2][spill_mask] - lab_in[:, :, 2][spill_mask]))
        rb_in = orig_bgr[:, :, 2].astype(np.float32) - orig_bgr[:, :, 0].astype(np.float32)
        rb_out = out_bgr[:, :, 2].astype(np.float32) - out_bgr[:, :, 0].astype(np.float32)
        spill_d_rb = float(np.mean(rb_out[spill_mask] - rb_in[spill_mask]))
    else:
        spill_dL = spill_db = spill_d_rb = 0.0

    spurious_streaks = _detect_spurious_warm_streaks(lab_in, lab_out, ignore_mask=dil_outer)
    return {
        "yellow_px": float(yellow_px),
        "center_px": float(center_px),
        "yellow_rem": yellow_rem,
        "center_L": center_mean,
        "center_cv": center_cv,
        "gap_frac": gap_frac,
        "spill_dL": spill_dL,
        "spill_db": spill_db,
        "spill_d_r_minus_b": spill_d_rb,
        "out_yellow_px": float(out_yellow_mask.sum()),
        "out_spurious_streak_px": float(spurious_streaks.sum()),
    }


def compute_geometry_preservation(
    orig: ImageInput,
    out: ImageInput,
    exclude_rois: Optional[Sequence[RoiSpec]] = None,
) -> Dict[str, float]:
    """Computes R2 global Sobel gradient correlation, 2px-dilated Canny F1, L* SSIM, and normalized dE_ab."""
    orig_bgr, out_bgr = align_to_reference(orig, out)
    h, w = orig_bgr.shape[:2]
    mask = _build_valid_mask(h, w, exclude_rois)
    if not np.any(mask):
        return {
            "sobel_corr": 1.0,
            "canny_f1_d2": 1.0,
            "canny_recall_d2": 1.0,
            "canny_precision_d2": 1.0,
            "global_ssim_L": 1.0,
            "dE_ab_norm": 0.0,
        }

    lab_in = bgr_to_cie_lab(orig_bgr)
    lab_out = bgr_to_cie_lab(out_bgr)

    l_in = lab_in[:, :, 0]
    l_out = lab_out[:, :, 0]

    blur_in = cv2.GaussianBlur(l_in, (0, 0), 1.0)
    blur_out = cv2.GaussianBlur(l_out, (0, 0), 1.0)
    mag_in = np.hypot(cv2.Sobel(blur_in, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(blur_in, cv2.CV_32F, 0, 1, ksize=3))
    mag_out = np.hypot(cv2.Sobel(blur_out, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(blur_out, cv2.CV_32F, 0, 1, ksize=3))

    m_in = mag_in[mask]
    m_out = mag_out[mask]
    if float(np.std(m_in)) < 1e-6 or float(np.std(m_out)) < 1e-6:
        sobel_corr = 1.0 if float(np.std(m_in)) < 1e-6 and float(np.std(m_out)) < 1e-6 else 0.0
    else:
        sobel_corr = float(np.corrcoef(m_in, m_out)[0, 1])

    edge_in = cv2.Canny(cv2.GaussianBlur(orig_bgr, (0, 0), 1.0), 40, 110) > 0
    edge_out = cv2.Canny(cv2.GaussianBlur(out_bgr, (0, 0), 1.0), 40, 110) > 0
    k5 = np.ones((5, 5), dtype=np.uint8)
    dil_in = cv2.dilate(edge_in.astype(np.uint8), k5) > 0
    dil_out = cv2.dilate(edge_out.astype(np.uint8), k5) > 0

    in_cnt = int((edge_in & mask).sum())
    out_cnt = int((edge_out & mask).sum())
    if in_cnt == 0 and out_cnt == 0:
        canny_rec = canny_prec = canny_f1 = 1.0
    elif in_cnt == 0 or out_cnt == 0:
        canny_rec = canny_prec = canny_f1 = 0.0
    else:
        canny_rec = float((edge_in & dil_out & mask).sum()) / float(in_cnt)
        canny_prec = float((edge_out & dil_in & mask).sum()) / float(out_cnt)
        canny_f1 = (2.0 * canny_prec * canny_rec) / max(canny_prec + canny_rec, 1e-9)

    if min(h, w) < 3:
        global_ssim_l = float(max(0.0, 1.0 - np.mean(np.abs(l_out[mask] - l_in[mask])) / 100.0))
    else:
        win_size = min(7, h if h % 2 == 1 else h - 1, w if w % 2 == 1 else w - 1)
        win_size = max(3, win_size)
        _, smap = structural_similarity(l_in, l_out, data_range=100.0, win_size=win_size, full=True)
        global_ssim_l = float(np.mean(smap[mask]))

    da = (lab_out[:, :, 1] - lab_in[:, :, 1])[mask]
    db = (lab_out[:, :, 2] - lab_in[:, :, 2])[mask]
    de_norm = float(np.mean(np.hypot(da - float(np.median(da)), db - float(np.median(db)))))

    return {
        "sobel_corr": sobel_corr,
        "canny_f1_d2": canny_f1,
        "canny_recall_d2": canny_rec,
        "canny_precision_d2": canny_prec,
        "global_ssim_L": global_ssim_l,
        "dE_ab_norm": de_norm,
    }


def _try_rapidocr_text(bgr_crop: np.ndarray) -> Tuple[bool, str]:
    """Optional rapidocr_onnxruntime text probe (cached at module level)."""
    global _RAPIDOCR_ENGINE, _RAPIDOCR_CHECKED
    if not _RAPIDOCR_CHECKED:
        _RAPIDOCR_CHECKED = True
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
            _RAPIDOCR_ENGINE = RapidOCR()
        except Exception:
            _RAPIDOCR_ENGINE = None

    if _RAPIDOCR_ENGINE is None:
        return False, ""

    try:
        result, _ = _RAPIDOCR_ENGINE(bgr_crop)
        if not result:
            return True, ""
        text = " ".join(str(item[1]) for item in result if len(item) >= 2)
        return True, text.strip()
    except Exception:
        return True, ""


def compute_roi_preservation(
    orig: ImageInput,
    out: ImageInput,
    roi: RoiSpec,
    expected_text: Optional[str] = None,
    inset_px: int = 0,
) -> Dict[str, Any]:
    """Computes R2 protected ROI ZNCC, 1px-dilated Edge F1, L* SSIM, and optional RapidOCR match."""
    orig_bgr, out_bgr = align_to_reference(orig, out)
    h, w = orig_bgr.shape[:2]
    ys, xs = roi_to_slices(roi, h, w, inset_px=inset_px)

    crop_in_bgr = orig_bgr[ys, xs]
    crop_out_bgr = out_bgr[ys, xs]
    ch, cw = crop_in_bgr.shape[:2]

    lab_in = bgr_to_cie_lab(crop_in_bgr)[:, :, 0]
    lab_out = bgr_to_cie_lab(crop_out_bgr)[:, :, 0]

    std_in = float(np.std(lab_in))
    std_out = float(np.std(lab_out))
    if std_in < 1e-6 or std_out < 1e-6:
        zncc = 1.0 if (std_in < 1e-6 and std_out < 1e-6) else 0.0
    else:
        zncc = float(np.mean((lab_in - np.mean(lab_in)) * (lab_out - np.mean(lab_out))) / (std_in * std_out))

    edge_in = cv2.Canny(cv2.GaussianBlur(crop_in_bgr, (0, 0), 0.8), 40, 110) > 0
    edge_out = cv2.Canny(cv2.GaussianBlur(crop_out_bgr, (0, 0), 0.8), 40, 110) > 0
    k3 = np.ones((3, 3), dtype=np.uint8)
    dil_in = cv2.dilate(edge_in.astype(np.uint8), k3) > 0
    dil_out = cv2.dilate(edge_out.astype(np.uint8), k3) > 0

    in_cnt = int(edge_in.sum())
    out_cnt = int(edge_out.sum())
    if in_cnt == 0 and out_cnt == 0:
        rec = prec = edge_f1 = 1.0
    elif in_cnt == 0 or out_cnt == 0:
        rec = prec = edge_f1 = 0.0
    else:
        rec = float((edge_in & dil_out).sum()) / float(in_cnt)
        prec = float((edge_out & dil_in).sum()) / float(out_cnt)
        edge_f1 = (2.0 * prec * rec) / max(prec + rec, 1e-9)

    if min(ch, cw) < 3:
        ssim_l = float(max(0.0, 1.0 - np.mean(np.abs(lab_out - lab_in)) / 100.0))
    else:
        win_size = min(7, ch if ch % 2 == 1 else ch - 1, cw if cw % 2 == 1 else cw - 1)
        win_size = max(3, win_size)
        ssim_l = float(structural_similarity(lab_in, lab_out, data_range=100.0, win_size=win_size))

    ocr_available, ocr_text = _try_rapidocr_text(crop_out_bgr)
    ocr_matched: Optional[bool] = None
    if ocr_available and expected_text:
        ocr_matched = expected_text.lower() in ocr_text.lower()

    return {
        "zncc": zncc,
        "edge_f1_d1": edge_f1,
        "edge_recall_d1": rec,
        "edge_precision_d1": prec,
        "ssim_L": ssim_l,
        "ocr_available": ocr_available,
        "ocr_text": ocr_text,
        "ocr_matched": ocr_matched,
    }
