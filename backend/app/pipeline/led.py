"""
LED strip markers drawn by the designer in Promob.

Designers mark LED profiles with a thin, saturated yellow line (the Promob render shows it as a flat
unlit stroke). This module finds those strokes, removes them before any enhancement (so neither the
photometric stages nor the diffusion smear them), and paints a smooth, continuous recessed LED profile
back at the end: a subtle warm-white core plus a 3-scale directional warm glow on the surrounding surfaces.
"""
from collections import OrderedDict
import threading
from typing import Any, Dict, List, Optional, Tuple
import zlib

import cv2
import numpy as np

# OpenCV HSV: hue 0-180 (yellow ~30). Calibrated on Promob renders: the marker stroke is H 24-32,
# S > 200, V > 220, while brass handles, oak, golden marble veins and bulbs stay below S 170 or V 170.
_HUE_RANGE = (22, 34)
_MIN_SAT = 170
_MIN_VAL = 170

# Warm white (~3000 K) in BGR
_GLOW_BGR = np.array([175.0, 215.0, 255.0], dtype=np.float32) / 255.0
_CORE_BGR = np.array([235.0, 247.0, 255.0], dtype=np.float32) / 255.0


def _to_u8(img: np.ndarray) -> np.ndarray:
    bgr = img[:, :, :3]
    if bgr.dtype == np.uint8:
        return bgr
    return np.clip(np.round(bgr * 255.0), 0, 255).astype(np.uint8)


def _build_zhang_suen_luts() -> Tuple[np.ndarray, np.ndarray]:
    """Precomputes 256-entry 8-neighbor deletion LUTs for Zhang-Suen sub-iterations 0 and 1."""
    lut0 = np.zeros(256, dtype=bool)
    lut1 = np.zeros(256, dtype=bool)
    for code in range(256):
        p2 = (code >> 0) & 1
        p3 = (code >> 1) & 1
        p4 = (code >> 2) & 1
        p5 = (code >> 3) & 1
        p6 = (code >> 4) & 1
        p7 = (code >> 5) & 1
        p8 = (code >> 6) & 1
        p9 = (code >> 7) & 1
        nz = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        if nz < 2 or nz > 6:
            continue
        trans = (
            (p2 == 0 and p3 == 1)
            + (p3 == 0 and p4 == 1)
            + (p4 == 0 and p5 == 1)
            + (p5 == 0 and p6 == 1)
            + (p6 == 0 and p7 == 1)
            + (p7 == 0 and p8 == 1)
            + (p8 == 0 and p9 == 1)
            + (p9 == 0 and p2 == 1)
        )
        if trans != 1:
            continue
        if p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0:
            lut0[code] = True
        if p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0:
            lut1[code] = True
    return lut0, lut1


_ZS_LUT_0, _ZS_LUT_1 = _build_zhang_suen_luts()
_THIN_CACHE: "OrderedDict[Tuple[Tuple[int, int], int], Tuple[np.ndarray, np.ndarray]]" = OrderedDict()
_THIN_CACHE_LOCK = threading.Lock()
_THIN_CACHE_MAX = 8


def _thin_binary_uncached(norm_u8: np.ndarray) -> np.ndarray:
    """Active-pixel LUT Zhang-Suen thinning on a tight 0/1 uint8 ROI."""
    img = cv2.copyMakeBorder(norm_u8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ys, xs = np.nonzero(img[1:-1, 1:-1])
    if ys.size == 0:
        return img[1:-1, 1:-1]
    ys = ys + 1
    xs = xs + 1
    for _ in range(64):
        changed = False
        for lut in (_ZS_LUT_0, _ZS_LUT_1):
            code = (
                img[ys - 1, xs]
                | (img[ys - 1, xs + 1] << 1)
                | (img[ys, xs + 1] << 2)
                | (img[ys + 1, xs + 1] << 3)
                | (img[ys + 1, xs] << 4)
                | (img[ys + 1, xs - 1] << 5)
                | (img[ys, xs - 1] << 6)
                | (img[ys - 1, xs - 1] << 7)
            )
            cond = lut[code]
            if np.any(cond):
                img[ys[cond], xs[cond]] = 0
                keep = ~cond
                ys = ys[keep]
                xs = xs[keep]
                changed = True
                if ys.size == 0:
                    return img[1:-1, 1:-1]
        if not changed:
            break
    return img[1:-1, 1:-1]


def _thin_binary(bin_u8: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning on a binary uint8 ROI; returns a 1-px connected 0/1 skeleton."""
    nz_y, nz_x = np.nonzero(bin_u8)
    if nz_y.size == 0:
        return np.zeros(bin_u8.shape[:2], dtype=np.uint8)
    y0, y1 = int(nz_y.min()), int(nz_y.max()) + 1
    x0, x1 = int(nz_x.min()), int(nz_x.max()) + 1
    sub = np.ascontiguousarray((bin_u8[y0:y1, x0:x1] > 0).astype(np.uint8))
    key = (sub.shape, zlib.crc32(sub))
    with _THIN_CACHE_LOCK:
        cached = _THIN_CACHE.get(key)
        if cached is not None and np.array_equal(cached[0], sub):
            _THIN_CACHE.move_to_end(key)
            sub_skel = cached[1].copy()
        else:
            sub_skel = None
    if sub_skel is None:
        sub_skel = _thin_binary_uncached(sub)
        with _THIN_CACHE_LOCK:
            _THIN_CACHE[key] = (sub, sub_skel.copy())
            _THIN_CACHE.move_to_end(key)
            while len(_THIN_CACHE) > _THIN_CACHE_MAX:
                _THIN_CACHE.popitem(last=False)
    if y0 == 0 and x0 == 0 and y1 == bin_u8.shape[0] and x1 == bin_u8.shape[1]:
        return sub_skel
    out = np.zeros(bin_u8.shape[:2], dtype=np.uint8)
    out[y0:y1, x0:x1] = sub_skel
    return out


def _skeleton_length(skel: np.ndarray) -> float:
    """Euclidean arc length of a 1-px 8-connected binary skeleton (1.0 for 4-edges, sqrt(2) for diagonals)."""
    s = (skel > 0).astype(np.uint8)
    n_px = int(cv2.countNonZero(s))
    if n_px <= 1:
        return float(n_px)
    h_edges = int(np.count_nonzero(s[:, :-1] & s[:, 1:]))
    v_edges = int(np.count_nonzero(s[:-1, :] & s[1:, :]))
    # Do not double-count a diagonal chord across a 4-connected corner step
    dr_edges = int(np.count_nonzero((s[:-1, :-1] & s[1:, 1:]) & ~(s[:-1, 1:] | s[1:, :-1])))
    dl_edges = int(np.count_nonzero((s[:-1, 1:] & s[1:, :-1]) & ~(s[:-1, :-1] | s[1:, 1:])))
    return 1.0 + float(h_edges + v_edges) + float(dr_edges + dl_edges) * float(np.sqrt(2.0))


def detect_led_strip_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Returns a uint8 mask (0/255) of the yellow LED marker strokes: saturated yellow components that
    are long (>= 1.8% of the short image side) and thin (elongation >= 6), including corner-joined
    L-shaped and U-shaped profiles. Empty when there are none.
    Returns the emitting stroke only; the wider inpainting margin is added by remove_led_strips.
    """
    bgr = _to_u8(img_bgr)
    h, w = bgr.shape[:2]
    s = min(h, w)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    raw = cv2.inRange(hsv, (_HUE_RANGE[0], _MIN_SAT, _MIN_VAL), (_HUE_RANGE[1], 255, 255))
    nnz = cv2.countNonZero(raw)
    if nnz < 20:
        if nnz:
            raw.fill(0)
        return raw

    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    mask = np.zeros_like(raw)
    min_len = max(12.0, s * 0.018)
    max_thick = max(7.0, s * 0.012)
    for i in range(1, n_lbl):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 20:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        roi_eq = labels[y : y + ch, x : x + cw] == i
        ys, xs = np.nonzero(roi_eq)
        pts = np.empty((len(xs), 2), dtype=np.float32)
        pts[:, 0] = xs + x
        pts[:, 1] = ys + y
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        length = max(rw, rh, 1.0)  # rotated rect: slanted strips in perspective keep their true length
        thick = area / max(length, 1.0)
        fill = area / (max(rw, 1.0) * max(rh, 1.0))
        is_strip = (
            length >= min_len
            and thick <= max_thick
            and length / max(thick, 1.0) >= 6.0
            and fill >= 0.55
        )
        if not is_strip and fill < 0.55:
            # Corner-joined L-shaped / U-shaped LED profiles have low bounding-rect fill (<0.55)
            # because of empty interior space; validate via local distanceTransform thickness,
            # skeleton geodesic length & elongation, and straight-segment coverage along skeleton.
            roi_u8 = roi_eq.astype(np.uint8)
            padded = cv2.copyMakeBorder(roi_u8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            dist_t = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
            max_half_thick = float(dist_t.max())
            compactness = float(area) / (np.pi * max(max_half_thick, 1.0) ** 2)
            if max_half_thick <= 0.65 * max_thick and compactness >= 4.0:
                skel = _thin_binary(roi_u8)
                skel_px = int(cv2.countNonZero(skel))
                skel_len = _skeleton_length(skel)
                local_thick = float(area) / max(skel_len, 1.0)
                if (
                    skel_len >= max(1.6 * min_len, 24.0)
                    and local_thick <= max_thick
                    and skel_len / max(local_thick, 1.0) >= 6.0
                ):
                    min_seg = max(10, int(round(min_len * 0.72)))
                    hough_thresh = max(8, int(round(min_seg * 0.65)))
                    lines = cv2.HoughLinesP(
                        skel,
                        1,
                        np.pi / 180.0,
                        threshold=hough_thresh,
                        minLineLength=min_seg,
                        maxLineGap=2,
                    )
                    if lines is not None:
                        segs = lines.reshape(-1, 4)
                        seg_lens = [float(np.hypot(lx1 - lx0, ly1 - ly0)) for lx0, ly0, lx1, ly1 in segs]
                        max_seg_len = max(seg_lens)
                        long_thresh = (
                            max(float(min_seg), 0.14 * skel_len) if skel_len > 80.0 else float(min_seg)
                        )
                        if max_seg_len >= max(float(min_seg), 0.25 * skel_len):
                            cov = np.zeros_like(skel)
                            cov_thick = 3 if skel_len > 80.0 else 1
                            for (lx0, ly0, lx1, ly1), seg_len in zip(segs, seg_lens):
                                if seg_len >= long_thresh:
                                    cv2.line(cov, (int(lx0), int(ly0)), (int(lx1), int(ly1)), 1, cov_thick)
                            cov_frac = float(cv2.countNonZero(skel & cov)) / max(1, skel_px)
                            if cov_frac >= 0.70:
                                is_strip = True
        if is_strip:
            mask[y : y + ch, x : x + cw][roi_eq] = 255
    return mask


def _extract_strip_metadata(mask: np.ndarray) -> Dict[str, Any]:
    """Computes analytical PCA geometry and orientation metadata for each LED strip component."""
    if mask is None:
        return {"count": 0, "has_leds": False, "image_shape": (0, 0), "strips": []}
    h, w = mask.shape[:2]
    s = min(h, w)
    if not mask.any():
        return {"count": 0, "has_leds": False, "image_shape": (h, w), "strips": []}

    bin_mask = (mask > 0).astype(np.uint8)
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, 8)
    strips: List[Dict[str, Any]] = []
    for i in range(1, n_lbl):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        roi_eq = labels[y : y + ch, x : x + cw] == i
        ys, xs = np.nonzero(roi_eq)
        gx = (xs + x).astype(np.float32)
        gy = (ys + y).astype(np.float32)

        mx = float(gx.mean())
        my = float(gy.mean())
        dx = gx - mx
        dy = gy - my

        if len(gx) >= 2:
            cov = np.array(
                [[float(np.dot(dx, dx)), float(np.dot(dx, dy))],
                 [float(np.dot(dx, dy)), float(np.dot(dy, dy))]],
                dtype=np.float64,
            ) / float(len(gx))
            eigvals, eigvecs = np.linalg.eigh(cov)
            ux, uy = float(eigvecs[0, 1]), float(eigvecs[1, 1])
            if ux < 0.0 or (abs(ux) < 1e-6 and uy < 0.0):
                ux, uy = -ux, -uy
        else:
            ux, uy = (1.0, 0.0) if cw >= ch else (0.0, 1.0)

        t = dx * ux + dy * uy
        perp = np.abs(dx * (-uy) + dy * ux)
        t_min = float(t.min())
        t_max = float(t.max())
        p0 = (round(mx + t_min * ux, 2), round(my + t_min * uy, 2))
        p1 = (round(mx + t_max * ux, 2), round(my + t_max * uy, 2))
        length_px = max(1.0, t_max - t_min)
        thickness_px = float(area) / length_px
        p95_perp = float(np.percentile(perp, 95)) if perp.size > 0 else 0.0
        is_straight = bool(p95_perp <= max(3.5, 0.006 * s) and length_px >= 8.0)
        if not is_straight:
            skel = _thin_binary(roi_eq.astype(np.uint8))
            skel_len = _skeleton_length(skel)
            if skel_len > 1.0:
                length_px = max(length_px, skel_len)
                thickness_px = max(1.0, float(area) / skel_len)

        if abs(ux) >= 0.85:
            orientation = "horizontal"
        elif abs(uy) >= 0.85:
            orientation = "vertical"
        else:
            orientation = "diagonal"

        strips.append(
            {
                "id": i,
                "orientation": orientation,
                "p0": p0,
                "p1": p1,
                "length_px": round(length_px, 2),
                "thickness_px": round(thickness_px, 2),
                "bbox": (x, y, x + cw, y + ch),
                "is_straight": is_straight,
                "area_px": area,
            }
        )

    return {
        "count": len(strips),
        "has_leds": len(strips) > 0,
        "image_shape": (h, w),
        "strips": strips,
    }


def remove_led_strips(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaints the marker strokes so later stages see the surface behind them. Keeps dtype/alpha."""
    if mask is None or not mask.any():
        return img.copy()
    s = min(img.shape[:2])
    radius = max(3, int(round(s / 250.0)))
    # Remove the low-saturation antialiasing fringe without enlarging the emitted LED core.
    k = max(3, int(round(s / 300.0)) * 2 + 1)
    removal_mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    clean_u8 = cv2.inpaint(_to_u8(img), removal_mask, radius, cv2.INPAINT_TELEA)
    return _restore_layout(img, clean_u8)


def _segment_distance_field(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
) -> np.ndarray:
    """Exact sub-pixel Euclidean distance from every grid point to segment [p0, p1]."""
    vx = float(p1[0] - p0[0])
    vy = float(p1[1] - p0[1])
    len2 = vx * vx + vy * vy
    if len2 <= 1e-6:
        return np.hypot(x_grid - float(p0[0]), y_grid - float(p0[1]))
    px = x_grid - float(p0[0])
    py = y_grid - float(p0[1])
    proj = np.clip((px * vx + py * vy) / len2, 0.0, 1.0)
    cx = float(p0[0]) + proj * vx
    cy = float(p0[1]) + proj * vy
    return np.hypot(x_grid - cx, y_grid - cy).astype(np.float32)


def render_led_strips(
    img: np.ndarray,
    mask: np.ndarray,
    intensity: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Paints smooth continuous recessed LED profiles where `mask` marks them: an anti-aliased
    analytical segment core and a 3-scale directional warm glow (screen blend, so bright surfaces
    do not clip harshly). Eliminates diagonal staircase beading. Keeps dtype/alpha.
    """
    if mask is None or not mask.any() or intensity <= 0.0:
        return img.copy()
    h, w = img.shape[:2]
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    if metadata is None or tuple(metadata.get("image_shape", ())) != (h, w):
        meta = _extract_strip_metadata(mask)
    else:
        meta = metadata

    strips = meta.get("strips", [])
    if not strips:
        return img.copy()

    s = min(h, w)

    near_sigma = max(2.0, s * 0.0045)
    mid_sigma = max(8.0, s * 0.018)
    far_sigma = max(20.0, s * 0.045)
    pad = int(np.ceil(3.2 * far_sigma))

    bin_mask = (mask > 0).astype(np.uint8)
    n_lbl, labels, _, _ = cv2.connectedComponentsWithStats(bin_mask, 8)
    # Match connected component labels when metadata did not come from this exact mask labeling
    use_label_ids = tuple(meta.get("image_shape", ())) == (h, w)

    glow_acc = np.zeros((h, w), dtype=np.float32)
    core_acc = np.zeros((h, w), dtype=np.float32)
    ux0, uy0, ux1, uy1 = w, h, 0, 0

    for idx, strip in enumerate(strips):
        x0, y0, x1, y1 = strip["bbox"]
        rx0 = max(0, x0 - pad)
        ry0 = max(0, y0 - pad)
        rx1 = min(w, x1 + pad)
        ry1 = min(h, y1 + pad)
        rw, rh = rx1 - rx0, ry1 - ry0
        if rw <= 0 or rh <= 0:
            continue
        ux0 = min(ux0, rx0)
        uy0 = min(uy0, ry0)
        ux1 = max(ux1, rx1)
        uy1 = max(uy1, ry1)

        thick = float(strip.get("thickness_px", 3.0))
        src_sigma = max(0.75, min(1.35, thick * 0.40))
        core_sigma = max(0.80, min(1.85, thick * 0.32))
        if strip.get("is_straight", False):
            p0 = (float(strip["p0"][0]) - rx0, float(strip["p0"][1]) - ry0)
            p1 = (float(strip["p1"][0]) - rx0, float(strip["p1"][1]) - ry0)
            xs = np.arange(rw, dtype=np.float32)[None, :]
            ys = np.arange(rh, dtype=np.float32)[:, None]
            dist = _segment_distance_field(xs, ys, p0, p1)
        else:
            # Smooth continuous distance-field rendering from thinned centerline for L/U-shaped & curved strips
            lbl_id = int(strip.get("id", idx + 1))
            if use_label_ids and 1 <= lbl_id < n_lbl:
                comp = (labels[ry0:ry1, rx0:rx1] == lbl_id).astype(np.uint8)
            else:
                comp = (bin_mask[ry0:ry1, rx0:rx1] > 0).astype(np.uint8)
            if not comp.any():
                comp = (bin_mask[ry0:ry1, rx0:rx1] > 0).astype(np.uint8)
            if not comp.any():
                continue
            skel = _thin_binary(comp)
            if not skel.any():
                skel = comp
            inv_skel = (skel == 0).astype(np.uint8)
            dist = cv2.distanceTransform(inv_skel, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        src = np.exp(-0.5 * (dist / src_sigma) ** 2).astype(np.float32)
        c_comp = np.exp(-0.5 * (dist / core_sigma) ** 2).astype(np.float32)

        near = cv2.GaussianBlur(src, (0, 0), near_sigma)
        mid = cv2.GaussianBlur(src, (0, 0), mid_sigma)
        far = cv2.GaussianBlur(src, (0, 0), far_sigma)

        if strip.get("orientation") == "horizontal":
            mid_shift = int(round(mid_sigma * 0.55))
            if 0 < mid_shift < rh:
                mid = np.roll(mid, mid_shift, axis=0)
                mid[:mid_shift, :] = 0.0
            far_shift = int(round(far_sigma * 0.65))
            if 0 < far_shift < rh:
                far = np.roll(far, far_shift, axis=0)
                far[:far_shift, :] = 0.0

        near /= max(float(near.max()), 1e-6)
        mid /= max(float(mid.max()), 1e-6)
        far /= max(float(far.max()), 1e-6)

        strip_glow = 0.28 * near + 0.24 * mid + 0.14 * far
        glow_acc[ry0:ry1, rx0:rx1] = np.maximum(glow_acc[ry0:ry1, rx0:rx1], strip_glow)
        core_acc[ry0:ry1, rx0:rx1] = np.maximum(core_acc[ry0:ry1, rx0:rx1], c_comp)

    out_u8 = _to_u8(img).copy()
    if ux1 > ux0 and uy1 > uy0:
        base = out_u8[uy0:uy1, ux0:ux1].astype(np.float32) / 255.0
        g = np.clip(glow_acc[uy0:uy1, ux0:ux1] * float(intensity), 0.0, 1.0)[:, :, None]
        lit = 1.0 - (1.0 - base) * (1.0 - g * _GLOW_BGR[None, None, :])
        core_alpha = (
            np.clip(core_acc[uy0:uy1, ux0:ux1] * 0.88, 0.0, 0.88) * min(1.0, float(intensity))
        )[:, :, None]
        out = core_alpha * _CORE_BGR[None, None, :] + (1.0 - core_alpha) * lit
        out_u8[uy0:uy1, ux0:ux1] = np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)
    return _restore_layout(img, out_u8)


def _restore_layout(template: np.ndarray, bgr_u8: np.ndarray) -> np.ndarray:
    out_bgr = bgr_u8 if template.dtype == np.uint8 else bgr_u8.astype(template.dtype) / 255.0
    if template.shape[2] == 4:
        out = np.empty_like(template)
        out[:, :, :3] = out_bgr
        out[:, :, 3] = template[:, :, 3]
        return out
    return out_bgr


def split_led_markers(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Returns (image without markers, marker mask, strip metadata)."""
    mask = detect_led_strip_mask(img)
    meta = _extract_strip_metadata(mask)
    return remove_led_strips(img, mask), mask, meta
