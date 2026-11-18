"""Yellow Promob marker lines -> lit LED profiles."""
import cv2
import numpy as np

from app.core.interfaces import EnhancementConfig
from app.pipeline.enhancer import ClassicEnhancer
from app.pipeline.led import detect_led_strip_mask, remove_led_strips, render_led_strips, split_led_markers

MARKER_BGR = (0, 220, 255)  # saturated Promob yellow (RGB 255, 220, 0)


def _cabinet_with_markers():
    """Navy cabinet with a vertical and a horizontal marker, brass handles and an oak shelf."""
    img = np.full((400, 600, 3), (80, 60, 45), dtype=np.uint8)
    img[300:330, 100:500] = (95, 150, 200)  # oak shelf (orange, low saturation)
    cv2.line(img, (150, 40), (160, 280), MARKER_BGR, 5)  # vertical strip, slightly slanted
    cv2.line(img, (250, 200), (520, 200), MARKER_BGR, 4)  # under-shelf strip
    cv2.rectangle(img, (400, 60), (408, 140), (40, 140, 190), -1)  # brass handle
    cv2.circle(img, (60, 360), 14, (60, 200, 235), -1)  # golden knob (small blob)
    return img


def test_detects_marker_lines_only():
    img = _cabinet_with_markers()
    mask = detect_led_strip_mask(img)
    assert mask.dtype == np.uint8 and mask.shape == img.shape[:2]
    assert mask[100:220, 150:162].any(), "Vertical marker missed"
    assert mask[199:202, 300:500].mean() > 200, "Horizontal marker missed"
    assert not mask[60:141, 398:410].any(), "Brass handle is not an LED"
    assert not mask[300:331, 100:500].any(), "Oak shelf is not an LED"
    assert not mask[340:380, 40:80].any(), "Short golden blob is not an LED"


def test_no_markers_returns_empty_mask():
    img = np.full((200, 300, 3), (95, 150, 200), dtype=np.uint8)
    assert not detect_led_strip_mask(img).any()


def test_short_perspective_strips_are_lit_without_expanding_the_emitting_core():
    img = np.full((900, 1600, 3), 50, np.uint8)
    cv2.line(img, (780, 430), (805, 430), MARKER_BGR, 3)
    mask = detect_led_strip_mask(img)
    assert mask[430, 790] == 255, "Short visible shelf segment must not stay yellow"
    assert np.count_nonzero(mask[:, 790]) <= 5, "Inpainting margin must not become an opaque LED tube"
    lit = render_led_strips(remove_led_strips(img, mask), mask)
    bright_core = lit[:, 790].min(axis=1) > 200
    assert 1 <= np.count_nonzero(bright_core) <= 5


def test_curved_yellow_texture_is_not_a_straight_led_profile():
    img = np.full((300, 400, 3), 80, np.uint8)
    cv2.ellipse(img, (100, 150), (25, 12), 0, 0, 140, MARKER_BGR, 2)
    assert not detect_led_strip_mask(img).any()


def test_remove_then_render_turns_marker_into_glow():
    img = _cabinet_with_markers()
    mask = detect_led_strip_mask(img)
    clean = remove_led_strips(img, mask)
    hsv = cv2.cvtColor(clean, cv2.COLOR_BGR2HSV)
    assert hsv[195:206, 300:500, 1].mean() < 120, "Yellow marker survived inpainting"

    lit = render_led_strips(clean, mask)
    core = lit[199:202, 300:500].astype(float).mean(axis=(0, 1))
    assert core.min() > 190, "LED core must read as warm white"
    assert core[2] >= core[0], "LED light must be warm (R >= B)"
    # Glow: surface just below the strip is brighter than far away, and more than above (down-light)
    below = lit[215:225, 350:450].astype(float).mean()
    above = lit[175:185, 350:450].astype(float).mean()
    far = lit[60:70, 450:550].astype(float).mean()
    assert below > far + 15 and below > above


def test_led_helpers_keep_dtype_and_alpha():
    img = _cabinet_with_markers()
    bgra = np.dstack([img, np.full(img.shape[:2], 200, np.uint8)])
    mask = detect_led_strip_mask(bgra)
    out = render_led_strips(remove_led_strips(bgra, mask), mask)
    assert out.shape == bgra.shape and out.dtype == np.uint8
    np.testing.assert_array_equal(out[:, :, 3], bgra[:, :, 3])

    f32 = img.astype(np.float32) / 255.0
    out_f = render_led_strips(f32, detect_led_strip_mask(f32))
    assert out_f.dtype == np.float32 and 0.0 <= out_f.min() and out_f.max() <= 1.0


def test_preview_renders_led_and_can_be_disabled():
    img = _cabinet_with_markers()
    enh = ClassicEnhancer()
    on = enh.process_preview(img, EnhancementConfig(led_strips=True))
    off = enh.process_preview(img, EnhancementConfig(led_strips=False))
    sat_on = cv2.cvtColor(on, cv2.COLOR_BGR2HSV)[199:202, 300:500, 1].mean()
    sat_off = cv2.cvtColor(off, cv2.COLOR_BGR2HSV)[199:202, 300:500, 1].mean()
    assert sat_on < sat_off - 60, "With LED on, the flat yellow stroke must become a white-ish lit core"


def _reference_detect_led_strip_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Reference full-frame NumPy implementation to verify bit-exact equivalence."""
    bgr = img_bgr[:, :, :3]
    if bgr.dtype != np.uint8:
        bgr = np.clip(np.round(bgr * 255.0), 0, 255).astype(np.uint8)
    h, w = bgr.shape[:2]
    s = min(h, w)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    raw = ((hue >= 22) & (hue <= 34) & (sat >= 170) & (val >= 170)).astype(np.uint8) * 255
    if not raw.any():
        return raw
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    mask = np.zeros_like(raw)
    min_len = max(12.0, s * 0.018)
    max_thick = max(7.0, s * 0.012)
    for i in range(1, n_lbl):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 20:
            continue
        pts = np.column_stack(np.nonzero(labels == i))[:, ::-1].astype(np.float32)
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        length = max(rw, rh, 1.0)
        thick = area / max(length, 1.0)
        fill = area / (max(rw, 1.0) * max(rh, 1.0))
        if length >= min_len and thick <= max_thick and length / max(thick, 1.0) >= 6.0 and fill >= 0.55:
            mask[labels == i] = 255
    return mask


def test_detect_led_strip_mask_bit_identical_and_sub20_fast_exit():
    """Optimized inRange + ROI bbox slicing matches full-frame reference bit-for-bit across edge cases."""
    # 1. Sub-20-pixel yellow speckle must fast-exit with an all-zero uint8 mask
    speck = np.full((120, 160, 3), 40, dtype=np.uint8)
    speck[10:13, 20:25] = MARKER_BGR  # 15 pixels (< 20)
    out_speck = detect_led_strip_mask(speck)
    assert out_speck.dtype == np.uint8 and out_speck.shape == (120, 160)
    assert not out_speck.any()
    np.testing.assert_array_equal(out_speck, _reference_detect_led_strip_mask(speck))

    # 2. Mixed strips, boundary HSV values, curves, noise dots, and odd dimensions
    rng = np.random.default_rng(99)
    scene = np.full((373, 541, 3), (75, 65, 55), dtype=np.uint8)
    cv2.line(scene, (40, 30), (55, 310), MARKER_BGR, 4)
    cv2.line(scene, (120, 180), (490, 185), MARKER_BGR, 3)
    cv2.ellipse(scene, (260, 90), (24, 11), 0, 0, 150, MARKER_BGR, 2)
    for _ in range(40):
        cx, cy = int(rng.integers(5, 535)), int(rng.integers(5, 368))
        cv2.circle(scene, (cx, cy), 2, MARKER_BGR, -1)
    # Boundary HSV colors converted to BGR
    for h_val, s_val, v_val, y_pos in [
        (21, 200, 220, 330),
        (22, 170, 170, 338),
        (34, 170, 170, 346),
        (35, 200, 220, 354),
    ]:
        swatch = np.array([[[h_val, s_val, v_val]]], dtype=np.uint8)
        bgr_px = tuple(int(c) for c in cv2.cvtColor(swatch, cv2.COLOR_HSV2BGR)[0, 0])
        cv2.line(scene, (100, y_pos), (260, y_pos), bgr_px, 3)

    np.testing.assert_array_equal(detect_led_strip_mask(scene), _reference_detect_led_strip_mask(scene))
    # Also verify BGRA + float32 inputs match bit-for-bit
    bgra = np.dstack([scene, np.full(scene.shape[:2], 180, np.uint8)])
    np.testing.assert_array_equal(detect_led_strip_mask(bgra), _reference_detect_led_strip_mask(bgra))
    f32 = scene.astype(np.float32) / 255.0
    np.testing.assert_array_equal(detect_led_strip_mask(f32), _reference_detect_led_strip_mask(f32))


def test_split_led_markers_returns_three_tuple_and_metadata():
    img = _cabinet_with_markers()
    cleaned, mask, meta = split_led_markers(img)
    assert cleaned.shape == img.shape and mask.shape == img.shape[:2]
    assert meta["count"] == 2 and meta["has_leds"] is True
    assert meta["image_shape"] == img.shape[:2]
    orientations = {s["orientation"] for s in meta["strips"]}
    assert "horizontal" in orientations and "vertical" in orientations
    for s in meta["strips"]:
        assert s["is_straight"] is True and s["length_px"] > 20.0
        assert len(s["p0"]) == 2 and len(s["p1"]) == 2

    # Marker-free image: zero hallucinated LEDs
    clean_room = np.full((200, 300, 3), (95, 150, 200), dtype=np.uint8)
    _, mask0, meta0 = split_led_markers(clean_room)
    assert not mask0.any()
    assert meta0["count"] == 0 and meta0["has_leds"] is False and meta0["strips"] == []


def test_slanted_led_strip_renders_smooth_without_beads():
    img = np.full((420, 320, 3), 50, dtype=np.uint8)
    p0, p1 = (120, 40), (155, 360)
    cv2.line(img, p0, p1, MARKER_BGR, 3)
    cleaned, mask, meta = split_led_markers(img)
    assert meta["count"] == 1 and meta["strips"][0]["is_straight"] is True

    lit = render_led_strips(cleaned, mask, metadata=meta)
    # Sample along the fitted centerline y=60..340
    sp0, sp1 = meta["strips"][0]["p0"], meta["strips"][0]["p1"]
    vals = []
    for y in range(60, 341):
        t = (y - sp0[1]) / max(1e-6, (sp1[1] - sp0[1]))
        x = int(round(sp0[0] + t * (sp1[0] - sp0[0])))
        vals.append(float(lit[y, x].mean()))
    assert min(vals) > 190.0
    assert float(np.std(vals)) < 3.0


def test_curved_led_strip_fallback_is_continuous_and_safe():
    img = np.full((240, 320, 3), 60, dtype=np.uint8)
    curved_mask = np.zeros((240, 320), dtype=np.uint8)
    cv2.ellipse(curved_mask, (160, 140), (80, 60), 0, 180, 360, 255, 3)
    lit = render_led_strips(img, curved_mask)
    assert lit.shape == img.shape and lit.dtype == np.uint8
    assert float(lit[80, 160].mean()) > 170.0


def test_render_led_strips_handles_resized_metadata_safely():
    img = _cabinet_with_markers()
    _, mask, meta = split_led_markers(img)
    small_img = cv2.resize(img, (300, 200), interpolation=cv2.INTER_AREA)
    # Pass metadata from 400x600 while target image is 200x300
    out = render_led_strips(small_img, mask, metadata=meta)
    assert out.shape == small_img.shape and out.dtype == np.uint8


def test_l_shaped_corner_led_strip_detected_and_rendered_smoothly():
    img = np.full((256, 256, 3), 65, dtype=np.uint8)
    img[60:64, 50:200] = (0, 238, 255)
    img[60:200, 196:200] = (0, 238, 255)

    cleaned, mask, meta = split_led_markers(img)
    assert meta["count"] == 1 and meta["has_leds"] is True
    assert mask[61, 60:195].all(), "Horizontal leg missed"
    assert mask[65:195, 197].all(), "Vertical leg missed"

    # Yellow stroke removed cleanly in cleaned_bgr
    hsv = cv2.cvtColor(cleaned, cv2.COLOR_BGR2HSV)
    assert hsv[60:64, 60:195, 1].mean() < 80
    assert hsv[65:195, 196:200, 1].mean() < 80

    lit = render_led_strips(cleaned, mask, metadata=meta)
    # Sample along thinned centerline of horizontal leg, corner, and vertical leg
    horiz_vals = [float(lit[61, x].min()) for x in range(55, 196)]
    vert_vals = [float(lit[y, 197].min()) for y in range(63, 195)]
    corner_val = float(lit[61, 197].min())
    all_vals = horiz_vals + [corner_val] + vert_vals
    assert min(all_vals) > 190.0, f"Corner or leg dimmed: min={min(all_vals):.1f}"
    assert float(np.std(all_vals)) < 6.0, f"Non-uniform / beaded core: std={float(np.std(all_vals)):.2f}"


def test_u_shaped_led_strip_detected_while_solid_knob_rejected():
    img = np.full((400, 600, 3), 70, dtype=np.uint8)
    img[60:64, 100:400] = MARKER_BGR
    img[60:240, 100:104] = MARKER_BGR
    img[60:240, 396:400] = MARKER_BGR
    cv2.circle(img, (60, 350), 14, MARKER_BGR, -1)  # solid knob must stay rejected

    cleaned, mask, meta = split_led_markers(img)
    assert meta["count"] == 1
    assert mask[61, 120:380].all()
    assert mask[80:220, 101].all()
    assert mask[80:220, 397].all()
    assert not mask[330:370, 40:80].any(), "Solid knob falsely detected as LED"

    lit = render_led_strips(cleaned, mask, metadata=meta)
    assert float(lit[61, 101].min()) > 190.0
    assert float(lit[61, 397].min()) > 190.0
    assert float(lit[150, 101].min()) > 190.0
    assert float(lit[150, 397].min()) > 190.0


def test_diagonal_45deg_4px_aa_l_strip_detected_and_rendered():
    img = np.full((400, 400, 3), 65, dtype=np.uint8)
    cv2.line(img, (80, 200), (200, 80), MARKER_BGR, 4, cv2.LINE_AA)
    cv2.line(img, (200, 80), (320, 200), MARKER_BGR, 4, cv2.LINE_AA)

    cleaned, mask, meta = split_led_markers(img)
    assert meta["count"] == 1
    assert meta["strips"][0]["is_straight"] is False
    assert meta["strips"][0]["length_px"] > 300.0
    assert meta["strips"][0]["thickness_px"] < 7.0

    hsv = cv2.cvtColor(cleaned, cv2.COLOR_BGR2HSV)
    assert hsv[138:143, 138:143, 1].mean() < 60.0

    lit = render_led_strips(cleaned, mask, metadata=meta)
    assert float(lit[80:85, 198:203].min(axis=2).max()) > 210.0
    assert float(lit[140, 140].min()) > 210.0
    assert float(lit[140, 260].min()) > 210.0


def test_short_20px_4px_l_strips_detected_across_resolutions():
    for h, w in [(256, 256), (800, 800), (1080, 1920)]:
        img = np.full((h, w, 3), 65, dtype=np.uint8)
        img[60:64, 60:80] = MARKER_BGR
        img[60:80, 76:80] = MARKER_BGR
        cleaned, mask, meta = split_led_markers(img)
        assert meta["count"] == 1, f"Failed for {(h, w)}: {meta}"
        assert mask[61, 65:79].all()
        assert mask[62:78, 77].all()
        lit = render_led_strips(cleaned, mask, metadata=meta)
        assert float(lit[61, 77].min()) > 190.0


def test_negative_curved_ellipses_and_knobs_rejected():
    for res in [(300, 400), (800, 800)]:
        for axes in [(25, 12), (80, 40), (100, 50)]:
            img = np.full(res + (3,), 80, dtype=np.uint8)
            cv2.ellipse(img, (res[1] // 2, res[0] // 2), axes, 0, 0, 140, MARKER_BGR, 2)
            assert split_led_markers(img)[2]["count"] == 0
            img_arc = np.full(res + (3,), 80, dtype=np.uint8)
            cv2.ellipse(img_arc, (res[1] // 2, res[0] // 2), axes, 0, 180, 360, MARKER_BGR, 3)
            assert split_led_markers(img_arc)[2]["count"] == 0

    for radius in (3, 6, 10, 15, 25, 40):
        knobs = np.full((240, 240, 3), 80, dtype=np.uint8)
        cv2.circle(knobs, (120, 120), radius, MARKER_BGR, -1)
        assert split_led_markers(knobs)[2]["count"] == 0


def test_negative_large_curved_ellipses_and_quarter_ellipse_rejected():
    """Regression: large curved yellow arcs must not pass via multiple short Hough tangent chords."""
    challenger_cases = [
        ((400, 400), (100, 50), 90, 0, 140, 3),
        ((400, 400), (100, 50), 90, 0, 140, 4),
        ((400, 400), (170, 85), 0, 0, 140, 2),
        ((512, 512), (200, 100), 90, 180, 360, 2),
        ((800, 800), (200, 100), 90, 180, 360, 2),
        ((800, 800), (320, 160), 0, 0, 140, 3),
        ((800, 800), (320, 160), 0, 0, 140, 4),
    ]
    for res, axes, angle, start_a, end_a, thick in challenger_cases:
        img = np.full(res + (3,), 80, dtype=np.uint8)
        cv2.ellipse(img, (res[1] // 2, res[0] // 2), axes, angle, start_a, end_a, MARKER_BGR, thick)
        assert split_led_markers(img)[2]["count"] == 0, f"FP on {(res, axes, angle, start_a, end_a, thick)}"

    # Reviewer 2 quarter-ellipse (120, 60) across angles and thicknesses
    for angle in (0, 90):
        for thick in (3, 4):
            img_q = np.full((400, 400, 3), 80, dtype=np.uint8)
            cv2.ellipse(img_q, (200, 200), (120, 60), angle, 0, 90, MARKER_BGR, thick)
            assert split_led_markers(img_q)[2]["count"] == 0


def test_diagonal_aa_angles_and_asymmetric_l_detected():
    for deg in (15, 22, 30, 45, 60):
        rad = np.deg2rad(deg)
        dx, dy = int(round(140 * np.cos(rad))), int(round(140 * np.sin(rad)))
        img = np.full((400, 400, 3), 65, dtype=np.uint8)
        cv2.line(img, (200 - dx, 200 + dy), (200, 200), MARKER_BGR, 4, cv2.LINE_AA)
        cv2.line(img, (200, 200), (200 + dx, 200 + dy), MARKER_BGR, 4, cv2.LINE_AA)
        _, _, meta = split_led_markers(img)
        assert meta["count"] == 1, f"Missed {deg} deg diagonal L-strip"

    asym = np.full((400, 600, 3), 65, dtype=np.uint8)
    asym[60:64, 50:300] = MARKER_BGR
    asym[60:120, 296:300] = MARKER_BGR
    cleaned, mask, meta = split_led_markers(asym)
    assert meta["count"] == 1
    lit = render_led_strips(cleaned, mask, metadata=meta)
    lit_nometa = render_led_strips(cleaned, mask, metadata=None)
    np.testing.assert_array_equal(lit, lit_nometa)


def test_render_led_strips_mismatched_metadata_image_shape_recomputes_labels():
    # Two nearby components: an L-shaped corner strip and a straight strip within far_sigma pad
    img = np.full((300, 300, 3), 65, dtype=np.uint8)
    img[60:64, 60:140] = MARKER_BGR
    img[60:140, 136:140] = MARKER_BGR
    img[160:163, 60:150] = MARKER_BGR
    cleaned, mask, meta = split_led_markers(img)
    assert meta["count"] == 2

    # Pass stale metadata with mismatched image_shape so render_led_strips must recompute meta
    # and use the recomputed label IDs (not merging both components inside the padded ROI).
    stale_meta = {"image_shape": (150, 150), "strips": meta["strips"]}
    out_stale = render_led_strips(cleaned, mask, metadata=stale_meta)
    out_fresh = render_led_strips(cleaned, mask, metadata=meta)
    np.testing.assert_array_equal(out_stale, out_fresh)
    # Center between strips must remain dark (not turned into an LED core)
    assert float(out_stale[110, 90].max()) < 140.0


def test_diagonal_aa_multiresolution_l_strips_detected():
    """66-case multi-resolution diagonal AA L-strip regression matrix + 3 representative split+render checks."""
    specs = [
        ((400, 400), 200, 160, (80, 100, 120, 140, 160)),
        ((800, 800), 400, 250, (280,)),
        ((1080, 1920), 960, 250, (80, 140, 250, 378, 450)),
    ]
    angles = (15, 22, 30, 35, 45, 60)

    for (h, w), vx, vy, arms in specs:
        buf = np.empty((h, w, 3), dtype=np.uint8)
        for arm in arms:
            for deg in angles:
                buf.fill(65)
                rad = np.deg2rad(deg)
                dx = int(round(arm * np.cos(rad)))
                dy = int(round(arm * np.sin(rad)))
                cv2.line(buf, (vx - dx, vy + dy), (vx, vy), MARKER_BGR, 4, cv2.LINE_AA)
                cv2.line(buf, (vx, vy), (vx + dx, vy + dy), MARKER_BGR, 4, cv2.LINE_AA)
                mask = detect_led_strip_mask(buf)
                assert mask.any(), f"Missed L-strip for res={(h, w)}, arm={arm}, deg={deg}"
                assert mask[vy, vx] == 255, f"Vertex missed for res={(h, w)}, arm={arm}, deg={deg}"
                n_cc, _ = cv2.connectedComponents(mask)
                assert n_cc == 2, f"Expected 1 connected L-component for {(h, w, arm, deg)}, got {n_cc - 1}"

    # 3 representative end-to-end split_led_markers + render_led_strips checks at formerly failing geometries
    e2e_cases = [
        ((400, 400), 200, 160, 140, 35),
        ((800, 800), 400, 250, 280, 60),
        ((1080, 1920), 960, 250, 378, 30),
    ]
    for (h, w), vx, vy, arm, deg in e2e_cases:
        img = np.full((h, w, 3), 65, dtype=np.uint8)
        rad = np.deg2rad(deg)
        dx = int(round(arm * np.cos(rad)))
        dy = int(round(arm * np.sin(rad)))
        cv2.line(img, (vx - dx, vy + dy), (vx, vy), MARKER_BGR, 4, cv2.LINE_AA)
        cv2.line(img, (vx, vy), (vx + dx, vy + dy), MARKER_BGR, 4, cv2.LINE_AA)

        cleaned, mask, meta = split_led_markers(img)
        assert meta["count"] == 1, f"Expected count=1 for {(h, w, arm, deg)}, got {meta}"
        hsv = cv2.cvtColor(cleaned, cv2.COLOR_BGR2HSV)
        assert hsv[vy - 2 : vy + 3, vx - 2 : vx + 3, 1].mean() < 80.0

        lit = render_led_strips(cleaned, mask, metadata=meta)
        for px, py in (
            (vx, vy),
            (vx - dx // 2, vy + dy // 2),
            (vx + dx // 2, vy + dy // 2),
        ):
            peak = float(lit[py - 4 : py + 5, px - 4 : px + 5].min(axis=2).max())
            assert peak > 190.0, f"Unlit point {(px, py)} for {(h, w, arm, deg)}: peak={peak:.1f}"


def test_render_led_strips_ignores_nonpositive_label_id_and_preserves_outside_union_roi():
    """Non-positive strip id (0, -1) must not invert background; out-of-bounds bbox and outside-ROI pixels stay intact."""
    img = np.full((160, 160, 3), 50, dtype=np.uint8)
    img[60:64, 60:110] = MARKER_BGR
    img[60:110, 106:110] = MARKER_BGR

    cleaned, mask, meta = split_led_markers(img)
    assert meta["count"] == 1
    expected = render_led_strips(cleaned, mask, metadata=meta)

    for bad_id in (0, -1):
        forged_meta = {
            **meta,
            "strips": [{**meta["strips"][0], "id": bad_id}],
        }
        out = render_led_strips(cleaned, mask, metadata=forged_meta)
        np.testing.assert_array_equal(out, expected)
        assert out[10, 10].tolist() == [50, 50, 50]

    # Outside union-ROI pixels must remain bit-identical to cleaned
    np.testing.assert_array_equal(expected[:5, :5], cleaned[:5, :5])
    np.testing.assert_array_equal(expected[-5:, -5:], cleaned[-5:, -5:])

    # Out-of-bounds bbox must safely return input without error
    oob_meta = {
        "count": 1,
        "has_leds": True,
        "image_shape": (160, 160),
        "strips": [{**meta["strips"][0], "bbox": (-200, -200, -150, -150)}],
    }
    out_oob = render_led_strips(cleaned, mask, metadata=oob_meta)
    np.testing.assert_array_equal(out_oob, cleaned)

    # 3D single-channel mask is handled defensively
    out_3d = render_led_strips(cleaned, mask[:, :, None], metadata=meta)
    np.testing.assert_array_equal(out_3d, expected)
