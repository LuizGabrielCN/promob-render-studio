"""Unit tests for M1 Scene Depth & 2D Illumination Light-Map Synthesis (CPU-fast, offline)."""
import os
import warnings
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from app.core.hardware import HardwareGuardrails
from app.pipeline import lighting_map as LM
from app.pipeline import objects as O


def _synthetic_room_with_spots_and_doorway():
    """
    240x320 synthetic room:
    - Left side (x=0..200): recessed ceiling (y=0..50, depth 0.25->0.70) above vertical wall (y=50..240, depth=0.70)
    - Spot A at (x=100, y=30) above vertical wall (depth=0.70)
    - Right side (x=200..320): deep doorway/air (depth=0.10) with Spot B at (x=260, y=30) on foreground soffit (depth=0.88)
    """
    h, w = 240, 320
    img = np.full((h, w, 3), 130, dtype=np.uint8)
    depth = np.full((h, w), 0.70, dtype=np.float32)

    # Ceiling gradient on left
    depth[:50, :200] = np.linspace(0.25, 0.70, 50, dtype=np.float32)[:, None]
    # Deep doorway on right
    img[50:, 200:] = 85
    depth[50:, 200:] = 0.10
    depth[:50, 200:] = 0.88

    # Bright ceiling spots
    cv2.circle(img, (100, 30), 6, (255, 255, 255), -1)
    cv2.circle(img, (260, 30), 6, (255, 255, 255), -1)
    return img, depth


def test_scene_lighting_analysis_contract_and_dtypes():
    img, depth = _synthetic_room_with_spots_and_doorway()
    ana = LM.analyze_scene_lighting(img, depth_map=depth, allow_depth_model=False, allow_object_detection=False)

    assert isinstance(ana, LM.SceneLightingAnalysis)
    assert ana.depth_map.shape == img.shape[:2] and ana.depth_map.dtype == np.float32
    assert 0.0 <= float(ana.depth_map.min()) and float(ana.depth_map.max()) <= 1.0

    assert ana.illumination_map_bgr.shape == img.shape and ana.illumination_map_bgr.dtype == np.uint8
    assert ana.spot_mask.shape == img.shape[:2] and ana.spot_mask.dtype == np.uint8
    assert set(np.unique(ana.spot_mask)).issubset({0, 255})

    assert ana.window_mask.shape == img.shape[:2] and ana.window_mask.dtype == np.uint8
    assert set(np.unique(ana.window_mask)).issubset({0, 255})

    assert isinstance(ana.fireplace_boxes, list)
    for box in ana.fireplace_boxes:
        assert len(box) == 4 and all(isinstance(v, int) for v in box)
        assert box[2] > box[0] and box[3] > box[1]

    assert ana.protection_mask.shape == img.shape[:2] and ana.protection_mask.dtype == np.float32
    assert ana.text_logo_mask.shape == img.shape[:2] and ana.text_logo_mask.dtype == np.float32
    assert np.all(ana.protection_mask >= ana.text_logo_mask - 1e-6)


def test_input_validation_and_float32_bgra_tiny_shapes():
    for bad in [
        None,
        "not_an_array",
        np.zeros((0, 0, 3), dtype=np.uint8),
        np.zeros((64, 64), dtype=np.uint8),
        np.zeros((32, 32, 1), dtype=np.uint8),
        np.zeros((32, 32, 3), dtype=np.float64),
        np.zeros((32, 32, 3), dtype=np.int32),
    ]:
        with pytest.raises(ValueError):
            LM.analyze_scene_lighting(bad, allow_depth_model=False, allow_object_detection=False)

    # float32 with NaN/Inf must sanitize without RuntimeWarning
    f32 = np.full((64, 64, 3), 0.5, dtype=np.float32)
    f32[0, 0, 0] = np.nan
    f32[0, 1, 1] = np.inf
    f32[0, 2, 2] = -np.inf
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always", RuntimeWarning)
        ana_f = LM.analyze_scene_lighting(f32, allow_depth_model=False, allow_object_detection=False)
    assert not [w for w in rec if issubclass(w.category, RuntimeWarning)]
    assert ana_f.illumination_map_bgr.shape == (64, 64, 3)

    # BGRA & odd/tiny/sliver dimensions
    for shape in [(2, 2, 4), (5, 5, 3), (237, 319, 3), (64, 1920, 3)]:
        arr = np.full(shape, 120, dtype=np.uint8)
        ana = LM.analyze_scene_lighting(arr, allow_depth_model=False, allow_object_detection=False)
        assert ana.depth_map.shape == shape[:2]
        assert ana.illumination_map_bgr.shape == (shape[0], shape[1], 3)


def test_depth_model_mocked_execution_and_memory_release(monkeypatch):
    import torch

    monkeypatch.setenv(LM.DEPTH_ENV_VAR, "1")
    release_calls = {"n": 0}
    monkeypatch.setattr(HardwareGuardrails, "release_memory", lambda: release_calls.__setitem__("n", release_calls["n"] + 1))

    class FakeProc:
        def __call__(self, images, return_tensors="pt"):
            return {"pixel_values": torch.zeros((1, 3, 16, 16), dtype=torch.float32)}

        def post_process_depth_estimation(self, outputs, target_sizes):
            h, w = target_sizes[0]
            grad = torch.linspace(0.0, 10.0, h).unsqueeze(1).expand(h, w)
            return [{"predicted_depth": grad}]

    class FakeModel:
        def __call__(self, **kwargs):
            return {"predicted_depth": torch.zeros((1, 16, 16))}

    monkeypatch.setattr(LM, "_load_depth_model", lambda: (FakeProc(), FakeModel(), "cpu", torch))
    img = np.full((60, 80, 3), 120, dtype=np.uint8)
    depth, src = LM.estimate_depth_map(img, allow_model=True)
    assert src == "depth_anything_v2"
    assert depth.shape == (60, 80) and depth.dtype == np.float32
    assert float(depth[-1, 0]) > float(depth[0, 0]) + 0.8
    assert release_calls["n"] >= 1

    # Also release when inference raises
    def boom_model(**kwargs):
        raise RuntimeError("cuda oom simulation")

    release_calls["n"] = 0
    monkeypatch.setattr(LM, "_load_depth_model", lambda: (FakeProc(), boom_model, "cpu", torch))
    depth_fb, src_fb = LM.estimate_depth_map(img, allow_model=True)
    assert src_fb == "heuristic"
    assert depth_fb.shape == (60, 80)
    assert release_calls["n"] >= 1


def test_depth_model_unavailable_falls_back_gracefully(monkeypatch):
    monkeypatch.setenv(LM.DEPTH_ENV_VAR, "1")
    release_calls = {"n": 0}
    monkeypatch.setattr(HardwareGuardrails, "release_memory", lambda: release_calls.__setitem__("n", release_calls["n"] + 1))

    def missing():
        raise OSError("no local cache")

    monkeypatch.setattr(LM, "_load_depth_model", missing)
    img = np.linspace(40, 200, 80 * 100 * 3, dtype=np.uint8).reshape(80, 100, 3)
    ana = LM.analyze_scene_lighting(img, allow_depth_model=True, allow_object_detection=False)
    assert ana.depth_map.dtype == np.float32 and np.isfinite(ana.depth_map).all()
    assert 0.0 <= float(ana.depth_map.min()) and float(ana.depth_map.max()) <= 1.0
    assert release_calls["n"] >= 1


def test_ceiling_spot_projects_downward_parabolic_cone_only_on_adjacent_wall():
    img, depth = _synthetic_room_with_spots_and_doorway()
    ana = LM.analyze_scene_lighting(img, depth_map=depth, allow_depth_model=False, allow_object_detection=False)

    assert ana.spot_mask[30, 100] == 255
    illum = ana.illumination_map_bgr.astype(np.float32)

    # Below Spot A on adjacent vertical wall (depth=0.70): strong warm cone
    wall_patch = illum[70:115, 85:115]
    # Deep doorway (depth=0.10) next to wall & below Spot B (empty air): no floating cone
    doorway_patch = illum[70:115, 245:275]
    ambient_far = illum[180:220, 20:50]

    assert float(wall_patch.mean()) > float(ambient_far.mean()) + 20.0
    assert float(wall_patch[:, :, 2].mean()) > float(wall_patch[:, :, 0].mean()) + 15.0  # warm R > B
    assert float(wall_patch.mean()) > float(doorway_patch.mean()) + 25.0


def test_dark_mirror_reflections_rejected_from_spot_mask():
    h, w = 240, 320
    img = np.full((h, w, 3), 185, dtype=np.uint8)
    # Real ceiling spot on bright ceiling
    cv2.circle(img, (80, 24), 5, (255, 255, 255), -1)
    # Dark bronze mirror box (s06 style) with bright spot reflections inside
    img[70:170, 120:260] = 45
    cv2.circle(img, (160, 100), 4, (250, 250, 250), -1)
    cv2.circle(img, (210, 100), 4, (250, 250, 250), -1)

    spot_mask = LM.detect_spot_mask(img)
    assert spot_mask[24, 80] == 255
    assert spot_mask[100, 160] == 0
    assert spot_mask[100, 210] == 0


def test_wall_sconces_and_pendants_emit_local_radial_glow():
    h, w = 240, 320
    img = np.full((h, w, 3), 150, dtype=np.uint8)
    depth = np.full((h, w), 0.65, dtype=np.float32)  # vertical wall everywhere

    # Vertical sconce (s17 style) at x=100, y=70..110 (bh >= 2*bw)
    cv2.rectangle(img, (98, 70), (102, 110), (255, 255, 255), -1)
    # Hanging pendant bulb (s23 style) at x=220, y=75 (cy >= 0.24 H in front of vertical wall)
    cv2.circle(img, (220, 75), 5, (255, 255, 255), -1)

    ana = LM.analyze_scene_lighting(img, depth_map=depth, allow_depth_model=False, allow_object_detection=False)
    assert ana.spot_mask[90, 100] == 255
    assert ana.spot_mask[75, 220] == 255

    illum = ana.illumination_map_bgr.astype(np.float32)
    sconce_glow = illum[80:100, 108:124]
    pendant_glow = illum[65:85, 230:246]
    far_wall = illum[190:220, 20:50]
    assert float(sconce_glow.mean()) > float(far_wall.mean()) + 20.0
    assert float(pendant_glow.mean()) > float(far_wall.mean()) + 20.0
    assert float(sconce_glow[:, :, 2].mean()) > float(sconce_glow[:, :, 0].mean())


def test_fireplace_box_synthesizes_warm_core_and_hearth_floor_bounce():
    h, w = 260, 320
    img = np.full((h, w, 3), 150, dtype=np.uint8)
    depth = np.full((h, w), 0.60, dtype=np.float32)
    depth[210:, :] = np.linspace(0.60, 0.85, h - 210, dtype=np.float32)[:, None]

    # Unlit tan firebox inside darker surround
    img[150:210, 120:200] = (55, 95, 130)  # warm tan BGR
    ana = LM.analyze_scene_lighting(
        img,
        depth_map=depth,
        fireplace_boxes=[(120, 150, 200, 210)],
        allow_depth_model=False,
        allow_object_detection=False,
    )
    illum = ana.illumination_map_bgr.astype(np.float32)
    core = illum[175:205, 145:175]
    hearth = illum[215:245, 135:185]
    far_wall = illum[80:120, 20:60]
    assert float(core[:, :, 2].mean() - core[:, :, 0].mean()) > 40.0
    assert float(hearth.mean()) > float(far_wall.mean()) + 15.0

    # Heuristic fallback also detects unlit tan firebox when fireplace_boxes=None
    heu_boxes = LM._detect_fireplace_boxes_heuristic(img, depth)
    assert len(heu_boxes) >= 1


def test_window_mask_detects_daylight_and_excludes_tv_screen():
    h, w = 240, 320
    img = np.full((h, w, 3), 120, dtype=np.uint8)
    depth = np.full((h, w), 0.70, dtype=np.float32)

    # Bright exterior opening on right
    img[60:180, 245:310] = (230, 235, 225)
    depth[60:180, 245:310] = 0.20

    # Bright blue mountain-lake TV screen on left (s23 style)
    img[75:155, 10:70] = (235, 195, 165)
    depth[75:155, 10:70] = 0.72
    prot = np.zeros((h, w), dtype=np.float32)
    prot[75:155, 10:70] = 1.0

    ana = LM.analyze_scene_lighting(
        img,
        depth_map=depth,
        protection_mask=prot,
        allow_depth_model=False,
        allow_object_detection=False,
    )
    assert ana.window_mask[100:140, 260:295].mean() > 200
    assert not ana.window_mask[75:155, 10:70].any()

    # Inward cool-neutral daylight fill next to right window
    near_win = ana.illumination_map_bgr[100:140, 210:240].astype(np.float32)
    far_corner = ana.illumination_map_bgr[190:220, 20:50].astype(np.float32)
    assert float(near_win[:, :, 0].mean()) > float(far_corner[:, :, 0].mean()) + 12.0


def test_led_mask_synthesizes_depth_gated_warm_linear_spill():
    h, w = 240, 320
    img = np.full((h, w, 3), 90, dtype=np.uint8)
    depth = np.full((h, w), 0.55, dtype=np.float32)
    # Foreground column adjacent to niche
    depth[:, 195:] = 0.95
    # Promob yellow LED stroke inside recessed niche
    cv2.line(img, (60, 90), (185, 90), (0, 220, 255), 3)

    ana = LM.analyze_scene_lighting(img, depth_map=depth, allow_depth_model=False, allow_object_detection=False)
    illum = ana.illumination_map_bgr.astype(np.float32)

    niche_below = illum[94:108, 90:150]
    fg_col = illum[94:108, 210:250]
    assert float(niche_below[:, :, 2].mean()) > float(niche_below[:, :, 0].mean()) + 10.0
    assert float(niche_below.mean()) > float(fg_col.mean()) + 15.0


def test_uniform_image_stability():
    flat = np.full((128, 128, 3), 128, dtype=np.uint8)
    ana = LM.analyze_scene_lighting(flat, allow_depth_model=False, allow_object_detection=False)
    assert not ana.spot_mask.any()
    assert not ana.window_mask.any()
    assert ana.fireplace_boxes == []
    assert np.isfinite(ana.depth_map).all()
    assert np.isfinite(ana.illumination_map_bgr).all()


def test_objects_module_integration_supports_extended_and_legacy_result(monkeypatch):
    monkeypatch.setenv(LM.OBJECT_ENV_VAR, "1")
    img = np.full((120, 160, 3), 120, dtype=np.uint8)

    # 1. Legacy DetectionResult without text_logo_mask / fireplace_boxes
    legacy_mask = np.zeros((120, 160), dtype=np.float32)
    legacy_mask[20:40, 20:40] = 0.9
    legacy = O.DetectionResult(mask=legacy_mask, n_objects=1, labels=["bottle"])
    monkeypatch.setattr(O, "detect_objects", lambda _: legacy)
    ana1 = LM.analyze_scene_lighting(img, allow_depth_model=False, allow_object_detection=True)
    assert ana1.protection_mask[30, 30] > 0.8
    assert ana1.text_logo_mask.max() == 0.0

    # 2. Extended M1 DetectionResult with text_logo_mask + fireplace_boxes
    txt = np.zeros((120, 160), dtype=np.float32)
    txt[50:60, 50:80] = 1.0
    ext = O.DetectionResult(
        mask=legacy_mask.copy(),
        n_objects=1,
        labels=["bottle"],
        text_logo_mask=txt,
        fireplace_boxes=[(60, 80, 100, 110)],
    )
    monkeypatch.setattr(O, "detect_objects", lambda _: ext)
    ana2 = LM.analyze_scene_lighting(img, allow_depth_model=False, allow_object_detection=True)
    assert np.all(ana2.protection_mask >= ana2.text_logo_mask - 1e-6)
    assert ana2.text_logo_mask[55, 65] == 1.0
    assert ana2.fireplace_boxes == [(60, 80, 100, 110)]


def test_estimate_depth_map_downscales_over_768_and_restores_shape(monkeypatch):
    import torch

    monkeypatch.setenv(LM.DEPTH_ENV_VAR, "1")
    seen_sizes = []

    class FakeProc:
        def __call__(self, images, return_tensors="pt"):
            seen_sizes.append(images.size)  # (width, height)
            return {"pixel_values": torch.zeros((1, 3, 16, 16), dtype=torch.float32)}

        def post_process_depth_estimation(self, outputs, target_sizes):
            h, w = target_sizes[0]
            grad = torch.linspace(0.0, 5.0, h).unsqueeze(1).expand(h, w)
            return [{"predicted_depth": grad}]

    class FakeModel:
        def __call__(self, **kwargs):
            return {"predicted_depth": torch.zeros((1, 16, 16))}

    monkeypatch.setattr(LM, "_load_depth_model", lambda: (FakeProc(), FakeModel(), "cpu", torch))
    large_img = np.full((900, 1536, 3), 120, dtype=np.uint8)
    depth, src = LM.estimate_depth_map(large_img, allow_model=True)
    assert src == "depth_anything_v2"
    assert depth.shape == (900, 1536) and depth.dtype == np.float32
    assert seen_sizes == [(LM.MAX_DEPTH_LONG_SIDE, 450)]


def test_analyze_scene_lighting_releases_memory_on_objects_success_and_exception(monkeypatch):
    monkeypatch.setenv(LM.OBJECT_ENV_VAR, "1")
    img = np.full((120, 160, 3), 120, dtype=np.uint8)
    release_calls = {"n": 0}
    monkeypatch.setattr(HardwareGuardrails, "release_memory", lambda: release_calls.__setitem__("n", release_calls["n"] + 1))

    # 1. Success path
    monkeypatch.setattr(
        O,
        "detect_objects",
        lambda _: O.DetectionResult(mask=np.zeros((120, 160), dtype=np.float32), n_objects=0),
    )
    LM.analyze_scene_lighting(img, allow_depth_model=False, allow_object_detection=True)
    assert release_calls["n"] >= 1

    # 2. Exception path
    release_calls["n"] = 0

    def boom(_):
        raise RuntimeError("simulated dino/sam failure")

    monkeypatch.setattr(O, "detect_objects", boom)
    ana = LM.analyze_scene_lighting(img, allow_depth_model=False, allow_object_detection=True)
    assert release_calls["n"] >= 1
    assert ana.protection_mask.shape == (120, 160)


def test_caller_supplied_float32_led_mask_mismatch_shape_preserved():
    h, w = 240, 320
    img = np.full((h, w, 3), 85, dtype=np.uint8)
    depth = np.full((h, w), 0.60, dtype=np.float32)

    # Half-res float32 mask with values < 1.0 (0.8) must not truncate to 0 on resize
    half_led = np.zeros((120, 160), dtype=np.float32)
    half_led[45:47, 30:110] = 0.8

    ana = LM.analyze_scene_lighting(
        img,
        depth_map=depth,
        led_mask=half_led,
        allow_depth_model=False,
        allow_object_detection=False,
    )
    illum = ana.illumination_map_bgr.astype(np.float32)
    under_led = illum[92:108, 90:190]
    far_corner = illum[190:220, 20:50]
    assert float(under_led.mean()) > float(far_corner.mean()) + 15.0
    assert float(under_led[:, :, 2].mean()) > float(under_led[:, :, 0].mean()) + 10.0


def test_luminaire_boxes_forwarded_to_spot_classification(monkeypatch):
    monkeypatch.setenv(LM.OBJECT_ENV_VAR, "1")
    h, w = 240, 320
    img = np.full((h, w, 3), 130, dtype=np.uint8)
    # Deep doorway behind hanging pendant so gy_s alone wouldn't detect a wall behind it
    depth = np.full((h, w), 0.70, dtype=np.float32)
    depth[:50, :] = 0.25
    depth[50:, 180:260] = 0.10
    cv2.circle(img, (220, 68), 5, (255, 255, 255), -1)  # cy=68 >= 0.22*240=52.8

    det = O.DetectionResult(
        mask=np.zeros((h, w), dtype=np.float32),
        n_objects=0,
        luminaire_boxes=[(205, 55, 235, 85)],
    )
    monkeypatch.setattr(O, "detect_objects", lambda _: det)

    captured_kinds = []
    orig_synth = LM.synthesize_illumination_map

    def spy_synth(**kwargs):
        captured_kinds.extend(c["kind"] for c in kwargs["spot_components"])
        return orig_synth(**kwargs)

    monkeypatch.setattr(LM, "synthesize_illumination_map", spy_synth)
    ana = LM.analyze_scene_lighting(img, depth_map=depth, allow_depth_model=False, allow_object_detection=True)
    assert ana.spot_mask[68, 220] == 255
    assert "sconce_pendant" in captured_kinds


def test_out_of_bounds_fireplace_boxes_discarded_without_corner_glow():
    img = np.full((200, 300, 3), 120, dtype=np.uint8)
    depth = np.full((200, 300), 0.65, dtype=np.float32)

    baseline = LM.analyze_scene_lighting(
        img,
        depth_map=depth,
        fireplace_boxes=[],
        allow_depth_model=False,
        allow_object_detection=False,
    )
    oob = LM.analyze_scene_lighting(
        img,
        depth_map=depth,
        fireplace_boxes=[(400, 300, 500, 400), (-50, -40, -10, -5), (300, 100, 350, 150)],
        allow_depth_model=False,
        allow_object_detection=False,
    )
    assert oob.fireplace_boxes == []
    np.testing.assert_array_equal(oob.illumination_map_bgr, baseline.illumination_map_bgr)
