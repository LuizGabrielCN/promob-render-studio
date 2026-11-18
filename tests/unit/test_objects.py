"""Testes do detector de objetos (sem GPU, sem modelos reais)."""
import numpy as np
import pytest

from app.pipeline import objects as O


@pytest.fixture(autouse=True)
def _clean():
    O.clear_cache()
    yield
    O.clear_cache()


def _img():
    img = np.full((120, 200, 3), 200, np.uint8)
    return img


def test_returns_none_without_models(monkeypatch):
    def boom(*a, **k):
        raise OSError("no local cache")
    monkeypatch.setattr(O, "_Models", boom)
    assert O.detect_object_mask(_img()) is None
    assert O.detect_objects(_img()) is None


def test_tiles_cover_image_with_overlap():
    t = O._tiles(900, 1600)
    assert t[0] == (0, 0, 1600, 900) and len(t) == 5
    xs = {(a, c) for a, _, c, _ in t[1:]}
    assert min(a for a, _ in xs) == 0 and max(c for _, c in xs) == 1600
    left, right = sorted(xs)
    assert left[1] > right[0]  # overlap


def test_nms_removes_duplicates():
    b = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], np.float32)
    s = np.array([0.9, 0.5, 0.4], np.float32)
    assert O._nms(b, s, 0.5) == [0, 2]


def test_filter_drops_big_tiny_and_flat_boxes():
    img = np.full((200, 400, 3), 128, np.uint8)
    # objeto real: quadrado com borda densa
    img[50:80, 50:80] = np.random.RandomState(0).randint(0, 255, (30, 30, 3), np.uint8)
    boxes = [(50, 50, 80, 80), (0, 0, 300, 150), (10, 10, 13, 40), (200, 100, 230, 130)]
    kb, ks, kl = O.filter_boxes(boxes, [0.4] * 4, ["bottle"] * 4, img)
    assert kb == [(50, 50, 80, 80)]  # grande, fino e liso (sem borda) caem


def test_filter_drops_container_with_inner_boxes():
    img = np.random.RandomState(1).randint(0, 255, (400, 400, 3), np.uint8)
    boxes = [(100, 100, 190, 190), (110, 110, 130, 130), (140, 140, 160, 160)]
    kb, _, _ = O.filter_boxes(boxes, [0.5] * 3, ["glass"] * 3, img)
    assert (100, 100, 190, 190) not in kb and len(kb) == 2


def test_detect_with_fake_models_and_cache(monkeypatch):
    calls = {"n": 0}

    class Fake:
        torch = __import__("torch")
        sam = None  # força fallback de retângulo
        def release(self):
            pass

    def fake_boxes(m, rgb):
        calls["n"] += 1
        return [[40, 30, 60, 60]], [0.8], ["bottle"]

    img = np.random.RandomState(2).randint(0, 255, (120, 200, 3), np.uint8)
    monkeypatch.setattr(O, "_Models", lambda: Fake())
    monkeypatch.setattr(O, "_detect_boxes", fake_boxes)
    m = O.detect_object_mask(img)
    assert m.dtype == np.float32 and m.shape == (120, 200)
    assert 0.0 <= m.min() and m.max() <= 1.0 and m[45, 50] > 0.9 and m[5, 5] == 0
    meta = O.last_metadata()
    assert meta["n_objects"] == 1 and meta["labels"] == ["bottle"] and meta["sam_ok"] is False
    O.detect_object_mask(img)
    assert calls["n"] == 1  # cache por hash


def test_cache_hit_restores_all_metadata_and_reports_current_call_cost():
    first = _img()
    second = first.copy()
    second[0, 0] = 0
    for img, label, sam_ok in ((first, "bottle", True), (second, "plant", False)):
        O._cache[O._hash(img)] = O.DetectionResult(
            mask=np.ones(img.shape[:2], np.float32), n_objects=1, labels=[label],
            elapsed_s=12.0, vram_peak_mb=2048.0, sam_ok=sam_ok,
        )
    O.detect_object_mask(second)
    result = O.detect_objects(first)
    meta = O.last_metadata()
    assert meta["labels"] == ["bottle"] and meta["sam_ok"] is True
    assert meta["cache_hit"] is True and meta["vram_peak_mb"] == 0.0
    assert meta["elapsed_s"] < 1.0 and result.elapsed_s < 1.0
    assert O._cache[O._hash(first)].elapsed_s == 12.0


def test_unavailable_detector_does_not_reuse_previous_metadata(monkeypatch):
    O._last_meta.update(n_objects=10, labels=["bottle"], elapsed_s=8.0)
    def unavailable():
        raise OSError("missing local models")
    monkeypatch.setattr(O, "_Models", unavailable)
    assert O.detect_objects(_img()) is None
    assert O.last_metadata() == {}


def test_detect_tv_text_logo_and_fireplace_contracts(monkeypatch):
    class Fake:
        torch = __import__("torch")
        sam = None
        def release(self):
            pass

    img = np.full((240, 320, 3), 128, np.uint8)
    # Dark TV rectangle with bright inner logo
    img[40:110, 80:180] = 25
    img[65:85, 110:150] = 240
    # Dense bottle edges & fireplace edges so _edge_ratio passes (area_frac <= 0.06)
    rng = np.random.RandomState(7)
    img[135:185, 110:165] = rng.randint(40, 220, (50, 55, 3), np.uint8)
    img[60:110, 220:245] = rng.randint(20, 240, (50, 25, 3), np.uint8)

    def fake_boxes(m, rgb):
        return (
            [[80, 40, 180, 110], [110, 135, 165, 185], [220, 60, 245, 110]],
            [0.65, 0.72, 0.80],
            ["television tv monitor screen", "fireplace", "bottle"],
        )

    fake_poly = np.array([[110, 65], [150, 65], [150, 85], [110, 85]], np.int32)
    monkeypatch.setattr(O, "_Models", lambda: Fake())
    monkeypatch.setattr(O, "_detect_boxes", fake_boxes)
    monkeypatch.setattr(O, "_detect_ocr_boxes", lambda rgb, tvs: ([fake_poly], ["LOGO"], True))

    res = O.detect_objects(img)
    assert res is not None
    assert res.tv_boxes == [(80, 40, 180, 110)]
    assert res.fireplace_boxes == [(110, 135, 165, 185)]
    assert res.ocr_ok is True and res.ocr_texts == ["LOGO"]
    # Fireplace must NOT be frozen in protection_mask
    assert float(res.mask[145:175, 120:155].max()) == 0.0
    # TV & bottle & OCR logo must be protected, and protection_mask >= text_logo_mask everywhere
    assert float(res.mask[75, 130]) > 0.9
    assert float(res.mask[85, 232]) > 0.9
    assert float(res.text_logo_mask[75, 130]) > 0.8
    assert np.all(res.mask >= res.text_logo_mask - 1e-6)


def test_suppresses_bar_stools_and_pendants_without_freezing(monkeypatch):
    class Fake:
        torch = __import__("torch")
        sam = None
        def release(self):
            pass

    img = np.random.RandomState(3).randint(30, 220, (240, 320, 3), np.uint8)

    def fake_boxes(m, rgb):
        return (
            [
                [30, 120, 90, 220],   # bar stool (area=0.078)
                [35, 125, 85, 155],   # overlapping towel false positive
                [140, 20, 195, 90],   # pendant light (area=0.050)
                [145, 30, 190, 85],   # overlapping vase false positive
                [240, 110, 265, 165], # real bottle
            ],
            [0.52, 0.48, 0.45, 0.41, 0.78],
            ["bar stool stool", "towel", "pendant light chandelier lamp", "glass vase", "bottle"],
        )

    monkeypatch.setattr(O, "_Models", lambda: Fake())
    monkeypatch.setattr(O, "_detect_boxes", fake_boxes)
    monkeypatch.setattr(O, "_detect_ocr_boxes", lambda rgb, tvs: ([], [], False))

    res = O.detect_objects(img)
    assert res is not None
    assert res.n_objects == 1 and res.labels == ["bottle"]
    assert float(res.mask[135:200, 45:75].max()) == 0.0   # stool & towel not frozen
    assert float(res.mask[40:75, 155:180].max()) == 0.0   # pendant & vase not frozen
    assert float(res.mask[135, 252]) > 0.9                # bottle protected


def test_fireplace_rejects_compound_grill_and_large_bench():
    img = np.random.RandomState(4).randint(0, 255, (300, 400, 3), np.uint8)
    boxes = [
        [20, 120, 100, 200],   # BBQ grill compound label, low score
        [50, 200, 350, 296],   # wide bottom bench (area=0.24 > 0.06 and bottom border)
        [160, 140, 220, 195],  # valid firebox (area=0.0275, score=0.46)
    ]
    classified = O.classify_and_filter_boxes(
        boxes,
        [0.31, 0.35, 0.46],
        ["monitor fireplace", "fireplace", "fireplace"],
        img,
    )
    assert classified["fireplace_boxes"] == [(160, 140, 220, 195)]
    assert classified["tv_boxes"] == []


def test_ocr_fallback_when_rapidocr_unavailable(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_rapidocr(name, *args, **kwargs):
        if name == "rapidocr_onnxruntime":
            raise ImportError("simulated missing rapidocr")
        return real_import(name, *args, **kwargs)

    class Fake:
        torch = __import__("torch")
        sam = None
        def release(self):
            pass

    img = np.full((200, 300, 3), 120, np.uint8)
    img[30:90, 80:200] = 20
    img[50:70, 115:165] = 245  # geometric logo inside TV

    monkeypatch.setattr(builtins, "__import__", no_rapidocr)
    monkeypatch.setattr(O, "_Models", lambda: Fake())
    monkeypatch.setattr(
        O,
        "_detect_boxes",
        lambda m, rgb: ([[80, 30, 200, 90], [110, 48, 170, 72]], [0.75, 0.45], ["television", "logo"]),
    )

    res = O.detect_objects(img)
    assert res is not None
    assert res.ocr_ok is False
    assert float(res.text_logo_mask[60, 140]) > 0.8


def test_models_release_calls_hardware_guardrails(monkeypatch):
    from app.core.hardware import HardwareGuardrails

    called = {"n": 0}
    monkeypatch.setattr(HardwareGuardrails, "release_memory", lambda: called.__setitem__("n", called["n"] + 1))

    m = O._Models.__new__(O._Models)
    m.dino = object()
    m.sam = object()
    m.dino_proc = object()
    m.sam_proc = object()
    m.release()
    assert m.dino is None and m.sam is None
    assert called["n"] == 1


def test_sam_masks_clips_to_box_and_handles_count_mismatch():
    import torch

    h, w = 80, 100
    rgb = np.full((h, w, 3), 128, dtype=np.uint8)
    boxes = [(20, 15, 50, 45), (60, 20, 85, 60)]

    class FakeInputs(dict):
        def to(self, device):
            return self

    class FakeSamProc:
        def __init__(self, count=2):
            self.count = count

        def __call__(self, images, input_boxes, return_tensors="pt"):
            return FakeInputs({"original_sizes": [(h, w)]})

        def post_process_masks(self, pred_masks, original_sizes):
            # Intentionally return oversized True region to verify clipping to box bounds
            return [torch.ones((self.count, 1, h, w), dtype=torch.float32)]

    class FakeSamOut:
        pred_masks = torch.ones((1, 2, 1, h, w), dtype=torch.float32)

    class FakeSam:
        def __call__(self, **kwargs):
            return FakeSamOut()

    m = O._Models.__new__(O._Models)
    m.torch = torch
    m.device = "cpu"
    m.sam_proc = FakeSamProc(count=2)
    m.sam = FakeSam()

    masks = O._sam_masks(m, rgb, boxes)
    assert masks is not None and len(masks) == 2
    assert masks[0].dtype == np.uint8 and masks[0].shape == (h, w)
    assert masks[0][25, 35] == 1 and masks[0][5, 5] == 0 and masks[0][60, 80] == 0
    assert masks[1][30, 70] == 1 and masks[1][25, 35] == 0

    # Mismatched mask count returns None so caller falls back to _rect_mask
    m.sam_proc = FakeSamProc(count=1)
    assert O._sam_masks(m, rgb, boxes) is None


def test_classify_filters_picture_frame_aspect_label_min_score_tv_screen_and_text_boxes():
    rng = np.random.RandomState(11)
    img = rng.randint(10, 240, (300, 400, 3), dtype=np.uint8)
    boxes = [
        [20, 20, 35, 220],     # picture frame aspect 200/15 = 13.3 > 3.0 -> dropped
        [50, 40, 100, 90],     # picture frame valid aspect, score=0.40 < 0.45 -> dropped
        [120, 40, 170, 90],    # picture frame valid aspect, score=0.50 >= 0.45 -> kept
        [190, 40, 220, 80],    # lamp score=0.40 < 0.45 -> dropped
        [230, 120, 280, 160],  # screen score=0.38 < 0.42 -> dropped
        [290, 120, 340, 160],  # screen score=0.45 >= 0.42 -> kept in tv_boxes
        [10, 240, 60, 265],    # logo score=0.30 >= 0.27, area=0.0104 <= 0.03 -> text_dino_boxes
        [100, 200, 250, 290],  # sign area=0.1125 > 0.03 -> dropped
    ]
    scores = [0.60, 0.40, 0.50, 0.40, 0.38, 0.45, 0.30, 0.80]
    labels = [
        "picture frame",
        "picture frame",
        "picture frame",
        "lamp",
        "screen",
        "screen",
        "logo",
        "sign",
    ]
    c = O.classify_and_filter_boxes(boxes, scores, labels, img)
    assert c["decor_boxes"] == [(120, 40, 170, 90)]
    assert c["tv_boxes"] == [(290, 120, 340, 160)]
    assert c["text_dino_boxes"] == [(10, 240, 60, 265)]


def test_detect_ocr_boxes_tiles_720p_and_filters_low_conf_single_non_ascii(monkeypatch):
    import sys
    import types

    calls = []

    class FakeRapidOCR:
        def __call__(self, img_in):
            ih, iw = img_in.shape[:2]
            calls.append((ih, iw))
            # Full frame (720x1280): misses small WOLF logo, emits noisy 1-char CJK at conf 0.55 & ASCII A
            if (ih, iw) == (720, 1280):
                return [
                    ([[10, 10], [30, 10], [30, 30], [10, 30]], "品", 0.55),
                    ([[50, 10], [70, 10], [70, 30], [50, 30]], "中", 0.74),
                    ([[90, 10], [110, 10], [110, 30], [90, 30]], "漢", 0.82),
                    ([[130, 10], [150, 10], [150, 30], [130, 30]], "A", 0.60),
                ], None
            # First 2x2 tile catches small "WOLF" text at local coords
            if len(calls) == 2:
                return [
                    ([[20, 20], [90, 20], [90, 45], [20, 45]], "WOLF", 0.92),
                ], None
            return [], None

    fake_mod = types.ModuleType("rapidocr_onnxruntime")
    fake_mod.RapidOCR = FakeRapidOCR
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", fake_mod)

    rgb = np.full((720, 1280, 3), 120, dtype=np.uint8)
    polys, texts, ok = O._detect_ocr_boxes(rgb, tv_boxes=[])
    assert ok is True
    # Full frame + 4 overlapping 2x2 tiles = 5 OCR passes
    assert len(calls) == 5
    assert "WOLF" in texts
    assert "A" in texts
    assert "漢" in texts
    assert "品" not in texts
    assert "中" not in texts
    assert len(polys) == len(texts)
