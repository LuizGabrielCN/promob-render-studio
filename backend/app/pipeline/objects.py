"""
Detecção de objetos soltos, telas/TVs, lareiras e textos/logos para proteção no blend.

Grounding DINO tiny (caixas por texto) + SAM 2.1 tiny (máscaras) + RapidOCR ONNX (textos/logos em CPU).
Tudo com local_files_only=True: nunca baixa nada sozinho; sem cache de modelo,
`detect_object_mask` devolve None e o chamador cai na heurística `compute_object_protection_mask`.

API:
  detect_object_mask(ref_bgr) -> Optional[np.ndarray]   máscara float32 [0,1] em resolução cheia
  detect_objects(ref_bgr)     -> Optional[DetectionResult]  máscara + metadados (n_objects, labels, boxes, timing, text_logo_mask, fireplace_boxes, tv_boxes)
  last_metadata()             -> dict da última detecção
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.hardware import HardwareGuardrails

logger = logging.getLogger(__name__)

DINO_ID = "IDEA-Research/grounding-dino-tiny"
SAM_ID = "facebook/sam2.1-hiera-tiny"

PROMPT = (
    "television. tv. monitor. screen. fireplace. "
    "sign. logo. text. label. "
    "chair. bar stool. stool. sofa. armchair. pendant light. chandelier. "
    "bottle. wine bottle. glass. vase. potted plant. plant. book. picture frame. bag. handbag. "
    "suitcase. box. basket. lamp. towel. candle. sculpture. bowl. cup. pillow. coat."
)
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25
NMS_IOU = 0.5
MAX_BOX_AREA_FRAC = 0.06
TV_MAX_AREA_FRAC = 0.12
FURNITURE_MAX_AREA_FRAC = 0.35
TEXT_MAX_AREA_FRAC = 0.03
MIN_BOX_SIDE = 6
TILE_OVERLAP = 0.20
LABEL_MAX_AREA_FRAC = {"picture frame": 0.025}
LABEL_MIN_SCORE = {"lamp": 0.45, "picture frame": 0.45}
EDGE_SKIP_SCORE = 0.45  # detecções confiantes dispensam o teste de borda (fundo ripado tem borda demais)
EDGE_RATIO_MIN = 1.0  # energia de borda interior / anel; abaixo disso = painel/puxador liso
CACHE_SIZE = 8

FURNITURE_TOKENS = ("chair", "bar stool", "stool", "sofa", "armchair")
LUMINAIRE_TOKENS = ("pendant light", "chandelier")
TV_PRIMARY_TOKENS = ("television", "tv", "monitor")
TV_ALL_TOKENS = ("television", "tv", "monitor", "screen")
TEXT_TOKENS = ("sign", "logo", "text", "label")

_cache: "OrderedDict[str, DetectionResult]" = OrderedDict()
_last_meta: dict = {}


@dataclass
class DetectionResult:
    mask: np.ndarray
    n_objects: int
    labels: List[str] = field(default_factory=list)
    boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    sam_ok: bool = True
    elapsed_s: float = 0.0
    vram_peak_mb: float = 0.0
    text_logo_mask: Optional[np.ndarray] = None
    fireplace_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    tv_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    luminaire_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    ocr_texts: List[str] = field(default_factory=list)
    ocr_ok: bool = False

    def __post_init__(self) -> None:
        if self.text_logo_mask is None:
            self.text_logo_mask = np.zeros(self.mask.shape[:2], dtype=np.float32)
        else:
            self.text_logo_mask = np.asarray(self.text_logo_mask, dtype=np.float32)


def last_metadata() -> dict:
    return dict(_last_meta)


def clear_cache() -> None:
    _cache.clear()
    _last_meta.clear()


def _set_metadata(result: DetectionResult, cache_hit: bool = False) -> None:
    global _last_meta
    _last_meta = {
        "n_objects": result.n_objects,
        "labels": list(result.labels),
        "elapsed_s": round(result.elapsed_s, 3),
        "vram_peak_mb": round(result.vram_peak_mb, 1),
        "sam_ok": result.sam_ok,
        "cache_hit": cache_hit,
        "n_fireplaces": len(result.fireplace_boxes),
        "n_tvs": len(result.tv_boxes),
        "n_luminaires": len(result.luminaire_boxes),
        "ocr_ok": result.ocr_ok,
        "ocr_texts": list(result.ocr_texts),
    }


def _hash(img: np.ndarray) -> str:
    return hashlib.md5(img.tobytes()).hexdigest() + f"_{img.shape[0]}x{img.shape[1]}"


def _tiles(h: int, w: int) -> List[Tuple[int, int, int, int]]:
    """Imagem inteira + 2x2 tiles com 20% de overlap. (x0, y0, x1, y1)."""
    out = [(0, 0, w, h)]
    tw, th = int(round(w / (2 - TILE_OVERLAP))), int(round(h / (2 - TILE_OVERLAP)))
    for y0 in (0, h - th):
        for x0 in (0, w - tw):
            out.append((x0, y0, x0 + tw, y0 + th))
    return out


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> List[int]:
    order = np.argsort(-scores)
    keep: List[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx0 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy0 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx1 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy1 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        b = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        order = rest[inter / (a + b - inter + 1e-9) <= iou]
    return keep


def _edge_ratio(edge: np.ndarray, box: Tuple[int, int, int, int]) -> float:
    """Densidade de borda dentro da caixa / no anel ao redor. Objeto real tem mais borda dentro."""
    h, w = edge.shape
    x0, y0, x1, y1 = box
    pad = max(4, int(0.25 * max(x1 - x0, y1 - y0)))
    ox0, oy0, ox1, oy1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    inner = edge[y0:y1, x0:x1]
    outer = edge[oy0:oy1, ox0:ox1]
    ring_n = outer.size - inner.size
    if inner.size == 0 or ring_n <= 0:
        return 99.0
    ring = (float(outer.sum()) - float(inner.sum())) / ring_n
    return (float(inner.sum()) / inner.size) / (ring + 1e-3)


def _box_overlap_metrics(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Returns (IoU, intersection / area(a))."""
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter <= 0:
        return 0.0, 0.0
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    iou = inter / float(area_a + area_b - inter)
    frac_a = inter / float(area_a)
    return iou, frac_a


def _has_any_token(name: str, tokens: Tuple[str, ...]) -> bool:
    words = set(name.split())
    for tok in tokens:
        if " " in tok:
            if tok in name:
                return True
        elif tok in words:
            return True
    return False


def classify_and_filter_boxes(boxes, scores, labels, ref_bgr) -> Dict[str, list]:
    """
    Classifies and filters raw Grounding DINO detections into:
    - decor_boxes / decor_scores / decor_labels (protected decor objects)
    - tv_boxes / tv_scores / tv_labels (protected TVs/monitors/screens)
    - fireplace_boxes / fireplace_scores (unlit fireboxes; EXCLUDED from protection_mask)
    - text_dino_boxes / text_dino_scores (DINO text/logo/sign detections)
    - luminaire_boxes (pendants/chandeliers/large ceiling lamps; NOT frozen)
    - suppressed_furniture (stools/chairs/sofas/armchairs; NOT frozen)
    """
    h, w = ref_bgr.shape[:2]
    img_area = float(max(1, h * w))
    gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    edge = (cv2.Canny(gray, 40, 120) > 0).astype(np.float32)

    suppressed_furniture: List[Tuple[int, int, int, int]] = []
    luminaire_boxes: List[Tuple[int, int, int, int]] = []
    fireplace_cands: List[Tuple[Tuple[int, int, int, int], float]] = []
    tv_cands: List[Tuple[Tuple[int, int, int, int], float, str]] = []
    text_dino_boxes: List[Tuple[int, int, int, int]] = []
    decor_cands: List[Tuple[Tuple[int, int, int, int], float, str]] = []

    for b, s, l in zip(boxes, scores, labels):
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        bw, bh = x1 - x0, y1 - y0
        if bw < MIN_BOX_SIDE or bh < MIN_BOX_SIDE:
            continue
        area_frac = (bw * bh) / img_area
        score = float(s)
        name = str(l).strip().lower()
        box = (x0, y0, x1, y1)

        is_tv_label = _has_any_token(name, TV_ALL_TOKENS)
        is_primary_tv = _has_any_token(name, TV_PRIMARY_TOKENS)
        is_furniture = _has_any_token(name, FURNITURE_TOKENS)
        is_luminaire = _has_any_token(name, LUMINAIRE_TOKENS) or (
            "lamp" in name.split() and (area_frac > 0.012 or (y0 < 0.20 * h and area_frac > 0.008))
        )
        is_text = _has_any_token(name, TEXT_TOKENS)
        is_fireplace_word = "fireplace" in name.split()

        # 1. Suppressed furniture (stools, chairs, sofas) — never frozen, used to suppress false decor overlap
        if is_furniture and not is_primary_tv:
            if area_frac <= FURNITURE_MAX_AREA_FRAC and area_frac >= 0.008:
                suppressed_furniture.append(box)
            continue

        # 2. Luminaires (pendants, chandeliers, large ceiling lamps) — never frozen
        if is_luminaire:
            if area_frac <= FURNITURE_MAX_AREA_FRAC:
                luminaire_boxes.append(box)
            continue

        # 3. Fireplaces — strict primary match, excluded from protection_mask
        if is_fireplace_word:
            has_non_fire_compound = is_tv_label or ("grill" in name.split())
            is_wide_bottom_bench = (y1 >= int(0.96 * h)) and (bw > 0.35 * w)
            if (
                not has_non_fire_compound
                and score >= 0.40
                and 0.004 <= area_frac <= MAX_BOX_AREA_FRAC
                and not is_wide_bottom_bench
            ):
                fireplace_cands.append((box, score))
            continue

        # 4. TVs / monitors / screens (protected)
        if is_tv_label:
            if area_frac <= TV_MAX_AREA_FRAC and (
                (is_primary_tv and score >= 0.30) or (not is_primary_tv and score >= 0.42)
            ):
                if score >= EDGE_SKIP_SCORE or _edge_ratio(edge, box) >= EDGE_RATIO_MIN:
                    tv_cands.append((box, score, str(l)))
            continue

        # 5. Text / logo / sign detections (protected in text_logo_mask)
        if is_text:
            if area_frac <= TEXT_MAX_AREA_FRAC and score >= 0.27:
                text_dino_boxes.append(box)
            continue

        # 6. Standard decor objects
        if area_frac > MAX_BOX_AREA_FRAC:
            continue
        if score < EDGE_SKIP_SCORE and _edge_ratio(edge, box) < EDGE_RATIO_MIN:
            continue
        if name in LABEL_MAX_AREA_FRAC and area_frac > LABEL_MAX_AREA_FRAC[name]:
            continue
        if score < LABEL_MIN_SCORE.get(name, 0.0):
            continue
        if name == "picture frame" and max(bw, bh) / max(1, min(bw, bh)) > 3.0:
            continue  # tira vertical = lateral de armário, não quadro
        decor_cands.append((box, score, str(l)))

    # Suppress decor false positives overlapping furniture/luminaires (e.g., bar stools tagged as towel/sculpture,
    # pendants tagged as vase), while keeping small throw pillows (< 0.01 area) on sofas.
    filtered_decor: List[Tuple[Tuple[int, int, int, int], float, str]] = []
    suppressors = suppressed_furniture + luminaire_boxes
    for box, score, label in decor_cands:
        area_frac = ((box[2] - box[0]) * (box[3] - box[1])) / img_area
        name = label.strip().lower()
        is_small_pillow = ("pillow" in name.split()) and area_frac < 0.01
        if not is_small_pillow and suppressors:
            dropped = False
            for sup in suppressors:
                iou, frac_a = _box_overlap_metrics(box, sup)
                if iou >= 0.30 or frac_a >= 0.55:
                    dropped = True
                    break
            if dropped:
                continue
        filtered_decor.append((box, score, label))

    # Container check on decor boxes only: drop boxes enclosing >= 2 smaller decor boxes (e.g. cabinet doors)
    drop = set()
    for i, (a, _, _) in enumerate(filtered_decor):
        inside = 0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        for j, (c, _, _) in enumerate(filtered_decor):
            if i != j and (c[2] - c[0]) * (c[3] - c[1]) < 0.6 * area_a:
                cx, cy = (c[0] + c[2]) / 2.0, (c[1] + c[3]) / 2.0
                if a[0] <= cx <= a[2] and a[1] <= cy <= a[3]:
                    inside += 1
        if inside >= 2:
            drop.add(i)

    kept_decor = [filtered_decor[i] for i in range(len(filtered_decor)) if i not in drop]

    # Container check on fireplace boxes (keep inner firebox, drop outer mantel if any)
    fp_drop = set()
    for i, (a, _) in enumerate(fireplace_cands):
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        for j, (c, _) in enumerate(fireplace_cands):
            if i != j and (c[2] - c[0]) * (c[3] - c[1]) < 0.65 * area_a:
                cx, cy = (c[0] + c[2]) / 2.0, (c[1] + c[3]) / 2.0
                if a[0] <= cx <= a[2] and a[1] <= cy <= a[3]:
                    fp_drop.add(i)
                    break
    kept_fp = [fireplace_cands[i] for i in range(len(fireplace_cands)) if i not in fp_drop]

    return {
        "decor_boxes": [x[0] for x in kept_decor],
        "decor_scores": [x[1] for x in kept_decor],
        "decor_labels": [x[2] for x in kept_decor],
        "tv_boxes": [x[0] for x in tv_cands],
        "tv_scores": [x[1] for x in tv_cands],
        "tv_labels": [x[2] for x in tv_cands],
        "fireplace_boxes": [x[0] for x in kept_fp],
        "fireplace_scores": [x[1] for x in kept_fp],
        "text_dino_boxes": text_dino_boxes,
        "luminaire_boxes": luminaire_boxes,
        "suppressed_furniture": suppressed_furniture,
    }


def filter_boxes(boxes, scores, labels, ref_bgr):
    """Filtros de área, lado mínimo e contraste de borda. Entradas/saídas: listas paralelas."""
    classified = classify_and_filter_boxes(boxes, scores, labels, ref_bgr)
    kb = classified["decor_boxes"] + classified["tv_boxes"]
    ks = classified["decor_scores"] + classified["tv_scores"]
    kl = classified["decor_labels"] + classified["tv_labels"]
    return kb, ks, kl


class _Models:
    def __init__(self):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dino_proc = AutoProcessor.from_pretrained(DINO_ID, local_files_only=True)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(
            DINO_ID, local_files_only=True).to(self.device).eval()
        self.sam = self.sam_proc = None
        try:
            from transformers import Sam2Model, Sam2Processor
            self.sam_proc = Sam2Processor.from_pretrained(SAM_ID, local_files_only=True)
            self.sam = Sam2Model.from_pretrained(SAM_ID, local_files_only=True).to(self.device).eval()
        except Exception as exc:  # SAM ausente: retângulos
            logger.warning("SAM 2.1 indisponível (%s); usando retângulos.", exc)

    def release(self):
        self.dino = self.sam = self.dino_proc = self.sam_proc = None
        import sys as _sys
        sam2_mod = _sys.modules.get("transformers.models.sam2.modeling_sam2")
        if sam2_mod is not None:
            def _clear_caches(fn) -> None:
                for candidate in (fn, getattr(fn, "__wrapped__", None), getattr(fn, "__func__", None)):
                    if candidate is None:
                        continue
                    if hasattr(candidate, "cache_clear"):
                        candidate.cache_clear()
                    for cell in (getattr(candidate, "__closure__", None) or ()):
                        try:
                            val = cell.cell_contents
                            if hasattr(val, "cache_clear"):
                                val.cache_clear()
                        except ValueError:
                            pass

            for obj in vars(sam2_mod).values():
                _clear_caches(obj)
                if isinstance(obj, type):
                    for attr in vars(obj).values():
                        _clear_caches(attr)
        HardwareGuardrails.release_memory()


def _detect_boxes(m: _Models, rgb: np.ndarray):
    from PIL import Image
    h, w = rgb.shape[:2]
    all_b, all_s, all_l = [], [], []
    for (x0, y0, x1, y1) in _tiles(h, w):
        img = Image.fromarray(rgb[y0:y1, x0:x1])
        inp = m.dino_proc(images=img, text=PROMPT, return_tensors="pt").to(m.device)
        with m.torch.no_grad():
            out = m.dino(**inp)
        res = m.dino_proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
            target_sizes=[(y1 - y0, x1 - x0)])[0]
        labs = res.get("text_labels", res.get("labels"))
        for b, s, l in zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), labs):
            all_b.append([b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0])
            all_s.append(float(s))
            all_l.append(str(l))
    return all_b, all_s, all_l


def _sam_masks(m: _Models, rgb: np.ndarray, boxes) -> Optional[List[np.ndarray]]:
    from PIL import Image
    h, w = rgb.shape[:2]
    try:
        inp = m.sam_proc(images=Image.fromarray(rgb), input_boxes=[[list(map(float, b)) for b in boxes]],
                         return_tensors="pt").to(m.device)
        with m.torch.no_grad():
            out = m.sam(**inp, multimask_output=False)
        masks = m.sam_proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0]
        masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1]).numpy() > 0
        if masks.shape[0] != len(boxes):
            return None
        res = []
        for mk, b in zip(masks, boxes):
            # sanidade: máscara não pode vazar muito além da caixa
            bx = np.zeros((h, w), bool)
            bx[b[1]:b[3], b[0]:b[2]] = True
            mk = mk & bx
            res.append(mk.astype(np.uint8))
        return res
    except Exception as exc:
        logger.warning("SAM falhou (%s); usando retângulos.", exc)
        return None


def _rect_mask(shape, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    dx, dy = int(0.05 * (x1 - x0)), int(0.05 * (y1 - y0))  # erosão total 10%
    mk = np.zeros(shape, np.uint8)
    mk[y0 + dy:max(y0 + dy + 1, y1 - dy), x0 + dx:max(x0 + dx + 1, x1 - dx)] = 1
    return mk


def _detect_ocr_boxes(
    rgb: np.ndarray,
    tv_boxes: List[Tuple[int, int, int, int]],
) -> Tuple[List[np.ndarray], List[str], bool]:
    """
    Runs RapidOCR (CPU ONNX, zero VRAM) on full frame and 2x upscaled TV crops.
    Gracefully returns ([], [], False) if rapidocr_onnxruntime is unavailable.
    """
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return [], [], False

    ocr = None
    h, w = rgb.shape[:2]
    polys: List[np.ndarray] = []
    texts: List[str] = []
    try:
        ocr = RapidOCR()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        def _collect(img_in: np.ndarray, scale: float, ox: int, oy: int, min_score: float) -> None:
            res, _ = ocr(img_in)
            if not res:
                return
            for item in res:
                if len(item) < 3:
                    continue
                pts_raw, txt, conf = item[0], str(item[1]).strip(), float(item[2])
                if not txt or conf < min_score:
                    continue
                if len(txt) == 1 and not txt.isascii() and conf < 0.75:
                    continue
                pts = np.asarray(pts_raw, dtype=np.float32).reshape(-1, 2)
                pts[:, 0] = np.clip(pts[:, 0] / scale + ox, 0, w - 1)
                pts[:, 1] = np.clip(pts[:, 1] / scale + oy, 0, h - 1)
                poly = np.round(pts).astype(np.int32)
                if cv2.contourArea(poly) < 12.0:
                    continue
                polys.append(poly)
                texts.append(txt)

        # Full-frame OCR catches signs, book/box labels, and oven/appliance logos
        _collect(bgr, 1.0, 0, 0, min_score=0.45)

        # 2x2 overlapping tiles on >=720p images avoid downscaling below DBNet's stroke threshold
        if min(h, w) >= 720:
            for (tx0, ty0, tx1, ty1) in _tiles(h, w)[1:]:
                tile = bgr[ty0:ty1, tx0:tx1]
                if tile.size == 0:
                    continue
                _collect(tile, 1.0, tx0, ty0, min_score=0.45)

        # 2x upscaled crops on detected TV/screen regions capture small screen text/logos
        for (x0, y0, x1, y1) in tv_boxes:
            pad = max(8, int(round(0.08 * max(x1 - x0, y1 - y0))))
            cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
            cx1, cy1 = min(w, x1 + pad), min(h, y1 + pad)
            crop = bgr[cy0:cy1, cx0:cx1]
            if crop.size == 0:
                continue
            crop_2x = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
            _collect(crop_2x, 2.0, cx0, cy0, min_score=0.40)

        return polys, texts, True
    except Exception as exc:
        logger.warning("RapidOCR falhou (%s); seguindo sem OCR.", exc)
        return [], [], False
    finally:
        if ocr is not None:
            del ocr


def _tv_foreground_logo_mask(rgb: np.ndarray, tv_boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """Flags high-contrast graphic marks/logos inside detected TV screens (bezel inset 6%)."""
    h, w = rgb.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    if not tv_boxes:
        return out
    kernel = np.ones((5, 5), dtype=np.uint8)
    for (x0, y0, x1, y1) in tv_boxes:
        bw, bh = x1 - x0, y1 - y0
        dx = max(1, int(round(0.06 * bw)))
        dy = max(1, int(round(0.06 * bh)))
        ix0, iy0 = min(w, x0 + dx), min(h, y0 + dy)
        ix1, iy1 = max(ix0 + 1, x1 - dx), max(iy0 + 1, y1 - dy)
        crop = rgb[iy0:iy1, ix0:ix1]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
        med = float(np.median(gray))
        fg = (np.abs(gray - med) > 22.0).astype(np.uint8) * 255
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
        out[iy0:iy1, ix0:ix1] = np.maximum(out[iy0:iy1, ix0:ix1], fg)
    return out


def _build_text_logo_mask(
    shape: Tuple[int, int],
    ocr_polys: List[np.ndarray],
    text_dino_boxes: List[Tuple[int, int, int, int]],
    tv_fg_mask: np.ndarray,
) -> np.ndarray:
    h, w = shape
    raw = np.zeros((h, w), dtype=np.uint8)
    for poly in ocr_polys:
        cv2.fillPoly(raw, [poly], 1)
    for (x0, y0, x1, y1) in text_dino_boxes:
        raw[y0:y1, x0:x1] = 1
    if tv_fg_mask is not None and tv_fg_mask.any():
        raw[tv_fg_mask > 0] = 1
    if not raw.any():
        return np.zeros((h, w), dtype=np.float32)
    s = min(h, w)
    k = max(3, int(round(s / 300.0)) * 2 + 1)
    dilated = cv2.dilate(raw, np.ones((k, k), dtype=np.uint8))
    blurred = cv2.GaussianBlur(dilated.astype(np.float32), (0, 0), 1.0)
    return np.clip(blurred, 0.0, 1.0).astype(np.float32)


def detect_objects(ref_bgr: np.ndarray) -> Optional[DetectionResult]:
    """Detecta objetos soltos, TVs, lareiras e textos/logos. None se os modelos não estão no cache local."""
    _last_meta.clear()
    t0 = time.perf_counter()
    key = _hash(ref_bgr)
    if key in _cache:
        _cache.move_to_end(key)
        result = replace(_cache[key], elapsed_s=time.perf_counter() - t0, vram_peak_mb=0.0)
        _set_metadata(result, cache_hit=True)
        return result
    try:
        models = _Models()
    except Exception as exc:
        logger.info("Detector de objetos indisponível (%s); usando heurística.", exc)
        return None
    torch = models.torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        h, w = ref_bgr.shape[:2]
        rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
        b, s, l = _detect_boxes(models, rgb)
        mask = np.zeros((h, w), np.float32)
        boxes: List[Tuple[int, int, int, int]] = []
        scores: List[float] = []
        labels: List[str] = []
        tv_boxes: List[Tuple[int, int, int, int]] = []
        fireplace_boxes: List[Tuple[int, int, int, int]] = []
        luminaire_boxes: List[Tuple[int, int, int, int]] = []
        text_dino_boxes: List[Tuple[int, int, int, int]] = []
        sam_ok = models.sam is not None

        if b:
            keep = _nms(np.array(b, np.float32), np.array(s, np.float32), NMS_IOU)
            b, s, l = [b[i] for i in keep], [s[i] for i in keep], [l[i] for i in keep]
            classified = classify_and_filter_boxes(b, s, l, ref_bgr)
            boxes = classified["decor_boxes"] + classified["tv_boxes"]
            scores = classified["decor_scores"] + classified["tv_scores"]
            labels = classified["decor_labels"] + classified["tv_labels"]
            tv_boxes = classified["tv_boxes"]
            fireplace_boxes = classified["fireplace_boxes"]
            luminaire_boxes = classified["luminaire_boxes"]
            text_dino_boxes = classified["text_dino_boxes"]

        if boxes:
            masks = _sam_masks(models, rgb, boxes) if models.sam is not None else None
            if masks is None:
                sam_ok = False
                masks = [_rect_mask((h, w), bx) for bx in boxes]
            acc = np.zeros((h, w), np.uint8)
            for mk in masks:
                acc |= mk
            # Guarantee full interior coverage for detected TVs even if SAM only caught the inner bezel
            for bx in tv_boxes:
                acc |= _rect_mask((h, w), bx)
            mask = cv2.GaussianBlur(cv2.dilate(acc, np.ones((7, 7), np.uint8)).astype(np.float32), (0, 0), 1.5)
            mask = np.clip(mask, 0, 1).astype(np.float32)

        vram = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
    except Exception as exc:
        logger.warning("Detecção de objetos falhou (%s); usando heurística.", exc)
        return None
    finally:
        models.release()

    # CPU ONNX OCR + TV foreground logo mask (runs after GPU models are released)
    ocr_polys, ocr_texts, ocr_ok = _detect_ocr_boxes(rgb, tv_boxes)
    tv_fg = _tv_foreground_logo_mask(rgb, tv_boxes)
    text_logo_mask = _build_text_logo_mask((h, w), ocr_polys, text_dino_boxes, tv_fg)
    mask = np.clip(np.maximum(mask, text_logo_mask), 0.0, 1.0).astype(np.float32)

    res = DetectionResult(
        mask=mask,
        n_objects=len(boxes),
        labels=labels,
        boxes=boxes,
        scores=scores,
        sam_ok=sam_ok,
        elapsed_s=time.perf_counter() - t0,
        vram_peak_mb=vram,
        text_logo_mask=text_logo_mask,
        fireplace_boxes=fireplace_boxes,
        tv_boxes=tv_boxes,
        luminaire_boxes=luminaire_boxes,
        ocr_texts=ocr_texts,
        ocr_ok=ocr_ok,
    )
    _cache[key] = res
    while len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)
    _set_metadata(res)
    return res


def detect_object_mask(ref_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Máscara float32 [0,1] (1 = objeto, congelar). None sem modelos. Metadados via last_metadata()."""
    res = detect_objects(ref_bgr)
    if res is None:
        return None
    return res.mask
