"""
Unit tests for LocalControlNetGenerator (R3):
- Adaptive Canny structural edge extraction
- Monotonic generative_strength parameter curve
- Architectural scene_profile prompts (including oak/LED/brass PBR keywords)
- CIE L*a*b* chroma anchoring for wood & multi-view consistency
- Generative short-circuit at strength=0.0, mocked ControlNet pipeline + VRAM cleanup
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app.pipeline.generative import (
    BASE_MODEL_ID,
    CONTROLNET_MODEL_ID,
    DEFAULT_CONTROL_GUIDANCE_END,
    NEGATIVE_PROMPT,
    SCENE_PROMPTS,
    LocalControlNetGenerator,
    blend_with_structure_preservation,
    compute_flat_surface_mask,
    compute_object_protection_mask,
    remove_invented_edges,
    extract_adaptive_canny,
    get_controlnet_generator,
    harmonize_chromatic_lab,
    map_strength_parameters,
)


def test_extract_adaptive_canny_captures_cabinet_edges():
    """Adaptive Canny must detect architectural panel borders on 3-channel uint8 BGR."""
    img = np.full((256, 256, 3), 185, dtype=np.uint8)
    img[50:210, 60:196] = (90, 140, 180)
    img[50:210, 127:129] = (40, 60, 80)

    canny = extract_adaptive_canny(img)
    assert canny.shape == (256, 256, 3)
    assert canny.dtype == np.uint8
    assert np.count_nonzero(canny) > 100


def test_extract_adaptive_canny_invalid_inputs():
    with pytest.raises(ValueError):
        extract_adaptive_canny(None)
    with pytest.raises(ValueError):
        extract_adaptive_canny(np.zeros((0, 0, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        extract_adaptive_canny(np.zeros((64, 64), dtype=np.uint8))


def test_map_strength_parameters_zero_and_monotonic_bands():
    """map_strength_parameters must disable diffusion at 0.0 and scale monotonically across architectural fidelity bands."""
    p0 = map_strength_parameters(0.0)
    assert p0["generative_strength"] == 0.0
    assert p0["denoising_strength"] == 0.0
    assert p0["blend_alpha"] == 0.0
    assert p0["num_inference_steps"] == 0

    p_low = map_strength_parameters(0.28)
    p_mid = map_strength_parameters(0.48)
    p_high = map_strength_parameters(0.72)
    p_max = map_strength_parameters(1.0)

    # Diffusion must be strong enough to actually re-render materials; objects are protected by mask
    assert 0.40 <= p_low["denoising_strength"] <= 0.50
    # Passo 4: cn_knots -0.05 (Canny loosened; objects/patterns now come from the original)
    assert p_low["controlnet_conditioning_scale"] >= 0.92

    assert p_mid["denoising_strength"] == pytest.approx(0.58)
    assert p_mid["controlnet_conditioning_scale"] == pytest.approx(0.90)
    assert p_mid["blend_alpha"] == pytest.approx(0.85 + 0.15 * 0.48)
    assert p_mid["num_inference_steps"] == 22

    assert 0.64 <= p_high["denoising_strength"] <= 0.70
    assert p_high["controlnet_conditioning_scale"] >= 0.83

    assert (
        p_low["denoising_strength"]
        < p_mid["denoising_strength"]
        < p_high["denoising_strength"]
        < p_max["denoising_strength"]
    )
    assert (
        p_low["controlnet_conditioning_scale"]
        >= p_mid["controlnet_conditioning_scale"]
        >= p_high["controlnet_conditioning_scale"]
        >= p_max["controlnet_conditioning_scale"]
    )
    assert p_low["blend_alpha"] < p_mid["blend_alpha"] < p_high["blend_alpha"] < p_max["blend_alpha"]
    assert p_max["blend_alpha"] <= 1.0


def test_map_strength_parameters_rejects_non_finite():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            map_strength_parameters(bad)


def test_scene_prompts_include_architectural_pbr_keywords():
    """All five scene profiles exist and use PHOTOGRAPHIC wording without CG/render/8k words,
    and `quarto_sala` stays free of material-biasing words like 'casella oak' or 'handbags'."""
    for profile in ("auto", "banheiro", "cozinha", "quarto_sala", "externa"):
        assert profile in SCENE_PROMPTS
        prompt = SCENE_PROMPTS[profile].lower()
        assert "photograph" in prompt
        assert "global illumination" in prompt
        for cg_word in ("pbr", "v-ray", "corona", "render", "8k"):
            assert cg_word not in prompt, (profile, cg_word)

    qs_lower = SCENE_PROMPTS["quarto_sala"].lower()
    assert "casella oak" not in qs_lower
    assert "handbags" not in qs_lower
    assert "living room" in qs_lower
    assert "spot" in qs_lower
    assert "porcelain" in SCENE_PROMPTS["banheiro"].lower()
    assert "outdoor" in SCENE_PROMPTS["externa"].lower()


def test_harmonize_chromatic_lab_anchors_oak_tone():
    """CIE L*a*b* chroma anchoring pulls washed-out generated output back to warm oak chroma."""
    import cv2

    # Warm oak reference in BGR (RGB=(185, 140, 95) -> BGR=(95, 140, 185))
    ref = np.full((128, 128, 3), (95, 140, 185), dtype=np.uint8)
    # Neutral washed-out gray/beige generated result
    gen = np.full((128, 128, 3), (170, 175, 180), dtype=np.uint8)

    anchored = harmonize_chromatic_lab(gen, ref, chroma_retention=0.72)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2Lab).astype(np.float32)
    out_lab = cv2.cvtColor(anchored, cv2.COLOR_BGR2Lab).astype(np.float32)

    assert out_lab[:, :, 2].mean() > 135.0
    assert abs(out_lab[:, :, 2].mean() - ref_lab[:, :, 2].mean()) < 10.0


def test_generator_status_contract():
    gen = LocalControlNetGenerator()
    status = gen.get_status()
    assert status["base_model"] == BASE_MODEL_ID
    assert status["controlnet_model"] == CONTROLNET_MODEL_ID
    assert status["dtype"] == "float16"
    assert status["scheduler"] == "DPMSolverMultistepScheduler"
    assert isinstance(status["available"], bool)
    assert isinstance(status["cached_locally"], bool)


def test_generator_zero_strength_short_circuits_without_loading_pipeline():
    gen = LocalControlNetGenerator()
    img = np.full((128, 128, 3), 140, dtype=np.uint8)
    with patch.object(gen, "_ensure_pipeline", side_effect=AssertionError("Should not load pipeline")):
        out = gen.generate(img, generative_strength=0.0)
    np.testing.assert_array_equal(out, img)
    assert out is not img


def test_generator_mocked_pipeline_runs_preserves_bgra_and_releases_vram():
    gen = LocalControlNetGenerator()
    img = np.full((128, 128, 4), 130, dtype=np.uint8)
    img[:, :, 3] = 211

    def fake_pipe(**kwargs):
        w, h = kwargs["image"].size
        assert w >= 512 or h >= 512
        assert w % 8 == 0 and h % 8 == 0
        assert kwargs["num_inference_steps"] == 22
        assert kwargs["controlnet_conditioning_scale"] == pytest.approx(0.90)
        assert kwargs["strength"] == pytest.approx(0.58)
        assert kwargs["control_guidance_end"] == pytest.approx(DEFAULT_CONTROL_GUIDANCE_END)
        cb = kwargs.get("callback_on_step_end")
        if cb:
            cb(None, 0, None, {})
        synthetic_out = Image.fromarray(np.full((h, w, 3), 180, dtype=np.uint8))
        return SimpleNamespace(images=[synthetic_out])

    fake_pipe_mock = MagicMock(side_effect=fake_pipe)
    gen._pipeline = fake_pipe_mock
    gen._pipeline_device = "cuda"

    progress_events = []
    with patch.object(gen, "release_vram", wraps=gen.release_vram) as mock_release:
        out = gen.generate(
            img,
            generative_strength=0.48,
            scene_profile="banheiro",
            progress_callback=lambda p, m: progress_events.append((p, m)),
        )

    assert out.shape == img.shape
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out[:, :, 3], img[:, :, 3])
    assert fake_pipe_mock.call_count == 1
    assert mock_release.call_count == 1
    assert len(progress_events) >= 2


def test_get_controlnet_generator_singleton():
    assert get_controlnet_generator() is get_controlnet_generator()


def test_extract_adaptive_canny_preserves_low_contrast_objects_in_bright_scenes():
    """Bright scenes (median_lum > 200) must still extract low-contrast cream-on-oak contours."""
    # Simulate bright light oak background (BGR ~ (180, 208, 228), lum ~ 211)
    img = np.full((256, 256, 3), (180, 208, 228), dtype=np.uint8)
    # Low-contrast cream handbag silhouette (delta ~ 18-22)
    img[70:180, 75:180] = (198, 226, 242)

    canny = extract_adaptive_canny(img)
    border_band = canny[66:74, 80:175]
    assert np.count_nonzero(border_band) > 40, "Low-contrast perimeter lost in bright scene"
    # Interior of flat panels must stay noise-free
    assert np.count_nonzero(canny[10:55, 10:55]) == 0


def test_scene_prompts_auto_hygiene_and_negative_prompt():
    """
    'auto' is the default preset for every room type, so it must not name a material or colour
    (an oak/brass/handbag prompt turned white lacquer cream, tile floors into wood and chrome into brass).
    The negative prompt still suppresses seams/veins and now material drift.
    """
    auto_lower = SCENE_PROMPTS["auto"].lower()
    for material in ("marble", "oak", "walnut", "brass", "gold", "handbag", "luggage", "wood"):
        assert material not in auto_lower, f"'auto' prompt must stay material-neutral ({material})"
    assert "same materials and colors" in auto_lower
    # Passo 4a: removed from "auto" (the negative "horizontal door seams" still suppresses seams)
    assert "seamless smooth cabinet doors" not in auto_lower
    for profile in ("banheiro", "cozinha"):
        assert "brass" not in SCENE_PROMPTS[profile].lower()
        assert "same materials and colors" in SCENE_PROMPTS[profile].lower()

    neg_lower = NEGATIVE_PROMPT.lower()
    assert "solid beige paint over wood" not in neg_lower
    assert "marble veining" in neg_lower
    assert "horizontal door seams" in neg_lower
    assert "perforated metal towers" in neg_lower
    assert "changed materials" in neg_lower and "wood paneling on walls" in neg_lower
    for cg_word in ("render", "3d", "cgi", "octane", "vray"):
        assert cg_word in neg_lower


@pytest.fixture(scope="module")
def clip_tokenizer():
    tokenizer_mod = pytest.importorskip("transformers")
    from huggingface_hub import try_to_load_from_cache

    files = ("tokenizer/vocab.json", "tokenizer/merges.txt")
    if not all(isinstance(try_to_load_from_cache(BASE_MODEL_ID, name), str) for name in files):
        pytest.skip("SD 1.5 tokenizer not cached locally")
    return tokenizer_mod.CLIPTokenizer.from_pretrained(BASE_MODEL_ID, subfolder="tokenizer", local_files_only=True)


def test_prompts_fit_clip_token_window(clip_tokenizer):
    """CLIP silently truncates after 77 tokens: every prompt (and the negative) must fit."""
    for text in list(SCENE_PROMPTS.values()) + [NEGATIVE_PROMPT]:
        assert len(clip_tokenizer(text).input_ids) <= 77, text


def test_harmonize_chromatic_lab_anchors_low_frequency_luminance_and_keeps_highlights():
    """Low-frequency L* anchoring prevents macro niche darkening while keeping fine specular highlights."""
    ref = np.full((128, 128, 3), (185, 205, 220), dtype=np.uint8)
    # Simulated darkened generative output with a crisp bright specular streak
    gen = np.full((128, 128, 3), (110, 125, 140), dtype=np.uint8)
    gen[62:66, 30:98] = (235, 240, 245)

    out = harmonize_chromatic_lab(gen, ref, chroma_retention=0.72, luma_retention=0.78)
    # Mean luminance recovers closely toward bright reference instead of staying darkened
    assert float(out.mean()) > 175.0
    # Fine specular highlight remains brighter than surrounding panel
    assert float(out[63:65, 40:90].mean()) > float(out[10:40, 10:40].mean()) + 30.0


def test_blend_with_structure_preservation_suppresses_flat_veins_and_keeps_fine_detail():
    """Smooth panels reject invented veins/seams while structured areas retain fine reference detail."""
    ref = np.full((256, 256, 3), (205, 215, 222), dtype=np.uint8)
    # Structured fine detail region on right half (e.g. patterned box / clasp)
    ref[80:160, 150:220:4] = (90, 100, 110)

    # Generated image invents a dark horizontal seam/vein inside smooth left panel AND blurs right detail
    gen = np.full((256, 256, 3), (205, 215, 222), dtype=np.uint8)
    gen[118:124, 20:110] = (120, 125, 130)  # invented seam/vein on flat left door
    gen[80:160, 145:225] = (160, 165, 170)  # smeared structured detail

    flat_mask = compute_flat_surface_mask(ref)
    assert flat_mask[40:100, 30:90].mean() > 0.9
    assert flat_mask[90:150, 160:210].mean() < 0.2

    out = blend_with_structure_preservation(gen, ref, blend_alpha=0.74, preserve_geometry=True)
    # Invented vein inside smooth left panel must be heavily suppressed
    seam_drop = float(ref[120:122, 30:100].mean() - out[120:122, 30:100].mean())
    assert seam_drop < 18.0, f"Invented flat-panel seam was not suppressed (drop={seam_drop:.2f})"
    # Fine structured stripes on right half remain clearly resolved
    stripe_contrast = float(out[90:150, 152:212:4].mean() - out[90:150, 150:210:4].mean())
    assert stripe_contrast > 25.0, f"Fine structured detail lost (contrast={stripe_contrast:.2f})"


def _closet_scene():
    """Light oak niche wall with shelves, a saturated red bag and a low-contrast cream bag with a handle."""
    import cv2

    img = np.full((400, 400, 3), (170, 200, 225), dtype=np.uint8)  # light oak
    for y in (130, 260):  # long straight shelf fronts
        img[y:y + 6, :] = (205, 225, 240)
    cv2.rectangle(img, (40, 50), (130, 120), (60, 40, 200), -1)  # red bag
    cv2.ellipse(img, (85, 50), (25, 18), 0, 180, 360, (40, 30, 160), 3)
    cv2.rectangle(img, (220, 180), (310, 250), (195, 220, 238), -1)  # cream bag on oak
    cv2.ellipse(img, (265, 180), (22, 16), 0, 180, 360, (110, 120, 130), 2)
    for x in range(228, 304, 8):  # chain links
        cv2.circle(img, (x, 200 + (x % 16)), 2, (120, 130, 140), -1)
    return img


def test_object_protection_mask_covers_objects_and_frees_cabinetry():
    """Bags (saturated and low-contrast) are protected; empty oak panels and shelf fronts are not."""
    ref = _closet_scene()
    mask = compute_object_protection_mask(ref)
    assert mask.shape == ref.shape[:2] and mask.dtype == np.float32
    assert mask[70:110, 60:110].mean() > 0.9, "Red bag interior must be fully protected (no holes)"
    # Low-contrast cream bag: the detailed part (handle, chain) must be protected. Known limit: a
    # perfectly featureless body in the same tone as the oak is not detected by this heuristic.
    assert mask[198:222, 235:295].mean() > 0.7, "Low-contrast cream bag detail must be protected"
    assert mask[300:380, 150:380].mean() < 0.05, "Empty oak niche must stay free for re-rendering"
    assert mask[131:135, 330:390].mean() < 0.1, "Long shelf fronts must not be treated as objects"


def test_object_protection_mask_rejects_windows_columns_and_texture():
    """Architecture (window grid, tall column) and texture patches are not loose objects."""
    import cv2

    img = np.full((400, 600, 3), (200, 212, 220), dtype=np.uint8)  # beige wall
    # Window: white frame with a muntin grid (dense short edges, big filled rectangle)
    cv2.rectangle(img, (30, 60), (200, 300), (250, 250, 250), -1)
    for x in range(30, 201, 28):
        cv2.line(img, (x, 60), (x, 300), (150, 150, 150), 3)
    for y in range(60, 301, 30):
        cv2.line(img, (30, y), (200, y), (150, 150, 150), 3)
    # Dark navy column, floor to ceiling
    img[:, 480:520] = (80, 55, 45)
    # Bag-sized object in the middle with a handle
    cv2.rectangle(img, (290, 220), (370, 290), (60, 40, 200), -1)
    cv2.ellipse(img, (330, 220), (22, 16), 0, 180, 360, (40, 30, 160), 3)

    mask = compute_object_protection_mask(img)
    assert mask[240:280, 300:360].mean() > 0.9, "The bag must stay protected"
    assert mask[100:260, 60:170].mean() < 0.1, "A window is not an object"
    assert mask[50:350, 485:515].mean() < 0.1, "A floor-to-ceiling column is not an object"


def test_blend_does_not_draw_pillow_contours_around_veins():
    """A generated light streak across a veined stone must stay smooth next to the veins; with the
    corrections cut at the flat-mask edge (old behaviour) it showed a step ("pillow" contour)."""
    import cv2

    ref = np.full((400, 400, 3), (232, 236, 238), dtype=np.uint8)  # white stone
    for x in (120, 200, 280):  # long straight veins (not mistaken for loose objects)
        ref[:, x:x + 2] = (90, 120, 150)
    gen_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2Lab).astype(np.float32)
    yy = np.arange(400, dtype=np.float32)[:, None]
    # Window-light streak with soft but short edges (mid-frequency), like the SD output on stone
    streak = (np.abs(yy - 200.0) < 12.0).astype(np.float32)
    gen_lab[:, :, 0] -= 25.0 * cv2.GaussianBlur(np.repeat(streak, 400, axis=1), (0, 0), 2.0)
    gen = cv2.cvtColor(np.clip(np.round(gen_lab), 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)

    def max_step(out):
        l_col = cv2.cvtColor(out, cv2.COLOR_BGR2Lab)[:, :, 0].astype(np.float32)[195:206].mean(axis=0)
        d = np.abs(np.diff(l_col))
        for x in (120, 200, 280):
            d[x - 2:x + 3] = 0.0  # ignore the vein line itself
        return float(d.max())

    sharp = blend_with_structure_preservation(gen, ref, 0.84, True, 0.48, low_band_soften_scale=0.001)
    soft = blend_with_structure_preservation(gen, ref, 0.84, True, 0.48)
    # Since passo 1 the veins are recognised as reference pattern (texture mask + thin-line map), so the
    # corrections next to them no longer depend on the flat-mask edge: even the sharp-cut variant stays
    # smooth (the old "sharp > 2x soft" discriminator no longer applies). Both must stay pillow-free.
    assert max_step(sharp) < 1.5, max_step(sharp)
    assert max_step(soft) < 1.5, max_step(soft)
    # Real structure is never clipped: the vein keeps its depth next to the softened correction
    l_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2Lab)[:, :, 0].astype(np.float32)
    l_out = cv2.cvtColor(soft, cv2.COLOR_BGR2Lab)[:, :, 0].astype(np.float32)
    depth_ref = l_ref[100:150, 150].mean() - l_ref[100:150, 200:202].mean()
    depth_out = l_out[100:150, 150].mean() - l_out[100:150, 200:202].mean()
    assert depth_out > 0.85 * depth_ref, (depth_ref, depth_out)


def test_uniform_border_band_is_detected_and_kept():
    """The empty frame around a Promob section view is kept verbatim; scene gradients are not."""
    from app.pipeline.generative import compute_uniform_border_mask

    rng = np.random.default_rng(0)
    scene = rng.integers(60, 200, size=(300, 400, 3)).astype(np.uint8)
    ref = scene.copy()
    ref[260:, :] = (225, 216, 212)  # bottom frame band
    ref[:, 350:] = (225, 216, 212)  # right frame band
    mask = compute_uniform_border_mask(ref)
    assert mask[265:, :].all() and mask[:, 355:].all()
    assert not mask[:250, :340].any()

    # A smooth ceiling gradient touching the top border has no hard step: not a frame
    grad = np.repeat(np.linspace(240, 170, 300, dtype=np.float32)[:, None], 400, axis=1)
    ceiling = np.repeat(grad[:, :, None], 3, axis=2).astype(np.uint8)
    assert not compute_uniform_border_mask(ceiling).any()

    gen = np.full_like(ref, 90)
    out = blend_with_structure_preservation(gen, ref, blend_alpha=0.84, preserve_geometry=True)
    np.testing.assert_array_equal(out[265:, :], ref[265:, :])


def test_blend_keeps_original_objects_but_takes_generated_materials():
    """Hallucinated object colour is rejected while cabinetry adopts the generated material."""
    ref = _closet_scene()
    ref[:, :12] = (90, 110, 130)  # cabinet side panel: the flat niche is not an empty border frame
    gen = ref.copy()
    gen[50:121, 40:131] = (60, 90, 130)  # red bag hallucinated into brown block
    gen[300:390, 150:390] = (120, 160, 200)  # re-rendered darker, richer oak in empty niche

    out = blend_with_structure_preservation(gen, ref, blend_alpha=0.84, preserve_geometry=True,
                                            generative_strength=0.48)
    bag_err = np.abs(out[70:110, 60:110].astype(float) - ref[70:110, 60:110].astype(float)).mean()
    assert bag_err < 20.0, f"Red bag drifted toward the hallucination (MAE={bag_err:.1f})"
    niche_shift = float(ref[320:370, 200:350].mean() - out[320:370, 200:350].mean())
    assert niche_shift > 25.0, f"Generated material was discarded in the free niche (shift={niche_shift:.1f})"


def test_remove_invented_edges_erases_seam_but_keeps_reference_edges():
    """A strong seam drawn across a flat door is inpainted; edges present in the reference stay."""
    ref = np.full((300, 300, 3), (180, 205, 225), dtype=np.uint8)
    ref[:, 148:152] = (120, 140, 160)  # real gap between two doors
    gen = ref.copy()
    gen[140:143, 10:140] = (110, 120, 135)  # invented horizontal seam

    out = remove_invented_edges(gen, ref)
    seam_drop = float(ref[140:143, 30:120].mean() - out[140:143, 30:120].mean())
    assert seam_drop < 10.0, f"Invented seam survived (drop={seam_drop:.1f})"
    np.testing.assert_array_equal(out[60:240, 148:152], gen[60:240, 148:152])


def test_blend_levels_faint_seam_and_invented_shelf_but_keeps_grain():
    """On a featureless Promob panel, a thin seam and a whole invented shelf (edge + shadow, pale
    chroma) are flattened, while low-amplitude wood grain survives."""
    import cv2

    rng = np.random.default_rng(0)
    ref = np.full((400, 400, 3), (175, 205, 228), dtype=np.uint8)  # smooth light oak panel
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2Lab).astype(np.float32)

    gen_lab = ref_lab.copy()
    grain = cv2.GaussianBlur(rng.normal(0, 6.0, (1, 400)).astype(np.float32), (0, 0), sigmaX=1.5, sigmaY=0.1)
    gen_lab[:, :, 0] += np.repeat(grain, 400, axis=0)  # vertical grain stripes, 2-4 px wide, ~2-3 L*
    gen_lab[200:202, 20:380, 0] -= 15.0  # thin seam across the door
    gen_lab[100:106, 20:380, 0] += 10.0  # invented shelf front (bright)
    gen_lab[106:116, 20:380, 0] -= 6.0  # its shadow
    gen_lab[100:106, 20:380, 1] -= 5.0  # pale chroma of the front edge
    gen = cv2.cvtColor(np.clip(np.round(gen_lab), 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)

    out = blend_with_structure_preservation(gen, ref, blend_alpha=0.84, preserve_geometry=True,
                                            generative_strength=0.48)
    out_lab = cv2.cvtColor(out, cv2.COLOR_BGR2Lab).astype(np.float32)
    row_l = out_lab[:, 60:340, 0].mean(axis=1)
    row_a = out_lab[:, 60:340, 1].mean(axis=1)
    base_l = float(np.median(row_l[150:190]))

    assert base_l - row_l[199:203].min() < 3.0, "Thin seam must be levelled"
    assert row_l[98:108].max() - base_l < 3.5, "Invented shelf front must be flattened"
    assert float(np.median(row_a[150:190])) - row_a[98:108].min() < 3.0, "Shelf chroma stripe must be clipped (limit 2.0 + rounding)"

    def _grain_std(lab):
        g = lab[150:190, 60:340, 0].mean(axis=0)
        return float(np.std(g - cv2.GaussianBlur(g[None], (0, 0), 6)[0]))

    grain_in = _grain_std(cv2.cvtColor(gen, cv2.COLOR_BGR2Lab).astype(np.float32))
    assert _grain_std(out_lab) > 0.5 * grain_in, "Wood grain was erased"


# ---------------------------------------------------------------------------------------------------------
# Passo 6 (nucleo): synthetic regressions for detail transfer, texture colour anchor, latent lock, finish
# ---------------------------------------------------------------------------------------------------------
WORK_SCALE = 1600.0 / 768.0  # typical Promob sample (1600x900) -> 768x432 working size


def _simulate_ai(ref, work_scale=WORK_SCALE, lab_shift=(0.0, 0.0, 0.0)):
    """What the diffusion can give back at best: the scene at working resolution, Lanczos-upscaled
    (no band above the working Nyquist), optionally with a Lab shift (relight / material swap)."""
    import cv2

    h, w = ref.shape[:2]
    small = cv2.resize(ref, (int(round(w / work_scale)), int(round(h / work_scale))), interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LANCZOS4)
    lab = cv2.cvtColor(up, cv2.COLOR_BGR2Lab).astype(np.float32)
    lab += np.array(lab_shift, np.float32)
    return cv2.cvtColor(np.clip(np.round(lab), 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)


def _lab_from(l_val, a_val, b_val, shape):
    import cv2

    lab = np.empty(shape + (3,), np.float32)
    lab[..., 0], lab[..., 1], lab[..., 2] = l_val, a_val, b_val
    return lab, cv2


def _hp(img, sigma=1.5):
    import cv2

    l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, 0].astype(np.float32)
    return l - cv2.GaussianBlur(l, (0, 0), sigma)


def test_diagonal_grout_grid_survives_blend_with_detail_transfer():
    """1-2 px diagonal grout, only 4 L* deep, lost by the 768 px diffusion, must come back aligned
    (high-band correlation > 0.9) when the blend runs with the working scale."""
    h, w = 450, 800
    lab, cv2 = _lab_from(150.0, 128.0, 128.0, (h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    grid = (((xx + yy) % 28) < 2) | (((xx - yy) % 28) < 1)
    lab[..., 0] -= 4.0 * grid
    ref = cv2.cvtColor(np.round(lab).astype(np.uint8), cv2.COLOR_Lab2BGR)
    gen = _simulate_ai(ref, lab_shift=(-6.0, 0.0, 0.0))  # AI relights darker, pattern blurred away

    out = blend_with_structure_preservation(gen, ref, 0.85 + 0.15 * 0.48, True, 0.48, work_scale=WORK_SCALE)
    sl = (slice(40, h - 40), slice(40, w - 40))
    corr = float(np.corrcoef(_hp(ref)[sl].ravel(), _hp(out)[sl].ravel())[0, 1])
    corr_ai = float(np.corrcoef(_hp(ref)[sl].ravel(), _hp(gen)[sl].ravel())[0, 1])
    assert corr > 0.9, (corr, corr_ai)
    assert corr > corr_ai + 0.1
    # The relight is kept (the AI owns the low band)
    l_out = cv2.cvtColor(out, cv2.COLOR_BGR2Lab)[:, :, 0].astype(np.float32)
    l_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2Lab)[:, :, 0].astype(np.float32)
    assert float(l_ref[sl].mean() - l_out[sl].mean()) > 3.0


def test_smooth_panel_gains_no_high_band():
    """A featureless lacquer panel must not receive any high band from the detail transfer."""
    lab, cv2 = _lab_from(200.0, 130.0, 135.0, (450, 800))
    ref = cv2.cvtColor(np.round(lab).astype(np.uint8), cv2.COLOR_Lab2BGR)
    rng = np.random.default_rng(1)
    soft = cv2.GaussianBlur(rng.normal(0, 3.0, (450, 800)).astype(np.float32), (0, 0), 6.0)
    lab[..., 0] += 4.0 * soft / max(float(soft.std()), 1e-6)  # gentle AI lighting variation, no fine band
    gen = _simulate_ai(cv2.cvtColor(np.clip(np.round(lab), 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR))

    out = blend_with_structure_preservation(gen, ref, 0.92, True, 0.48, work_scale=WORK_SCALE)
    assert float(_hp(out)[20:-20, 20:-20].std()) < 0.5


def test_grey_textured_floor_turned_brown_by_ai_is_anchored_back():
    """s15 failure: grey speckled porcelain/terrazzo re-rendered as brown wood. The texture colour anchor
    must bring a*/b* back within dE_ab < 4 while the AI may still relight it."""
    from app.pipeline.generative import compute_texture_mask

    h, w = 450, 800
    lab, cv2 = _lab_from(120.0, 128.0, 127.0, (h, w))
    rng = np.random.default_rng(2)
    lab[..., 0] += rng.normal(0, 5.0, (h, w)).astype(np.float32)  # granilite speckle
    yy, xx = np.mgrid[0:h, 0:w]
    lab[..., 0] -= 4.0 * ((((xx + yy) % 40) < 2) | (((xx - yy) % 40) < 2))
    ref = cv2.cvtColor(np.clip(np.round(lab), 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)
    assert compute_texture_mask(ref)[20:-20, 20:-20].mean() > 0.9

    gen = _simulate_ai(ref, lab_shift=(8.0, 10.0, 22.0))  # brown wood colour, brighter
    out = blend_with_structure_preservation(gen, ref, 0.92, True, 0.48, work_scale=WORK_SCALE)

    def ab(img):
        lab_i = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)[40:-40, 40:-40]
        return lab_i[..., 1].mean(), lab_i[..., 2].mean()

    (ar, br), (ag, bg), (ao, bo) = ab(ref), ab(gen), ab(out)
    de_gen = float(np.hypot(ag - ar, bg - br))
    de_out = float(np.hypot(ao - ar, bo - br))
    assert de_gen > 20.0
    assert de_out < 4.0, de_out


def test_blend_uses_explicit_object_mask_instead_of_heuristic():
    """object_mask (e.g. from the detector) replaces the heuristic; None keeps the heuristic."""
    ref = _closet_scene()
    gen = ref.copy()
    gen[50:121, 40:131] = (60, 90, 130)  # hallucinated red bag
    gen[300:390, 150:390] = (120, 160, 200)

    heur = blend_with_structure_preservation(gen, ref, 0.92, True, 0.48)
    explicit_none = blend_with_structure_preservation(gen, ref, 0.92, True, 0.48, object_mask=None)
    np.testing.assert_array_equal(heur, explicit_none)

    empty = np.zeros(ref.shape[:2], np.float32)
    unprotected = blend_with_structure_preservation(gen, ref, 0.92, True, 0.48, object_mask=empty)
    bag_err = lambda o: np.abs(o[70:110, 60:110].astype(float) - ref[70:110, 60:110].astype(float)).mean()
    assert bag_err(heur) < 20.0
    # (the global colour-cast cap already pulls an unprotected bag partly back, hence the modest margin)
    assert bag_err(unprotected) > bag_err(heur) + 5.0

    niche = np.zeros(ref.shape[:2], np.float32)
    niche[300:390, 150:390] = 1.0  # detector says: an object sits in the niche
    kept = blend_with_structure_preservation(gen, ref, 0.92, True, 0.48, object_mask=niche)
    assert np.abs(kept[320:370, 200:350].astype(float) - ref[320:370, 200:350].astype(float)).mean() < 12.0


class _FakeScheduler:
    """Minimal DPMSolver-like scheduler: add_noise uses the sigma at step_index (begin_index before the loop)."""

    def __init__(self, n_steps=6):
        import torch

        self.timesteps = torch.arange(n_steps * 100, 0, -100)
        self.sigmas = torch.linspace(1.0, 0.0, n_steps + 1)
        self.step_index = None
        self.begin_index = 0

    def add_noise(self, original_samples, noise, timesteps):
        idx = self.begin_index if self.step_index is None else self.step_index
        return original_samples + self.sigmas[idx] * noise


def test_latent_object_lock_keeps_masked_latents_on_the_noised_init():
    """The per-step callback replaces masked latents by add_noise(z0, noise, t_next), clean z0 after the last
    step, and leaves unmasked latents untouched."""
    import torch
    from app.pipeline.generative import LatentObjectLock

    sched = _FakeScheduler()
    mask = np.zeros((8, 10), np.float32)
    mask[2:5, 3:7] = 1.0
    lock = LatentObjectLock(sched, mask)
    lock.install()
    gen = torch.Generator().manual_seed(0)
    z0 = torch.randn(1, 4, 8, 10, generator=gen)
    noise = torch.randn(1, 4, 8, 10, generator=gen)
    sched.add_noise(z0, noise, sched.timesteps[:1])  # what prepare_latents does
    assert lock.z0 is not None and torch.equal(lock.z0, z0) and torch.equal(lock.noise, noise)

    m = torch.from_numpy(mask)[None, None] > 0.5
    for i in range(len(sched.timesteps)):
        sched.step_index = i + 1  # scheduler.step() has advanced
        latents = torch.randn(1, 4, 8, 10, generator=gen)
        out = lock.apply({"latents": latents.clone()})["latents"]
        expected = z0 if i + 1 >= len(sched.timesteps) else z0 + sched.sigmas[i + 1] * noise
        assert torch.allclose(out[m.expand_as(out)], expected[m.expand_as(expected)], atol=1e-6)
        assert torch.equal(out[~m.expand_as(out)], latents[~m.expand_as(latents)])
    assert lock.applied_steps == len(sched.timesteps)
    lock.uninstall()
    assert "add_noise" not in vars(sched) or vars(sched)["add_noise"].__name__ == "add_noise"


def test_latent_lock_is_noop_without_capture_or_latents():
    import torch
    from app.pipeline.generative import LatentObjectLock

    lock = LatentObjectLock(_FakeScheduler(), np.ones((4, 4), np.float32))
    lock.install()
    lat = torch.zeros(1, 4, 4, 4)
    assert lock.apply({"latents": lat})["latents"] is lat  # nothing captured yet
    assert lock.apply({}) == {}
    lock.uninstall()


@pytest.mark.parametrize("use_karras", [False, True])
def test_latent_lock_with_real_dpmsolver_and_img2img_start_index(use_karras):
    """Exercise actual scheduler.step/add_noise, including an img2img slice and a second render."""
    import torch
    from diffusers import DPMSolverMultistepScheduler
    from app.pipeline.generative import LatentObjectLock

    scheduler = DPMSolverMultistepScheduler(
        beta_schedule="scaled_linear", use_karras_sigmas=use_karras,
        algorithm_type="dpmsolver++",
    )
    rng = torch.Generator().manual_seed(42)
    clean = torch.randn((1, 4, 6, 8), generator=rng)
    noise = torch.randn(clean.shape, generator=rng)
    mask = np.zeros((6, 8), np.float32)
    mask[:, :4] = 1.0
    for begin in (7, 11):
        scheduler.set_timesteps(20)
        scheduler.set_begin_index(begin)
        timesteps = scheduler.timesteps[begin:]
        lock = LatentObjectLock(scheduler, mask)
        lock.install()
        try:
            latents = scheduler.add_noise(clean, noise, timesteps[:1])
            assert torch.equal(lock.z0, clean)
            for timestep in timesteps:
                raw = scheduler.step(torch.zeros_like(latents), timestep, latents).prev_sample
                sigma = scheduler.sigmas[scheduler.step_index]
                expected = (clean + sigma * noise) / torch.sqrt(1.0 + sigma ** 2)
                latents = lock.apply({"latents": raw})["latents"]
                torch.testing.assert_close(latents[..., :4], expected[..., :4])
                torch.testing.assert_close(latents[..., 4:], raw[..., 4:])
            torch.testing.assert_close(latents[..., :4], clean[..., :4])
            assert lock.applied_steps == len(timesteps)
        finally:
            lock.uninstall()
        assert "add_noise" not in vars(scheduler)


def test_object_mask_to_latent_shape_and_coverage():
    from app.pipeline.generative import object_mask_to_latent

    m = np.zeros((900, 1600), np.float32)
    m[400:470, 700:730] = 1.0  # a bottle: ~3 latents wide at 768x432
    lat = object_mask_to_latent(m, (432, 768))
    assert lat.shape == (54, 96) and lat.dtype == np.float32
    assert lat.max() > 0.6 and lat[:20].max() == 0.0


def test_generate_installs_latent_lock_with_detected_mask_and_toggles_freeu():
    """generate(): a given object_mask drives the latent lock (captured from the pipeline's own add_noise);
    FreeU stays off by default and is enabled only on request."""
    import torch

    gen = LocalControlNetGenerator()
    img = np.full((96, 128, 3), 150, dtype=np.uint8)
    obj = np.zeros((96, 128), np.float32)
    obj[30:70, 40:80] = 1.0
    sched = _FakeScheduler(n_steps=4)
    seen = {}

    def fake_pipe(**kwargs):
        w, h = kwargs["image"].size
        z0 = torch.zeros(1, 4, h // 8, w // 8)
        noise = torch.ones(1, 4, h // 8, w // 8)
        sched.step_index = None
        sched.add_noise(z0, noise, sched.timesteps[:1])
        cb = kwargs["callback_on_step_end"]
        assert kwargs["callback_on_step_end_tensor_inputs"] == ["latents"]
        for i in range(len(sched.timesteps)):
            sched.step_index = i + 1
            out = cb(None, i, sched.timesteps[i], {"latents": torch.full_like(z0, 5.0)})
        seen["final"] = out["latents"]
        return SimpleNamespace(images=[Image.fromarray(np.full((h, w, 3), 160, np.uint8))])

    pipe = MagicMock(side_effect=fake_pipe)
    pipe.scheduler = sched
    gen._pipeline = pipe
    gen.generate(img, generative_strength=0.48, object_mask=obj)
    lat = seen["final"][0, 0].numpy()
    assert lat.max() == pytest.approx(5.0) and lat.min() == pytest.approx(0.0)  # frozen -> clean z0 (0)
    assert gen.last_run_info["latent_lock_steps"] == 4
    assert gen.last_run_info["object_mask_source"] == "detected"
    pipe.disable_freeu.assert_called()
    pipe.enable_freeu.assert_not_called()

    gen.generate(img, generative_strength=0.48, object_mask=obj, freeu=True, latent_object_lock=False)
    pipe.enable_freeu.assert_called_once()
    assert "latent_lock_steps" not in gen.last_run_info


def test_combine_object_masks_keeps_detected_and_only_large_heuristic_blobs():
    from app.pipeline.generative import combine_object_masks

    det = np.zeros((200, 400), np.float32)
    det[20:40, 20:30] = 1.0  # small detected bottle: always kept
    heur = np.zeros((200, 400), np.float32)
    heur[100:160, 200:280] = 1.0  # handbag: 6% of the frame -> kept
    heur[10:14, 300:304] = 1.0  # tiny heuristic speck (0.02%) -> dropped
    out = combine_object_masks(det, heur, min_area_frac=0.005)
    assert out[30, 25] == pytest.approx(1.0)
    assert out[130, 240] == pytest.approx(1.0)
    assert out[12, 302] == 0.0


def test_generate_auto_detection_unions_with_heuristic_and_reports_metadata(monkeypatch):
    """Without an explicit mask, generate() asks the detector BEFORE loading the pipeline and records it."""
    gen = LocalControlNetGenerator()
    img = np.full((96, 128, 3), 150, dtype=np.uint8)
    order = []
    det = np.zeros((96, 128), np.float32)
    det[30:60, 40:70] = 1.0

    def fake_detect(ref):
        order.append("detect")
        gen.last_run_info["objects_detected"] = 1
        return det

    def fake_pipe(**kwargs):
        w, h = kwargs["image"].size
        return SimpleNamespace(images=[Image.fromarray(np.full((h, w, 3), 160, np.uint8))])

    pipe = MagicMock(side_effect=fake_pipe)
    real_ensure = gen._ensure_pipeline
    gen._pipeline = pipe

    def ensure(*a, **k):
        order.append("pipeline")
        return real_ensure(*a, **k)

    monkeypatch.setattr(gen, "_detect_object_mask", fake_detect)
    monkeypatch.setattr(gen, "_ensure_pipeline", ensure)
    gen.generate(img, generative_strength=0.48, latent_object_lock=False)
    assert order == ["detect", "pipeline"]
    assert gen.last_run_info["object_mask_source"] == "detected+heuristic"
    assert gen.last_run_info["objects_detected"] == 1

    # Detector unavailable -> heuristic fallback, no crash
    monkeypatch.setattr(gen, "_detect_object_mask", lambda ref: None)
    gen.generate(img, generative_strength=0.48, latent_object_lock=False)
    assert gen.last_run_info["object_mask_source"] == "heuristic"

    # Explicitly disabled -> detector not called
    monkeypatch.setattr(gen, "_detect_object_mask", lambda ref: pytest.fail("detector called"))
    gen.generate(img, generative_strength=0.48, detect_objects=False)

    # A successful detector call with zero objects is still only heuristic protection.
    monkeypatch.setattr(gen, "_detect_object_mask", lambda ref: np.zeros(ref.shape[:2], np.float32))
    gen.generate(img, generative_strength=0.48, latent_object_lock=False)
    assert gen.last_run_info["object_mask_source"] == "heuristic"
    gen.generate(img, generative_strength=0.48, preserve_geometry=False)
    assert gen.last_run_info["object_mask_source"] == "none"


def test_legacy_prompt_style_for_ab():
    gen = LocalControlNetGenerator()
    p_new, n_new = gen.get_prompt_for_profile("auto")
    p_old, n_old = gen.get_prompt_for_profile("auto", "legacy")
    assert "photograph" in p_new and "v-ray" in p_old.lower()
    assert n_new != n_old


def test_photo_finish_is_deterministic_and_keeps_mean_lightness():
    import cv2
    from app.pipeline.photo_finish import apply_photo_finish

    rng = np.random.default_rng(3)
    img = cv2.GaussianBlur(rng.integers(30, 250, (300, 500, 3)).astype(np.uint8), (0, 0), 3)
    a = apply_photo_finish(img, 0.5)
    b = apply_photo_finish(img, 0.5)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, img)
    l_in = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, 0].astype(np.float32).mean()
    for amount in (0.5, 1.0):
        out = apply_photo_finish(img, amount)
        l_out = cv2.cvtColor(out, cv2.COLOR_BGR2Lab)[:, :, 0].astype(np.float32).mean()
        assert abs(l_out - l_in) < 2.0, (amount, l_in, l_out)
    # Colour is not edited: mean a*/b* stay put
    lab_i = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
    lab_o = cv2.cvtColor(apply_photo_finish(img, 1.0), cv2.COLOR_BGR2Lab).astype(np.float32)
    assert np.abs(lab_i[..., 1:].mean((0, 1)) - lab_o[..., 1:].mean((0, 1))).max() < 1.0
    # amount 0 is identity; alpha and float dtype preserved
    np.testing.assert_array_equal(apply_photo_finish(img, 0.0), img)
    bgra = np.dstack([img, np.full(img.shape[:2], 77, np.uint8)])
    assert (apply_photo_finish(bgra, 0.5)[:, :, 3] == 77).all()
    f = apply_photo_finish(img.astype(np.float32) / 255.0, 0.5)
    assert f.dtype == np.float32 and f.shape == img.shape
    # Different seeds give different grain; tone-only has no grain
    assert not np.array_equal(apply_photo_finish(img, 0.5, seed=1), apply_photo_finish(img, 0.5, seed=2))
    np.testing.assert_array_equal(apply_photo_finish(img, 0.5, grain=False, seed=1),
                                  apply_photo_finish(img, 0.5, grain=False, seed=2))


def test_grain_stays_subtle_on_walls_and_suppressed_on_dark_lacquer():
    from app.pipeline.photo_finish import apply_photo_finish
    dark = np.full((540, 320, 3), 40, np.uint8)
    wall = np.full_like(dark, 150)
    dark_delta = apply_photo_finish(dark, 0.5, tone=False).astype(float) - dark
    wall_delta = apply_photo_finish(wall, 0.5, tone=False).astype(float) - wall
    assert 0.1 < float(wall_delta.std()) < 1.0
    assert float(dark_delta.std()) < float(wall_delta.std()) * 0.2
    assert abs(float(wall_delta.mean())) < 0.05


# ---------------------------------------------------------------------------------------------------------
# Milestone M2 (F6-F10): Aliases, Local-Only, Fireplace Synthesis, Unclamped Relight, Hue Fidelity, VRAM
# ---------------------------------------------------------------------------------------------------------

def test_local_relighting_generator_aliases_and_status_contract():
    from app.pipeline.generative import (
        CN_DEPTH_ID,
        ICLIGHT_REPO,
        LocalRelightingGenerator,
        get_relighting_generator,
    )

    assert LocalRelightingGenerator is LocalControlNetGenerator
    assert get_relighting_generator() is get_controlnet_generator()
    status = LocalRelightingGenerator().get_status()
    assert status["base_model"] == BASE_MODEL_ID
    assert status["controlnet_model"] == CONTROLNET_MODEL_ID
    assert status["relighting_model"] == ICLIGHT_REPO
    assert status["controlnet_models"] == [CN_DEPTH_ID, CONTROLNET_MODEL_ID]
    assert isinstance(status["iclight_cached"], bool)


def test_ensure_pipeline_enforces_local_files_only_and_offline_error(monkeypatch):
    gen = LocalControlNetGenerator()
    monkeypatch.setattr(gen, "is_cached_locally", lambda: False)
    with pytest.raises(RuntimeError, match="download_models.py --required"):
        gen._ensure_pipeline()


def test_ensure_pipeline_passes_local_files_only_true_to_all_loaders(monkeypatch, tmp_path):
    import diffusers
    import torch
    from safetensors.torch import save_file
    from app.pipeline.generative import CN_DEPTH_ID, ICLIGHT_FC_FILENAME, ICLIGHT_REPO

    gen = LocalControlNetGenerator()
    monkeypatch.setattr(gen, "is_cached_locally", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    fake_ic = tmp_path / "iclight_sd15_fc.safetensors"
    save_file({"conv_in.weight": torch.zeros((4, 8, 3, 3), dtype=torch.float32)}, str(fake_ic))

    calls = []

    class _FakeConv(torch.nn.Conv2d):
        pass

    class _FakeUnet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_in = _FakeConv(4, 4, 3, padding=1)

        def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
            return sample

    class _FakePipe:
        def __init__(self):
            self.unet = _FakeUnet()
            self.scheduler = SimpleNamespace(config={"num_train_timesteps": 1000})
            self.vae = SimpleNamespace(enable_tiling=lambda: None, enable_slicing=lambda: None)

        def set_progress_bar_config(self, **kwargs):
            pass

        def enable_vae_tiling(self):
            pass

        def enable_vae_slicing(self):
            pass

    class _FakeCN:
        @classmethod
        def from_pretrained(cls, repo_id, **kwargs):
            calls.append(("ControlNet", repo_id, kwargs.get("local_files_only")))
            return MagicMock()

    class _FakeSDPipe:
        @classmethod
        def from_pretrained(cls, repo_id, **kwargs):
            calls.append(("SDPipe", repo_id, kwargs.get("local_files_only")))
            return _FakePipe()

    monkeypatch.setattr(diffusers, "ControlNetModel", _FakeCN)
    monkeypatch.setattr(
        diffusers,
        "DPMSolverMultistepScheduler",
        SimpleNamespace(from_config=lambda cfg, **kw: MagicMock()),
    )
    monkeypatch.setattr(diffusers, "MultiControlNetModel", lambda cns: MagicMock())
    monkeypatch.setattr(diffusers, "StableDiffusionControlNetImg2ImgPipeline", _FakeSDPipe)
    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda repo_id, filename: str(fake_ic) if (repo_id == ICLIGHT_REPO and filename == ICLIGHT_FC_FILENAME) else None,
    )

    pipe = gen._ensure_pipeline()
    assert getattr(pipe, "_is_iclight", False) is True
    assert pipe.unet.conv_in.in_channels == 8
    assert ("ControlNet", CN_DEPTH_ID, True) in calls
    assert ("ControlNet", CONTROLNET_MODEL_ID, True) in calls
    assert ("SDPipe", BASE_MODEL_ID, True) in calls


def test_fireplace_crop_inpaint_unloads_iclight_before_loading_inpaint_and_cleans_up(monkeypatch):
    import diffusers
    import torch

    gen = LocalControlNetGenerator()
    events = []
    fake_ic_pipe = MagicMock()
    fake_ic_pipe._is_iclight = True
    gen._pipeline = fake_ic_pipe
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def fake_release_pipe():
        events.append("release_iclight")
        gen._pipeline = None

    class _FakeInpaintSDPipe:
        @classmethod
        def from_pretrained(cls, repo_id, **kwargs):
            events.append(("load_inpaint", kwargs.get("local_files_only")))
            assert gen._pipeline is None

            def _call(**call_kwargs):
                events.append("run_inpaint")
                return SimpleNamespace(images=[Image.fromarray(np.full((512, 512, 3), 120, dtype=np.uint8))])

            p = MagicMock(side_effect=_call)
            p.scheduler = SimpleNamespace(config={"num_train_timesteps": 1000}, add_noise=lambda x, n, t: x, timesteps=[1])
            p.vae = SimpleNamespace(enable_tiling=lambda: None, enable_slicing=lambda: None)
            return p

    monkeypatch.setattr(gen, "_release_pipeline", fake_release_pipe)
    monkeypatch.setattr(diffusers, "StableDiffusionImg2ImgPipeline", _FakeInpaintSDPipe)
    monkeypatch.setattr(
        diffusers,
        "DPMSolverMultistepScheduler",
        SimpleNamespace(from_config=lambda cfg, **kw: SimpleNamespace(add_noise=lambda x, n, t: x, timesteps=[1])),
    )
    monkeypatch.setattr(gen, "release_vram", lambda: events.append("release_vram"))

    img = np.full((256, 256, 3), 180, dtype=np.uint8)
    img[140:210, 80:176] = (24, 24, 26)
    out, patches = gen._synthesize_fireplaces_512(
        img, [(80, 140, 176, 210)], seed=42, generative_strength=0.48
    )
    assert out.shape == img.shape
    assert len(patches) == 1
    assert events == ["release_iclight", ("load_inpaint", True), "run_inpaint", "release_vram"]


def test_fireplace_synthesis_skips_tiny_boxes_already_lit_and_strength_zero(monkeypatch):
    gen = LocalControlNetGenerator()
    monkeypatch.setattr(
        gen,
        "_load_fireplace_inpaint_pipeline",
        lambda: pytest.fail("Inpaint pipeline must not load when skipped"),
    )

    # 1. strength=0.0
    img = np.full((256, 256, 3), 180, dtype=np.uint8)
    img[140:200, 80:170] = (24, 24, 26)
    out0, p0 = gen._synthesize_fireplaces_512(img, [(80, 140, 170, 200)], seed=42, generative_strength=0.0)
    assert p0 == [] and np.array_equal(out0, img)

    # 2. <24px tiny handle cutout
    tiny = np.full((256, 256, 3), 185, dtype=np.uint8)
    tiny[120:136, 120:136] = (20, 20, 22)
    out_tiny, p_tiny = gen._synthesize_fireplaces_512(tiny, [(120, 120, 136, 136)], seed=42, generative_strength=0.55)
    assert p_tiny == [] and np.array_equal(out_tiny, tiny)

    # 3. already-lit warm bright flames
    lit = np.full((256, 256, 3), 185, dtype=np.uint8)
    lit[140:210, 80:170] = (55, 165, 250)  # bright orange/yellow BGR
    out_lit, p_lit = gen._synthesize_fireplaces_512(lit, [(80, 140, 170, 210)], seed=42, generative_strength=0.55)
    assert p_lit == [] and np.array_equal(out_lit, lit)


def test_dark_hearth_fallback_detects_neutral_firebox_and_rejects_tall_cabinet_or_tiny_handle():
    import cv2
    from app.pipeline.generative import _detect_dark_hearth_fallback

    # Neutral dark firebox inside mantel surround -> detected
    fp_scene = np.full((512, 768, 3), (185, 185, 185), dtype=np.uint8)
    cv2.rectangle(fp_scene, (265, 328), (425, 442), (78, 78, 82), -1)
    cv2.rectangle(fp_scene, (276, 338), (414, 434), (25, 25, 28), -1)
    boxes = _detect_dark_hearth_fallback(fp_scene)
    assert len(boxes) == 1
    assert boxes[0][0] >= 270 and boxes[0][2] <= 420

    # Tall vertical cabinet niche (aspect=0.288) -> strictly rejected
    cab = np.full((512, 512, 3), 190, dtype=np.uint8)
    cab[130:390, 270:345] = (55, 70, 90)
    assert _detect_dark_hearth_fallback(cab) == []

    # Sub-24px dark handle -> strictly rejected
    handle = np.full((256, 256, 3), 185, dtype=np.uint8)
    handle[120:136, 120:136] = (18, 18, 20)
    assert _detect_dark_hearth_fallback(handle) == []


def test_f8_unclamped_relighting_preserves_spot_cone_above_legacy_3_0_cap():
    import cv2

    h, w = 256, 256
    ref = np.full((h, w, 3), (175, 178, 182), dtype=np.uint8)
    gen = ref.copy()
    # Inject a strong warm spot cone (> 16 Lab units shift) on upper wall
    yy, xx = np.mgrid[0:h, 0:w]
    cone = np.exp(-0.5 * (((xx - 128.0) / 38.0) ** 2 + ((yy - 90.0) / 48.0) ** 2))
    gen_f = gen.astype(np.float32)
    gen_f[:, :, 2] = np.clip(gen_f[:, :, 2] + 34.0 * cone, 0, 255)
    gen_f[:, :, 1] = np.clip(gen_f[:, :, 1] + 14.0 * cone, 0, 255)
    gen_f[:, :, 0] = np.clip(gen_f[:, :, 0] - 24.0 * cone, 0, 255)
    gen = np.round(gen_f).astype(np.uint8)

    out_legacy = blend_with_structure_preservation(gen, ref, 0.92, True, 0.48, unclamped_relight=False)
    spot_mask = (cone > 0.5).astype(np.uint8) * 255
    out_unclamped = blend_with_structure_preservation(
        gen, ref, 1.0, True, 0.48, unclamped_relight=True, spot_mask=spot_mask,
    )

    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2Lab).astype(np.float32)
    leg_lab = cv2.cvtColor(out_legacy, cv2.COLOR_BGR2Lab).astype(np.float32)
    unc_lab = cv2.cvtColor(out_unclamped, cv2.COLOR_BGR2Lab).astype(np.float32)

    roi = (slice(60, 120), slice(100, 156))
    db_leg = float((leg_lab[roi][..., 2] - ref_lab[roi][..., 2]).mean())
    db_unc = float((unc_lab[roi][..., 2] - ref_lab[roi][..., 2]).mean())
    assert db_leg < 6.5
    assert db_unc > db_leg + 6.0 and db_unc > 10.0, f"Unclamped spot cone was crushed: db_unc={db_unc:.2f}, db_leg={db_leg:.2f}"


def test_f8_luminance_modulated_text_keeps_tv_logo_crisp_on_darkened_screen():
    import cv2
    from app.pipeline.generative import apply_luminance_modulated_preservation

    ref = np.full((256, 256, 3), (180, 180, 180), dtype=np.uint8)
    cv2.rectangle(ref, (48, 64), (208, 192), (42, 42, 46), -1)  # dark TV
    cv2.putText(ref, "ELLEGANZA", (62, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)

    # Relit output darkens TV background slightly and blurs text
    relit = cv2.GaussianBlur(ref, (7, 7), 2.5)
    relit = np.clip(relit.astype(np.float32) * 0.80, 0, 255).astype(np.uint8)

    txt_mask = np.zeros((256, 256), dtype=np.float32)
    txt_mask[64:192, 48:208] = 1.0

    out = apply_luminance_modulated_preservation(
        ref, relit, object_mask=txt_mask, text_logo_mask=txt_mask, generative_strength=0.55
    )
    # Screen background deepens slightly, while bright white strokes stay crisp & bright (>200)
    bg_l_ref = float(cv2.cvtColor(ref, cv2.COLOR_BGR2Lab)[72:96, 60:190, 0].mean())
    bg_l_out = float(cv2.cvtColor(out, cv2.COLOR_BGR2Lab)[72:96, 60:190, 0].mean())
    assert bg_l_out < bg_l_ref - 2.0
    assert float(out[112:142, 62:196].max()) > 210.0


def test_f8_fireplace_bypass_protects_synthesized_flames_from_inpaint_and_detail_overwrite():
    import cv2
    from app.pipeline.generative import build_fireplace_bypass_mask

    ref = np.full((256, 256, 3), (170, 170, 172), dtype=np.uint8)
    ref[120:200, 80:176] = (40, 40, 42)  # flat dead firebox in reference

    gen = ref.copy()
    # Synthesized bright flame edges inside firebox
    cv2.line(gen, (110, 185), (128, 135), (60, 190, 255), 3)
    cv2.line(gen, (145, 185), (132, 140), (80, 210, 255), 3)

    bypass = build_fireplace_bypass_mask((256, 256), [(80, 120, 176, 200)])
    out = blend_with_structure_preservation(
        gen,
        ref,
        1.0,
        True,
        0.55,
        work_scale=2.0,
        bypass_mask=bypass,
        unclamped_relight=True,
    )
    # Synthesized bright orange flame pixels must survive inside the firebox
    assert float(out[140:180, 110:150, 2].max()) > 210.0
    assert float(out[140:180, 110:150, 0].min()) < 100.0


def test_f9_stabilize_material_hues_blocks_neutral_wall_jaundice_and_keeps_wood_navy_hues():
    import cv2
    from app.pipeline.generative import resolve_profile_spec, stabilize_material_hues

    h, w = 256, 256
    ref = np.full((h, w, 3), (225, 225, 225), dtype=np.uint8)  # neutral white/grey wall
    ref[120:240, 20:110] = (95, 140, 185)  # light oak
    ref[120:240, 140:230] = (95, 65, 42)   # dark charcoal-navy

    # Simulate off-Planckian jaundice + brown shift on navy
    relit = ref.copy().astype(np.float32)
    relit[:, :, 2] += 14.0
    relit[:, :, 1] += 18.0  # green-yellow shift
    relit[:, :, 0] -= 20.0
    relit[120:240, 140:230] = (65, 90, 120)  # navy drifted brown
    relit_u8 = np.clip(relit, 0, 255).astype(np.uint8)

    dummy_analysis = SimpleNamespace(
        illumination_map_bgr=np.full((h, w, 3), (122, 114, 108), dtype=np.uint8),  # unlit neutral
        fireplace_boxes=[],
        window_mask=np.zeros((h, w), dtype=np.uint8),
    )
    spec = resolve_profile_spec("banheiro", ref_u8=ref, analysis=dummy_analysis)
    fixed = stabilize_material_hues(relit_u8, ref, dummy_analysis, spec)

    fixed_lab = cv2.cvtColor(fixed, cv2.COLOR_BGR2Lab).astype(np.float32)
    relit_lab = cv2.cvtColor(relit_u8, cv2.COLOR_BGR2Lab).astype(np.float32)
    orig_b_shift = abs(float(relit_lab[10:90, 10:240, 2].mean()) - 128.0)
    wall_b_shift = abs(float(fixed_lab[10:90, 10:240, 2].mean()) - 128.0)
    assert wall_b_shift < 6.0 and wall_b_shift < 0.4 * orig_b_shift, (
        f"Neutral wall remained yellowed: b_shift={wall_b_shift:.2f} (orig={orig_b_shift:.2f})"
    )
    # Navy cabinet retains cool negative b* (< 126) instead of brown (> 135)
    navy_b = float(fixed_lab[140:220, 160:210, 2].mean())
    assert navy_b < 126.0, f"Charcoal-navy drifted brown: b={navy_b:.2f}"


def test_f9_resolve_profile_spec_auto_adapts_for_outdoor_neutral_vs_fireplace_scenes():
    from app.pipeline.generative import resolve_profile_spec

    outdoor_img = np.full((128, 128, 3), (220, 220, 220), dtype=np.uint8)
    # Bright sky strip + 2D isotropic textured foliage (>3% dense foliage + exterior corroboration)
    outdoor_img[:28, :] = (235, 195, 150)
    yy, xx = np.mgrid[28:128, 64:128]
    leaf_mod = (((xx // 2 + yy // 2) % 2) * 32 - 16) + (((xx * 3 + yy * 5) % 3) * 14 - 14)
    foliage_patch = np.zeros((100, 64, 3), dtype=np.int32)
    foliage_patch[:, :, 0] = 35 + leaf_mod // 2
    foliage_patch[:, :, 1] = 115 + leaf_mod
    foliage_patch[:, :, 2] = 42 + leaf_mod // 2
    outdoor_img[28:128, 64:128] = np.clip(foliage_patch, 0, 255).astype(np.uint8)
    win_mask = np.zeros((128, 128), dtype=np.uint8)
    win_mask[:28, 64:128] = 255
    ana_out = SimpleNamespace(fireplace_boxes=[], window_mask=win_mask)
    spec_out = resolve_profile_spec("outdoor", ref_u8=outdoor_img, analysis=ana_out)
    spec_auto_out = resolve_profile_spec("auto", ref_u8=outdoor_img, analysis=ana_out)
    assert spec_out.profile == "externa"
    assert spec_auto_out.emitter_warmth_scale < 0.75
    assert "outdoor" in spec_auto_out.prompt.lower()

    ana_fp = SimpleNamespace(fireplace_boxes=[(20, 40, 60, 80)], window_mask=np.zeros((128, 128), dtype=np.uint8))
    spec_fp = resolve_profile_spec("auto", ref_u8=np.full((128, 128, 3), 160, dtype=np.uint8), analysis=ana_fp)
    assert spec_fp.emitter_warmth_scale >= 1.05

    # Outdoor patio with fireplace/grill must still select outdoor prompt
    ana_out_fp = SimpleNamespace(fireplace_boxes=[(10, 80, 40, 110)], window_mask=win_mask)
    spec_out_fp = resolve_profile_spec("auto", ref_u8=outdoor_img, analysis=ana_out_fp)
    assert "outdoor" in spec_out_fp.prompt.lower()
    assert 0.75 <= spec_out_fp.emitter_warmth_scale <= 0.95


def test_generate_iclight_clears_tensors_and_releases_vram_on_exception(monkeypatch):
    gen = LocalControlNetGenerator()
    calls = []

    def failing_pipe(**kwargs):
        raise RuntimeError("Simulated mid-diffusion error")

    fake_pipe = MagicMock(side_effect=failing_pipe)
    fake_pipe._is_iclight = True
    fake_pipe._n_controlnets = 2
    fake_pipe.scheduler = _FakeScheduler(n_steps=4)
    gen._pipeline = fake_pipe

    monkeypatch.setattr("app.pipeline.generative._encode_iclight_latent", lambda p, b: np.zeros((1, 4, 8, 8)))
    monkeypatch.setattr(gen, "release_vram", lambda: calls.append("release_vram"))

    img = np.full((128, 128, 3), 150, dtype=np.uint8)
    dummy_analysis = SimpleNamespace(
        depth_map=np.full((128, 128), 0.5, dtype=np.float32),
        illumination_map_bgr=np.full((128, 128, 3), 120, dtype=np.uint8),
        spot_mask=np.zeros((128, 128), dtype=np.uint8),
        fireplace_boxes=[],
        window_mask=np.zeros((128, 128), dtype=np.uint8),
        protection_mask=np.zeros((128, 128), dtype=np.float32),
        text_logo_mask=np.zeros((128, 128), dtype=np.float32),
    )
    with pytest.raises(RuntimeError, match="Simulated mid-diffusion error"):
        gen.generate(img, generative_strength=0.48, analysis=dummy_analysis)
    assert calls == ["release_vram"]


def test_m2_prompts_with_dynamic_emitters_fit_clip_77_tokens(clip_tokenizer):
    for text in SCENE_PROMPTS.values():
        full = f"{text}, warm glowing fireplace light, soft window daylight, warm concealed linear glow"
        assert len(clip_tokenizer(full).input_ids) <= 77, full


# ---------------------------------------------------------------------------------------------------------
# Milestone M2 Iteration 2 Regression Tests
# ---------------------------------------------------------------------------------------------------------

def test_compute_uniform_border_mask_ignores_hearth_floor_near_dark_cutout():
    """Uniform hearth floor below a dark firebox cutout must NOT be treated as a cut-view border."""
    import cv2
    from app.pipeline.generative import compute_uniform_border_mask

    bgr = np.full((512, 768, 3), (185, 185, 185), dtype=np.uint8)
    cv2.rectangle(bgr, (265, 328), (425, 442), (78, 78, 82), -1)
    cv2.rectangle(bgr, (276, 338), (414, 434), (25, 25, 28), -1)
    bgr[442:512, :] = (165, 165, 168)

    mask_auto = compute_uniform_border_mask(bgr)
    assert not mask_auto[442:, :].any()

    mask_fp = compute_uniform_border_mask(bgr, fireplace_boxes=[(276, 338, 414, 434)])
    assert not mask_fp[442:, :].any()


def test_dark_hearth_fallback_rejects_top_clipped_and_exclude_sliced_tall_cabinets():
    """Top-clipped or exclude_mask-sliced tall dark cabinets must NOT be sliced into stubby fireboxes."""
    from app.pipeline.generative import _detect_dark_hearth_fallback

    # 1. Tall dark cabinet extending above 0.32*h (true aspect=0.50..0.80)
    for bw in (120, 156, 192):
        img = np.full((512, 768, 3), 185, dtype=np.uint8)
        img[40:280, 280 : 280 + bw] = (22, 22, 24)
        assert _detect_dark_hearth_fallback(img) == [], f"Tall cabinet bw={bw} misclassified as firebox"

    # 2. Tall dark cabinet (120x200, aspect=0.60) partially covered by exclude_mask on top half
    img2 = np.full((512, 768, 3), 185, dtype=np.uint8)
    img2[180:380, 300:420] = (22, 22, 24)
    excl = np.zeros((512, 768), dtype=np.float32)
    excl[180:290, 300:420] = 1.0
    assert _detect_dark_hearth_fallback(img2, exclude_mask=excl) == []


def test_inject_fallback_fireplace_illum_clamps_oob_boxes():
    """Out-of-bounds and edge-clipped fireplace boxes must never raise ValueError or produce NaNs."""
    from app.pipeline.generative import _inject_fallback_fireplace_illum

    illum = np.full((512, 768, 3), 120, dtype=np.uint8)
    depth = np.full((512, 768), 0.5, dtype=np.float32)
    for box in [(-30, -10, 80, 120), (700, 450, 850, 600), (900, 600, 1000, 700), (-50, -50, -10, -10)]:
        out = _inject_fallback_fireplace_illum(illum, depth, [box])
        assert out.shape == illum.shape and out.dtype == np.uint8


def test_is_fireplace_already_lit_detects_overexposed_white_flame_core():
    """Bright white / warm-white overexposed flame cores (L > 215) count as already lit."""
    from app.pipeline.generative import _is_fireplace_already_lit

    img_white = np.full((256, 256, 3), 180, dtype=np.uint8)
    img_white[120:200, 80:180] = (24, 24, 26)
    img_white[145:175, 115:145] = (245, 252, 255)
    assert _is_fireplace_already_lit(img_white, (80, 120, 180, 200)) is True

    img_warm_white = np.full((256, 256, 3), 180, dtype=np.uint8)
    img_warm_white[120:200, 80:180] = (225, 245, 255)
    assert _is_fireplace_already_lit(img_warm_white, (80, 120, 180, 200)) is True


def test_build_fireplace_bypass_mask_multi_box_blur_isolated():
    """Adding additional boxes must not re-blur box 1's bypass mask."""
    from app.pipeline.generative import build_fireplace_bypass_mask

    m1 = build_fireplace_bypass_mask((512, 768), [(100, 200, 220, 300)])
    m3 = build_fireplace_bypass_mask(
        (512, 768),
        [(100, 200, 220, 300), (350, 200, 470, 300), (550, 200, 670, 300)],
    )
    assert float(np.max(np.abs(m1[190:310, 90:230] - m3[190:310, 90:230]))) < 1e-6


def test_stabilize_material_hues_white_unlit_blue_drift_and_subtle_charcoal_navy():
    """Verify strong unlit pure-white anchoring in quarto_sala, cool blue drift cap, and low-chroma navy."""
    import cv2
    from app.pipeline.generative import resolve_profile_spec, stabilize_material_hues

    h, w = 256, 256
    unlit_ana = SimpleNamespace(
        illumination_map_bgr=np.full((h, w, 3), (122, 114, 108), dtype=np.uint8),
        fireplace_boxes=[],
        window_mask=np.zeros((h, w), dtype=np.uint8),
    )

    # 1. Unlit pure white wall (L~245) in quarto_sala with strong yellow drift
    ref_white = np.full((h, w, 3), (242, 242, 242), dtype=np.uint8)
    relit_yel = np.full((h, w, 3), (200, 242, 245), dtype=np.uint8)
    spec_qs = resolve_profile_spec("quarto_sala", ref_u8=ref_white, analysis=unlit_ana)
    fixed_white = stabilize_material_hues(relit_yel, ref_white, unlit_ana, spec_qs)
    db_orig = float(cv2.cvtColor(relit_yel, cv2.COLOR_BGR2Lab)[:, :, 2].mean()) - 128.0
    db_fix = float(cv2.cvtColor(fixed_white, cv2.COLOR_BGR2Lab)[:, :, 2].mean()) - 128.0
    assert abs(db_fix) < 5.0 and abs(db_fix) < 0.35 * abs(db_orig), (db_orig, db_fix)

    # 2. Neutral gray unlit cool blue drift (db < 0) in banheiro
    ref_gray = np.full((h, w, 3), (160, 160, 160), dtype=np.uint8)
    relit_blue = np.full((h, w, 3), (215, 160, 120), dtype=np.uint8)
    spec_b = resolve_profile_spec("banheiro", ref_u8=ref_gray, analysis=unlit_ana)
    fixed_blue = stabilize_material_hues(relit_blue, ref_gray, unlit_ana, spec_b)
    db_blue_fix = float(cv2.cvtColor(fixed_blue, cv2.COLOR_BGR2Lab)[:, :, 2].mean()) - 128.0
    assert db_blue_fix > -7.0, f"Cool blue drift not capped: {db_blue_fix:.2f}"

    # 3. Subtle low-chroma charcoal-navy (68, 56, 48 -> b0=119) must never cross b=128
    ref_navy = np.full((h, w, 3), (68, 56, 48), dtype=np.uint8)
    relit_brown = np.full((h, w, 3), (48, 62, 78), dtype=np.uint8)
    fixed_navy = stabilize_material_hues(relit_brown, ref_navy, unlit_ana, spec_qs)
    b_navy = float(cv2.cvtColor(fixed_navy, cv2.COLOR_BGR2Lab)[:, :, 2].mean())
    assert b_navy < 128.0, f"Subtle charcoal-navy crossed b=128: {b_navy:.2f}"


def test_resolve_profile_spec_auto_does_not_misclassify_indoor_sage_cabinet_as_outdoor():
    """Smooth indoor sage-green cabinetry without windows must NOT trigger the 'externa' prompt."""
    from app.pipeline.generative import resolve_profile_spec

    img = np.full((128, 128, 3), (195, 195, 195), dtype=np.uint8)
    img[:, :52] = (95, 130, 100)  # 40% flat sage-green cabinetry
    ana = SimpleNamespace(fireplace_boxes=[], window_mask=np.zeros((128, 128), dtype=np.uint8))
    spec = resolve_profile_spec("auto", ref_u8=img, analysis=ana)
    assert "outdoor" not in spec.prompt.lower()


def test_float32_lab_chain_eliminates_ceiling_contour_banding():
    """Continuous float32 Lab pipeline eliminates integer contour steps on smooth bright white ceilings."""
    import cv2
    from app.pipeline.generative import resolve_profile_spec, stabilize_material_hues

    h, w = 256, 256
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot(xx - 128.0, yy - 128.0) / 96.0
    halo = np.exp(-0.5 * r * r)
    # Smooth bright ceiling gradient L* ~ 215 with subtle warm radial halo
    ref_f32 = np.full((h, w, 3), 215.0 / 255.0, dtype=np.float32)
    gen_f32 = ref_f32.copy()
    gen_f32[:, :, 2] = np.clip(gen_f32[:, :, 2] + 0.04 * halo, 0.0, 1.0)
    gen_f32[:, :, 1] = np.clip(gen_f32[:, :, 1] + 0.025 * halo, 0.0, 1.0)
    gen_f32[:, :, 0] = np.clip(gen_f32[:, :, 0] - 0.015 * halo, 0.0, 1.0)

    ana = SimpleNamespace(
        illumination_map_bgr=np.full((h, w, 3), (122, 114, 108), dtype=np.uint8),
        fireplace_boxes=[],
        window_mask=np.zeros((h, w), dtype=np.uint8),
    )
    spec = resolve_profile_spec("auto", ref_u8=(ref_f32 * 255).astype(np.uint8), analysis=ana)
    stab_f32 = stabilize_material_hues(gen_f32, ref_f32, ana, spec)
    assert stab_f32.dtype == np.float32
    out = blend_with_structure_preservation(
        stab_f32, ref_f32, blend_alpha=1.0, preserve_geometry=True, work_scale=2.0, unclamped_relight=True
    )
    assert out.dtype == np.float32
    # Check smooth monotonic profile along center row (max second-difference tiny)
    row = out[128, 40:216, 2] * 255.0
    second_diff = np.abs(np.diff(row, n=2))
    assert float(second_diff.max()) < 0.6, f"Contour banding detected: max_d2={second_diff.max():.3f}"


def test_generate_accepts_image_bgr_kwarg_and_3d_yellow_led_mask_and_skips_redundant_remove(monkeypatch):
    """Covers image_bgr alias, 3D yellow led_mask.max(axis=2), prompt_style alignment, and single LED removal."""
    from app.pipeline import led as _led
    from app.pipeline import lighting_map as _lm

    gen = LocalControlNetGenerator()
    fake_pipe = MagicMock(side_effect=lambda **kw: SimpleNamespace(
        images=[Image.fromarray(np.full((kw["image"].size[1], kw["image"].size[0], 3), 160, dtype=np.uint8))]
    ))
    fake_pipe._is_iclight = True
    fake_pipe._n_controlnets = 2
    fake_pipe.scheduler = _FakeScheduler(n_steps=4)
    gen._pipeline = fake_pipe
    monkeypatch.setattr("app.pipeline.generative._encode_iclight_latent", lambda p, b: np.zeros((1, 4, 8, 8)))

    seen_masks = []

    def fake_analyze(bgr, **kwargs):
        seen_masks.append(kwargs.get("led_mask"))
        h, w = bgr.shape[:2]
        return SimpleNamespace(
            depth_map=np.full((h, w), 0.5, dtype=np.float32),
            illumination_map_bgr=np.full((h, w, 3), 120, dtype=np.uint8),
            spot_mask=np.zeros((h, w), dtype=np.uint8),
            fireplace_boxes=[],
            window_mask=np.zeros((h, w), dtype=np.uint8),
            protection_mask=np.zeros((h, w), dtype=np.float32),
            text_logo_mask=np.zeros((h, w), dtype=np.float32),
        )

    monkeypatch.setattr(_lm, "analyze_scene_lighting", fake_analyze)

    # 1. image_bgr kwarg + 3D yellow BGR led_mask -> max(axis=2) keeps nonzero mask; prompt_style == "photo"
    img = np.full((96, 128, 3), 160, dtype=np.uint8)
    led_3d = np.zeros((96, 128, 3), dtype=np.uint8)
    led_3d[40:44, 20:100] = (0, 238, 255)
    out = gen.generate(image_bgr=img, generative_strength=0.48, led_mask=led_3d, prompt_style="legacy")
    assert out.shape == img.shape
    assert seen_masks[-1].shape == (96, 128)
    assert bool(seen_masks[-1].any()) is True
    assert gen.last_run_info["prompt_style"] == "photo"

    # 2. When reference_bgr is None and image contains yellow marker, remove_led_strips is called only once (inside split_led_markers)
    remove_calls = []
    orig_remove = _led.remove_led_strips
    monkeypatch.setattr(_led, "remove_led_strips", lambda im, mk: (remove_calls.append(1), orig_remove(im, mk))[1])
    img_yel = np.full((96, 128, 3), 160, dtype=np.uint8)
    img_yel[40:44, 20:100] = (0, 238, 255)
    gen.generate(img_yel, generative_strength=0.48)
    assert len(remove_calls) == 1


def test_generate_mid_render_exception_frees_hooks_and_releases_pipeline(monkeypatch):
    """Mid-render and mid-load exceptions must call maybe_free_model_hooks, clear _pipeline, and release VRAM."""
    import diffusers
    import torch

    # 1. Mid-diffusion exception on cached IC-Light pipe
    gen = LocalControlNetGenerator()
    events = []
    fake_pipe = MagicMock(side_effect=RuntimeError("mid-step boom"))
    fake_pipe._is_iclight = True
    fake_pipe._n_controlnets = 2
    fake_pipe.scheduler = _FakeScheduler(n_steps=4)
    fake_pipe.maybe_free_model_hooks = MagicMock(side_effect=lambda: events.append("maybe_free_hooks"))
    fake_pipe.remove_all_hooks = MagicMock(side_effect=lambda: events.append("remove_all_hooks"))
    gen._pipeline = fake_pipe

    monkeypatch.setattr("app.pipeline.generative._encode_iclight_latent", lambda p, b: np.zeros((1, 4, 8, 8)))
    monkeypatch.setattr(gen, "release_vram", lambda: events.append("release_vram"))

    img = np.full((128, 128, 3), 150, dtype=np.uint8)
    dummy_ana = SimpleNamespace(
        depth_map=np.full((128, 128), 0.5, dtype=np.float32),
        illumination_map_bgr=np.full((128, 128, 3), 120, dtype=np.uint8),
        spot_mask=np.zeros((128, 128), dtype=np.uint8),
        fireplace_boxes=[],
        window_mask=np.zeros((128, 128), dtype=np.uint8),
        protection_mask=np.zeros((128, 128), dtype=np.float32),
        text_logo_mask=np.zeros((128, 128), dtype=np.float32),
    )
    with pytest.raises(RuntimeError, match="mid-step boom"):
        gen.generate(img, generative_strength=0.48, analysis=dummy_ana)
    assert gen._pipeline is None
    assert "maybe_free_hooks" in events
    assert "remove_all_hooks" in events
    assert events.count("release_vram") == 1

    # 2. Mid-load failure inside _ensure_pipeline must also call release_vram()
    gen2 = LocalControlNetGenerator()
    rel2 = []
    monkeypatch.setattr(gen2, "is_cached_locally", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(gen2, "release_vram", lambda: rel2.append("release_vram"))
    monkeypatch.setattr(diffusers.ControlNetModel, "from_pretrained", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(diffusers, "MultiControlNetModel", lambda cns: MagicMock())
    monkeypatch.setattr(
        diffusers.StableDiffusionControlNetImg2ImgPipeline,
        "from_pretrained",
        MagicMock(side_effect=FileNotFoundError("corrupt local shard")),
    )
    with pytest.raises(FileNotFoundError, match="corrupt local shard"):
        gen2._ensure_pipeline()
    assert gen2._pipeline is None
    assert rel2 == ["release_vram"]


def test_wide_linear_firebox_lower_half_not_suppressed_by_generic_protection_mask(monkeypatch):
    """Zero-text uniformly dark horizontal firebox in lower half is NOT suppressed by generic protection_mask."""
    import cv2

    gen = LocalControlNetGenerator()
    fake_pipe = MagicMock(side_effect=lambda **kw: SimpleNamespace(
        images=[Image.fromarray(np.full((kw["image"].size[1], kw["image"].size[0], 3), 160, dtype=np.uint8))]
    ))
    fake_pipe._is_iclight = True
    fake_pipe._n_controlnets = 2
    fake_pipe.scheduler = _FakeScheduler(n_steps=4)
    gen._pipeline = fake_pipe
    monkeypatch.setattr("app.pipeline.generative._encode_iclight_latent", lambda p, b: np.zeros((1, 4, 8, 8)))

    seen_fp_boxes = []

    def fake_synth(work_bgr, fp_boxes, **kwargs):
        seen_fp_boxes.append(list(fp_boxes))
        return work_bgr, []

    monkeypatch.setattr(gen, "_synthesize_fireplaces_512", fake_synth)

    bgr = np.full((512, 768, 3), (182, 182, 185), dtype=np.uint8)
    bgr[380:512, :] = (160, 160, 162)
    cv2.rectangle(bgr, (180, 290), (588, 376), (72, 72, 76), -1)
    cv2.rectangle(bgr, (192, 300), (576, 366), (22, 22, 24), -1)
    prot = np.zeros((512, 768), dtype=np.float32)
    prot[290:376, 180:588] = 1.0  # generic DINO screen false-positive

    ana = SimpleNamespace(
        depth_map=np.full((512, 768), 0.5, dtype=np.float32),
        illumination_map_bgr=np.full((512, 768, 3), 120, dtype=np.uint8),
        spot_mask=np.zeros((512, 768), dtype=np.uint8),
        fireplace_boxes=[],
        window_mask=np.zeros((512, 768), dtype=np.uint8),
        protection_mask=prot,
        text_logo_mask=np.zeros((512, 768), dtype=np.float32),
    )
    gen.generate(bgr, generative_strength=0.60, analysis=ana)
    assert len(seen_fp_boxes) == 1 and len(seen_fp_boxes[0]) == 1

    # Whereas if text_logo_mask covers the box (e.g. TV with logo), it MUST stay blocked
    ana_tv = SimpleNamespace(
        depth_map=ana.depth_map,
        illumination_map_bgr=ana.illumination_map_bgr,
        spot_mask=ana.spot_mask,
        fireplace_boxes=[],
        window_mask=ana.window_mask,
        protection_mask=prot,
        text_logo_mask=prot.copy(),
    )
    gen.generate(bgr, generative_strength=0.60, analysis=ana_tv)
    assert seen_fp_boxes[-1] == []


def test_highlight_safe_relight_init_lifts_hearth_spill_and_spot_cones():
    """Hearth floor spill and ceiling spot cones receive warm luminous lift rather than amb_atten darkening."""
    from app.pipeline.generative import _build_highlight_safe_relight_init, _inject_fallback_fireplace_illum

    h, w = 256, 384
    clean = np.full((h, w, 3), 160, dtype=np.uint8)
    depth = np.linspace(0.25, 0.85, h, dtype=np.float32)[:, None] * np.ones((1, w), dtype=np.float32)
    illum = np.full((h, w, 3), (108, 114, 122), dtype=np.uint8)
    illum = _inject_fallback_fireplace_illum(illum, depth, [(130, 145, 250, 200)])

    init = _build_highlight_safe_relight_init(
        clean,
        illum,
        depth,
        strength=0.60,
        warmth_scale=1.10,
        specular_bloom=0.25,
        fireplace_boxes=[(130, 145, 250, 200)],
    )
    hearth_roi = init[205:245, 145:235].astype(np.float32)
    clean_roi = clean[205:245, 145:235].astype(np.float32)
    assert float(hearth_roi.mean()) > float(clean_roi.mean()) + 5.0
    assert float((hearth_roi[:, :, 2] - hearth_roi[:, :, 0]).mean()) > 12.0


# ---------------------------------------------------------------------------------------------------------
# Milestone M2 Iteration 3 Regression Tests (Defects #1–#5)
# ---------------------------------------------------------------------------------------------------------

def test_blank_dark_tv_185_and_219_stay_protected_not_hijacked_as_firebox(monkeypatch):
    """Blank dark TVs (16:9 / 1.85:1 / 21:9, protection_mask=1.0, text_logo_mask=0.0) must NOT trigger flame synthesis."""
    gen = LocalControlNetGenerator()
    fake_pipe = MagicMock(side_effect=lambda **kw: SimpleNamespace(
        images=[Image.fromarray(np.full((kw["image"].size[1], kw["image"].size[0], 3), 160, dtype=np.uint8))]
    ))
    fake_pipe._is_iclight = True
    fake_pipe._n_controlnets = 2
    fake_pipe.scheduler = _FakeScheduler(n_steps=4)
    gen._pipeline = fake_pipe
    monkeypatch.setattr("app.pipeline.generative._encode_iclight_latent", lambda p, b: np.zeros((1, 4, 8, 8)))

    seen_fp_boxes = []
    seen_obj_masks = []

    def fake_synth(work_bgr, fp_boxes, **kwargs):
        seen_fp_boxes.append(list(fp_boxes))
        return work_bgr, []

    orig_blend = blend_with_structure_preservation

    def spy_blend(*args, **kwargs):
        seen_obj_masks.append(kwargs.get("object_mask"))
        return orig_blend(*args, **kwargs)

    monkeypatch.setattr(gen, "_synthesize_fireplaces_512", fake_synth)
    monkeypatch.setattr("app.pipeline.generative.blend_with_structure_preservation", spy_blend)

    # Case 1: 1.85:1 blank dark TV at cy=0.52*h (200x108)
    # Case 2: 21:9 ultrawide blank dark TV at cy=0.60*h (240x100, aspect=2.40)
    for x1, y1, x2, y2 in [(284, 212, 484, 320), (264, 257, 504, 357)]:
        bgr = np.full((512, 768, 3), (185, 185, 188), dtype=np.uint8)
        bgr[y1:y2, x1:x2] = (20, 20, 22)
        prot = np.zeros((512, 768), dtype=np.float32)
        prot[y1:y2, x1:x2] = 1.0
        ana = SimpleNamespace(
            depth_map=np.full((512, 768), 0.5, dtype=np.float32),
            illumination_map_bgr=np.full((512, 768, 3), 120, dtype=np.uint8),
            spot_mask=np.zeros((512, 768), dtype=np.uint8),
            fireplace_boxes=[],
            window_mask=np.zeros((512, 768), dtype=np.uint8),
            protection_mask=prot,
            text_logo_mask=np.zeros((512, 768), dtype=np.float32),
        )
        gen.generate(bgr, generative_strength=0.60, analysis=ana)
        assert seen_fp_boxes[-1] == [], f"Blank dark TV {(x1, y1, x2, y2)} hijacked: {seen_fp_boxes[-1]}"
        assert seen_obj_masks[-1] is not None
        assert float(seen_obj_masks[-1][y1 + 5 : y2 - 5, x1 + 5 : x2 - 5].min()) > 0.9


def test_compute_uniform_border_mask_hearth_apron_and_dark_plinth_frame():
    """Hearth floor behind >25px stone apron stays unfrozen, while partnered cut-view frames above dark plinths stay frozen."""
    import cv2
    from app.pipeline.generative import compute_uniform_border_mask

    # 1. ledge2: 32px stone apron [410:442] between dark firebox [310:410] and uniform hearth floor [442:512]
    rng = np.random.default_rng(7)
    bgr_ledge2 = rng.integers(140, 210, size=(512, 768, 3), dtype=np.uint8)
    bgr_ledge2[310:410, 276:414] = (25, 25, 28)
    bgr_ledge2[410:442, :] = (130, 130, 132)
    bgr_ledge2[442:512, :] = (165, 165, 168)
    mask_ledge2 = compute_uniform_border_mask(bgr_ledge2, fireplace_boxes=None)
    assert int(mask_ledge2[442:, :].sum()) == 0, f"Hearth floor frozen: {int(mask_ledge2[442:, :].sum())} px"

    # 2. True partnered bottom+right cut-view frame above a dark furniture plinth
    bgr_plinth = rng.integers(120, 195, size=(300, 400, 3), dtype=np.uint8)
    bgr_plinth[220:260, 100:250] = (30, 30, 32)  # dark sofa/cabinet plinth
    bgr_plinth[260:300, :] = (225, 216, 212)      # uniform bottom frame
    bgr_plinth[:, 350:400] = (225, 216, 212)      # matching uniform right frame
    mask_plinth = compute_uniform_border_mask(bgr_plinth)
    assert bool(mask_plinth[265:, :].all()) is True
    assert bool(mask_plinth[:, 355:].all()) is True


def test_build_fireplace_bypass_mask_mixed_size_and_edge_cases():
    """Mixed-size and overlapping boxes blur independently without cross-contaminating smaller boxes."""
    from app.pipeline.generative import build_fireplace_bypass_mask

    box_small = (100, 200, 220, 300)
    box_large = (320, 120, 720, 420)
    m_single = build_fireplace_bypass_mask((512, 768), [box_small])
    m_mixed = build_fireplace_bypass_mask((512, 768), [box_small, box_large])
    diff = float(np.max(np.abs(m_single[190:310, 90:230] - m_mixed[190:310, 90:230])))
    assert diff < 1e-6, f"Mixed-size box contaminated box 1 blur: max_diff={diff}"

    m_overlap = build_fireplace_bypass_mask((512, 768), [(100, 200, 220, 300), (150, 220, 270, 320)])
    assert float(m_overlap.min()) >= 0.0 and float(m_overlap.max()) <= 1.0
    assert float(np.max(np.abs(m_single[:, :135] - m_overlap[:, :135]))) < 1e-6

    m_empty = build_fireplace_bypass_mask(
        (512, 768),
        [(-50, -50, -10, -10), (900, 600, 1000, 700), (200, 200, 200, 200), (300, 300, 200, 200), (10, 10, 25, 25)],
    )
    assert float(m_empty.sum()) == 0.0


def test_analytic_float32_lab_preserves_120px_smooth_gradient_across_4_stage_chain():
    """Vectorized analytic sRGB <-> CIE Lab preserves 120/120 monotonic L* steps with zero LUT sawtooth reversals."""
    from app.pipeline.generative import _bgr_to_lab255_f32, resolve_profile_spec, stabilize_material_hues

    h, w = 256, 120
    ramp = np.linspace(200.0 / 255.0, 230.0 / 255.0, w, dtype=np.float32)
    ref_f32 = np.broadcast_to(ramp[None, :, None], (h, w, 3)).copy()
    gen_f32 = ref_f32.copy()

    l_direct = _bgr_to_lab255_f32(ref_f32)[128, :, 0]
    assert len(np.unique(np.round(l_direct, 5))) == w
    assert float(np.diff(l_direct).min()) > 0.2

    ana = SimpleNamespace(
        illumination_map_bgr=np.full((h, w, 3), (122, 114, 108), dtype=np.uint8),
        fireplace_boxes=[],
        window_mask=np.zeros((h, w), dtype=np.uint8),
    )
    obj_mask = np.zeros((h, w), dtype=np.float32)
    obj_mask[64:192, :] = 0.75  # exercise Stage 3 (luminance-modulated preservation) & Stage 4 (detail transfer)
    spec = resolve_profile_spec("auto", ref_u8=(ref_f32 * 255).astype(np.uint8), analysis=ana)
    s1 = stabilize_material_hues(gen_f32, ref_f32, ana, spec)
    s4 = blend_with_structure_preservation(
        s1,
        ref_f32,
        blend_alpha=1.0,
        preserve_geometry=True,
        object_mask=obj_mask,
        work_scale=2.0,
        unclamped_relight=True,
    )
    assert s4.dtype == np.float32
    l_out = _bgr_to_lab255_f32(s4)[128, :, 0]
    d1 = np.diff(l_out)
    d2 = np.abs(np.diff(l_out, n=2))
    assert len(np.unique(np.round(l_out, 5))) == w
    assert float(d1.min()) > 0.0, f"Non-monotonic reversal: min_d1={d1.min():.6f}"
    assert float(d2.max()) < 0.01, f"LUT sawtooth jitter: max_d2={d2.max():.6f}"

    harm = harmonize_chromatic_lab(gen_f32, ref_f32, chroma_retention=0.72, luma_retention=0.70)
    assert harm.dtype == np.float32
    l_harm = _bgr_to_lab255_f32(harm)[128, :, 0]
    assert float(np.diff(l_harm).min()) > 0.0
    assert float(np.abs(np.diff(l_harm, n=2)).max()) < 0.01


def test_f9_quarto_sala_cool_drift_continuous_slate_sweep_and_green_cabinet_rejection():
    """Covers F9 cool blue drift in quarto_sala, C1-smooth slate sweep across b0=125, and green indoor cabinet rejection."""
    import cv2
    from app.pipeline.generative import _bgr_to_lab255_f32, resolve_profile_spec, stabilize_material_hues

    h, w = 128, 128
    unlit_ana = SimpleNamespace(
        illumination_map_bgr=np.full((h, w, 3), (122, 114, 108), dtype=np.uint8),
        fireplace_boxes=[],
        window_mask=np.zeros((h, w), dtype=np.uint8),
    )

    # 1. quarto_sala neutral gray cool blue drift must stay strongly anchored (db_fix > -7.0)
    ref_gray = np.full((h, w, 3), (160, 160, 160), dtype=np.uint8)
    relit_blue = np.full((h, w, 3), (215, 160, 120), dtype=np.uint8)
    spec_qs = resolve_profile_spec("quarto_sala", ref_u8=ref_gray, analysis=unlit_ana)
    fixed_qs = stabilize_material_hues(relit_blue, ref_gray, unlit_ana, spec_qs)
    db_qs = float(cv2.cvtColor(fixed_qs, cv2.COLOR_BGR2Lab)[:, :, 2].mean()) - 128.0
    assert db_qs > -7.0, f"quarto_sala cool blue drift not capped: {db_qs:.2f}"

    # Real saturated blue wall (c0 >= 12) stays anchored near its blue hue; window daylight allows mild cool shift
    ref_blue_wall = np.full((h, w, 3), (190, 145, 110), dtype=np.uint8)
    fixed_blue_wall = stabilize_material_hues(ref_blue_wall, ref_blue_wall, unlit_ana, spec_qs)
    assert float(_bgr_to_lab255_f32(fixed_blue_wall)[:, :, 2].mean()) < 115.0
    win_ana = SimpleNamespace(
        illumination_map_bgr=np.full((h, w, 3), (122, 114, 108), dtype=np.uint8),
        fireplace_boxes=[],
        window_mask=np.full((h, w), 255, dtype=np.uint8),
    )
    fixed_win = stabilize_material_hues(relit_blue, ref_gray, win_ana, spec_qs)
    assert float(_bgr_to_lab255_f32(fixed_win)[:, :, 2].mean()) < float(_bgr_to_lab255_f32(fixed_qs)[:, :, 2].mean())

    # 2. Smooth continuous slate sweep across B in [68, 64, 60, 58, 56, 54, 52, 48]
    b_vals = (68, 64, 60, 58, 56, 54, 52, 48)
    b_outputs = []
    # The render pipeline stabilizes hues before rounding to 8-bit output.
    relit_warm = np.full((h, w, 3), (48, 62, 78), dtype=np.float32) / 255.0
    for b_val in b_vals:
        ref_slate = np.full((h, w, 3), (b_val, 56, 48), dtype=np.uint8)
        fixed_slate = stabilize_material_hues(relit_warm, ref_slate, unlit_ana, spec_qs)
        b_outputs.append(float(_bgr_to_lab255_f32(fixed_slate)[64, 64, 2]))
    assert b_outputs[4] < 130.0, f"Slate (56,56,48) jumped warm brown: b={b_outputs[4]:.2f}"
    assert all(b_outputs[i + 1] >= b_outputs[i] - 0.1 for i in range(len(b_outputs) - 1)), b_outputs
    # Smooth across b0=125.0 boundary (B in [60, 58, 56]) without cliff jump
    assert max(abs(b_outputs[i + 1] - b_outputs[i]) for i in (2, 3)) < 1.5, b_outputs
    # Warm oak and walnut retain warm wood tones without cool-slate suppression
    for wood_bgr in ((95, 140, 185), (40, 62, 95)):
        ref_wood = np.full((h, w, 3), wood_bgr, dtype=np.uint8)
        fixed_wood = stabilize_material_hues(ref_wood, ref_wood, unlit_ana, spec_qs)
        assert float(_bgr_to_lab255_f32(fixed_wood)[64, 64, 2]) > 142.0

    # 3. 40% vivid-green (55, 135, 70) flat AND seamed indoor cabinets with window_mask=0 stay interior
    for with_seams in (False, True):
        cab = np.full((128, 128, 3), (195, 195, 195), dtype=np.uint8)
        cab[:, :52] = (55, 135, 70)
        if with_seams:
            cab[:, 25] = (35, 95, 45)
            cab[64, :52] = (35, 95, 45)
        spec_cab = resolve_profile_spec("auto", ref_u8=cab, analysis=unlit_ana)
        assert "outdoor" not in spec_cab.prompt.lower()
