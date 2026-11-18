import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { useEnhancerStore, DEFAULT_PARAMETERS, PRESETS } from './useEnhancerStore';
import * as apiClient from '@/api/client';
import * as imageLib from '@/lib/image';

describe('useEnhancerStore', () => {
  beforeEach(() => {
    useEnhancerStore.getState().resetParameters();
    useEnhancerStore.getState().clearImage();
    useEnhancerStore.getState().setSliderPos(50);
    useEnhancerStore.getState().setZoom(1);
    useEnhancerStore.getState().setIsSpacePressed(false);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('initializes with default calibrated parameters and realismo_ia preset', () => {
    const state = useEnhancerStore.getState();
    expect(state.parameters.temperature).toBe(DEFAULT_PARAMETERS.temperature);
    expect(state.parameters.shadow_lift).toBe(DEFAULT_PARAMETERS.shadow_lift);
    expect(state.parameters.clahe_clip_limit).toBe(DEFAULT_PARAMETERS.clahe_clip_limit);
    expect(state.parameters.preserve_geometry).toBe(true);
    expect(state.parameters.render_mode).toBe('generative');
    expect(state.parameters.generative_strength).toBe(0.48);
    expect(state.parameters.scene_profile).toBe('auto');
    expect(state.parameters.specular_bloom).toBe(0.25);
    expect(state.activePreset).toBe('realismo_ia');
  });

  it('applies marcenaria preset correctly with wood texture parameters and classic mode', () => {
    useEnhancerStore.getState().applyPreset('marcenaria');
    const state = useEnhancerStore.getState();
    expect(state.activePreset).toBe('marcenaria');
    expect(state.parameters.temperature).toBe(PRESETS.marcenaria.params.temperature);
    expect(state.parameters.shadow_lift).toBe(PRESETS.marcenaria.params.shadow_lift);
    expect(state.parameters.sharpness).toBe(PRESETS.marcenaria.params.sharpness);
    expect(state.parameters.render_mode).toBe('classic');
    expect(state.parameters.specular_bloom).toBe(PRESETS.marcenaria.params.specular_bloom);
  });

  it('applies luz_quente preset correctly with warm Kelvin temperature and classic mode', () => {
    useEnhancerStore.getState().applyPreset('luz_quente');
    const state = useEnhancerStore.getState();
    expect(state.activePreset).toBe('luz_quente');
    expect(state.parameters.temperature).toBe(PRESETS.luz_quente.params.temperature);
    expect(state.parameters.highlight_recovery).toBe(PRESETS.luz_quente.params.highlight_recovery);
    expect(state.parameters.render_mode).toBe('classic');
    expect(state.parameters.specular_bloom).toBe(PRESETS.luz_quente.params.specular_bloom);
  });

  it('applies ultra_detalhe preset correctly with high CLAHE, AO and classic mode', () => {
    useEnhancerStore.getState().applyPreset('ultra_detalhe');
    const state = useEnhancerStore.getState();
    expect(state.activePreset).toBe('ultra_detalhe');
    expect(state.parameters.clahe_clip_limit).toBe(PRESETS.ultra_detalhe.params.clahe_clip_limit);
    expect(state.parameters.ambient_occlusion).toBe(PRESETS.ultra_detalhe.params.ambient_occlusion);
    expect(state.parameters.sharpness).toBe(PRESETS.ultra_detalhe.params.sharpness);
    expect(state.parameters.render_mode).toBe('classic');
    expect(state.parameters.specular_bloom).toBe(PRESETS.ultra_detalhe.params.specular_bloom);
  });

  it('forces render_mode to classic and sets calibrated specular_bloom across all classic presets', () => {
    const classicKeys = ['natural', 'marcenaria', 'luz_quente', 'ultra_detalhe'] as const;
    for (const key of classicKeys) {
      useEnhancerStore.getState().applyPreset('realismo_ia');
      expect(useEnhancerStore.getState().parameters.render_mode).toBe('generative');

      useEnhancerStore.getState().applyPreset(key);
      const state = useEnhancerStore.getState();
      expect(state.activePreset).toBe(key);
      expect(state.parameters.render_mode).toBe('classic');
      expect(state.parameters.specular_bloom).toBe(PRESETS[key].params.specular_bloom);
    }
  });

  it('applies realismo_ia preset from classic state and restores generative defaults', () => {
    useEnhancerStore.getState().applyPreset('natural');
    useEnhancerStore.getState().updateParameter('specular_bloom', 0.85);
    useEnhancerStore.getState().updateParameter('scene_profile', 'cozinha');
    useEnhancerStore.getState().updateParameter('generative_strength', 0.75);

    useEnhancerStore.getState().applyPreset('realismo_ia');
    const state = useEnhancerStore.getState();
    expect(state.activePreset).toBe('realismo_ia');
    expect(state.parameters.render_mode).toBe('generative');
    expect(state.parameters.generative_strength).toBe(0.48);
    expect(state.parameters.scene_profile).toBe('auto');
    expect(state.parameters.specular_bloom).toBe(0.25);
  });

  it('switches activePreset to personalizado when updating a single parameter', () => {
    useEnhancerStore.getState().applyPreset('natural');
    useEnhancerStore.getState().updateParameter('exposure', 0.45);
    const state = useEnhancerStore.getState();
    expect(state.activePreset).toBe('personalizado');
    expect(state.parameters.exposure).toBe(0.45);
  });

  it('updates render_mode, generative_strength, scene_profile, and specular_bloom via updateParameter', () => {
    useEnhancerStore.getState().updateParameter('render_mode', 'classic');
    expect(useEnhancerStore.getState().parameters.render_mode).toBe('classic');
    expect(useEnhancerStore.getState().activePreset).toBe('personalizado');

    useEnhancerStore.getState().applyPreset('realismo_ia');
    useEnhancerStore.getState().updateParameter('generative_strength', 0.72);
    expect(useEnhancerStore.getState().parameters.generative_strength).toBe(0.72);
    expect(useEnhancerStore.getState().activePreset).toBe('personalizado');

    useEnhancerStore.getState().applyPreset('realismo_ia');
    useEnhancerStore.getState().updateParameter('scene_profile', 'banheiro');
    expect(useEnhancerStore.getState().parameters.scene_profile).toBe('banheiro');
    expect(useEnhancerStore.getState().activePreset).toBe('personalizado');

    useEnhancerStore.getState().applyPreset('realismo_ia');
    useEnhancerStore.getState().updateParameter('specular_bloom', 0.65);
    expect(useEnhancerStore.getState().parameters.specular_bloom).toBe(0.65);
    expect(useEnhancerStore.getState().activePreset).toBe('personalizado');
  });

  it('resets parameters to DEFAULT_PARAMETERS and realismo_ia cleanly', () => {
    useEnhancerStore.getState().updateParameter('temperature', 0.9);
    useEnhancerStore.getState().updateParameter('clahe_clip_limit', 4.5);
    useEnhancerStore.getState().updateParameter('render_mode', 'classic');
    useEnhancerStore.getState().updateParameter('generative_strength', 0.9);
    useEnhancerStore.getState().updateParameter('scene_profile', 'cozinha');
    useEnhancerStore.getState().updateParameter('specular_bloom', 0.8);
    useEnhancerStore.getState().resetParameters();
    const state = useEnhancerStore.getState();
    expect(state.parameters.temperature).toBe(DEFAULT_PARAMETERS.temperature);
    expect(state.parameters.clahe_clip_limit).toBe(DEFAULT_PARAMETERS.clahe_clip_limit);
    expect(state.parameters.render_mode).toBe('generative');
    expect(state.parameters.generative_strength).toBe(0.48);
    expect(state.parameters.scene_profile).toBe('auto');
    expect(state.parameters.specular_bloom).toBe(0.25);
    expect(state.activePreset).toBe('realismo_ia');
  });

  it('clamps slider position strictly between 0 and 100', () => {
    useEnhancerStore.getState().setSliderPos(150);
    expect(useEnhancerStore.getState().sliderPos).toBe(100);

    useEnhancerStore.getState().setSliderPos(-25);
    expect(useEnhancerStore.getState().sliderPos).toBe(0);

    useEnhancerStore.getState().setSliderPos(65);
    expect(useEnhancerStore.getState().sliderPos).toBe(65);
  });

  it('clamps zoom scale between 0.5 and 4.0', () => {
    useEnhancerStore.getState().setZoom(5.5);
    expect(useEnhancerStore.getState().zoom).toBe(4.0);

    useEnhancerStore.getState().setZoom(0.2);
    expect(useEnhancerStore.getState().zoom).toBe(0.5);

    useEnhancerStore.getState().setZoom(1.5);
    expect(useEnhancerStore.getState().zoom).toBe(1.5);
  });

  it('manages spacebar hold toggle state', () => {
    expect(useEnhancerStore.getState().isSpacePressed).toBe(false);
    useEnhancerStore.getState().setIsSpacePressed(true);
    expect(useEnhancerStore.getState().isSpacePressed).toBe(true);
    useEnhancerStore.getState().setIsSpacePressed(false);
    expect(useEnhancerStore.getState().isSpacePressed).toBe(false);
  });

  it('sets original image and initializes job metadata', () => {
    useEnhancerStore.getState().setOriginalImage({
      file: null,
      objectUrl: 'blob:mock-url',
      width: 1920,
      height: 1080,
      name: 'kitchen.jpg',
      size: 500000,
    });

    const state = useEnhancerStore.getState();
    expect(state.original).not.toBeNull();
    expect(state.original?.width).toBe(1920);
    expect(state.original?.height).toBe(1080);
    expect(state.original?.name).toBe('kitchen.jpg');
    expect(state.job.status).toBe('idle');
  });

  it('clears active polling interval when clearImage is called', async () => {
    vi.useFakeTimers();
    vi.spyOn(apiClient, 'requestPreview').mockResolvedValue('blob:preview-1');
    vi.spyOn(apiClient, 'submitEnhancement').mockResolvedValue({
      job_id: 'job-clear-test',
      status: 'processing',
    });
    const pollSpy = vi.spyOn(apiClient, 'pollJob').mockResolvedValue({
      job_id: 'job-clear-test',
      status: 'processing',
      progress: 35,
      stage: 'Executando super-resolução...',
    });

    const mockFile = new File(['img'], 'room.jpg', { type: 'image/jpeg' });
    useEnhancerStore.getState().setOriginalImage({
      file: mockFile,
      objectUrl: 'blob:img-1',
      width: 1280,
      height: 720,
      name: 'room.jpg',
      size: 1024,
    });

    await useEnhancerStore.getState().startEnhancement();
    await vi.advanceTimersByTimeAsync(750);
    expect(pollSpy).toHaveBeenCalledTimes(1);

    useEnhancerStore.getState().clearImage();
    await vi.advanceTimersByTimeAsync(3000);

    expect(pollSpy).toHaveBeenCalledTimes(1);
    expect(useEnhancerStore.getState().job.status).toBe('idle');
    expect(useEnhancerStore.getState().original).toBeNull();
  });

  it('clears active polling interval when setOriginalImage switches images', async () => {
    vi.useFakeTimers();
    vi.spyOn(apiClient, 'requestPreview').mockResolvedValue('blob:preview-switch');
    vi.spyOn(apiClient, 'submitEnhancement').mockResolvedValue({
      job_id: 'job-switch-test',
      status: 'processing',
    });
    const pollSpy = vi.spyOn(apiClient, 'pollJob').mockResolvedValue({
      job_id: 'job-switch-test',
      status: 'processing',
      progress: 50,
      stage: 'Processando...',
    });

    useEnhancerStore.getState().setOriginalImage({
      file: new File(['img1'], 'first.jpg', { type: 'image/jpeg' }),
      objectUrl: 'blob:img-1',
      width: 1280,
      height: 720,
      name: 'first.jpg',
      size: 1024,
    });

    await useEnhancerStore.getState().startEnhancement();
    await vi.advanceTimersByTimeAsync(750);
    expect(pollSpy).toHaveBeenCalledTimes(1);

    useEnhancerStore.getState().setOriginalImage({
      file: new File(['img2'], 'second.jpg', { type: 'image/jpeg' }),
      objectUrl: 'blob:img-2',
      width: 1920,
      height: 1080,
      name: 'second.jpg',
      size: 2048,
    });

    await vi.advanceTimersByTimeAsync(3000);

    expect(pollSpy).toHaveBeenCalledTimes(1);
    expect(useEnhancerStore.getState().original?.name).toBe('second.jpg');
    expect(useEnhancerStore.getState().job.status).toBe('idle');
  });

  it('clears enhanced4kUrl on updateParameter, applyPreset, and resetParameters while triggering debounced preview', async () => {
    vi.useFakeTimers();
    const previewSpy = vi
      .spyOn(apiClient, 'requestPreview')
      .mockResolvedValue('blob:preview-updated');
    vi.spyOn(apiClient, 'submitEnhancement').mockResolvedValue({
      job_id: 'job-4k-done',
      status: 'processing',
    });
    vi.spyOn(apiClient, 'pollJob').mockResolvedValue({
      job_id: 'job-4k-done',
      status: 'completed',
      progress: 100,
      stage: 'Concluído!',
    });

    const mockFile = new File(['img'], 'suite.jpg', { type: 'image/jpeg' });
    useEnhancerStore.getState().setOriginalImage({
      file: mockFile,
      objectUrl: 'blob:suite',
      width: 1920,
      height: 1080,
      name: 'suite.jpg',
      size: 2048,
    });
    await vi.advanceTimersByTimeAsync(150);

    await useEnhancerStore.getState().startEnhancement();
    await vi.advanceTimersByTimeAsync(750);
    expect(useEnhancerStore.getState().enhanced4kUrl).toBe('/api/download/job-4k-done');

    // Editing specular_bloom clears enhanced4kUrl and triggers live preview
    useEnhancerStore.getState().updateParameter('specular_bloom', 0.55);
    expect(useEnhancerStore.getState().enhanced4kUrl).toBeNull();
    await vi.advanceTimersByTimeAsync(150);
    expect(previewSpy).toHaveBeenCalledTimes(2);

    // Re-complete 4K job, then applyPreset('personalizado') should not clear or trigger preview
    await useEnhancerStore.getState().startEnhancement();
    await vi.advanceTimersByTimeAsync(750);
    expect(useEnhancerStore.getState().enhanced4kUrl).toBe('/api/download/job-4k-done');
    useEnhancerStore.getState().applyPreset('personalizado');
    expect(useEnhancerStore.getState().enhanced4kUrl).toBe('/api/download/job-4k-done');

    // Concrete preset clears enhanced4kUrl and triggers preview
    useEnhancerStore.getState().applyPreset('marcenaria');
    expect(useEnhancerStore.getState().enhanced4kUrl).toBeNull();
    await vi.advanceTimersByTimeAsync(150);
    expect(previewSpy).toHaveBeenCalledTimes(3);

    // Re-complete 4K job, then resetParameters clears enhanced4kUrl and restores realismo_ia
    await useEnhancerStore.getState().startEnhancement();
    await vi.advanceTimersByTimeAsync(750);
    expect(useEnhancerStore.getState().enhanced4kUrl).toBe('/api/download/job-4k-done');
    useEnhancerStore.getState().resetParameters();
    expect(useEnhancerStore.getState().enhanced4kUrl).toBeNull();
    expect(useEnhancerStore.getState().activePreset).toBe('realismo_ia');
    await vi.advanceTimersByTimeAsync(150);
    expect(previewSpy).toHaveBeenCalledTimes(4);
  });

  it('forwards R5 parameters in requestPreview and submitEnhancement and tracks HF download stage updates', async () => {
    vi.useFakeTimers();
    const previewSpy = vi
      .spyOn(apiClient, 'requestPreview')
      .mockResolvedValue('blob:preview-r5');
    const submitSpy = vi
      .spyOn(apiClient, 'submitEnhancement')
      .mockResolvedValue({ job_id: 'job-hf-download', status: 'processing' });
    const pollSpy = vi
      .spyOn(apiClient, 'pollJob')
      .mockResolvedValueOnce({
        job_id: 'job-hf-download',
        status: 'processing',
        progress: 22,
        stage: 'Baixando pesos emilianJR/epiCRealism do HuggingFace (1.2 GB / 2.1 GB)...',
      })
      .mockResolvedValueOnce({
        job_id: 'job-hf-download',
        status: 'completed',
        progress: 100,
        stage: 'Renderização Realismo IA + 4K concluída',
      });

    const mockFile = new File(['img'], 'cozinha.jpg', { type: 'image/jpeg' });
    useEnhancerStore.getState().setOriginalImage({
      file: mockFile,
      objectUrl: 'blob:cozinha',
      width: 1920,
      height: 1080,
      name: 'cozinha.jpg',
      size: 4096,
    });

    useEnhancerStore.getState().updateParameter('scene_profile', 'cozinha');
    useEnhancerStore.getState().updateParameter('generative_strength', 0.62);
    useEnhancerStore.getState().updateParameter('specular_bloom', 0.42);
    await vi.advanceTimersByTimeAsync(150);

    expect(previewSpy).toHaveBeenCalledTimes(1);
    const previewParams = previewSpy.mock.calls[0][1];
    expect(previewParams.render_mode).toBe('generative');
    expect(previewParams.scene_profile).toBe('cozinha');
    expect(previewParams.generative_strength).toBe(0.62);
    expect(previewParams.specular_bloom).toBe(0.42);

    await useEnhancerStore.getState().startEnhancement();
    expect(submitSpy).toHaveBeenCalledTimes(1);
    const submitParams = submitSpy.mock.calls[0][1];
    expect(submitParams.render_mode).toBe('generative');
    expect(submitParams.scene_profile).toBe('cozinha');
    expect(submitParams.generative_strength).toBe(0.62);
    expect(submitParams.specular_bloom).toBe(0.42);

    await vi.advanceTimersByTimeAsync(750);
    expect(pollSpy).toHaveBeenCalledTimes(1);
    expect(useEnhancerStore.getState().job.progress).toBe(22);
    expect(useEnhancerStore.getState().job.stage).toContain('HuggingFace');

    await vi.advanceTimersByTimeAsync(750);
    expect(pollSpy).toHaveBeenCalledTimes(2);
    expect(useEnhancerStore.getState().job.status).toBe('completed');
    expect(useEnhancerStore.getState().enhanced4kUrl).toBe('/api/download/job-hf-download');
  });

  it('serializes R5 FormData fields in client requests and parses /api/capabilities', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch');
    const createObjectUrlSpy = vi
      .spyOn(URL, 'createObjectURL')
      .mockReturnValue('blob:mock-client-preview');

    fetchMock.mockResolvedValueOnce(
      new Response(new Blob(['preview-bytes'], { type: 'image/jpeg' }), {
        status: 200,
      })
    );

    const file = new File(['data'], 'banheiro.png', { type: 'image/png' });
    const params = {
      ...DEFAULT_PARAMETERS,
      render_mode: 'generative' as const,
      generative_strength: 0.55,
      scene_profile: 'banheiro' as const,
      specular_bloom: 0.35,
    };

    const previewBlobUrl = await apiClient.requestPreview(file, params);
    expect(previewBlobUrl).toBe('blob:mock-client-preview');
    expect(createObjectUrlSpy).toHaveBeenCalledTimes(1);

    const [previewReqUrl, previewReqInit] = fetchMock.mock.calls[0];
    expect(previewReqUrl).toBe('/api/preview');
    const sentForm = previewReqInit?.body as FormData;
    expect(sentForm.get('render_mode')).toBe('generative');
    expect(sentForm.get('generative_strength')).toBe('0.55');
    expect(sentForm.get('scene_profile')).toBe('banheiro');
    expect(sentForm.get('specular_bloom')).toBe('0.35');
    expect(sentForm.get('photo_finish')).toBe('0.5');

    const mockCapabilities = {
      engine: 'ClassicEnhancer',
      version: '1.0.0',
      max_scale: 4,
      preserves_geometry: true,
      deterministic: true,
      photometric_stages: ['color_temperature_tint_exposure'],
      super_resolution_engine: 'Real-ESRGAN-ONNX',
      hardware: {
        cuda_available: true,
        device_name: 'NVIDIA GeForce RTX 2060',
        free_mb: 5100,
        total_mb: 6144,
        allocated_mb: 1044,
        reserved_mb: 1200,
        source: 'cuda',
      },
      render_modes: ['classic', 'generative'],
      scene_profiles: ['auto', 'banheiro', 'cozinha', 'quarto_sala'],
      generative_engine: {
        available: true,
        base_model: 'emilianJR/epiCRealism',
        controlnet_model: 'lllyasviel/control_v11p_sd15_canny',
        dtype: 'float16',
        scheduler: 'DPMSolverMultistepScheduler',
        cached_locally: false,
      },
    };

    fetchMock.mockResolvedValueOnce(
      new Response(JSON.stringify(mockCapabilities), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    );

    const caps = await apiClient.fetchCapabilities();
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/capabilities');
    expect(caps.render_modes).toEqual(['classic', 'generative']);
    expect(caps.generative_engine.base_model).toBe('emilianJR/epiCRealism');
  });
});

describe('primeira experiência e recursos opcionais', () => {
  const featureStatus = (ready: boolean, state: 'idle' | 'running' | 'failed' | 'done' = 'idle', extra = {}) => ({
    ready,
    available: true,
    reason: null,
    warning: null,
    download_bytes: ready ? 0 : 67051650,
    disk_bytes: 67051650,
    summary: '',
    task: {
      feature: 'upscale_4x' as const,
      state,
      progress: state === 'done' ? 100 : 0,
      stage: '',
      error: state === 'failed' ? 'sem conexão' : null,
      unavailable: false,
      bytes_done: 0,
      bytes_total: 0,
    },
    ...extra,
  });
  const featuresResponse = (upscaleReady: boolean, realismoReady: boolean, upscaleState: any = 'idle') => ({
    bundled: true,
    features: {
      upscale_4x: featureStatus(upscaleReady, upscaleState),
      realismo_ia: featureStatus(realismoReady),
    },
  });

  beforeEach(() => {
    useEnhancerStore.setState({ features: null, pendingActivation: null, featuresError: null, sampleError: null });
    useEnhancerStore.getState().resetParameters();
    useEnhancerStore.getState().clearImage();
  });

  afterEach(() => {
    useEnhancerStore.setState({ features: null, pendingActivation: null, featuresError: null });
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('a amostra usa a rota real em modo Clássico, resolução original, e libera o download', async () => {
    vi.useFakeTimers();
    const sampleFile = new File(['jpg'], 'amostra_promob.jpg', { type: 'image/jpeg' });
    vi.spyOn(apiClient, 'fetchSampleFile').mockResolvedValue(sampleFile);
    vi.spyOn(imageLib, 'readImageSource').mockResolvedValue({
      file: sampleFile,
      objectUrl: 'blob:amostra',
      width: 1024,
      height: 698,
      name: 'amostra_promob.jpg',
      size: 3,
    });
    vi.spyOn(apiClient, 'requestPreview').mockResolvedValue('blob:preview-amostra');
    const submitSpy = vi.spyOn(apiClient, 'submitEnhancement').mockResolvedValue({ job_id: 'job-amostra', status: 'processing' });
    vi.spyOn(apiClient, 'pollJob').mockResolvedValue({
      job_id: 'job-amostra',
      status: 'completed',
      progress: 100,
      stage: 'ok',
      width: 1024,
      height: 698,
    });

    await useEnhancerStore.getState().trySample();
    expect(submitSpy).toHaveBeenCalledTimes(1);
    const params = submitSpy.mock.calls[0][1];
    expect(params.render_mode).toBe('classic');
    expect(params.upscale_factor).toBe(1);

    await vi.advanceTimersByTimeAsync(750);
    const state = useEnhancerStore.getState();
    expect(state.job.status).toBe('completed');
    expect(state.enhanced4kUrl).toBe('/api/download/job-amostra');
    expect(state.original?.objectUrl).toBe('blob:amostra');
  });

  it('mostra erro amigável se a amostra não carregar', async () => {
    vi.spyOn(apiClient, 'fetchSampleFile').mockRejectedValue(new Error('Amostra não encontrada'));
    await useEnhancerStore.getState().trySample();
    expect(useEnhancerStore.getState().sampleError).toBe('Amostra não encontrada');
    expect(useEnhancerStore.getState().sampleLoading).toBe(false);
  });

  it('sem modelos baixados, o padrão vira Clássico em resolução original', async () => {
    vi.spyOn(apiClient, 'fetchFeatures').mockResolvedValue(featuresResponse(false, false));
    expect(useEnhancerStore.getState().parameters.render_mode).toBe('generative');
    await useEnhancerStore.getState().fetchFeatures();
    const state = useEnhancerStore.getState();
    expect(state.parameters.render_mode).toBe('classic');
    expect(state.parameters.upscale_factor).toBe(1);
    expect(state.activePreset).toBe('natural');
  });

  it('não envia Realismo IA nem 4× antes da ativação', async () => {
    vi.spyOn(apiClient, 'fetchFeatures').mockResolvedValue(featuresResponse(false, false));
    await useEnhancerStore.getState().fetchFeatures();
    const submitSpy = vi.spyOn(apiClient, 'submitEnhancement');
    vi.spyOn(apiClient, 'requestPreview').mockResolvedValue('blob:p');
    useEnhancerStore.getState().setOriginalImage({
      file: new File(['x'], 'a.jpg', { type: 'image/jpeg' }),
      objectUrl: 'blob:a',
      width: 10,
      height: 10,
      name: 'a.jpg',
      size: 1,
    });
    useEnhancerStore.getState().updateParameter('upscale_factor', 4);
    await useEnhancerStore.getState().startEnhancement();
    expect(submitSpy).not.toHaveBeenCalled();
    expect(useEnhancerStore.getState().job.status).toBe('failed');
    expect(useEnhancerStore.getState().job.error).toMatch(/4×/);
  });

  it('ativação com falha permite tentar de novo e passa a usar 4× quando conclui', async () => {
    vi.useFakeTimers();
    const fetchSpy = vi.spyOn(apiClient, 'fetchFeatures');
    const activateSpy = vi.spyOn(apiClient, 'activateFeature').mockResolvedValue();
    fetchSpy.mockResolvedValueOnce(featuresResponse(false, false));
    await useEnhancerStore.getState().fetchFeatures();

    // 1ª tentativa: falha de rede
    fetchSpy.mockResolvedValueOnce(featuresResponse(false, false, 'running'));
    await useEnhancerStore.getState().activateFeature('upscale_4x');
    expect(useEnhancerStore.getState().pendingActivation).toBe('upscale_4x');
    fetchSpy.mockResolvedValueOnce(featuresResponse(false, false, 'failed'));
    await vi.advanceTimersByTimeAsync(1100);
    let state = useEnhancerStore.getState();
    expect(state.features?.features.upscale_4x.task.state).toBe('failed');
    expect(state.pendingActivation).toBeNull();
    expect(state.parameters.upscale_factor).toBe(1);

    // 2ª tentativa: conclui e seleciona 4×
    fetchSpy.mockResolvedValueOnce(featuresResponse(false, false, 'running'));
    await useEnhancerStore.getState().activateFeature('upscale_4x');
    fetchSpy.mockResolvedValue(featuresResponse(true, false, 'done'));
    await vi.advanceTimersByTimeAsync(1100);
    state = useEnhancerStore.getState();
    expect(activateSpy).toHaveBeenCalledTimes(2);
    expect(state.features?.features.upscale_4x.ready).toBe(true);
    expect(state.parameters.upscale_factor).toBe(4);
    expect(state.pendingActivation).toBeNull();
  });
});
