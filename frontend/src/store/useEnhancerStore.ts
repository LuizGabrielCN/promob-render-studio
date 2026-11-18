import { create } from 'zustand';
import {
  EnhancementParameters,
  Preset,
  PresetKey,
  HardwareStatus,
  JobProgress,
  ImageSource,
} from '@/types';
import {
  fetchHealth,
  requestPreview,
  submitEnhancement,
  pollJob,
  getDownloadUrl,
} from '@/api/client';

export const DEFAULT_PARAMETERS: EnhancementParameters = {
  temperature: 0.04,        // Subtle warmth while preserving paint and material colors
  tint: 0.0,
  exposure: 0.05,
  shadow_lift: 0.23,         // +23% shadow lift
  highlight_recovery: 0.10,  // -6% to 10% highlight compression
  clahe_clip_limit: 2.0,     // Local contrast L*
  clahe_grid_size: 8,
  ambient_occlusion: 0.15,   // Morphological AO in crevices
  sharpness: 0.40,           // 3.6x micro-edge enhancement
  denoise_strength: 0.10,    // Mild bilateral filter
  upscale_factor: 4,         // 4K super-resolution
  tile_size: 256,            // RTX 2060 6GB memory safe tile
  preserve_geometry: true,   // 100% strict geometry preservation
  render_mode: 'generative', // Local ControlNet photorealistic engine by default
  generative_strength: 0.48, // Calibrated fidelity <-> realism curve default
  scene_profile: 'auto',     // Automatic architectural scene detection
  specular_bloom: 0.25,      // Soft GI Bloom & selective specular highlight
  photo_finish: 0.5,         // Filmic curve, soft bloom, vignette and grain (Realismo IA only)
};

export const PRESETS: Record<Exclude<PresetKey, 'personalizado'>, Preset> = {
  realismo_ia: {
    id: 'realismo_ia',
    name: 'Realismo IA',
    description: 'Motor ControlNet local estilo ChatGPT com preservação geométrica e iluminação GI.',
    params: {
      temperature: 0.04,
      tint: 0.0,
      exposure: 0.05,
      shadow_lift: 0.23,
      highlight_recovery: 0.10,
      clahe_clip_limit: 2.0,
      ambient_occlusion: 0.15,
      sharpness: 0.40,
      denoise_strength: 0.10,
      render_mode: 'generative',
      generative_strength: 0.48,
      scene_profile: 'auto',
      specular_bloom: 0.25,
      photo_finish: 0.5,
    },
  },
  natural: {
    id: 'natural',
    name: 'Natural Studio',
    description: 'Equilíbrio sutil entre luz natural e contraste fotográfico suave.',
    params: {
      temperature: 0.10,
      tint: 0.0,
      exposure: 0.0,
      shadow_lift: 0.18,
      highlight_recovery: 0.12,
      clahe_clip_limit: 1.8,
      ambient_occlusion: 0.12,
      sharpness: 0.35,
      denoise_strength: 0.12,
      render_mode: 'classic',
      specular_bloom: 0.18,
    },
  },
  marcenaria: {
    id: 'marcenaria',
    name: 'Marcenaria & Madeira',
    description: 'Realce profundo de veios de madeira, texturas de MDF e sombras de contato.',
    params: {
      temperature: 0.24,
      tint: 0.03,
      exposure: 0.08,
      shadow_lift: 0.28,
      highlight_recovery: 0.15,
      clahe_clip_limit: 2.6,
      ambient_occlusion: 0.22,
      sharpness: 0.55,
      denoise_strength: 0.08,
      render_mode: 'classic',
      specular_bloom: 0.25,
    },
  },
  luz_quente: {
    id: 'luz_quente',
    name: 'Iluminação Quente',
    description: 'Atmosfera aconchegante com temperatura Kelvin elevada e realces suaves.',
    params: {
      temperature: 0.35,
      tint: 0.05,
      exposure: 0.10,
      shadow_lift: 0.25,
      highlight_recovery: 0.20,
      clahe_clip_limit: 2.0,
      ambient_occlusion: 0.14,
      sharpness: 0.38,
      denoise_strength: 0.10,
      render_mode: 'classic',
      specular_bloom: 0.30,
    },
  },
  ultra_detalhe: {
    id: 'ultra_detalhe',
    name: 'Ultra Detalhe 4K',
    description: 'Máxima definição de bordas, contraste local CLAHE elevado e micro-frestas.',
    params: {
      temperature: 0.15,
      tint: 0.0,
      exposure: 0.05,
      shadow_lift: 0.22,
      highlight_recovery: 0.10,
      clahe_clip_limit: 3.2,
      ambient_occlusion: 0.28,
      sharpness: 0.70,
      denoise_strength: 0.06,
      render_mode: 'classic',
      specular_bloom: 0.20,
    },
  },
};

interface EnhancerStoreState {
  // Original Image
  original: ImageSource | null;
  setOriginalImage: (image: ImageSource) => void;
  clearImage: () => void;

  // Preview & 4K Output
  previewUrl: string | null;
  enhanced4kUrl: string | null;
  isPreviewLoading: boolean;
  previewError: string | null;

  // Parameters & Presets
  parameters: EnhancementParameters;
  activePreset: PresetKey;
  updateParameter: <K extends keyof EnhancementParameters>(
    key: K,
    value: EnhancementParameters[K]
  ) => void;
  applyPreset: (presetKey: PresetKey) => void;
  resetParameters: () => void;

  // Asynchronous 4K Processing Job
  job: JobProgress;
  startEnhancement: () => Promise<void>;
  resetJob: () => void;

  // Hardware Status
  hardware: HardwareStatus | null;
  fetchHardware: () => Promise<void>;

  // Interactive Viewer State
  sliderPos: number;
  setSliderPos: (pos: number) => void;
  isSpacePressed: boolean;
  setIsSpacePressed: (pressed: boolean) => void;
  splitMode: 'split' | 'side-by-side' | 'original-hold';
  setSplitMode: (mode: 'split' | 'side-by-side' | 'original-hold') => void;
  zoom: number;
  setZoom: (zoom: number) => void;

  // Internal trigger for debounced preview
  requestPreviewDebounced: () => void;
}

let previewAbortController: AbortController | null = null;
let previewDebounceTimer: ReturnType<typeof setTimeout> | null = null;
let pollingIntervalTimer: ReturnType<typeof setInterval> | null = null;
let activeEnhancementSession = 0;

function clearPollingTimer() {
  if (pollingIntervalTimer) {
    clearInterval(pollingIntervalTimer);
    pollingIntervalTimer = null;
  }
}

function clearPendingPreview() {
  if (previewDebounceTimer) {
    clearTimeout(previewDebounceTimer);
    previewDebounceTimer = null;
  }
  if (previewAbortController) {
    previewAbortController.abort();
    previewAbortController = null;
  }
}

export const useEnhancerStore = create<EnhancerStoreState>((set, get) => ({
  // Initial Image
  original: null,
  setOriginalImage: (image) => {
    activeEnhancementSession += 1;
    clearPollingTimer();
    clearPendingPreview();

    // Revoke previous URLs if applicable
    const prev = get().original;
    if (prev && prev.objectUrl.startsWith('blob:')) {
      URL.revokeObjectURL(prev.objectUrl);
    }
    const prevPreview = get().previewUrl;
    if (prevPreview && prevPreview.startsWith('blob:')) {
      URL.revokeObjectURL(prevPreview);
    }

    set({
      original: image,
      previewUrl: null,
      enhanced4kUrl: null,
      isPreviewLoading: false,
      previewError: null,
      job: {
        job_id: null,
        status: 'idle',
        progress: 0,
        stage: '',
      },
    });

    // Automatically trigger initial preview
    get().requestPreviewDebounced();
  },

  clearImage: () => {
    activeEnhancementSession += 1;
    clearPollingTimer();
    clearPendingPreview();

    const prev = get().original;
    if (prev && prev.objectUrl.startsWith('blob:')) {
      URL.revokeObjectURL(prev.objectUrl);
    }
    const prevPreview = get().previewUrl;
    if (prevPreview && prevPreview.startsWith('blob:')) {
      URL.revokeObjectURL(prevPreview);
    }

    set({
      original: null,
      previewUrl: null,
      enhanced4kUrl: null,
      isPreviewLoading: false,
      previewError: null,
      job: {
        job_id: null,
        status: 'idle',
        progress: 0,
        stage: '',
      },
    });
  },

  // Preview & Output
  previewUrl: null,
  enhanced4kUrl: null,
  isPreviewLoading: false,
  previewError: null,

  // Parameters & Presets
  parameters: { ...DEFAULT_PARAMETERS },
  activePreset: 'realismo_ia',

  updateParameter: (key, value) => {
    set((state) => ({
      parameters: { ...state.parameters, [key]: value },
      activePreset: 'personalizado',
      enhanced4kUrl: null,
    }));
    get().requestPreviewDebounced();
  },

  applyPreset: (presetKey) => {
    if (presetKey === 'personalizado') {
      set({ activePreset: 'personalizado' });
      return;
    }
    const preset = PRESETS[presetKey];
    if (!preset) return;

    set((state) => ({
      activePreset: presetKey,
      enhanced4kUrl: null,
      parameters: {
        ...state.parameters,
        ...preset.params,
      },
    }));
    get().requestPreviewDebounced();
  },

  resetParameters: () => {
    set({
      parameters: { ...DEFAULT_PARAMETERS },
      activePreset: 'realismo_ia',
      enhanced4kUrl: null,
    });
    get().requestPreviewDebounced();
  },

  // Interactive Viewer State
  sliderPos: 50,
  setSliderPos: (pos) => set({ sliderPos: Math.max(0, Math.min(100, pos)) }),
  isSpacePressed: false,
  setIsSpacePressed: (pressed) => set({ isSpacePressed: pressed }),
  splitMode: 'split',
  setSplitMode: (mode) => set({ splitMode: mode }),
  zoom: 1,
  setZoom: (zoom) => set({ zoom: Math.max(0.5, Math.min(4, zoom)) }),

  // Asynchronous Job
  job: {
    job_id: null,
    status: 'idle',
    progress: 0,
    stage: '',
  },

  resetJob: () => {
    activeEnhancementSession += 1;
    clearPollingTimer();
    set({
      job: {
        job_id: null,
        status: 'idle',
        progress: 0,
        stage: '',
      },
    });
  },

  startEnhancement: async () => {
    const { original, parameters, job } = get();
    if (!original || !original.file || job.status === 'processing') return;

    activeEnhancementSession += 1;
    const sessionId = activeEnhancementSession;
    clearPollingTimer();

    set({
      job: {
        job_id: null,
        status: 'uploading',
        progress: 5,
        stage: 'Enviando render original e parâmetros para o servidor...',
      },
    });

    try {
      const submission = await submitEnhancement(original.file, parameters, original.name);
      if (sessionId !== activeEnhancementSession) return;

      const jobId = submission.job_id;

      set({
        job: {
          job_id: jobId,
          status: 'processing',
          progress: 10,
          stage: 'Alocando pipeline de super-resolução e VRAM...',
        },
      });

      // Start Polling loop with error resilience threshold
      let consecutiveErrors = 0;
      const MAX_CONSECUTIVE_ERRORS = 5;

      pollingIntervalTimer = setInterval(async () => {
        if (sessionId !== activeEnhancementSession) {
          clearPollingTimer();
          return;
        }
        try {
          const pollData = await pollJob(jobId);
          if (sessionId !== activeEnhancementSession) {
            clearPollingTimer();
            return;
          }
          consecutiveErrors = 0; // Reset error counter on successful response

          set({
            job: {
              job_id: jobId,
              status: pollData.status === 'completed' ? 'completed' : (pollData.status === 'failed' ? 'failed' : 'processing'),
              progress: pollData.progress ?? 50,
              stage: pollData.stage ?? 'Processando...',
              width: pollData.width,
              height: pollData.height,
              result_url: pollData.result_url,
              processing_time_seconds: pollData.processing_time_seconds,
              error: pollData.error,
            },
          });

          if (pollData.status === 'completed') {
            clearPollingTimer();
            const fullUrl = getDownloadUrl(jobId);
            set({ enhanced4kUrl: fullUrl });
          } else if (pollData.status === 'failed') {
            clearPollingTimer();
          }
        } catch (pollErr: any) {
          if (sessionId !== activeEnhancementSession) {
            clearPollingTimer();
            return;
          }
          consecutiveErrors += 1;
          console.error(`Erro de polling (${consecutiveErrors}/${MAX_CONSECUTIVE_ERRORS}):`, pollErr);
          if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
            clearPollingTimer();
            set((state) => ({
              job: {
                ...state.job,
                status: 'failed',
                stage: 'Conexão perdida com o servidor de processamento.',
                error: 'Não foi possível comunicar com o backend após 5 tentativas consecutivas. Verifique se o servidor está ativo.',
              },
            }));
          }
        }
      }, 700);

    } catch (err: any) {
      if (sessionId !== activeEnhancementSession) return;
      set({
        job: {
          job_id: null,
          status: 'failed',
          progress: 0,
          stage: 'Falha na submissão do job.',
          error: err.message || 'Erro inesperado na submissão.',
        },
      });
    }
  },

  // Hardware Status
  hardware: null,
  fetchHardware: async () => {
    try {
      const hw = await fetchHealth();
      set({ hardware: hw });
    } catch (e) {
      console.warn('Servidor backend offline ou endpoint /health inacessível:', e);
      // Fallback graceful telemetry for offline mode
      set({
        hardware: {
          status: 'offline',
          gpu: 'NVIDIA RTX 2060 (Local Fallback)',
          vram_free_mb: 4800,
          cuda_available: true,
          total_vram_mb: 6144,
          allocated_vram_mb: 1344,
        },
      });
    }
  },

  // Debounced preview generation
  requestPreviewDebounced: () => {
    if (previewDebounceTimer) {
      clearTimeout(previewDebounceTimer);
    }

    const { original, parameters } = get();
    if (!original || !original.file) return;

    previewDebounceTimer = setTimeout(async () => {
      if (previewAbortController) {
        previewAbortController.abort();
      }
      previewAbortController = new AbortController();

      set({ isPreviewLoading: true, previewError: null });

      try {
        const previewUrl = await requestPreview(
          original.file!,
          parameters,
          previewAbortController.signal,
          original.name
        );
        const oldUrl = get().previewUrl;
        if (oldUrl && oldUrl.startsWith('blob:')) {
          URL.revokeObjectURL(oldUrl);
        }
        set({
          previewUrl,
          isPreviewLoading: false,
          previewError: null,
        });
      } catch (err: any) {
        if (err.name === 'AbortError') return; // Cancelled by newer adjustment
        console.warn('Preview fallback:', err.message);
        set({
          isPreviewLoading: false,
          previewError: err.message,
        });
      }
    }, 120); // 120ms debounce for ultra smooth slider interaction
  },
}));
