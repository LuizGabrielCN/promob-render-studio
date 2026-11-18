export type RenderMode = 'generative' | 'classic';
export type SceneProfile = 'auto' | 'banheiro' | 'cozinha' | 'quarto_sala' | 'externa';

export interface EnhancementParameters {
  temperature: number;       // [-1.0, 1.0]
  tint: number;              // [-1.0, 1.0]
  exposure: number;          // [-2.0, 2.0]
  shadow_lift: number;       // [0.0, 1.0]
  highlight_recovery: number;// [0.0, 1.0]
  clahe_clip_limit: number;  // [0.0, 5.0]
  clahe_grid_size: number;   // [4, 16]
  ambient_occlusion: number; // [0.0, 1.0]
  sharpness: number;         // [0.0, 2.0]
  denoise_strength: number;  // [0.0, 1.0]
  upscale_factor: number;    // [1, 4]
  tile_size: number;         // [128, 512]
  preserve_geometry: boolean;
  render_mode: RenderMode;
  generative_strength: number; // [0.0, 1.0]
  scene_profile: SceneProfile;
  specular_bloom: number;      // [0.0, 1.0]
  photo_finish: number;        // [0.0, 1.0] photographic finish (Realismo IA only)
}

export type PresetKey =
  | 'realismo_ia'
  | 'natural'
  | 'marcenaria'
  | 'luz_quente'
  | 'ultra_detalhe'
  | 'personalizado';

export interface Preset {
  id: PresetKey;
  name: string;
  description: string;
  params: Partial<EnhancementParameters>;
}

export interface HardwareStatus {
  status: string;
  gpu: string;
  vram_free_mb: number;
  cuda_available: boolean;
  total_vram_mb: number;
  allocated_vram_mb: number;
}

export interface HardwareCapabilitiesInfo {
  cuda_available: boolean;
  device_name: string;
  free_mb: number;
  total_mb: number;
  allocated_mb: number;
  reserved_mb: number;
  source: string;
}

export interface GenerativeEngineCapabilities {
  available: boolean;
  base_model: string;
  controlnet_model: string;
  dtype: string;
  scheduler: string;
  cached_locally: boolean;
}

export interface EngineCapabilities {
  engine: string;
  version: string;
  max_scale: number;
  preserves_geometry: boolean;
  deterministic: boolean;
  photometric_stages: string[];
  super_resolution_engine: string;
  hardware: HardwareCapabilitiesInfo;
  render_modes: RenderMode[];
  scene_profiles: SceneProfile[];
  generative_engine: GenerativeEngineCapabilities;
}

export type JobState = 'idle' | 'uploading' | 'processing' | 'completed' | 'failed';

export interface JobProgress {
  job_id: string | null;
  status: JobState;
  progress: number;
  stage: string;
  width?: number;
  height?: number;
  result_url?: string;
  processing_time_seconds?: number;
  error?: string;
}

export interface ImageSource {
  file: File | null;
  objectUrl: string;
  width: number;
  height: number;
  name: string;
  size: number;
}

export interface ViewerState {
  sliderPos: number; // 0 to 100
  mode: 'split' | 'side-by-side' | 'original-hold';
  zoom: number;
}
