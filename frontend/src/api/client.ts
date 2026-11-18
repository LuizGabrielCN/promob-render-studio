import {
  EngineCapabilities,
  EnhancementParameters,
  HardwareStatus,
  JobProgress,
} from "@/types";

const API_BASE = "/api";

export async function fetchHealth(): Promise<HardwareStatus> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) {
    throw new Error(`Falha ao obter status do hardware: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchCapabilities(): Promise<EngineCapabilities> {
  const res = await fetch(`${API_BASE}/capabilities`);
  if (!res.ok) {
    throw new Error(`Falha ao obter capacidades do motor: ${res.statusText}`);
  }
  return res.json();
}

function buildFormData(file: Blob | File, params: EnhancementParameters, filename?: string): FormData {
  const formData = new FormData();
  const actualFilename = filename || (file instanceof File ? file.name : "promob_render.jpg");
  formData.append("file", file, actualFilename);

  formData.append("temperature", params.temperature.toString());
  formData.append("tint", params.tint.toString());
  formData.append("exposure", params.exposure.toString());
  formData.append("shadow_lift", params.shadow_lift.toString());
  formData.append("highlight_recovery", params.highlight_recovery.toString());
  formData.append("clahe_clip_limit", params.clahe_clip_limit.toString());
  formData.append("clahe_grid_size", params.clahe_grid_size.toString());
  formData.append("ambient_occlusion", params.ambient_occlusion.toString());
  formData.append("sharpness", params.sharpness.toString());
  formData.append("denoise_strength", params.denoise_strength.toString());
  formData.append("upscale_factor", params.upscale_factor.toString());
  formData.append("tile_size", params.tile_size.toString());
  formData.append("preserve_geometry", params.preserve_geometry ? "true" : "false");
  formData.append("render_mode", params.render_mode);
  formData.append("generative_strength", params.generative_strength.toString());
  formData.append("scene_profile", params.scene_profile);
  formData.append("specular_bloom", params.specular_bloom.toString());
  formData.append("photo_finish", params.photo_finish.toString());

  return formData;
}

export async function requestPreview(
  file: Blob | File,
  params: EnhancementParameters,
  signal?: AbortSignal,
  filename?: string
): Promise<string> {
  const formData = buildFormData(file, params, filename);
  const res = await fetch(`${API_BASE}/preview`, {
    method: "POST",
    body: formData,
    signal,
  });

  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Falha ao gerar prévia (${res.status}): ${errorText || res.statusText}`);
  }

  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

export async function submitEnhancement(
  file: Blob | File,
  params: EnhancementParameters,
  filename?: string
): Promise<{ job_id: string; status: string }> {
  const formData = buildFormData(file, params, filename);
  const res = await fetch(`${API_BASE}/enhance`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Falha ao submeter job de aprimoramento (${res.status}): ${errorText || res.statusText}`);
  }

  return res.json();
}

export async function pollJob(jobId: string): Promise<JobProgress> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`);
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Falha ao verificar status do job (${res.status}): ${errorText || res.statusText}`);
  }
  return res.json();
}

export function getDownloadUrl(jobId: string): string {
  return `${API_BASE}/download/${jobId}`;
}
