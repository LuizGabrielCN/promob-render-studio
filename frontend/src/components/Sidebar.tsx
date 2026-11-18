import React, { useState } from 'react';
import { useEnhancerStore, PRESETS, DEFAULT_PARAMETERS } from '@/store/useEnhancerStore';
import { PresetKey, RenderMode, SceneProfile } from '@/types';
import { Slider } from '@/components/ui/slider';
import { Button } from '@/components/ui/button';
import { FeatureCard } from '@/components/FeatureCard';
import {
  SunMedium,
  Layers,
  Cpu,
  Download,
  Sparkles,
  Loader2,
  Sliders,
  ChevronDown,
  ChevronUp,
  RotateCcw,
  WandSparkles,
  Home,
  Bath,
  UtensilsCrossed,
  Sofa,
  Trees,
  CloudDownload,
  AlertCircle,
  Maximize2,
} from 'lucide-react';

const PRESET_SUBTITLES: Record<Exclude<PresetKey, 'personalizado'>, string> = {
  realismo_ia: 'ControlNet SD 1.5 + PBR Local',
  natural: 'Balanceado',
  marcenaria: 'Textura de madeira',
  luz_quente: 'Tom aconchegante',
  ultra_detalhe: 'Máxima definição',
};

const ENGINE_MODE_OPTIONS: ReadonlyArray<{
  value: RenderMode;
  label: string;
  subtitle: string;
}> = [
  {
    value: 'classic',
    label: '⚡ Clássico (funciona em CPU)',
    subtitle: 'Ajustes fotométricos determinísticos, sem baixar modelos',
  },
  {
    value: 'generative',
    label: '✨ Realismo IA',
    subtitle: 'ControlNet local; requer GPU NVIDIA e download dos modelos',
  },
];

const SCENE_PROFILE_OPTIONS: ReadonlyArray<{
  value: SceneProfile;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}> = [
  { value: 'auto', label: 'Automático', icon: Home },
  { value: 'banheiro', label: 'Banheiro/Lavabo', icon: Bath },
  { value: 'cozinha', label: 'Cozinha/Gourmet', icon: UtensilsCrossed },
  { value: 'quarto_sala', label: 'Sala/Quarto', icon: Sofa },
  { value: 'externa', label: 'Área Externa', icon: Trees },
];

interface SliderRowProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  formatValue?: (val: number) => string;
  disabled?: boolean;
  hint?: string;
  onChange: (val: number) => void;
  onReset?: () => void;
  defaultValue?: number;
}

const SliderRow: React.FC<SliderRowProps> = ({
  label,
  value,
  min,
  max,
  step,
  unit = '',
  formatValue,
  disabled = false,
  hint,
  onChange,
  onReset,
  defaultValue,
}) => {
  const isChanged = defaultValue !== undefined && Math.abs(value - defaultValue) > 0.001;
  const formattedValue = formatValue
    ? formatValue(value)
    : `${value > 0 && ['+', ''].includes(unit) ? `+${value.toFixed(2)}` : value.toFixed(2)}${unit}`;

  return (
    <div className={`space-y-1.5 py-1 ${disabled ? 'opacity-50' : ''}`}>
      <div className="flex items-center justify-between text-[11px]">
        <span className="text-zinc-400 flex items-center gap-1 font-medium">
          {label}
          {!disabled && isChanged && onReset && (
            <button
              type="button"
              onClick={onReset}
              className="text-zinc-600 hover:text-zinc-400 p-0.5 rounded"
              title="Restaurar valor padrão"
            >
              <RotateCcw className="h-2.5 w-2.5" />
            </button>
          )}
        </span>
        <span className="font-mono text-zinc-300 font-semibold text-[10px]">
          {formattedValue}
        </span>
      </div>
      <Slider
        value={[value]}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        className={disabled ? 'cursor-not-allowed' : undefined}
        onValueChange={(vals) => onChange(vals[0])}
      />
      {hint && <p className="text-[10px] text-zinc-500 leading-snug">{hint}</p>}
    </div>
  );
};

export const Sidebar: React.FC = () => {
  const {
    original,
    parameters,
    updateParameter,
    activePreset,
    applyPreset,
    resetParameters,
    job,
    startEnhancement,
    enhanced4kUrl,
    features,
    fetchFeatures,
  } = useEnhancerStore();

  // Mostra o cartão de ativação quando a pessoa escolhe um recurso que ainda não foi baixado.
  const [requested, setRequested] = useState<{ realismo: boolean; upscale: boolean }>({
    realismo: false,
    upscale: false,
  });
  const realismoReady = features?.features.realismo_ia.ready ?? false;
  const upscaleReady = features?.features.upscale_4x.ready ?? false;
  const realismoRunning = features?.features.realismo_ia.task.state === 'running';
  const upscaleRunning = features?.features.upscale_4x.task.state === 'running';

  React.useEffect(() => {
    if (!features) fetchFeatures();
  }, [features, fetchFeatures]);

  const chooseEngine = (mode: RenderMode) => {
    if (mode === 'generative' && features && !realismoReady) {
      setRequested((r) => ({ ...r, realismo: true }));
      return;
    }
    updateParameter('render_mode', mode);
  };

  const chooseScale = (factor: 1 | 4) => {
    if (factor === 4 && features && !upscaleReady) {
      setRequested((r) => ({ ...r, upscale: true }));
      return;
    }
    updateParameter('upscale_factor', factor);
  };

  const scale = parameters.upscale_factor;
  const outWidth = original ? original.width * scale : 0;
  const outHeight = original ? original.height * scale : 0;

  const [expandedSections, setExpandedSections] = useState({
    generative: true,
    lighting: true,
    details: true,
    engine: false,
  });

  const toggleSection = (section: 'generative' | 'lighting' | 'details' | 'engine') => {
    setExpandedSections((prev) => ({ ...prev, [section]: !prev[section] }));
  };

  const isProcessing = job.status === 'processing' || job.status === 'uploading';
  const isClassicMode = parameters.render_mode === 'classic';
  const isDownloadingWeights = /baixando|download|huggingface|pesos|modelo/i.test(job.stage);
  const clampedProgress = Math.max(0, Math.min(100, job.progress));

  const handleDownload = () => {
    if (!enhanced4kUrl) return;
    const a = document.createElement('a');
    a.href = enhanced4kUrl;
    const originalName = original?.name?.replace(/\.[^/.]+$/, '') || 'render_promob';
    a.download = `${originalName}_${scale > 1 ? `${scale}x` : 'aprimorado'}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <aside className="w-80 border-l border-zinc-800 bg-zinc-950 flex flex-col h-full select-none z-20">
      {/* Sidebar Header */}
      <div className="p-3.5 border-b border-zinc-800 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sliders className="h-4 w-4 text-zinc-400" />
          <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-300">
            Painel de Ajustes
          </h2>
        </div>
        <button
          onClick={resetParameters}
          className="text-[11px] text-zinc-500 hover:text-zinc-300 flex items-center gap-1 transition-colors"
          title="Restaurar todos os parâmetros para o padrão"
        >
          <RotateCcw className="h-3 w-3" />
          <span>Reset</span>
        </button>
      </div>

      {/* Preset Chips */}
      <div className="p-3 border-b border-zinc-800/80 bg-zinc-900/30">
        <label className="text-[10px] uppercase font-semibold text-zinc-500 tracking-wider block mb-2">
          Predefinições Calibradas
        </label>
        <div className="grid grid-cols-2 gap-1.5">
          {(Object.keys(PRESETS) as Array<Exclude<PresetKey, 'personalizado'>>).map((key) => {
            const preset = PRESETS[key];
            const isActive = activePreset === key;
            const isFeatured = key === 'realismo_ia';
            return (
              <button
                key={key}
                type="button"
                onClick={() => {
                  if (preset.params.render_mode === 'generative' && features && !realismoReady) {
                    // Preset de IA sem modelos: mostra a ativação em vez de selecionar um modo indisponível.
                    setRequested((r) => ({ ...r, realismo: true }));
                    setExpandedSections((prev) => ({ ...prev, generative: true }));
                    return;
                  }
                  applyPreset(key);
                }}
                className={`text-left p-2 rounded-md border transition-all text-xs ${
                  isFeatured ? 'col-span-2' : ''
                } ${
                  isActive
                    ? isFeatured
                      ? 'border-amber-500/70 bg-amber-950/30 text-amber-100 shadow-sm'
                      : 'border-zinc-500 bg-zinc-800 text-zinc-100 shadow-sm'
                    : 'border-zinc-800/80 bg-zinc-900/40 text-zinc-400 hover:border-zinc-700 hover:text-zinc-200'
                }`}
              >
                <div className="font-medium text-[11px] truncate flex items-center justify-between gap-1">
                  <span>{preset.name}</span>
                  {isFeatured && (
                    <span className="text-[9px] font-mono uppercase px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-300 border border-amber-500/30">
                      Padrão IA
                    </span>
                  )}
                </div>
                <div className="text-[9px] text-zinc-500 truncate mt-0.5">
                  {PRESET_SUBTITLES[key]}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Scrollable Sliders Area */}
      <div className="flex-1 overflow-y-auto p-3.5 space-y-4">
        {/* Resolução de saída: original (1×) ou ampliação 4× (opcional, baixa o modelo na 1ª vez) */}
        <div className="rounded-lg border border-zinc-800 bg-zinc-900/20 p-2.5 space-y-2">
          <div className="flex items-center gap-2 text-xs font-semibold text-zinc-300">
            <Maximize2 className="h-3.5 w-3.5 text-emerald-400" />
            <span>Resolução de saída</span>
          </div>
          <div className="grid grid-cols-2 gap-1.5">
            {([1, 4] as const).map((factor) => {
              const isSelected = scale === factor;
              const needsActivation = factor === 4 && features && !upscaleReady;
              return (
                <button
                  key={factor}
                  type="button"
                  onClick={() => chooseScale(factor)}
                  className={`text-left px-2 py-1.5 rounded-md border text-[11px] font-medium transition-all ${
                    isSelected
                      ? 'border-zinc-500 bg-zinc-800 text-zinc-100 shadow-sm'
                      : 'border-zinc-800/80 bg-zinc-900/40 text-zinc-400 hover:border-zinc-700 hover:text-zinc-200'
                  }`}
                >
                  <div className="flex items-center justify-between gap-1">
                    <span>{factor === 1 ? 'Original (1×)' : 'Ampliar 4×'}</span>
                    {needsActivation && (
                      <span className="text-[9px] font-mono uppercase px-1 py-0.5 rounded bg-zinc-800 text-zinc-400 border border-zinc-700">
                        {upscaleRunning ? 'baixando' : 'ativar'}
                      </span>
                    )}
                  </div>
                  <div className="text-[9px] text-zinc-500 mt-0.5">
                    {factor === 1 ? 'Mais rápido, roda em CPU' : 'Real-ESRGAN, modelo de 64 MB'}
                  </div>
                </button>
              );
            })}
          </div>
          {original && (
            <p className="text-[10px] text-zinc-500 font-mono">
              {original.width} × {original.height} → <span className="text-emerald-400">{outWidth} × {outHeight}</span>
            </p>
          )}
          {features && !upscaleReady && (requested.upscale || upscaleRunning) && <FeatureCard feature="upscale_4x" />}
        </div>

        {/* Section 0: Realismo IA (ControlNet Local) */}
        <div className="rounded-lg border border-zinc-800 bg-zinc-900/20 overflow-hidden">
          <button
            type="button"
            onClick={() => toggleSection('generative')}
            className="w-full flex items-center justify-between p-2.5 text-xs font-semibold text-zinc-300 hover:bg-zinc-900/50 transition-colors"
          >
            <div className="flex items-center gap-2">
              <WandSparkles className="h-3.5 w-3.5 text-amber-400" />
              <span>Motor de renderização</span>
            </div>
            {expandedSections.generative ? (
              <ChevronUp className="h-3.5 w-3.5 text-zinc-500" />
            ) : (
              <ChevronDown className="h-3.5 w-3.5 text-zinc-500" />
            )}
          </button>

          {expandedSections.generative && (
            <div className="p-3 pt-2 border-t border-zinc-800/80 space-y-3">
              {/* Engine Selector */}
              <div className="space-y-1.5">
                <label className="text-[10px] uppercase font-semibold text-zinc-500 tracking-wider block">
                  Motor de Renderização
                </label>
                <div className="flex flex-col gap-1.5">
                  {ENGINE_MODE_OPTIONS.map((option) => {
                    const isSelected = parameters.render_mode === option.value;
                    return (
                      <button
                        key={option.value}
                        type="button"
                        onClick={() => chooseEngine(option.value)}
                        className={`text-left px-2.5 py-2 rounded-md border transition-all ${
                          isSelected
                            ? option.value === 'generative'
                              ? 'border-amber-500/70 bg-amber-950/30 text-amber-100 shadow-sm'
                              : 'border-zinc-500 bg-zinc-800 text-zinc-100 shadow-sm'
                            : 'border-zinc-800/80 bg-zinc-900/40 text-zinc-400 hover:border-zinc-700 hover:text-zinc-200'
                        }`}
                      >
                        <div className="text-[11px] font-medium leading-tight flex items-center justify-between gap-1">
                          <span>{option.label}</span>
                          {option.value === 'generative' && features && !realismoReady && (
                            <span className="text-[9px] font-mono uppercase px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 border border-zinc-700">
                              {realismoRunning
                                ? 'baixando'
                                : features.features.realismo_ia.available
                                  ? 'ativar'
                                  : 'indisponível'}
                            </span>
                          )}
                        </div>
                        <div className="text-[9px] text-zinc-500 mt-0.5">
                          {option.subtitle}
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>

              {features && !realismoReady && (requested.realismo || realismoRunning) && (
                <FeatureCard feature="realismo_ia" />
              )}

              {/* Scene Profile Selector */}
              <div className={`space-y-1.5 ${isClassicMode ? 'opacity-45' : ''}`}>
                <label className="text-[10px] uppercase font-semibold text-zinc-500 tracking-wider block">
                  Perfil do Ambiente
                </label>
                <div className="grid grid-cols-2 gap-1.5">
                  {SCENE_PROFILE_OPTIONS.map((opt) => {
                    const Icon = opt.icon;
                    const isSelected = parameters.scene_profile === opt.value;
                    return (
                      <button
                        key={opt.value}
                        type="button"
                        disabled={isClassicMode}
                        onClick={() => updateParameter('scene_profile', opt.value)}
                        className={`flex items-center gap-1.5 px-2 py-1.5 rounded-md border text-[10px] font-medium transition-all ${
                          isClassicMode ? 'cursor-not-allowed' : ''
                        } ${
                          isSelected && !isClassicMode
                            ? 'border-amber-500/60 bg-amber-950/25 text-amber-100'
                            : 'border-zinc-800/80 bg-zinc-900/40 text-zinc-400 hover:border-zinc-700 hover:text-zinc-200'
                        }`}
                      >
                        <Icon className="h-3 w-3 shrink-0 text-amber-400/80" />
                        <span className="truncate">{opt.label}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Generative Strength Slider */}
              <SliderRow
                label="Fidelidade Promob <-> Realismo IA"
                value={parameters.generative_strength}
                min={0.0}
                max={1.0}
                step={0.01}
                defaultValue={DEFAULT_PARAMETERS.generative_strength}
                formatValue={(v) => `${Math.round(v * 100)}%`}
                disabled={isClassicMode}
                hint={
                  isClassicMode
                    ? 'Ativo apenas no modo Realismo IA.'
                    : '0% Fiel ao Promob ↔ 100% Máximo Realismo IA'
                }
                onChange={(val) => updateParameter('generative_strength', val)}
                onReset={() =>
                  updateParameter('generative_strength', DEFAULT_PARAMETERS.generative_strength)
                }
              />

              {/* Specular Bloom Slider (Always active in real-time preview) */}
              <SliderRow
                label="Brilho GI & Reflexos (Bloom)"
                value={parameters.specular_bloom}
                min={0.0}
                max={1.0}
                step={0.01}
                defaultValue={DEFAULT_PARAMETERS.specular_bloom}
                hint="Atualizado em tempo real na prévia (GI suave + reflexos)"
                onChange={(val) => updateParameter('specular_bloom', val)}
                onReset={() =>
                  updateParameter('specular_bloom', DEFAULT_PARAMETERS.specular_bloom)
                }
              />

              {/* Photographic finish (generative render only: not shown in the fast preview) */}
              <SliderRow
                label="Acabamento Fotográfico"
                value={parameters.photo_finish}
                min={0.0}
                max={1.0}
                step={0.01}
                defaultValue={DEFAULT_PARAMETERS.photo_finish}
                formatValue={(v) => `${Math.round(v * 100)}%`}
                disabled={isClassicMode}
                hint={
                  isClassicMode
                    ? 'Ativo apenas no modo Realismo IA.'
                    : 'Curva fílmica, bloom suave, vinheta e grão (só no render final)'
                }
                onChange={(val) => updateParameter('photo_finish', val)}
                onReset={() =>
                  updateParameter('photo_finish', DEFAULT_PARAMETERS.photo_finish)
                }
              />
            </div>
          )}
        </div>

        {/* Section 1: Iluminação & Exposição */}
        <div className="rounded-lg border border-zinc-850 bg-zinc-900/20 overflow-hidden">
          <button
            onClick={() => toggleSection('lighting')}
            className="w-full flex items-center justify-between p-2.5 text-xs font-semibold text-zinc-300 hover:bg-zinc-900/50 transition-colors"
          >
            <div className="flex items-center gap-2">
              <SunMedium className="h-3.5 w-3.5 text-amber-400" />
              <span>Iluminação & Exposição</span>
            </div>
            {expandedSections.lighting ? (
              <ChevronUp className="h-3.5 w-3.5 text-zinc-500" />
            ) : (
              <ChevronDown className="h-3.5 w-3.5 text-zinc-500" />
            )}
          </button>

          {expandedSections.lighting && (
            <div className="p-3 pt-1 border-t border-zinc-850 space-y-2">
              <SliderRow
                label="Temperatura (Kelvin)"
                value={parameters.temperature}
                min={-1.0}
                max={1.0}
                step={0.02}
                defaultValue={DEFAULT_PARAMETERS.temperature}
                onChange={(val) => updateParameter('temperature', val)}
                onReset={() => updateParameter('temperature', DEFAULT_PARAMETERS.temperature)}
              />
              <SliderRow
                label="Tint (Verde / Magenta)"
                value={parameters.tint}
                min={-1.0}
                max={1.0}
                step={0.01}
                defaultValue={DEFAULT_PARAMETERS.tint}
                onChange={(val) => updateParameter('tint', val)}
                onReset={() => updateParameter('tint', DEFAULT_PARAMETERS.tint)}
              />
              <SliderRow
                label="Compensação Exposição (EV)"
                value={parameters.exposure}
                min={-2.0}
                max={2.0}
                step={0.05}
                unit=" EV"
                defaultValue={DEFAULT_PARAMETERS.exposure}
                onChange={(val) => updateParameter('exposure', val)}
                onReset={() => updateParameter('exposure', DEFAULT_PARAMETERS.exposure)}
              />
              <SliderRow
                label="Recuperação de Sombras"
                value={parameters.shadow_lift}
                min={0.0}
                max={1.0}
                step={0.02}
                defaultValue={DEFAULT_PARAMETERS.shadow_lift}
                onChange={(val) => updateParameter('shadow_lift', val)}
                onReset={() => updateParameter('shadow_lift', DEFAULT_PARAMETERS.shadow_lift)}
              />
              <SliderRow
                label="Compressão de Realces"
                value={parameters.highlight_recovery}
                min={0.0}
                max={1.0}
                step={0.02}
                defaultValue={DEFAULT_PARAMETERS.highlight_recovery}
                onChange={(val) => updateParameter('highlight_recovery', val)}
                onReset={() => updateParameter('highlight_recovery', DEFAULT_PARAMETERS.highlight_recovery)}
              />
            </div>
          )}
        </div>

        {/* Section 2: Detalhes & Texturas */}
        <div className="rounded-lg border border-zinc-850 bg-zinc-900/20 overflow-hidden">
          <button
            onClick={() => toggleSection('details')}
            className="w-full flex items-center justify-between p-2.5 text-xs font-semibold text-zinc-300 hover:bg-zinc-900/50 transition-colors"
          >
            <div className="flex items-center gap-2">
              <Layers className="h-3.5 w-3.5 text-indigo-400" />
              <span>Detalhes & Texturas</span>
            </div>
            {expandedSections.details ? (
              <ChevronUp className="h-3.5 w-3.5 text-zinc-500" />
            ) : (
              <ChevronDown className="h-3.5 w-3.5 text-zinc-500" />
            )}
          </button>

          {expandedSections.details && (
            <div className="p-3 pt-1 border-t border-zinc-850 space-y-2">
              <SliderRow
                label="Contraste Local (CLAHE)"
                value={parameters.clahe_clip_limit}
                min={0.0}
                max={5.0}
                step={0.1}
                defaultValue={DEFAULT_PARAMETERS.clahe_clip_limit}
                onChange={(val) => updateParameter('clahe_clip_limit', val)}
                onReset={() => updateParameter('clahe_clip_limit', DEFAULT_PARAMETERS.clahe_clip_limit)}
              />
              <SliderRow
                label="Oclusão de Ambiente (AO)"
                value={parameters.ambient_occlusion}
                min={0.0}
                max={1.0}
                step={0.02}
                defaultValue={DEFAULT_PARAMETERS.ambient_occlusion}
                onChange={(val) => updateParameter('ambient_occlusion', val)}
                onReset={() => updateParameter('ambient_occlusion', DEFAULT_PARAMETERS.ambient_occlusion)}
              />
              <SliderRow
                label="Nitidez & Micro-contraste"
                value={parameters.sharpness}
                min={0.0}
                max={2.0}
                step={0.05}
                defaultValue={DEFAULT_PARAMETERS.sharpness}
                onChange={(val) => updateParameter('sharpness', val)}
                onReset={() => updateParameter('sharpness', DEFAULT_PARAMETERS.sharpness)}
              />
              <SliderRow
                label="Redução de Ruído"
                value={parameters.denoise_strength}
                min={0.0}
                max={1.0}
                step={0.02}
                defaultValue={DEFAULT_PARAMETERS.denoise_strength}
                onChange={(val) => updateParameter('denoise_strength', val)}
                onReset={() => updateParameter('denoise_strength', DEFAULT_PARAMETERS.denoise_strength)}
              />
            </div>
          )}
        </div>

        {/* Section 3: Motor 4K & Guardrails */}
        <div className="rounded-lg border border-zinc-850 bg-zinc-900/20 overflow-hidden">
          <button
            onClick={() => toggleSection('engine')}
            className="w-full flex items-center justify-between p-2.5 text-xs font-semibold text-zinc-300 hover:bg-zinc-900/50 transition-colors"
          >
            <div className="flex items-center gap-2">
              <Cpu className="h-3.5 w-3.5 text-emerald-400" />
              <span>Motor 4K & GPU</span>
            </div>
            {expandedSections.engine ? (
              <ChevronUp className="h-3.5 w-3.5 text-zinc-500" />
            ) : (
              <ChevronDown className="h-3.5 w-3.5 text-zinc-500" />
            )}
          </button>

          {expandedSections.engine && (
            <div className="p-3 pt-1 border-t border-zinc-850 space-y-2.5 text-xs">
              <div className="flex items-center justify-between text-[11px] text-zinc-400">
                <span>Resolução de Saída:</span>
                <span className="font-mono text-zinc-200 font-semibold">
                  {scale > 1 ? `Super-resolução ${scale}×` : 'Resolução original'}
                </span>
              </div>
              <div className="flex items-center justify-between text-[11px] text-zinc-400">
                <span>Tamanho do Tile:</span>
                <span className="font-mono text-zinc-200">256 × 256 px</span>
              </div>
              <div className="flex items-center justify-between text-[11px] text-zinc-400">
                <span>Trava de Geometria:</span>
                <span className="text-emerald-400 font-medium font-mono">Proteção ativa</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Primary Actions Footer */}
      <div className="p-3.5 border-t border-zinc-800 bg-zinc-950/90 space-y-2.5">
        {/* Progress Bar when Job is active */}
        {isProcessing && (
          <div
            className={`space-y-1.5 bg-zinc-900/80 p-2.5 rounded-lg border ${
              isDownloadingWeights ? 'border-sky-800/70' : 'border-zinc-800'
            }`}
          >
            <div className="flex items-center justify-between text-xs gap-2">
              <span className="text-zinc-300 flex items-center gap-1.5 font-medium">
                {isDownloadingWeights ? (
                  <>
                    <CloudDownload className="h-3.5 w-3.5 text-sky-400 animate-pulse shrink-0" />
                    <span>Baixando modelos IA (1ª execução)...</span>
                  </>
                ) : (
                  <>
                    <Loader2 className="h-3.5 w-3.5 animate-spin text-amber-400 shrink-0" />
                    <span>
                      {isClassicMode ? 'Processando...' : 'Processando Realismo IA...'}
                    </span>
                  </>
                )}
              </span>
              <span
                className={`font-mono font-bold shrink-0 ${
                  isDownloadingWeights ? 'text-sky-400' : 'text-amber-400'
                }`}
              >
                {Math.round(clampedProgress)}%
              </span>
            </div>
            <div className="h-1.5 w-full bg-zinc-800 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all duration-300 rounded-full ${
                  isDownloadingWeights ? 'bg-sky-400' : 'bg-amber-400'
                }`}
                style={{ width: `${clampedProgress}%` }}
              />
            </div>
            <div className="text-[10px] text-zinc-400 whitespace-normal break-words leading-snug font-mono">
              {job.stage}
            </div>
          </div>
        )}

        {/* Error Banner when Job failed */}
        {job.status === 'failed' && (
          <div className="flex items-start gap-2 bg-red-950/40 p-2.5 rounded-lg border border-red-900/60 text-[10px] text-red-300 whitespace-normal break-words leading-snug">
            <AlertCircle className="h-3.5 w-3.5 text-red-400 shrink-0 mt-0.5" />
            <span>{job.error || job.stage || 'Falha no processamento.'}</span>
          </div>
        )}

        {/* Primary Enhance Button */}
        <Button
          onClick={startEnhancement}
          disabled={!original || isProcessing}
          className="w-full h-10 text-xs font-semibold bg-zinc-100 text-zinc-900 hover:bg-zinc-200 shadow-lg disabled:opacity-40"
        >
          {isProcessing ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Processando...
            </>
          ) : (
            <>
              <Sparkles className="h-4 w-4 mr-2 text-amber-600" />
              {scale > 1 ? 'Aprimorar e ampliar 4×' : 'Aprimorar imagem'}
            </>
          )}
        </Button>

        {/* Download Button (Active when completed) */}
        {enhanced4kUrl && (
          <Button
            onClick={handleDownload}
            variant="secondary"
            className="w-full h-9 text-xs border border-emerald-700/60 bg-emerald-950/40 text-emerald-300 hover:bg-emerald-900/50"
          >
            <Download className="h-3.5 w-3.5 mr-2 text-emerald-400" />
            Baixar PNG ({job.width && job.height ? `${job.width} × ${job.height}` : 'resultado'})
          </Button>
        )}
      </div>
    </aside>
  );
};
