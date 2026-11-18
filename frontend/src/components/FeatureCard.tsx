import React from 'react';
import { useEnhancerStore } from '@/store/useEnhancerStore';
import { FeatureKey } from '@/types';
import { Button } from '@/components/ui/button';
import { formatBytes } from '@/lib/utils';
import { AlertCircle, CloudDownload, Info, Loader2, RotateCcw } from 'lucide-react';

const TITLES: Record<FeatureKey, string> = {
  upscale_4x: 'Ampliação 4×',
  realismo_ia: 'Realismo IA',
};

/**
 * Ativação de um recurso opcional: mostra o tamanho estimado antes do download,
 * o progresso durante, e uma nova tentativa em caso de falha.
 */
export const FeatureCard: React.FC<{ feature: FeatureKey }> = ({ feature }) => {
  const { features, featuresError, activateFeature } = useEnhancerStore();
  const status = features?.features[feature];

  if (!status) {
    return (
      <div className="rounded-md border border-zinc-800 bg-zinc-900/60 p-2.5 text-[10px] text-zinc-400" data-testid={`feature-${feature}`}>
        {featuresError ? `Não foi possível verificar o recurso: ${featuresError}` : 'Verificando o que já está instalado...'}
      </div>
    );
  }
  if (status.ready) return null;

  const { task } = status;
  const title = TITLES[feature];
  const otherRunning = Object.values(features!.features).some(
    (f) => f.task.state === 'running' && f.task.feature !== feature
  );

  if (task.state === 'running') {
    const pct = Math.max(0, Math.min(100, task.progress));
    return (
      <div className="rounded-md border border-sky-800/70 bg-sky-950/20 p-2.5 space-y-1.5" data-testid={`feature-${feature}`}>
        <div className="flex items-center justify-between text-[11px] text-zinc-200 gap-2">
          <span className="flex items-center gap-1.5 font-medium">
            <CloudDownload className="h-3.5 w-3.5 text-sky-400 animate-pulse shrink-0" />
            Ativando {title}...
          </span>
          <span className="font-mono font-bold text-sky-400">{Math.round(pct)}%</span>
        </div>
        <div className="h-1.5 w-full bg-zinc-800 rounded-full overflow-hidden">
          <div className="h-full bg-sky-400 transition-all duration-500 rounded-full" style={{ width: `${pct}%` }} />
        </div>
        <p className="text-[10px] text-zinc-400 leading-snug break-words">{task.stage}</p>
        <p className="text-[10px] text-zinc-500 leading-snug">
          Você pode continuar usando o modo Clássico enquanto o download acontece.
        </p>
      </div>
    );
  }

  const unavailable = !status.available || (task.state === 'failed' && task.unavailable);
  if (unavailable) {
    return (
      <div className="rounded-md border border-zinc-700 bg-zinc-900/70 p-2.5 space-y-1" data-testid={`feature-${feature}`}>
        <div className="flex items-center gap-1.5 text-[11px] font-medium text-zinc-200">
          <Info className="h-3.5 w-3.5 text-amber-400 shrink-0" />
          {title} indisponível neste computador
        </div>
        <p className="text-[10px] text-zinc-400 leading-snug">{task.error || status.reason}</p>
      </div>
    );
  }

  const failed = task.state === 'failed';
  return (
    <div className="rounded-md border border-zinc-800 bg-zinc-900/60 p-2.5 space-y-1.5" data-testid={`feature-${feature}`}>
      {failed ? (
        <div className="flex items-start gap-1.5 text-[10px] text-red-300 leading-snug break-words">
          <AlertCircle className="h-3.5 w-3.5 text-red-400 shrink-0 mt-0.5" />
          <span>{task.error || 'O download falhou.'}</span>
        </div>
      ) : (
        <p className="text-[10px] text-zinc-400 leading-snug">{status.summary}</p>
      )}
      <p className="text-[10px] text-zinc-500 leading-snug">
        Download estimado: <span className="text-zinc-300 font-mono">{formatBytes(status.download_bytes)}</span>
        {' · '}Espaço em disco: <span className="text-zinc-300 font-mono">{formatBytes(status.disk_bytes)}</span>
        <br />
        Precisa de internet só para este download. As imagens continuam sendo processadas no seu computador.
      </p>
      {status.warning && <p className="text-[10px] text-amber-300/90 leading-snug">{status.warning}</p>}
      {featuresError && <p className="text-[10px] text-red-300 leading-snug">{featuresError}</p>}
      <Button
        type="button"
        size="sm"
        disabled={otherRunning}
        onClick={() => activateFeature(feature)}
        className="w-full h-8 text-[11px] bg-sky-600 hover:bg-sky-500 text-white"
      >
        {failed ? (
          <>
            <RotateCcw className="h-3.5 w-3.5 mr-1.5" />
            Tentar novamente
          </>
        ) : otherRunning ? (
          <>
            <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
            Aguardando o outro download
          </>
        ) : (
          <>
            <CloudDownload className="h-3.5 w-3.5 mr-1.5" />
            Baixar e ativar ({formatBytes(status.download_bytes)})
          </>
        )}
      </Button>
    </div>
  );
};
