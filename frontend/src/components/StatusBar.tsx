import React from 'react';
import { useEnhancerStore } from '@/store/useEnhancerStore';
import { formatBytes, formatSeconds } from '@/lib/utils';
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  CheckCircle2,
  AlertCircle,
  Loader2,
  HardDrive,
  Keyboard,
} from 'lucide-react';

export const StatusBar: React.FC = () => {
  const { original, job, zoom, setZoom, isPreviewLoading, parameters } = useEnhancerStore();

  const scale = parameters.upscale_factor;
  const targetWidth = original ? original.width * scale : 0;
  const targetHeight = original ? original.height * scale : 0;

  return (
    <footer className="h-8 border-t border-zinc-800 bg-zinc-950 px-3 flex items-center justify-between text-[11px] text-zinc-400 select-none z-30">
      {/* Left: Image Dimensions & Info */}
      <div className="flex items-center gap-3">
        {original ? (
          <>
            <div className="flex items-center gap-1.5 font-mono text-zinc-300">
              <span className="text-zinc-500">Entrada:</span>
              <span>{original.width} × {original.height}</span>
              <span className="text-zinc-600">→</span>
              <span className="text-emerald-400 font-medium">
                Saída: {job.width || targetWidth} × {job.height || targetHeight}
              </span>
            </div>

            <div className="hidden lg:flex items-center gap-1.5 text-zinc-500 font-mono">
              <span>•</span>
              <HardDrive className="h-3 w-3" />
              <span>{original.name}</span>
              <span>({formatBytes(original.size)})</span>
            </div>
          </>
        ) : (
          <span className="text-zinc-500 flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-zinc-600" />
            Nenhuma imagem carregada no estúdio
          </span>
        )}
      </div>

      {/* Center: Realtime Job / Preview Status */}
      <div className="flex items-center gap-2">
        {isPreviewLoading && (
          <div className="flex items-center gap-1.5 text-amber-400 font-medium">
            <Loader2 className="h-3 w-3 animate-spin" />
            <span>Atualizando prévia fotométrica...</span>
          </div>
        )}

        {job.status === 'processing' && (
          <div className="flex items-center gap-2 text-zinc-200">
            <Loader2 className="h-3 w-3 animate-spin text-amber-400" />
            <span className="font-mono text-amber-400 font-medium">{job.progress}%</span>
            <span className="hidden sm:inline text-zinc-400">{job.stage}</span>
          </div>
        )}

        {job.status === 'completed' && (
          <div className="flex items-center gap-1.5 text-emerald-400 font-medium">
            <CheckCircle2 className="h-3.5 w-3.5" />
            <span>Concluído</span>
            {job.processing_time_seconds && (
              <span className="text-zinc-400 font-mono text-[10px]">
                ({formatSeconds(job.processing_time_seconds)})
              </span>
            )}
          </div>
        )}

        {job.status === 'failed' && (
          <div className="flex items-center gap-1.5 text-red-400 font-medium">
            <AlertCircle className="h-3.5 w-3.5" />
            <span>{job.error || 'Falha no processamento'}</span>
          </div>
        )}
      </div>

      {/* Right: Keyboard Shortcut Guide & Zoom Controls */}
      <div className="flex items-center gap-3">
        {original && (
          <div className="hidden xl:flex items-center gap-1.5 text-zinc-500">
            <Keyboard className="h-3 w-3" />
            <span>Segure <kbd className="px-1 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-300 font-mono text-[9px]">Espaço</kbd> para alternar original</span>
          </div>
        )}

        {original && (
          <div className="flex items-center gap-1 bg-zinc-900 px-1 py-0.5 rounded border border-zinc-800">
            <button
              onClick={() => setZoom(zoom - 0.25)}
              disabled={zoom <= 0.5}
              className="p-0.5 hover:text-zinc-200 disabled:opacity-30 disabled:hover:text-zinc-500"
              title="Reduzir Zoom"
            >
              <ZoomOut className="h-3 w-3" />
            </button>
            <span className="font-mono text-[10px] w-9 text-center text-zinc-300">
              {Math.round(zoom * 100)}%
            </span>
            <button
              onClick={() => setZoom(zoom + 0.25)}
              disabled={zoom >= 3.0}
              className="p-0.5 hover:text-zinc-200 disabled:opacity-30 disabled:hover:text-zinc-500"
              title="Aumentar Zoom"
            >
              <ZoomIn className="h-3 w-3" />
            </button>
            <button
              onClick={() => setZoom(1.0)}
              className="p-0.5 hover:text-zinc-200 ml-0.5 text-zinc-400"
              title="Ajustar à Janela (100%)"
            >
              <Maximize2 className="h-2.5 w-2.5" />
            </button>
          </div>
        )}
      </div>
    </footer>
  );
};
