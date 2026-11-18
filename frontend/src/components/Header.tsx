import React, { useEffect } from 'react';
import { useEnhancerStore } from '@/store/useEnhancerStore';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Sparkles,
  RotateCcw,
  Cpu,
  Columns,
  Split,
} from 'lucide-react';

export const Header: React.FC = () => {
  const {
    hardware,
    fetchHardware,
    original,
    clearImage,
    resetParameters,
    splitMode,
    setSplitMode,
  } = useEnhancerStore();

  useEffect(() => {
    fetchHardware();
    const interval = setInterval(fetchHardware, 10000);
    return () => clearInterval(interval);
  }, [fetchHardware]);

  const offline = hardware?.status === 'offline';
  const vramFreeGb = hardware?.cuda_available ? (hardware.vram_free_mb / 1024).toFixed(1) : null;

  return (
    <header className="h-14 border-b border-zinc-800 bg-zinc-950/90 backdrop-blur px-4 flex items-center justify-between z-30 select-none">
      {/* Brand & Title */}
      <div className="flex items-center gap-3">
        <div className="h-8 w-8 rounded-md bg-zinc-900 border border-zinc-800 flex items-center justify-center text-zinc-100 shadow-sm">
          <Sparkles className="h-4 w-4 text-amber-400" />
        </div>
        <div className="flex flex-col">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-sm tracking-tight text-zinc-100">
              Promob Render Studio
            </span>
            <span className="text-[10px] font-mono uppercase bg-zinc-800/80 text-zinc-400 px-1.5 py-0.5 rounded border border-zinc-700/50">
              LOCAL
            </span>
          </div>
          <span className="text-[11px] text-zinc-400 hidden sm:inline">
            Aprimoramento Fotorrealista com Preservação Geométrica
          </span>
        </div>
      </div>

      {/* Center View Controls (active only when image is present) */}
      {original && (
        <div className="hidden md:flex items-center gap-1 bg-zinc-900/90 p-0.5 rounded-lg border border-zinc-800/80 text-xs">
          <button
            onClick={() => setSplitMode('split')}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
              splitMode === 'split'
                ? 'bg-zinc-800 text-zinc-100 shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
            title="Comparador Dividido (Slider Antes / Depois)"
          >
            <Split className="h-3.5 w-3.5" />
            <span>Divisor</span>
          </button>
          <button
            onClick={() => setSplitMode('side-by-side')}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
              splitMode === 'side-by-side'
                ? 'bg-zinc-800 text-zinc-100 shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
            title="Visualização Lado a Lado"
          >
            <Columns className="h-3.5 w-3.5" />
            <span>Lado a Lado</span>
          </button>
        </div>
      )}

      {/* Right Hardware Telemetry & Actions */}
      <div className="flex items-center gap-2.5">
        {/* GPU & VRAM Badge */}
        <Badge
          variant="secondary"
          className="hidden sm:inline-flex items-center gap-1.5 py-1 px-2.5 bg-zinc-900 border-zinc-800 text-zinc-300 font-mono text-[11px]"
          title="Telemetria de Hardware em Tempo Real"
        >
          <span className="relative flex h-2 w-2">
            {!offline && (
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            )}
            <span className={`relative inline-flex rounded-full h-2 w-2 ${offline ? 'bg-red-500' : 'bg-emerald-500'}`}></span>
          </span>
          <Cpu className="h-3.5 w-3.5 text-zinc-400" />
          <span>{hardware ? (hardware.cuda_available ? hardware.gpu : offline ? hardware.gpu : 'CPU') : 'Conectando...'}</span>
          {vramFreeGb && (
            <>
              <span className="text-zinc-400">|</span>
              <span className="text-zinc-300">{vramFreeGb} GB VRAM</span>
            </>
          )}
        </Badge>

        {original && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              resetParameters();
              clearImage();
            }}
            className="text-zinc-400 hover:text-zinc-200 gap-1.5 text-xs h-8 px-2.5"
            title="Limpar imagem e reiniciar estúdio"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Reiniciar</span>
          </Button>
        )}
      </div>
    </header>
  );
};
