import React, { useRef, useState, useEffect, useCallback } from 'react';
import { useEnhancerStore } from '@/store/useEnhancerStore';
import {
  ChevronsLeftRight,
  Eye,
  Sparkles,
} from 'lucide-react';

export const Viewer: React.FC = () => {
  const {
    original,
    previewUrl,
    enhanced4kUrl,
    sliderPos,
    setSliderPos,
    splitMode,
    zoom,
    isSpacePressed,
    setIsSpacePressed,
  } = useEnhancerStore();

  const containerRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  // Active enhanced source: either finished 4K result, live preview, or fallback to original
  const enhancedSrc = enhanced4kUrl || previewUrl || original?.objectUrl;
  const isOriginalForced = isSpacePressed;

  // Global keyboard listener for Spacebar shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger if user is focusing an input or typing
      if (['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) return;
      if (e.code === 'Space' && !e.repeat) {
        e.preventDefault();
        setIsSpacePressed(true);
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        e.preventDefault();
        setIsSpacePressed(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [setIsSpacePressed]);

  // Handle pointer dragging for the comparison divider
  const handlePointerMove = useCallback(
    (e: PointerEvent | React.PointerEvent) => {
      if (!isDragging && e.type !== 'pointerdown') return;
      if (!containerRef.current) return;

      const rect = containerRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
      setSliderPos(percentage);
    },
    [isDragging, setSliderPos]
  );

  const handlePointerDown = (e: React.PointerEvent) => {
    e.preventDefault();
    setIsDragging(true);
    handlePointerMove(e);
  };

  useEffect(() => {
    const onPointerUp = () => setIsDragging(false);
    const onPointerMove = (e: PointerEvent) => {
      if (isDragging) handlePointerMove(e);
    };

    if (isDragging) {
      window.addEventListener('pointermove', onPointerMove);
      window.addEventListener('pointerup', onPointerUp);
    }
    return () => {
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
    };
  }, [isDragging, handlePointerMove]);

  if (!original) return null;

  // Effective slider position (if spacebar is pressed, force to 100% so full original Promob is visible)
  const currentPos = isOriginalForced ? 100 : sliderPos;

  return (
    <div className="flex-1 flex flex-col items-center justify-center relative p-4 bg-zinc-950 overflow-hidden select-none">
      {/* Side-by-Side Mode */}
      {splitMode === 'side-by-side' ? (
        <div className="w-full h-full flex flex-col md:flex-row gap-4 items-center justify-center p-2">
          {/* Left: Original */}
          <div className="relative flex-1 flex flex-col items-center justify-center bg-zinc-900/60 rounded-lg border border-zinc-800 p-2 overflow-hidden h-full max-h-[85vh]">
            <div className="absolute top-4 left-4 z-20 flex items-center gap-1.5 px-2.5 py-1 rounded bg-zinc-950/80 backdrop-blur border border-zinc-800 text-zinc-300 font-mono text-[11px]">
              <Eye className="h-3 w-3 text-zinc-400" />
              <span>Original (Promob)</span>
            </div>
            <img
              src={original.objectUrl}
              alt="Promob Original"
              className="max-h-full max-w-full object-contain rounded transition-transform"
              style={{ transform: `scale(${zoom})` }}
              draggable={false}
            />
          </div>

          {/* Right: Enhanced */}
          <div className="relative flex-1 flex flex-col items-center justify-center bg-zinc-900/60 rounded-lg border border-zinc-800 p-2 overflow-hidden h-full max-h-[85vh]">
            <div className="absolute top-4 left-4 z-20 flex items-center gap-1.5 px-2.5 py-1 rounded bg-zinc-950/80 backdrop-blur border border-emerald-800/60 text-emerald-300 font-mono text-[11px]">
              <Sparkles className="h-3 w-3 text-amber-400" />
              <span>{enhanced4kUrl ? 'Aprimorado (4K Studio)' : 'Prévia Interativa'}</span>
            </div>
            <img
              src={enhancedSrc}
              alt="Promob Aprimorado"
              className="max-h-full max-w-full object-contain rounded transition-transform"
              style={{ transform: `scale(${zoom})` }}
              draggable={false}
            />
          </div>
        </div>
      ) : (
        /* Split Slider Mode (Single Canvas with clip-path) */
        <div
          ref={containerRef}
          onPointerDown={handlePointerDown}
          className="relative inline-block max-w-full max-h-[82vh] rounded-lg border border-zinc-800 bg-zinc-900/40 overflow-hidden cursor-ew-resize group shadow-2xl"
        >
          {/* Layer 1: Original Image (Bottom Layer - drives intrinsic sizing) */}
          <img
            src={original.objectUrl}
            alt="Original Promob"
            className="block max-w-full max-h-[82vh] w-auto h-auto object-contain pointer-events-none transition-transform"
            style={{ transform: `scale(${zoom})` }}
            draggable={false}
          />

          {/* Layer 2: Enhanced Image (Top Layer clipped by sliderPos) */}
          <div
            className="absolute inset-0 w-full h-full pointer-events-none overflow-hidden"
            style={{
              clipPath: `inset(0 0 0 ${currentPos}%)`,
            }}
          >
            <img
              src={enhancedSrc}
              alt="Aprimorado Promob"
              className="absolute inset-0 w-full h-full object-contain pointer-events-none transition-transform"
              style={{ transform: `scale(${zoom})` }}
              draggable={false}
            />
          </div>

          {/* Divider Line & Handle */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-zinc-100 z-20 pointer-events-none transition-all"
            style={{ left: `${currentPos}%` }}
          >
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 h-8 w-8 rounded-full bg-zinc-900/90 border border-zinc-400 flex items-center justify-center text-zinc-100 shadow-xl comparison-slider-handle pointer-events-auto cursor-ew-resize hover:scale-110 active:scale-95 transition-transform">
              <ChevronsLeftRight className="h-4 w-4 text-zinc-200" />
            </div>
          </div>

          {/* Left Label: Original */}
          <div className="absolute top-4 left-4 z-10 flex items-center gap-1.5 px-2.5 py-1 rounded bg-zinc-950/80 backdrop-blur border border-zinc-800 text-zinc-300 font-mono text-[11px] shadow-md pointer-events-none">
            <Eye className="h-3 w-3 text-zinc-400" />
            <span>Antes (Promob)</span>
          </div>

          {/* Right Label: Enhanced */}
          <div className="absolute top-4 right-4 z-10 flex items-center gap-1.5 px-2.5 py-1 rounded bg-zinc-950/80 backdrop-blur border border-emerald-800/80 text-emerald-300 font-mono text-[11px] shadow-md pointer-events-none">
            <Sparkles className="h-3 w-3 text-amber-400" />
            <span>{enhanced4kUrl ? 'Depois (4K Studio)' : 'Depois (Prévia)'}</span>
          </div>

          {/* Spacebar Indicator Toast when holding Space */}
          {isOriginalForced && (
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-30 px-3 py-1.5 rounded-full bg-zinc-900/95 border border-zinc-700 text-zinc-200 font-medium text-xs flex items-center gap-2 shadow-2xl animate-fade-in pointer-events-none">
              <Eye className="h-3.5 w-3.5 text-amber-400" />
              <span>Exibindo render original [Espaço mantido]</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
