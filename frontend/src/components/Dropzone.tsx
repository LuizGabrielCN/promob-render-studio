import React, { useRef, useState } from 'react';
import { useEnhancerStore } from '@/store/useEnhancerStore';
import { Button } from '@/components/ui/button';
import { readImageSource } from '@/lib/image';
import {
  UploadCloud,
  FileImage,
  Sparkles,
  ArrowRight,
  ShieldCheck,
  Zap,
} from 'lucide-react';

export const Dropzone: React.FC = () => {
  const { setOriginalImage, trySample, sampleLoading, sampleError } = useEnhancerStore();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const processFile = (file: File) => {
    setErrorMessage(null);
    const validExtensions = ['.jpg', '.jpeg', '.png', '.webp'];
    const hasValidExt = validExtensions.some((ext) =>
      file.name.toLowerCase().endsWith(ext)
    );
    const isImageType = file.type.startsWith('image/');

    if (!isImageType && !hasValidExt) {
      setErrorMessage('Por favor, selecione um arquivo de imagem válido (JPG, PNG ou WEBP).');
      return;
    }

    readImageSource(file)
      .then(setOriginalImage)
      .catch((err: Error) => setErrorMessage(err.message));
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0]);
    }
  };

  return (
    <div className="flex-1 flex flex-col items-center justify-center p-6 lg:p-12 relative select-none">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="hidden"
        onChange={handleFileInputChange}
      />

      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`w-full max-w-2xl border-2 border-dashed rounded-xl p-10 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200 group ${
          isDragging
            ? 'border-zinc-400 bg-zinc-900/80 scale-[1.01]'
            : 'border-zinc-800 hover:border-zinc-700 bg-zinc-950/40 hover:bg-zinc-900/30'
        }`}
      >
        <div className="h-16 w-16 rounded-2xl bg-zinc-900 border border-zinc-800 flex items-center justify-center mb-5 text-zinc-400 group-hover:text-zinc-200 group-hover:border-zinc-700 transition-all shadow-inner">
          <UploadCloud className="h-8 w-8" />
        </div>

        <h2 className="text-lg font-medium text-zinc-100 tracking-tight mb-1">
          Arraste seu render Promob aqui
        </h2>
        <p className="text-xs text-zinc-400 mb-6 max-w-md">
          Suporte direto para imagens brutas geradas pelo Promob Studio ou Real Scene (JPEG, PNG, WEBP).
          A amostra roda no modo Clássico, em CPU e na resolução original, sem baixar modelos de IA.
        </p>

        <div className="flex items-center gap-3 mb-6">
          <Button
            variant="secondary"
            size="sm"
            type="button"
            className="pointer-events-none text-xs"
          >
            <FileImage className="h-3.5 w-3.5 mr-1.5" />
            Selecionar do Computador
          </Button>

          <span className="text-zinc-600 text-xs font-mono">ou</span>

          <Button
            variant="default"
            size="sm"
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              trySample();
            }}
            disabled={sampleLoading}
            className="text-xs bg-zinc-100 text-zinc-900 hover:bg-zinc-200 font-semibold"
          >
            <Sparkles className="h-3.5 w-3.5 mr-1.5 text-amber-600" />
            {sampleLoading ? 'Carregando amostra...' : 'Experimentar com uma amostra'}
          </Button>
        </div>

        {/* Feature Badges */}
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 w-full max-w-md pt-4 border-t border-zinc-800/80 text-[11px] text-zinc-400">
          <div className="flex items-center gap-1.5 justify-center">
            <Zap className="h-3.5 w-3.5 text-amber-400" />
            <span>Ampliação 4× opcional</span>
          </div>
          <div className="flex items-center gap-1.5 justify-center">
            <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
            <span>Processamento local</span>
          </div>
          <div className="flex items-center gap-1.5 justify-center col-span-2 sm:col-span-1">
            <ArrowRight className="h-3.5 w-3.5 text-zinc-400" />
            <span>Geometria preservada</span>
          </div>
        </div>
      </div>

      {(errorMessage || sampleError) && (
        <div className="mt-4 p-3 rounded-md bg-red-950/60 border border-red-800 text-red-300 text-xs flex items-center gap-2">
          <span>{errorMessage || sampleError}</span>
        </div>
      )}
    </div>
  );
};
