import React, { useRef, useState } from 'react';
import { useEnhancerStore } from '@/store/useEnhancerStore';
import { Button } from '@/components/ui/button';
import {
  UploadCloud,
  FileImage,
  Sparkles,
  ArrowRight,
  ShieldCheck,
  Zap,
} from 'lucide-react';

export const Dropzone: React.FC = () => {
  const { setOriginalImage } = useEnhancerStore();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoadingSample, setIsLoadingSample] = useState(false);
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

    const objectUrl = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      setOriginalImage({
        file,
        objectUrl,
        width: img.naturalWidth,
        height: img.naturalHeight,
        name: file.name,
        size: file.size,
      });
    };
    img.onerror = () => {
      setErrorMessage('Não foi possível ler as dimensões da imagem.');
      URL.revokeObjectURL(objectUrl);
    };
    img.src = objectUrl;
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

  const handleLoadSample = async () => {
    try {
      setIsLoadingSample(true);
      setErrorMessage(null);
      const sampleUrl = '/samples/promob_sample.jpg';
      const response = await fetch(sampleUrl);
      if (!response.ok) {
        throw new Error('Amostra Promob não encontrada em /samples/promob_sample.jpg');
      }
      const blob = await response.blob();
      const file = new File([blob], 'promob_cozinha_original.jpg', { type: 'image/jpeg' });
      processFile(file);
    } catch (err: any) {
      console.error('Erro ao carregar amostra:', err);
      setErrorMessage(err.message || 'Erro ao carregar render de amostra.');
    } finally {
      setIsLoadingSample(false);
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
              handleLoadSample();
            }}
            disabled={isLoadingSample}
            className="text-xs bg-zinc-100 text-zinc-900 hover:bg-zinc-200 font-semibold"
          >
            <Sparkles className="h-3.5 w-3.5 mr-1.5 text-amber-600" />
            {isLoadingSample ? 'Carregando Amostra...' : 'Carregar Amostra Promob'}
          </Button>
        </div>

        {/* Feature Badges */}
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 w-full max-w-md pt-4 border-t border-zinc-800/80 text-[11px] text-zinc-400">
          <div className="flex items-center gap-1.5 justify-center">
            <Zap className="h-3.5 w-3.5 text-amber-400" />
            <span>Super-Resolução 4K</span>
          </div>
          <div className="flex items-center gap-1.5 justify-center">
            <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
            <span>Zero Alucinação</span>
          </div>
          <div className="flex items-center gap-1.5 justify-center col-span-2 sm:col-span-1">
            <ArrowRight className="h-3.5 w-3.5 text-zinc-400" />
            <span>Fidelidade 100%</span>
          </div>
        </div>
      </div>

      {errorMessage && (
        <div className="mt-4 p-3 rounded-md bg-red-950/60 border border-red-800 text-red-300 text-xs flex items-center gap-2">
          <span>{errorMessage}</span>
        </div>
      )}
    </div>
  );
};
