import { ImageSource } from '@/types';

/** Lê as dimensões de um arquivo de imagem e cria a URL local usada no comparador. */
export function readImageSource(file: File): Promise<ImageSource> {
  return new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () =>
      resolve({
        file,
        objectUrl,
        width: img.naturalWidth,
        height: img.naturalHeight,
        name: file.name,
        size: file.size,
      });
    img.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error('Não foi possível ler as dimensões da imagem.'));
    };
    img.src = objectUrl;
  });
}
