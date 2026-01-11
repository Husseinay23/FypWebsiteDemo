import React, { useEffect, useRef } from 'react';
import { cn } from '../lib/utils';

interface SpectrogramViewerProps {
  melSpectrogram?: number[][];
  className?: string;
}

export function SpectrogramViewer({ melSpectrogram, className }: SpectrogramViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!melSpectrogram || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const height = melSpectrogram.length; // n_mels
    const width = melSpectrogram[0]?.length || 0;

    canvas.width = width;
    canvas.height = height;

    // Normalize values for visualization
    const flat = melSpectrogram.flat();
    const min = Math.min(...flat);
    const max = Math.max(...flat);

    // Create image data
    const imageData = ctx.createImageData(width, height);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const value = melSpectrogram[y][x];
        const normalized = (value - min) / (max - min);
        
        // Map to colormap (viridis-like)
        const idx = (y * width + x) * 4;
        const r = Math.floor(normalized * 255);
        const g = Math.floor(normalized * 200);
        const b = Math.floor(normalized * 100);
        
        imageData.data[idx] = r;
        imageData.data[idx + 1] = g;
        imageData.data[idx + 2] = b;
        imageData.data[idx + 3] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }, [melSpectrogram]);

  if (!melSpectrogram) {
    return (
      <div className={cn("bg-slate-100 dark:bg-slate-800 rounded-lg p-8 text-center", className)}>
        <p className="text-slate-500 dark:text-slate-400">
          Spectrogram will appear here after analysis
        </p>
      </div>
    );
  }

  return (
    <div className={cn("bg-white dark:bg-slate-800 rounded-lg p-4 shadow-sm", className)}>
      <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
        Mel-Spectrogram
      </h3>
      <canvas
        ref={canvasRef}
        className="w-full h-64 border border-slate-200 dark:border-slate-700 rounded"
      />
    </div>
  );
}

