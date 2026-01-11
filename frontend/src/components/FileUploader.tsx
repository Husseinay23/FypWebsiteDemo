import React, { useRef, useState, useEffect } from 'react';
import { Upload, FileAudio, X } from 'lucide-react';
import { visualizeWaveform, getAudioDuration } from '../lib/audioUtils';
import { cn } from '../lib/utils';

interface FileUploaderProps {
  onFileSelected: (file: File) => void;
}

export function FileUploader({ onFileSelected }: FileUploaderProps) {
  const [file, setFile] = useState<File | null>(null);
  const [duration, setDuration] = useState<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (file && canvasRef.current) {
      const blob = new Blob([file], { type: file.type });
      visualizeWaveform(blob, canvasRef.current, 800, 150);
      getAudioDuration(blob).then(setDuration);
    }
  }, [file]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      onFileSelected(selectedFile);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith('audio/')) {
      setFile(droppedFile);
      onFileSelected(droppedFile);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const clearFile = () => {
    setFile(null);
    setDuration(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="space-y-4">
      {!file ? (
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          className={cn(
            "border-2 border-dashed border-slate-300 dark:border-slate-600",
            "rounded-lg p-8 text-center cursor-pointer",
            "hover:border-primary transition-colors",
            "bg-slate-50 dark:bg-slate-800/50"
          )}
          onClick={() => fileInputRef.current?.click()}
        >
          <Upload className="w-12 h-12 mx-auto mb-4 text-slate-400" />
          <p className="text-slate-600 dark:text-slate-400 mb-2">
            Click to upload or drag and drop
          </p>
          <p className="text-sm text-slate-500 dark:text-slate-500">
            WAV, MP3, WEBM (max 50MB)
          </p>
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            onChange={handleFileChange}
            className="hidden"
          />
        </div>
      ) : (
        <div className="space-y-2">
          <div className="flex items-center justify-between bg-white dark:bg-slate-800 rounded-lg p-4 shadow-sm">
            <div className="flex items-center gap-3">
              <FileAudio className="w-5 h-5 text-primary" />
              <div>
                <p className="font-medium text-slate-900 dark:text-slate-100">
                  {file.name}
                </p>
                {duration && (
                  <p className="text-sm text-slate-500 dark:text-slate-500">
                    {formatDuration(duration)} • {(file.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                )}
              </div>
            </div>
            <button
              onClick={clearFile}
              className="p-2 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-lg transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
          <div className="bg-white dark:bg-slate-800 rounded-lg p-4 shadow-sm">
            <canvas
              ref={canvasRef}
              className="w-full h-[150px]"
            />
          </div>
        </div>
      )}
    </div>
  );
}

