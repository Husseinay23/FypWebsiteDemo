import React, { useState, useRef, useEffect } from 'react';
import { Mic, Square, Play, Pause } from 'lucide-react';
import { MediaRecorderAudioRecorder, visualizeWaveform, getAudioDuration } from '../lib/audioUtils';
import { cn } from '../lib/utils';

interface AudioRecorderProps {
  onRecordingComplete: (blob: Blob) => void;
}

export function AudioRecorder({ onRecordingComplete }: AudioRecorderProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const recorderRef = useRef<MediaRecorderAudioRecorder | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const durationIntervalRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (durationIntervalRef.current) {
        clearInterval(durationIntervalRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (audioBlob && canvasRef.current) {
      visualizeWaveform(audioBlob, canvasRef.current, 800, 150);
      getAudioDuration(audioBlob).then(setDuration);
    }
  }, [audioBlob]);

  const startRecording = async () => {
    try {
      const recorder = new MediaRecorderAudioRecorder();
      recorderRef.current = recorder;
      await recorder.start();
      setIsRecording(true);
      setDuration(0);
      setAudioBlob(null);

      // Update duration every second
      const startTime = Date.now();
      durationIntervalRef.current = window.setInterval(() => {
        setDuration((Date.now() - startTime) / 1000);
      }, 100);
    } catch (error) {
      alert(`Failed to start recording: ${error}`);
    }
  };

  const stopRecording = async () => {
    if (!recorderRef.current) return;

    try {
      const blob = await recorderRef.current.stop();
      setAudioBlob(blob);
      setIsRecording(false);
      onRecordingComplete(blob);

      if (durationIntervalRef.current) {
        clearInterval(durationIntervalRef.current);
        durationIntervalRef.current = null;
      }
    } catch (error) {
      alert(`Failed to stop recording: ${error}`);
      setIsRecording(false);
    }
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-center gap-4">
        {!isRecording ? (
          <button
            onClick={startRecording}
            className={cn(
              "flex items-center gap-2 px-6 py-3 rounded-lg font-medium",
              "bg-red-500 hover:bg-red-600 text-white",
              "transition-colors shadow-lg"
            )}
          >
            <Mic className="w-5 h-5" />
            Start Recording
          </button>
        ) : (
          <button
            onClick={stopRecording}
            className={cn(
              "flex items-center gap-2 px-6 py-3 rounded-lg font-medium",
              "bg-red-600 hover:bg-red-700 text-white",
              "transition-colors shadow-lg animate-pulse"
            )}
          >
            <Square className="w-5 h-5" />
            Stop Recording
          </button>
        )}
      </div>

      {(isRecording || audioBlob) && (
        <div className="space-y-2">
          <div className="text-center text-sm text-slate-600 dark:text-slate-400">
            {isRecording ? (
              <span className="text-red-600 dark:text-red-400 font-medium">
                Recording... {formatDuration(duration)}
              </span>
            ) : (
              <span>Duration: {formatDuration(duration)}</span>
            )}
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

