import React, { useState, useEffect } from 'react';
import { Layout } from './components/Layout';
import { AudioRecorder } from './components/AudioRecorder';
import { FileUploader } from './components/FileUploader';
import { ModelSelector } from './components/ModelSelector';
import { WindowModeSelector } from './components/WindowModeSelector';
import { PredictionResult } from './components/PredictionResult';
import { getModels, predict, ModelInfo, PredictionResponse } from './lib/api';
import { blobToFile } from './lib/audioUtils';
import { cn } from './lib/utils';
import { Loader2, AlertCircle } from 'lucide-react';

type TabType = 'record' | 'upload';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('record');
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [defaultModel, setDefaultModel] = useState<string>('resnet18');
  const [selectedModel, setSelectedModel] = useState<string>('Best (recommended)');
  const [selectedWindowMode, setSelectedWindowMode] = useState<string>('auto');
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);

  useEffect(() => {
    // Load available models
    getModels()
      .then((response) => {
        setModels(response.models);
        setDefaultModel(response.default_model);
      })
      .catch((err) => {
        setError(`Failed to load models: ${err.message}`);
      });
  }, []);

  const handleRecordingComplete = (blob: Blob) => {
    const file = blobToFile(blob, 'recording.webm');
    setAudioFile(file);
  };

  const handleFileSelected = (file: File) => {
    setAudioFile(file);
  };

  const handleAnalyze = async () => {
    if (!audioFile) {
      setError('Please record or upload an audio file first');
      return;
    }

    setIsLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const result = await predict(audioFile, selectedModel, selectedWindowMode);
      setPrediction(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Layout>
      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Pane: Input */}
          <div className="space-y-6">
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-slate-700">
              {/* Tabs */}
              <div className="flex gap-2 mb-6 border-b border-slate-200 dark:border-slate-700">
                <button
                  onClick={() => setActiveTab('record')}
                  className={cn(
                    "px-4 py-2 font-medium transition-colors",
                    "border-b-2 -mb-[1px]",
                    activeTab === 'record'
                      ? "border-primary text-primary"
                      : "border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300"
                  )}
                >
                  Record
                </button>
                <button
                  onClick={() => setActiveTab('upload')}
                  className={cn(
                    "px-4 py-2 font-medium transition-colors",
                    "border-b-2 -mb-[1px]",
                    activeTab === 'upload'
                      ? "border-primary text-primary"
                      : "border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300"
                  )}
                >
                  Upload
                </button>
              </div>

              {/* Tab Content */}
              {activeTab === 'record' ? (
                <AudioRecorder onRecordingComplete={handleRecordingComplete} />
              ) : (
                <FileUploader onFileSelected={handleFileSelected} />
              )}
            </div>

            {/* Controls */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-slate-700 space-y-4">
              {models.length > 0 && (
                <ModelSelector
                  models={models}
                  defaultModel={defaultModel}
                  selectedModel={selectedModel}
                  onModelChange={setSelectedModel}
                />
              )}

              <WindowModeSelector
                selectedMode={selectedWindowMode}
                onModeChange={setSelectedWindowMode}
              />

              <button
                onClick={handleAnalyze}
                disabled={!audioFile || isLoading}
                className={cn(
                  "w-full px-6 py-3 rounded-lg font-medium",
                  "bg-primary hover:bg-primary/90 text-white",
                  "disabled:opacity-50 disabled:cursor-not-allowed",
                  "transition-colors shadow-lg",
                  "flex items-center justify-center gap-2"
                )}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  'Analyze'
                )}
              </button>

              {error && (
                <div className={cn(
                  "flex items-center gap-2 p-4 rounded-lg",
                  "bg-red-50 dark:bg-red-900/20",
                  "text-red-700 dark:text-red-400",
                  "border border-red-200 dark:border-red-800"
                )}>
                  <AlertCircle className="w-5 h-5" />
                  <p className="text-sm">{error}</p>
                </div>
              )}
            </div>
          </div>

          {/* Right Pane: Results */}
          <div>
            {prediction ? (
              <PredictionResult result={prediction} />
            ) : (
              <div className="bg-white dark:bg-slate-800 rounded-xl p-12 shadow-lg border border-slate-200 dark:border-slate-700 text-center">
                <p className="text-slate-500 dark:text-slate-400">
                  {isLoading
                    ? 'Processing audio...'
                    : 'Record or upload audio to get started'}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </Layout>
  );
}

export default App;

