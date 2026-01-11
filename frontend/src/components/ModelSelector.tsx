import React from 'react';
import { ModelInfo } from '../lib/api';
import { cn } from '../lib/utils';

interface ModelSelectorProps {
  models: ModelInfo[];
  defaultModel: string;
  selectedModel: string;
  onModelChange: (model: string) => void;
}

export function ModelSelector({
  models,
  defaultModel,
  selectedModel,
  onModelChange,
}: ModelSelectorProps) {
  const modelOptions = [
    { value: 'Best (recommended)', label: 'Best (Recommended)' },
    ...models.map(m => ({
      value: m.name,
      label: m.name.charAt(0).toUpperCase() + m.name.slice(1).replace('_', '-'),
    })),
  ];

  return (
    <div>
      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
        Model
      </label>
      <select
        value={selectedModel}
        onChange={(e) => onModelChange(e.target.value)}
        className={cn(
          "w-full px-4 py-2 rounded-lg border",
          "bg-white dark:bg-slate-800",
          "border-slate-300 dark:border-slate-600",
          "text-slate-900 dark:text-slate-100",
          "focus:outline-none focus:ring-2 focus:ring-primary",
          "transition-colors"
        )}
      >
        {modelOptions.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
}

