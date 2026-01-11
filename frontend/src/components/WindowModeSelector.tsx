import React from 'react';
import { cn } from '../lib/utils';

interface WindowModeSelectorProps {
  selectedMode: string;
  onModeChange: (mode: string) => void;
}

const windowModeOptions = [
  { value: 'auto', label: 'Auto (Recommended)' },
  { value: '7s', label: '7 seconds' },
  { value: '3s_center', label: '3 seconds (center)' },
  { value: '3s_5crop', label: '3 seconds (5-crop)' },
];

export function WindowModeSelector({
  selectedMode,
  onModeChange,
}: WindowModeSelectorProps) {
  return (
    <div>
      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
        Window Mode
      </label>
      <select
        value={selectedMode}
        onChange={(e) => onModeChange(e.target.value)}
        className={cn(
          "w-full px-4 py-2 rounded-lg border",
          "bg-white dark:bg-slate-800",
          "border-slate-300 dark:border-slate-600",
          "text-slate-900 dark:text-slate-100",
          "focus:outline-none focus:ring-2 focus:ring-primary",
          "transition-colors"
        )}
      >
        {windowModeOptions.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
}

