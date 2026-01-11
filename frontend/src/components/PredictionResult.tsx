import React from 'react';
import { Download, CheckCircle2 } from 'lucide-react';
import { PredictionResponse } from '../lib/api';
import { ProbabilityBarChart } from './ProbabilityBarChart';
import { cn } from '../lib/utils';

interface PredictionResultProps {
  result: PredictionResponse;
}

export function PredictionResult({ result }: PredictionResultProps) {
  const downloadJSON = () => {
    const dataStr = JSON.stringify(result, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `prediction_${result.request_id}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Main Prediction */}
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-slate-700">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
            Prediction Result
          </h2>
          <button
            onClick={downloadJSON}
            className={cn(
              "flex items-center gap-2 px-4 py-2 rounded-lg",
              "bg-primary hover:bg-primary/90 text-white",
              "transition-colors text-sm font-medium"
            )}
          >
            <Download className="w-4 h-4" />
            Download JSON
          </button>
        </div>

        <div className="flex items-center gap-4 mb-6">
          <div className={cn(
            "flex items-center justify-center w-16 h-16 rounded-full",
            "bg-green-100 dark:bg-green-900/30"
          )}>
            <CheckCircle2 className="w-8 h-8 text-green-600 dark:text-green-400" />
          </div>
          <div>
            <p className="text-sm text-slate-500 dark:text-slate-400">Predicted Dialect</p>
            <p className="text-3xl font-bold text-slate-900 dark:text-slate-100">
              {result.dialect}
            </p>
          </div>
          <div className="ml-auto text-right">
            <p className="text-sm text-slate-500 dark:text-slate-400">Confidence</p>
            <p className="text-3xl font-bold text-primary">
              {(result.confidence * 100).toFixed(1)}%
            </p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <p className="text-slate-500 dark:text-slate-400">Model</p>
            <p className="font-medium text-slate-900 dark:text-slate-100">
              {result.model_name}
            </p>
          </div>
          <div>
            <p className="text-slate-500 dark:text-slate-400">Window Mode</p>
            <p className="font-medium text-slate-900 dark:text-slate-100">
              {result.window_mode}
            </p>
          </div>
          <div>
            <p className="text-slate-500 dark:text-slate-400">Duration</p>
            <p className="font-medium text-slate-900 dark:text-slate-100">
              {result.duration_sec}s
            </p>
          </div>
          <div>
            <p className="text-slate-500 dark:text-slate-400">Request ID</p>
            <p className="font-mono text-xs text-slate-600 dark:text-slate-400">
              {result.request_id.slice(0, 8)}...
            </p>
          </div>
        </div>
      </div>

      {/* Probability Chart */}
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-slate-700">
        <ProbabilityBarChart data={result.top_k} />
      </div>

      {/* Full Distribution Table */}
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-slate-700">
        <h3 className="text-lg font-semibold mb-4 text-slate-900 dark:text-slate-100">
          Full Probability Distribution
        </h3>
        <div className="max-h-96 overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-slate-50 dark:bg-slate-900">
              <tr>
                <th className="text-left p-2 font-medium text-slate-700 dark:text-slate-300">
                  Dialect
                </th>
                <th className="text-right p-2 font-medium text-slate-700 dark:text-slate-300">
                  Probability
                </th>
                <th className="w-32 p-2">
                  <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded-full" />
                </th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(result.all_probs)
                .sort(([, a], [, b]) => b - a)
                .map(([dialect, prob]) => (
                  <tr
                    key={dialect}
                    className={cn(
                      "border-t border-slate-200 dark:border-slate-700",
                      dialect === result.dialect && "bg-primary/10"
                    )}
                  >
                    <td className="p-2 font-medium text-slate-900 dark:text-slate-100">
                      {dialect}
                    </td>
                    <td className="p-2 text-right text-slate-600 dark:text-slate-400">
                      {(prob * 100).toFixed(2)}%
                    </td>
                    <td className="p-2">
                      <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-primary transition-all"
                          style={{ width: `${prob * 100}%` }}
                        />
                      </div>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

