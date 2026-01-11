import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { TopKItem } from '../lib/api';
import { cn } from '../lib/utils';

interface ProbabilityBarChartProps {
  data: TopKItem[];
  className?: string;
}

export function ProbabilityBarChart({ data, className }: ProbabilityBarChartProps) {
  const chartData = data.map((item, index) => ({
    dialect: item.dialect,
    probability: item.prob * 100,
    index,
  }));

  const colors = [
    'hsl(221, 83%, 53%)',
    'hsl(221, 83%, 45%)',
    'hsl(221, 83%, 40%)',
    'hsl(221, 83%, 35%)',
    'hsl(221, 83%, 30%)',
  ];

  return (
    <div className={cn("w-full", className)}>
      <h3 className="text-lg font-semibold mb-4 text-slate-900 dark:text-slate-100">
        Top {data.length} Predictions
      </h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData} layout="vertical">
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" domain={[0, 100]} />
          <YAxis
            dataKey="dialect"
            type="category"
            width={120}
            tick={{ fontSize: 12 }}
          />
          <Tooltip
            formatter={(value: number) => `${value.toFixed(2)}%`}
            contentStyle={{
              backgroundColor: 'hsl(var(--background))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '8px',
            }}
          />
          <Bar dataKey="probability" radius={[0, 8, 8, 0]}>
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

