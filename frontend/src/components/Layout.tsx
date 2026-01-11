import React from 'react';

interface LayoutProps {
  children: React.ReactNode;
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 py-8">
        <header className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-slate-900 dark:text-slate-100 mb-2">
            Arabic Dialect Identification
          </h1>
          <p className="text-lg text-slate-600 dark:text-slate-400">
            22-Class Dialect Classification from Speech Audio
          </p>
          <p className="text-sm text-slate-500 dark:text-slate-500 mt-1">
            Research Prototype • Final Year Project
          </p>
        </header>
        {children}
      </div>
    </div>
  );
}

