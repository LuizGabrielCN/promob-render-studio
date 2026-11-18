import React from 'react';
import { useEnhancerStore } from '@/store/useEnhancerStore';
import { Header } from '@/components/Header';
import { StatusBar } from '@/components/StatusBar';
import { Dropzone } from '@/components/Dropzone';
import { Viewer } from '@/components/Viewer';
import { Sidebar } from '@/components/Sidebar';

export const App: React.FC = () => {
  const { original } = useEnhancerStore();

  return (
    <div className="h-screen w-screen flex flex-col bg-zinc-950 text-zinc-100 font-sans overflow-hidden">
      {/* 1. Professional Header */}
      <Header />

      {/* 2. Main Studio Workspace */}
      <main className="flex-1 flex overflow-hidden relative">
        {original ? (
          <>
            {/* Interactive Before / After Viewer */}
            <Viewer />
            {/* Parametric Sidebar */}
            <Sidebar />
          </>
        ) : (
          /* Dropzone & Sample Loader */
          <Dropzone />
        )}
      </main>

      {/* 3. Hardware & Execution Status Bar */}
      <StatusBar />
    </div>
  );
};

export default App;
