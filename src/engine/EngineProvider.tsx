
import React, { createContext, useContext, useEffect, useRef } from 'react';
import { useBreathEngine } from '../hooks/useBreathEngine';

/**
 * EngineProvider
 * -------------
 * ✅ SINGLETON for the OS Driver.
 * The driver hook owns:
 * - RAF tick loop feeding the Kernel
 * - side effects (audio)
 * - progress/entropy refs for visuals
 *
 * IMPORTANT:
 * - Call useBreathEngine() exactly ONCE in the whole app.
 * - Every UI component must consume refs via useEngine().
 */

type EngineRefs = ReturnType<typeof useBreathEngine>;

const EngineCtx = createContext<EngineRefs | null>(null);

let _engineProviderMounted = 0;

export function EngineProvider({ children }: { children?: React.ReactNode }) {
  const refs = useBreathEngine();

  useEffect(() => {
    // Dev-time safety: prevents accidentally mounting the driver twice
    _engineProviderMounted += 1;
    if (typeof window !== 'undefined' && _engineProviderMounted > 1) {
      console.warn('[ZenB] EngineProvider mounted more than once. This will double-run the Kernel + audio.');
    }
    return () => { _engineProviderMounted = Math.max(0, _engineProviderMounted - 1); };
  }, []);

  return <EngineCtx.Provider value={refs}>{children}</EngineCtx.Provider>;
}

export function useEngine(): EngineRefs {
  const ctx = useContext(EngineCtx);
  if (!ctx) throw new Error('useEngine must be used within <EngineProvider>. Wrap the app root.');
  return ctx;
}
