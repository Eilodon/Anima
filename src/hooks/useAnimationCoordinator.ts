
import { useRef, useCallback, useEffect } from 'react';

/**
 * Single RAF loop coordinator to prevent multiple requestAnimationFrame calls 
 * running simultaneously in different components.
 */
export function useAnimationCoordinator() {
  const frameCallbacks = useRef<Set<(deltaTime: number) => void>>(new Set());
  const rafId = useRef<number | null>(null);
  const lastTime = useRef<number>(0);

  const start = useCallback(() => {
    if (rafId.current !== null) return;
    
    lastTime.current = performance.now();
    
    const loop = (now: number) => {
      // Calculate delta time in seconds
      const delta = (now - lastTime.current) / 1000;
      lastTime.current = now;

      // Execute all subscribers
      frameCallbacks.current.forEach(cb => cb(delta));
      
      rafId.current = requestAnimationFrame(loop);
    };
    
    rafId.current = requestAnimationFrame(loop);
  }, []);

  const stop = useCallback(() => {
    if (rafId.current) {
      cancelAnimationFrame(rafId.current);
      rafId.current = null;
    }
  }, []);

  const subscribe = useCallback((callback: (dt: number) => void) => {
    frameCallbacks.current.add(callback);
    
    // Auto-start if it's the first subscriber
    if (frameCallbacks.current.size === 1) {
      start();
    }

    return () => {
      frameCallbacks.current.delete(callback);
      // Auto-stop if no subscribers left
      if (frameCallbacks.current.size === 0) {
        stop();
      }
    };
  }, [start, stop]);

  // Cleanup on unmount
  useEffect(() => {
    return () => stop();
  }, [stop]);

  return { subscribe };
}
