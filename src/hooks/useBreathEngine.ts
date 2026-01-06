
import React, { useEffect, useRef } from 'react';
import { useSessionStore } from '../stores/sessionStore';
import { useSettingsStore } from '../stores/settingsStore';
import { kernel, RuntimeState } from '../services/PureZenBKernel';
import { useCameraVitals } from './useCameraVitals';

type EngineRefs = {
  progressRef: React.MutableRefObject<number>;
  entropyRef: React.MutableRefObject<number>;
};

/**
 * 🜂 DRIVER (View-Controller Bridge) V2
 */
export function useBreathEngine(): EngineRefs {
  const isActive = useSessionStore((s) => s.isActive);
  const isPaused = useSessionStore((s) => s.isPaused);
  const currentPattern = useSessionStore((s) => s.currentPattern);
  const stopSession = useSessionStore((s) => s.stopSession);
  const syncState = useSessionStore((s) => s.syncState);
  
  const storeUserSettings = useSettingsStore((s) => s.userSettings);
  
  // Visual Interpolation Refs
  const progressRef = useRef<number>(0);
  const entropyRef = useRef<number>(0); 
  
  // --- SENSOR DRIVER: CAMERA VITALS ---
  const { vitals } = useCameraVitals(isActive && storeUserSettings.cameraVitalsEnabled);
  
  // --- KERNEL CONTROL BUS ---
  
  // 1. Handle START / STOP
  useEffect(() => {
    if (isActive) {
        kernel.dispatch({ type: 'LOAD_PROTOCOL', patternId: currentPattern.id, timestamp: Date.now() });
        kernel.dispatch({ type: 'START_SESSION', timestamp: Date.now() });
    } else {
        progressRef.current = 0;
        kernel.dispatch({ type: 'HALT', reason: 'cleanup', timestamp: Date.now() });
    }
  }, [isActive, currentPattern.id]);

  // 2. Handle PAUSE / RESUME
  useEffect(() => {
    if (!isActive) return;
    if (isPaused) {
       kernel.dispatch({ type: 'INTERRUPTION', kind: 'pause', timestamp: Date.now() });
    } else {
       kernel.dispatch({ type: 'RESUME', timestamp: Date.now() });
    }
  }, [isPaused, isActive]);

  // --- KERNEL OBSERVER (Visuals & React State) ---
  useEffect(() => {
      const unsub = kernel.subscribe((state: RuntimeState) => {
          // Safety Monitor
          if (state.status === 'SAFETY_LOCK') {
              stopSession();
          }

          // Visual Cortex Driver
          progressRef.current = state.phaseElapsed / (state.phaseDuration || 1);
          entropyRef.current = state.belief.prediction_error;

          // UI State Sync
          if (state.phase !== useSessionStore.getState().phase || state.cycleCount !== useSessionStore.getState().cycleCount) {
              syncState(state.phase, state.cycleCount);
          }
      });
      return unsub;
  }, [stopSession, syncState]);

  // --- CLOCK DRIVER (Tick Loop) ---
  useEffect(() => {
      if (!isActive) return;

      let lastTime = performance.now();
      let frameId: number;

      const tickLoop = (now: number) => {
          if (isPaused) {
              lastTime = now;
              frameId = requestAnimationFrame(tickLoop);
              return;
          }

          const dt = Math.min((now - lastTime) / 1000, 0.1);
          lastTime = now;

          // Dispatch TICK + SENSOR DATA to Kernel
          kernel.tick(dt, {
              timestamp: Date.now(),
              delta_time: dt,
              visibilty_state: document.hidden ? 'hidden' : 'visible',
              user_interaction: undefined,
              heart_rate: vitals.heartRate,
              hr_confidence: vitals.confidence
          });

          frameId = requestAnimationFrame(tickLoop);
      };

      frameId = requestAnimationFrame(tickLoop);
      return () => cancelAnimationFrame(frameId);
  }, [isActive, isPaused, vitals]);

  return { progressRef, entropyRef };
}
