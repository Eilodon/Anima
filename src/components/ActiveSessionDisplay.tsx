
import React, { useRef, useEffect, useMemo, useState } from 'react';
import clsx from 'clsx';
import { useSessionStore } from '../stores/sessionStore';
import { useSettingsStore } from '../stores/settingsStore';
import { useAnimationCoordinator } from '../hooks/useAnimationCoordinator';
import { useEngine } from '../engine/EngineProvider';
import { TRANSLATIONS } from '../translations';
import { ProgressArc } from './ProgressArc';
import { kernel } from '../services/PureZenBKernel';

export function ActiveSessionDisplay() {
  const isActive = useSessionStore(s => s.isActive);
  const isPaused = useSessionStore(s => s.isPaused);
  const phase = useSessionStore(s => s.phase);
  const cycleCount = useSessionStore(s => s.cycleCount);
  const userSettings = useSettingsStore(s => s.userSettings);

  const { progressRef } = useEngine();
  const animationCoordinator = useAnimationCoordinator();
  
  const textScaleRef = useRef<HTMLDivElement>(null);
  const ringCircleRef = useRef<SVGCircleElement>(null);

  const t = TRANSLATIONS[userSettings.language] || TRANSLATIONS.en;

  const [vitals, setVitals] = useState({ heartRate: 0, confidence: 0 });

  const phaseLabel = useMemo(() => {
    if (phase === 'holdIn' || phase === 'holdOut') return t.phases.hold;
    return t.phases[phase];
  }, [phase, t]);

  useEffect(() => {
    const unsub = kernel.subscribe(state => {
      // Use re-added lastObservation or fallback
      if (state.lastObservation && state.lastObservation.heart_rate !== undefined) {
         setVitals({
             heartRate: Math.round(state.lastObservation.heart_rate),
             confidence: state.lastObservation.hr_confidence || 0
         });
      } else {
         setVitals({ heartRate: 0, confidence: 0 });
      }
    });
    return unsub;
  }, []);

  // UI Animation Loop
  useEffect(() => {
    if (!isActive) {
      if (textScaleRef.current) textScaleRef.current.style.transform = 'scale(1)';
      return;
    }

    const RING_SIZE = 220;
    const STROKE = 2;
    const RADIUS = (RING_SIZE - STROKE) / 2;
    const CIRCUMFERENCE = RADIUS * 2 * Math.PI;

    return animationCoordinator.subscribe(() => {
      const p = progressRef.current;
      
      if (ringCircleRef.current) {
        const offset = CIRCUMFERENCE - p * CIRCUMFERENCE;
        ringCircleRef.current.style.strokeDashoffset = String(offset);
      }

      let scale = 1;
      if (phase === 'inhale') scale = 1 + (p * 0.05); 
      else if (phase === 'exhale') scale = 1.05 - (p * 0.05);
      else if (phase === 'holdIn') scale = 1.05;
      else if (phase === 'holdOut') scale = 1;
      
      if (textScaleRef.current) {
        textScaleRef.current.style.transform = `scale(${scale})`;
      }
    });
  }, [isActive, phase, animationCoordinator, progressRef]);

  if (!isActive) return null;

  return (
    <div className="absolute inset-0 z-20 flex flex-col items-center justify-center pointer-events-none">
        <div className={clsx("relative flex items-center justify-center transition-all duration-1000", isPaused ? "opacity-30 blur-sm scale-95" : "opacity-100 scale-100")}>
        
        <div 
            ref={textScaleRef}
            className="absolute inset-0 flex flex-col items-center justify-center z-10 will-change-transform"
        >
            <div className="text-4xl font-serif font-light tracking-[0.15em] text-white/90 mb-4 drop-shadow-2xl mix-blend-overlay">
            {phaseLabel}
            </div>
            {userSettings.showTimer && (
            <div className="text-[9px] font-sans text-white/30 uppercase tracking-[0.3em] font-light">
                {t.ui.cycle} {cycleCount + 1}
            </div>
            )}
        </div>

        {userSettings.showTimer && (
            <ProgressArc size={220} circleRef={ringCircleRef} />
        )}
        </div>

        {isPaused && (
        <div className="absolute inset-0 flex items-center justify-center z-30">
            <div className="px-6 py-3 rounded-full border border-white/10 bg-black/60 backdrop-blur-2xl text-[10px] font-bold tracking-[0.3em] uppercase shadow-2xl animate-in fade-in zoom-in-95 font-sans text-white/80">
                {t.ui.paused}
            </div>
        </div>
        )}

        {vitals.confidence > 0.5 && (
            <div className="absolute bottom-32 text-center animate-in fade-in slide-in-from-bottom-4 duration-700">
                <div className="text-xs text-white/40 uppercase tracking-widest mb-1">Heart Rate</div>
                <div className="text-2xl font-light text-white/70 font-mono">
                {vitals.heartRate} <span className="text-sm">BPM</span>
                </div>
            </div>
        )}
    </div>
  );
}
