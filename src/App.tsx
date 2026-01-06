
import React, { useEffect, useState, useRef } from 'react';
import { useSessionStore } from './stores/sessionStore';
import { useSettingsStore } from './stores/settingsStore';
import { useUIStore } from './stores/uiStore';
import OrbBreathViz from './components/OrbBreathVizZenSciFi';
import { useEngine } from './engine/EngineProvider';
import { cleanupAudio, unlockAudio } from './services/audio';
import { BreathingType } from './types';
import { TRANSLATIONS } from './translations';
import { Header } from './components/sections/Header';
import { Footer } from './components/sections/Footer';
import { ActiveSessionDisplay } from './components/ActiveSessionDisplay';
import { OnboardingModal } from './components/modals/OnboardingModal';
import { SummaryModal } from './components/modals/SummaryModal';
import { HistorySheet } from './components/sections/HistorySheet';
import { SettingsSheet } from './components/sections/SettingsSheet';
import { KernelMonitor } from './components/KernelMonitor';
import { BREATHING_PATTERNS } from './types';

export default function App() {
  // --- SELECTORS ---
  const isActive = useSessionStore(s => s.isActive);
  const isPaused = useSessionStore(s => s.isPaused);
  const phase = useSessionStore(s => s.phase);
  const currentPattern = useSessionStore(s => s.currentPattern);
  const lastSessionStats = useSessionStore(s => s.lastSessionStats);
  
  const userSettings = useSettingsStore(s => s.userSettings);
  const hasSeenOnboarding = useSettingsStore(s => s.hasSeenOnboarding);
  const completeOnboarding = useSettingsStore(s => s.completeOnboarding);
  const toggleKernelMonitor = useSettingsStore(s => s.toggleKernelMonitor);

  const showSummary = useUIStore(s => s.showSummary);
  const setShowSummary = useUIStore(s => s.setShowSummary);

  // Core Engine (Biological OS Driver) - VIA SINGLETON CONTEXT
  const { progressRef, entropyRef } = useEngine();
  
  const [selectedPatternId, setSelectedPatternId] = useState<BreathingType>(userSettings.lastUsedPattern || '4-7-8');
  
  useEffect(() => {
    setSelectedPatternId(userSettings.lastUsedPattern);
  }, [userSettings.lastUsedPattern]);

  const wakeLockRef = useRef<WakeLockSentinel | null>(null);
  const t = TRANSLATIONS[userSettings.language] || TRANSLATIONS.en;

  useEffect(() => {
    if (lastSessionStats) {
      setShowSummary(true);
    }
  }, [lastSessionStats, setShowSummary]);

  const handleCloseSummary = () => {
    setShowSummary(false);
  };

  // GLOBAL AUDIO UNLOCKER & SECRET DEV GESTURE
  useEffect(() => {
    let tapCount = 0;
    let lastTap = 0;

    const handleTap = (e: MouseEvent | TouchEvent) => {
        // Audio Unlock
        unlockAudio();

        // Secret 5-tap top-right corner to toggle Kernel Monitor
        const x = 'clientX' in e ? e.clientX : e.touches[0].clientX;
        const y = 'clientY' in e ? e.clientY : e.touches[0].clientY;
        const w = window.innerWidth;
        
        if (x > w - 100 && y < 100) {
            const now = Date.now();
            if (now - lastTap < 500) {
                tapCount++;
            } else {
                tapCount = 1;
            }
            lastTap = now;
            
            if (tapCount === 5) {
                toggleKernelMonitor();
                tapCount = 0;
            }
        }
    };

    window.addEventListener('click', handleTap);
    window.addEventListener('touchstart', handleTap);
    return () => {
        window.removeEventListener('click', handleTap);
        window.removeEventListener('touchstart', handleTap);
    };
  }, [toggleKernelMonitor]);

  // Audio Cleanup
  useEffect(() => {
    if (!isActive) cleanupAudio();
  }, [isActive]);

  // ROBUST WAKE LOCK
  useEffect(() => {
    const requestWakeLock = async () => {
      if (!isActive || isPaused) {
        if (wakeLockRef.current) {
          wakeLockRef.current.release().catch(() => {});
          wakeLockRef.current = null;
        }
        return;
      }

      if ('wakeLock' in navigator && !wakeLockRef.current) {
        try {
          const lock = await navigator.wakeLock.request('screen');
          wakeLockRef.current = lock;
          lock.addEventListener('release', () => {
            wakeLockRef.current = null;
          });
        } catch (err) {
          console.warn("Wake Lock failed:", err);
        }
      }
    };

    requestWakeLock();

    const handleVis = () => {
      if (document.visibilityState === 'visible') {
        requestWakeLock();
      } else {
        if (wakeLockRef.current) {
          wakeLockRef.current.release().catch(() => {});
          wakeLockRef.current = null;
        }
      }
    };
    
    document.addEventListener('visibilitychange', handleVis);
    return () => {
      document.removeEventListener('visibilitychange', handleVis);
      if (wakeLockRef.current) wakeLockRef.current.release().catch(() => {});
    };
  }, [isActive, isPaused]);

  return (
    <div className="relative w-full min-h-dvh overflow-hidden bg-black text-white selection:bg-white/20 font-sans">
      
      {/* ---------------- LAYER 0: VISUAL CORTEX ---------------- */}
      <div className="absolute inset-0 z-0 flex items-center justify-center">
        <OrbBreathViz
          phase={phase}
          theme={isActive ? currentPattern.colorTheme : BREATHING_PATTERNS[selectedPatternId].colorTheme}
          quality={userSettings.quality}
          reduceMotion={userSettings.reduceMotion}
          progressRef={progressRef}
          entropyRef={entropyRef}
          isActive={isActive}
        />
      </div>

      {/* ---------------- LAYER 1: UI ORCHESTRATION ---------------- */}
      
      <Header />
      
      <ActiveSessionDisplay />
      
      <Footer 
        selectedPatternId={selectedPatternId} 
        setSelectedPatternId={setSelectedPatternId} 
      />

      {/* ---------------- OVERLAYS ---------------- */}
      
      {!hasSeenOnboarding && <OnboardingModal onComplete={completeOnboarding} t={t} />}
      
      {showSummary && lastSessionStats && (
        <SummaryModal 
            stats={lastSessionStats} 
            onClose={handleCloseSummary} 
            t={t} 
            streak={userSettings.streak} 
            language={userSettings.language}
        />
      )}

      {userSettings.showKernelMonitor && <KernelMonitor onClose={toggleKernelMonitor} />}

      <HistorySheet />
      <SettingsSheet />
      
    </div>
  );
}
