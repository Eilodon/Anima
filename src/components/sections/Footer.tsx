import React from 'react';
import { Play, Pause, Square, Lock } from 'lucide-react';
import clsx from 'clsx';
import { useSessionStore } from '../../stores/sessionStore';
import { useSettingsStore } from '../../stores/settingsStore';
import { TRANSLATIONS } from '../../translations';
import { BREATHING_PATTERNS, BreathingType, BreathPattern, SafetyProfile } from '../../types';
import { unlockAudio, cleanupAudio } from '../../services/audio';
import { hapticTick } from '../../services/haptics';

type FooterProps = {
  selectedPatternId: BreathingType;
  setSelectedPatternId: (id: BreathingType) => void;
};

export function Footer({ selectedPatternId, setSelectedPatternId }: FooterProps) {
  const isActive = useSessionStore(s => s.isActive);
  const isPaused = useSessionStore(s => s.isPaused);
  const startSession = useSessionStore(s => s.startSession);
  const togglePause = useSessionStore(s => s.togglePause);
  const finishSession = useSessionStore(s => s.finishSession);
  
  const userSettings = useSettingsStore(s => s.userSettings);
  const setLastUsedPattern = useSettingsStore(s => s.setLastUsedPattern);
  const history = useSettingsStore(s => s.history);

  const t = TRANSLATIONS[userSettings.language] || TRANSLATIONS.en;

  // --- SAFETY GUARD LOGIC (Trauma Registry Check) ---
  const isPatternLocked = (pattern: BreathPattern): { locked: boolean; reason?: string } => {
    // 1. TIER 1: Always Open (Safe Zone)
    if (pattern.tier === 1) {
      // Even Safe Zone patterns can be locked if they caused an Incident recently
      const record = userSettings.safetyRegistry[pattern.id];
      if (record && record.safety_lock_until > Date.now()) {
          const remainingHours = Math.ceil((record.safety_lock_until - Date.now()) / (1000 * 60 * 60));
          return { locked: true, reason: `Safety Lock Active (${remainingHours}h)` };
      }
      return { locked: false };
    }
    
    // Filter for "quality" sessions (> 60s duration) for experience calculations
    const qualitySessions = history.filter(h => h.durationSec > 60);
    // Explicit cast to avoid 'unknown' type inference issues
    const patternRecord = userSettings.safetyRegistry[pattern.id] as SafetyProfile | undefined;
    
    // Check specific lock
    if (patternRecord && patternRecord.safety_lock_until > Date.now()) {
         return { locked: true, reason: 'Pattern Locked due to Adverse Event' };
    }

    // 2. TIER 2: Intermediate (Requires Experience + Stability)
    if (pattern.tier === 2) {
      const minSessions = 5;
      if (qualitySessions.length < minSessions) {
        return { 
          locked: true, 
          reason: `Complete ${minSessions - qualitySessions.length} more sessions to unlock` 
        };
      }
      
      // Check for history of stress (Resonance History average < 0.5)
      if (patternRecord && patternRecord.resonance_history && patternRecord.resonance_history.length > 0) {
          const history = patternRecord.resonance_history;
          const avgResonance = history.reduce((a: number, b: number) => a + b, 0) / history.length;
          if (avgResonance < 0.4) {
               return { locked: true, reason: 'System detected stress. Try Tier 1 patterns.' };
          }
      }
      
      return { locked: false };
    }
    
    // 3. TIER 3: Advanced (Requires Mastery)
    if (pattern.tier === 3) {
      const minSessions = 20;
      if (qualitySessions.length < minSessions) {
        return { 
          locked: true, 
          reason: `Advanced. Need ${minSessions - qualitySessions.length} more sessions.` 
        };
      }
      
      // Requires High Resonance across all sessions (Average Free Energy low)
      // Simplified: Check if user has at least 5 "Perfect" sessions (resonance 1.0)
      const perfectSessions = (Object.values(userSettings.safetyRegistry) as SafetyProfile[]).reduce((acc: number, profile: SafetyProfile) => {
          const perfects = profile.resonance_history ? profile.resonance_history.filter(r => r === 1.0).length : 0;
          return acc + perfects;
      }, 0);

      if (perfectSessions < 5) {
          return { locked: true, reason: 'Requires greater stability (5 perfect sessions).' };
      }
      
      return { locked: false };
    }
    
    return { locked: false };
  };

  const triggerHaptic = (strength: 'light' | 'medium' | 'heavy' = 'light') => {
    if (userSettings.hapticEnabled) hapticTick(true, strength);
  };

  const handleStart = (patternId: BreathingType) => {
    triggerHaptic('medium');
    unlockAudio();
    setLastUsedPattern(patternId);
    startSession(patternId);
  };

  const handleStop = () => {
    triggerHaptic('medium');
    cleanupAudio();
    finishSession();
  };
  
  const handleTogglePause = () => {
    triggerHaptic('light');
    togglePause();
  };

  const handleSelectPattern = (id: BreathingType) => {
    const pattern = BREATHING_PATTERNS[id];
    const lockStatus = isPatternLocked(pattern);
    if (lockStatus.locked) {
        triggerHaptic('heavy'); // Reject
        return;
    }
    setSelectedPatternId(id);
    handleStart(id);
  };

  return (
    <footer 
        className={clsx(
          "fixed bottom-0 inset-x-0 z-30 pb-[calc(2.5rem+env(safe-area-inset-bottom))] px-6 transition-all duration-700 ease-out",
        )}
      >
        <div className="max-w-md mx-auto w-full flex flex-col justify-end min-h-[160px]">
          
          {!isActive && (
            <div className="animate-in slide-in-from-bottom-8 fade-in duration-1000 space-y-8">
               
               {/* --- HERO CTA --- */}
               <button 
                  onClick={() => handleStart(selectedPatternId)}
                  className="relative w-full p-8 rounded-[2rem] overflow-hidden group text-left transition-transform active:scale-[0.99] glass-card-hero border border-white/10"
                  aria-label={`Start breathing session: ${t.patterns[selectedPatternId].label}`}
                >
                  <div className="relative z-10 flex flex-col items-start">
                    <div className="font-caps text-[9px] tracking-[0.2em] text-white/40 mb-3 ml-1">{t.ui.continue}</div>
                    <div className="text-5xl font-serif mb-3 text-white font-medium tracking-tight">{t.patterns[selectedPatternId].label}</div>
                    <div className="flex items-center gap-2 text-white/50 text-xs font-light ml-1">
                       <Play size={12} fill="currentColor" /> <span className="tracking-widest uppercase text-[10px]">{t.ui.begin}</span>
                    </div>
                  </div>
                  <div className="absolute inset-0 opacity-[0.03] bg-[url('https://grainy-gradients.vercel.app/noise.svg')] bg-repeat mix-blend-overlay" />
                  <div className="absolute inset-0 bg-gradient-to-tr from-white/0 via-white/[0.05] to-white/0 opacity-0 group-hover:opacity-100 transition-opacity duration-700 pointer-events-none" />
               </button>

               {/* --- SECONDARY GRID (Horizontal Scroll) --- */}
               <div className="w-full overflow-hidden">
                  <div className="font-caps text-[9px] tracking-[0.2em] text-white/30 mb-4 pl-1">{t.ui.selectRhythm}</div>
                  <div className="flex gap-4 overflow-x-auto pb-8 -mx-6 px-6 snap-x scrollbar-hide select-none">
                      {Object.values(BREATHING_PATTERNS).map((p: BreathPattern) => {
                        const isSelected = p.id === selectedPatternId;
                        const lockStatus = isPatternLocked(p);
                        const locked = lockStatus.locked;
                        
                        return (
                          <button
                            key={p.id}
                            onClick={() => handleSelectPattern(p.id as BreathingType)}
                            disabled={locked}
                            className={clsx(
                              "relative flex-shrink-0 w-[42%] min-w-[160px] snap-start p-5 rounded-[1.5rem] border text-left transition-all active:scale-[0.98] overflow-hidden backdrop-blur-md group",
                              isSelected 
                                ? "bg-white/10 border-white/20 shadow-lg shadow-white/5" 
                                : "bg-white/[0.02] border-white/5 hover:bg-white/[0.05]",
                              locked && "opacity-50 grayscale cursor-not-allowed"
                            )}
                            aria-label={`Start pattern: ${t.patterns[p.id as BreathingType]?.label || p.label}`}
                          >
                              {/* Highlight Indicator */}
                              {isSelected && !locked && <div className="absolute top-3 right-3 w-1.5 h-1.5 rounded-full bg-white shadow-[0_0_8px_rgba(255,255,255,0.8)]" />}
                              
                              {locked && (
                                  <>
                                    <div className="absolute inset-0 z-20 bg-black/40 flex items-center justify-center backdrop-blur-[2px]">
                                        <Lock size={20} className="text-white/40" />
                                    </div>
                                    <div className="absolute bottom-full left-0 right-0 mb-2 p-2 bg-black/90 text-white/80 text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-30 text-center">
                                       {lockStatus.reason}
                                    </div>
                                  </>
                              )}
                              
                              <div className="text-[8px] tracking-widest font-bold px-2 py-1 rounded-md bg-white/5 inline-block text-white/50 mb-3 uppercase truncate max-w-full">
                                {t.patterns[p.id as BreathingType]?.tag || p.tag}
                              </div>
                              <h3 className={clsx("text-lg font-serif mb-1 truncate", isSelected ? "text-white" : "text-white/80")}>
                                {t.patterns[p.id as BreathingType]?.label || p.label}
                              </h3>
                              <div className="flex items-center gap-2 text-[10px] font-mono text-white/30 tracking-wide">
                                  <span>{p.timings.inhale}-{p.timings.holdIn}-{p.timings.exhale}</span>
                              </div>
                          </button>
                        );
                      })}
                  </div>
               </div>

            </div>
          )}

          {isActive && (
            <div className="grid grid-cols-2 gap-4 animate-in slide-in-from-bottom-10 fade-in duration-700">
              <button
                onClick={handleTogglePause}
                className="py-5 bg-white/[0.02] backdrop-blur-2xl border border-white/5 hover:bg-white/[0.05] text-white rounded-2xl font-medium flex items-center justify-center gap-3 transition-all active:scale-95 group"
                aria-label={isPaused ? "Resume session" : "Pause session"}
              >
                {isPaused ? <Play size={18} fill="currentColor" className="opacity-80"/> : <Pause size={18} fill="currentColor" className="opacity-80" />}
                <span className="text-xs tracking-widest uppercase opacity-60 group-hover:opacity-100 transition-opacity">{isPaused ? t.ui.resume : t.ui.pause}</span>
              </button>
              <button
                onClick={handleStop}
                className="py-5 bg-white/[0.02] backdrop-blur-2xl border border-white/5 hover:bg-red-500/10 text-white/40 hover:text-red-300 rounded-2xl font-medium flex items-center justify-center gap-3 transition-all active:scale-95 group"
                aria-label="End session"
              >
                <Square size={16} fill="currentColor" className="opacity-60" />
                <span className="text-xs tracking-widest uppercase opacity-60 group-hover:opacity-100 transition-opacity">{t.ui.end}</span>
              </button>
            </div>
          )}
        </div>
      </footer>
  );
}
