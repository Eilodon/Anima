
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { UserSettings, SessionHistoryItem, ColorTheme, QualityTier, Language, SoundPack, BreathingType, SafetyProfile } from '../types';
import { kernel } from '../services/PureZenBKernel';

type SettingsState = {
  userSettings: UserSettings;
  history: SessionHistoryItem[];
  hasSeenOnboarding: boolean;

  // Actions
  toggleSound: () => void;
  toggleHaptic: () => void;
  setHapticStrength: (s: UserSettings['hapticStrength']) => void;
  setTheme: (t: ColorTheme) => void;
  setQuality: (q: QualityTier) => void;
  setReduceMotion: (v: boolean) => void;
  toggleTimer: () => void;
  setLanguage: (l: Language) => void;
  setSoundPack: (p: SoundPack) => void;
  completeOnboarding: () => void;
  clearHistory: () => void;
  setLastUsedPattern: (p: BreathingType) => void;
  toggleCameraVitals: () => void;
  toggleKernelMonitor: () => void;
  
  // Logic
  registerSessionComplete: (durationSec: number, patternId: BreathingType, cycles: number) => void;
};

const getTodayString = () => {
  const d = new Date();
  return d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') + '-' + String(d.getDate()).padStart(2, '0');
};

const getYesterdayString = () => {
  const d = new Date();
  d.setDate(d.getDate() - 1);
  return d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') + '-' + String(d.getDate()).padStart(2, '0');
};

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      userSettings: {
        soundEnabled: true,
        hapticEnabled: true,
        hapticStrength: 'medium',
        theme: 'neutral',
        quality: 'auto',
        reduceMotion: false,
        showTimer: true,
        language: 'en',
        soundPack: 'musical',
        streak: 0,
        lastBreathDate: '',
        lastUsedPattern: '4-7-8',
        safetyRegistry: {},
        cameraVitalsEnabled: false,
        showKernelMonitor: false
      },
      history: [],
      hasSeenOnboarding: false,

      toggleSound: () => set((s) => ({ userSettings: { ...s.userSettings, soundEnabled: !s.userSettings.soundEnabled } })),
      toggleHaptic: () => set((s) => ({ userSettings: { ...s.userSettings, hapticEnabled: !s.userSettings.hapticEnabled } })),
      setHapticStrength: (strength) => set((state) => ({ userSettings: { ...state.userSettings, hapticStrength: strength } })),
      setTheme: (t) => set((s) => ({ userSettings: { ...s.userSettings, theme: t } })),
      setQuality: (q) => set((s) => ({ userSettings: { ...s.userSettings, quality: q } })),
      setReduceMotion: (v) => set((s) => ({ userSettings: { ...s.userSettings, reduceMotion: v } })),
      toggleTimer: () => set((s) => ({ userSettings: { ...s.userSettings, showTimer: !s.userSettings.showTimer } })),
      setLanguage: (l) => set((s) => ({ userSettings: { ...s.userSettings, language: l } })),
      setSoundPack: (p) => set((s) => ({ userSettings: { ...s.userSettings, soundPack: p } })),
      completeOnboarding: () => set({ hasSeenOnboarding: true }),
      clearHistory: () => set({ history: [] }),
      setLastUsedPattern: (p) => set((s) => ({ userSettings: { ...s.userSettings, lastUsedPattern: p } })),
      toggleCameraVitals: () => set((s) => ({ userSettings: { ...s.userSettings, cameraVitalsEnabled: !s.userSettings.cameraVitalsEnabled } })),
      toggleKernelMonitor: () => set((s) => ({ userSettings: { ...s.userSettings, showKernelMonitor: !s.userSettings.showKernelMonitor } })),

      registerSessionComplete: (durationSec, patternId, cycles) => {
        const state = get();
        const kernelState = kernel.getState(); 

        // --- TRAUMA/SAFETY REGISTRY UPDATE ---
        const isSuccess = durationSec > 45 && kernelState.belief.prediction_error < 0.5;
        const registry = { ...state.userSettings.safetyRegistry };
        
        const record: SafetyProfile = registry[patternId] || {
            patternId,
            cummulative_stress_score: 0,
            last_incident_timestamp: 0,
            safety_lock_until: 0,
            resonance_history: []
        };

        if (isSuccess) {
            record.resonance_history.push(1.0);
        } else {
            record.resonance_history.push(0.0);
            record.cummulative_stress_score += 1;
            
            if (record.cummulative_stress_score > 5) {
                record.safety_lock_until = Date.now() + (24 * 60 * 60 * 1000);
                record.cummulative_stress_score = 0;
            }
        }
        
        if (record.resonance_history.length > 5) record.resonance_history.shift();
        
        registry[patternId] = record;
        kernel.loadSafetyRegistry(registry);

        // --- HISTORY & STREAK ---
        let newHistory = state.history;
        if (durationSec > 10) {
            const newItem: SessionHistoryItem = {
                id: Date.now().toString() + Math.random().toString().slice(2, 6),
                timestamp: Date.now(),
                durationSec,
                patternId,
                cycles,
                finalBelief: kernelState.belief
            };
            newHistory = [newItem, ...state.history].slice(0, 100);
        }

        let newStreak = state.userSettings.streak;
        let newLastDate = state.userSettings.lastBreathDate;
        
        if (durationSec > 30) {
            const today = getTodayString();
            const yesterday = getYesterdayString();
            
            if (newLastDate === today) {
                // Already breathed today
            } else if (newLastDate === yesterday) {
                newStreak += 1;
                newLastDate = today;
            } else {
                newStreak = 1;
                newLastDate = today;
            }
        }

        set({
            history: newHistory,
            userSettings: {
                ...state.userSettings,
                streak: newStreak,
                lastBreathDate: newLastDate,
                lastUsedPattern: patternId,
                safetyRegistry: registry
            }
        });
      }
    }),
    {
      name: 'zenb-settings-storage',
      partialize: (state) => ({ 
        userSettings: state.userSettings, 
        hasSeenOnboarding: state.hasSeenOnboarding,
        history: state.history
      }),
      onRehydrateStorage: () => (state) => {
         if (state && state.userSettings) {
             kernel.loadSafetyRegistry(state.userSettings.safetyRegistry);
         }
      }
    }
  )
);
