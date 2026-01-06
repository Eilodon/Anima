
import React from 'react';
import { History, Settings2, Flame } from 'lucide-react';
import clsx from 'clsx';
import { useSessionStore } from '../../stores/sessionStore';
import { useSettingsStore } from '../../stores/settingsStore';
import { useUIStore } from '../../stores/uiStore';
import { TRANSLATIONS } from '../../translations';
import { hapticTick } from '../../services/haptics';

export function Header() {
  const isActive = useSessionStore(s => s.isActive);
  const userSettings = useSettingsStore(s => s.userSettings);
  const setHistoryOpen = useUIStore(s => s.setHistoryOpen);
  const setSettingsOpen = useUIStore(s => s.setSettingsOpen);
  
  const t = TRANSLATIONS[userSettings.language] || TRANSLATIONS.en;

  const triggerHaptic = () => {
    if (userSettings.hapticEnabled) hapticTick(true, 'light');
  };

  return (
    <header 
        className={clsx(
          "fixed top-0 inset-x-0 z-40 p-6 pt-[calc(1.5rem+env(safe-area-inset-top))] flex justify-between items-start transition-all duration-1000 ease-in-out",
          isActive ? "opacity-0 pointer-events-none -translate-y-8" : "opacity-100 translate-y-0"
        )}
      >
        <div className="flex flex-col">
          <h1 className="text-xl font-serif font-medium tracking-wider text-white/90">ZENB</h1>
          {userSettings.streak > 0 && (
             <div className="flex items-center gap-1.5 mt-1.5 animate-in fade-in slide-in-from-left-2 duration-700 delay-300">
                <Flame size={10} className={clsx("transition-colors", userSettings.streak > 1 ? "fill-orange-400 text-orange-400" : "text-white/20")} />
                <span className="text-[9px] font-sans text-white/30 tracking-[0.2em] uppercase">{userSettings.streak} {t.ui.dayStreak}</span>
             </div>
          )}
        </div>
        
        <div className="flex gap-3">
          <button 
              onClick={() => { triggerHaptic(); setHistoryOpen(true); }}
              className="p-3 bg-white/[0.02] hover:bg-white/[0.08] rounded-full border border-white/5 transition-all active:scale-95 backdrop-blur-md"
              aria-label="Open History"
          >
              <History size={18} className="text-white/70" strokeWidth={1.5} />
          </button>
          <button 
              onClick={() => { triggerHaptic(); setSettingsOpen(true); }}
              className="p-3 bg-white/[0.02] hover:bg-white/[0.08] rounded-full border border-white/5 transition-all active:scale-95 backdrop-blur-md"
              aria-label="Open Settings"
          >
              <Settings2 size={18} className="text-white/70" strokeWidth={1.5} />
          </button>
        </div>
      </header>
  );
}
