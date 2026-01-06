
import React, { useMemo } from 'react';
import { X, Flame, Trash2 } from 'lucide-react';
import clsx from 'clsx';
import { useSettingsStore } from '../../stores/settingsStore';
import { useUIStore } from '../../stores/uiStore';
import { TRANSLATIONS } from '../../translations';
import { hapticTick } from '../../services/haptics';

const formatDate = (timestamp: number, lang: 'en' | 'vi', t: any) => {
  const date = new Date(timestamp);
  const now = new Date();
  const isToday = date.toDateString() === now.toDateString();
  const timeStr = date.toLocaleTimeString(lang === 'vi' ? 'vi-VN' : 'en-US', { hour: '2-digit', minute: '2-digit' });
  if (isToday) return `${t.history.today}, ${timeStr}`;
  return date.toLocaleDateString(lang === 'vi' ? 'vi-VN' : 'en-US', { month: 'short', day: 'numeric' }) + `, ${timeStr}`;
};

export function HistorySheet() {
  const isHistoryOpen = useUIStore(s => s.isHistoryOpen);
  const setHistoryOpen = useUIStore(s => s.setHistoryOpen);
  
  const userSettings = useSettingsStore(s => s.userSettings);
  const history = useSettingsStore(s => s.history);
  const clearHistory = useSettingsStore(s => s.clearHistory);
  
  const t = TRANSLATIONS[userSettings.language] || TRANSLATIONS.en;

  const triggerHaptic = () => {
    if (userSettings.hapticEnabled) hapticTick(true, 'medium');
  };

  const historyStats = useMemo(() => {
    const totalSessions = history.length;
    const totalSecs = history.reduce((acc, curr) => acc + curr.durationSec, 0);
    const totalMins = Math.floor(totalSecs / 60);
    return { totalSessions, totalMins };
  }, [history]);

  return (
    <div 
        className={clsx(
            "fixed inset-0 z-50 transition-colors duration-700 pointer-events-none",
            isHistoryOpen ? "bg-black/60 backdrop-blur-sm pointer-events-auto" : "bg-transparent"
        )}
        onClick={() => setHistoryOpen(false)}
        role="dialog"
        aria-label="History Sheet"
      >
        <div 
          onClick={(e) => e.stopPropagation()}
          className={clsx(
              "absolute inset-x-0 bottom-0 h-[85vh] bg-[#050505] border-t border-white/10 rounded-t-[3rem] p-8 pb-[calc(2.5rem+env(safe-area-inset-bottom))] transition-transform duration-500 cubic-bezier(0.19, 1, 0.22, 1) shadow-2xl flex flex-col",
              isHistoryOpen ? "translate-y-0" : "translate-y-full"
          )}
        >
           <div className="flex justify-between items-center mb-10 flex-shrink-0">
              <h3 className="text-2xl font-serif text-white tracking-wide flex items-center gap-3">{t.history.title}</h3>
              <button onClick={() => setHistoryOpen(false)} aria-label="Close History" className="p-2.5 bg-white/5 hover:bg-white/10 rounded-full transition-colors border border-white/5"><X size={20} className="text-white/70"/></button>
           </div>
           
           <div className="flex-1 overflow-y-auto scrollbar-hide">
              <div className="grid grid-cols-2 gap-4 mb-10">
                  <div className="bg-white/[0.02] border border-white/5 rounded-[2rem] p-6 backdrop-blur-md">
                      <div className="text-3xl font-light font-sans mb-1 text-white/90">{historyStats.totalMins}</div>
                      <div className="text-white/30 font-caps text-[9px] tracking-widest">{t.history.totalMinutes}</div>
                  </div>
                  <div className="bg-white/[0.02] border border-white/5 rounded-[2rem] p-6 backdrop-blur-md relative overflow-hidden">
                      {userSettings.streak > 1 && <div className="absolute inset-0 bg-orange-500/5" />}
                      <div className={clsx("text-3xl font-light font-sans mb-1 flex items-center gap-2", userSettings.streak > 1 ? "text-orange-200" : "text-white/90")}>
                          {userSettings.streak} <Flame size={18} className={userSettings.streak > 1 ? "text-orange-500 fill-orange-500" : "text-white/20"} />
                      </div>
                      <div className="text-white/30 font-caps text-[9px] tracking-widest">{t.ui.streak}</div>
                  </div>
              </div>

              {history.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-20 opacity-30 text-center">
                  <div className="mb-6 text-5xl grayscale opacity-50">🍃</div>
                  <p className="text-sm font-light max-w-[200px] leading-relaxed">{t.history.noHistory}</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {history.map((item) => (
                    <div key={item.id} className="flex items-center justify-between p-5 bg-white/[0.02] hover:bg-white/[0.04] rounded-[1.5rem] border border-white/5 transition-colors">
                        <div className="flex items-center gap-5">
                           <div className="w-10 h-10 rounded-full bg-white/5 border border-white/5 flex items-center justify-center text-xs font-bold text-white/50 font-mono">
                              {item.cycles}
                           </div>
                           <div>
                              <div className="text-base font-serif text-white/90">
                                {t.patterns[item.patternId]?.label || 'Breath'}
                              </div>
                              <div className="text-[10px] text-white/30 font-mono mt-0.5 tracking-wide">
                                {formatDate(item.timestamp, userSettings.language, t)}
                              </div>
                           </div>
                        </div>
                        <div className="text-right">
                           <div className="text-sm font-medium text-white/80 font-mono">
                             {Math.floor(item.durationSec / 60)}<span className="text-[9px] text-white/20 ml-0.5">{t.history.min}</span> {item.durationSec % 60}<span className="text-[9px] text-white/20 ml-0.5">{t.history.sec}</span>
                           </div>
                        </div>
                    </div>
                  ))}
                  
                  <button 
                    onClick={() => { triggerHaptic(); clearHistory(); }}
                    className="w-full mt-10 py-4 text-[10px] text-white/20 hover:text-red-400 hover:bg-red-500/5 rounded-2xl transition-all flex items-center justify-center gap-2 font-caps tracking-widest"
                  >
                    <Trash2 size={12} /> {t.history.clear}
                  </button>
                </div>
              )}
           </div>
        </div>
      </div>
  );
}
