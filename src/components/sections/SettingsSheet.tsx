
import React from 'react';
import { X, Volume2, VolumeX, Smartphone, SmartphoneNfc, Music, Check, Terminal } from 'lucide-react';
import clsx from 'clsx';
import { useSettingsStore } from '../../stores/settingsStore';
import { useUIStore } from '../../stores/uiStore';
import { TRANSLATIONS } from '../../translations';
import { SoundPack } from '../../types';
import { hapticTick } from '../../services/haptics';

export function SettingsSheet() {
  const isSettingsOpen = useUIStore(s => s.isSettingsOpen);
  const setSettingsOpen = useUIStore(s => s.setSettingsOpen);
  
  const userSettings = useSettingsStore(s => s.userSettings);
  const setLanguage = useSettingsStore(s => s.setLanguage);
  const toggleSound = useSettingsStore(s => s.toggleSound);
  const toggleHaptic = useSettingsStore(s => s.toggleHaptic);
  const setSoundPack = useSettingsStore(s => s.setSoundPack);
  const setQuality = useSettingsStore(s => s.setQuality);
  const setReduceMotion = useSettingsStore(s => s.setReduceMotion);
  const toggleTimer = useSettingsStore(s => s.toggleTimer);
  const toggleCameraVitals = useSettingsStore(s => s.toggleCameraVitals);
  const toggleKernelMonitor = useSettingsStore(s => s.toggleKernelMonitor);

  const t = TRANSLATIONS[userSettings.language] || TRANSLATIONS.en;
  
  // Updated Sound Pack List
  const soundPacks: SoundPack[] = ['musical', 'bells', 'breath', 'real-zen', 'voice-en', 'voice-vi', 'voice-12'];

  const triggerHaptic = () => {
    if (userSettings.hapticEnabled) hapticTick(true, 'light');
  };

  return (
    <div 
        className={clsx(
            "fixed inset-0 z-50 transition-colors duration-700 pointer-events-none",
            isSettingsOpen ? "bg-black/60 backdrop-blur-sm pointer-events-auto" : "bg-transparent"
        )}
        onClick={() => setSettingsOpen(false)}
        role="dialog"
        aria-label="Settings Sheet"
      >
        <div 
          onClick={(e) => e.stopPropagation()}
          className={clsx(
              "absolute inset-x-0 bottom-0 bg-[#050505] border-t border-white/10 rounded-t-[3rem] p-8 pb-[calc(2.5rem+env(safe-area-inset-bottom))] transition-transform duration-500 cubic-bezier(0.19, 1, 0.22, 1) shadow-2xl",
              isSettingsOpen ? "translate-y-0" : "translate-y-full"
          )}
        >
           <div className="flex justify-between items-center mb-8">
              <h3 className="text-2xl font-serif text-white tracking-wide">{t.settings.header}</h3>
              <button onClick={() => setSettingsOpen(false)} aria-label="Close Settings" className="p-2.5 bg-white/5 hover:bg-white/10 rounded-full transition-colors border border-white/5"><X size={20} className="text-white/70"/></button>
           </div>
           
           <div className="space-y-10 max-h-[70vh] overflow-y-auto scrollbar-hide pb-12">
              <section>
                  <div className="text-white/30 font-caps text-[9px] tracking-[0.2em] mb-4 flex items-center gap-2 pl-1">
                    {t.settings.language}
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                      <button 
                        onClick={() => { triggerHaptic(); setLanguage('en'); }} 
                        className={clsx("p-4 rounded-2xl flex items-center justify-center gap-3 transition-all border", userSettings.language === 'en' ? "bg-white/10 border-white/20 text-white" : "bg-transparent border-white/5 text-white/30")}
                      >
                         <span className="text-xl">🇬🇧</span>
                         <span className="text-xs font-medium tracking-wide">English</span>
                      </button>
                      <button 
                        onClick={() => { triggerHaptic(); setLanguage('vi'); }} 
                        className={clsx("p-4 rounded-2xl flex items-center justify-center gap-3 transition-all border", userSettings.language === 'vi' ? "bg-white/10 border-white/20 text-white" : "bg-transparent border-white/5 text-white/30")}
                      >
                         <span className="text-xl">🇻🇳</span>
                         <span className="text-xs font-medium tracking-wide">Tiếng Việt</span>
                      </button>
                  </div>
              </section>

              <section>
                  <div className="text-white/30 font-caps text-[9px] tracking-[0.2em] mb-4 flex items-center gap-2 pl-1">
                    {t.settings.immersion}
                  </div>
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-3">
                        <button onClick={() => { triggerHaptic(); toggleSound(); }} className={clsx("p-5 rounded-[1.5rem] flex flex-col items-center gap-3 transition-all border", userSettings.soundEnabled ? "bg-white/10 border-white/20 text-white" : "bg-transparent border-white/5 text-white/30")}>
                            {userSettings.soundEnabled ? <Volume2 size={24} strokeWidth={1} /> : <VolumeX size={24} strokeWidth={1} />}
                            <span className="text-xs font-medium tracking-wide">{t.settings.sounds}</span>
                        </button>
                        <button onClick={() => { triggerHaptic(); toggleHaptic(); }} className={clsx("p-5 rounded-[1.5rem] flex flex-col items-center gap-3 transition-all border", userSettings.hapticEnabled ? "bg-white/10 border-white/20 text-white" : "bg-transparent border-white/5 text-white/30")}>
                            {userSettings.hapticEnabled ? <Smartphone size={24} strokeWidth={1} /> : <SmartphoneNfc size={24} strokeWidth={1} />}
                            <span className="text-xs font-medium tracking-wide">{t.settings.haptics}</span>
                        </button>
                    </div>

                    {userSettings.soundEnabled && (
                      <div className="bg-white/[0.02] rounded-[1.5rem] border border-white/5 p-5">
                        <div className="text-[9px] text-white/40 uppercase font-bold mb-4 tracking-[0.2em] flex items-center gap-2">
                           <Music size={12} /> {t.settings.soundPack}
                        </div>
                        <div className="grid grid-cols-1 gap-1">
                          {soundPacks.map(pack => (
                            <button
                              key={pack}
                              onClick={() => { triggerHaptic(); setSoundPack(pack); }}
                              className={clsx(
                                "w-full text-left px-4 py-3.5 rounded-xl text-xs font-medium tracking-wide transition-all flex items-center justify-between group",
                                userSettings.soundPack === pack 
                                  ? "bg-white/10 text-white shadow-sm" 
                                  : "text-white/40 hover:bg-white/5 hover:text-white"
                              )}
                            >
                              {t.settings.soundPacks[pack]}
                              {userSettings.soundPack === pack && <Check size={14} />}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
              </section>

              <section>
                  <div className="text-white/30 font-caps text-[9px] tracking-[0.2em] mb-4 flex items-center gap-2 pl-1">
                    {t.settings.visuals}
                  </div>
                  <div className="space-y-3">
                      <div className="flex items-center justify-between p-5 bg-white/[0.02] rounded-[1.5rem] border border-white/5">
                          <span className="text-sm font-light text-white/80">{t.settings.graphics}</span>
                          <select 
                            value={userSettings.quality} 
                            onChange={(e) => setQuality(e.target.value as any)}
                            className="bg-black/40 text-white text-xs py-2 px-4 rounded-lg border border-white/10 outline-none focus:border-white/30 appearance-none font-mono"
                          >
                              <option value="auto">{t.settings.quality.auto}</option>
                              <option value="low">{t.settings.quality.low}</option>
                              <option value="medium">{t.settings.quality.medium}</option>
                              <option value="high">{t.settings.quality.high}</option>
                          </select>
                      </div>
                      <label className="flex items-center justify-between p-5 bg-white/[0.02] rounded-[1.5rem] border border-white/5 cursor-pointer hover:bg-white/[0.04] transition-colors">
                          <span className="text-sm font-light text-white/80">{t.settings.reduceMotion}</span>
                          <div className={clsx("w-11 h-6 rounded-full relative transition-colors border border-white/10", userSettings.reduceMotion ? "bg-white" : "bg-white/10")}>
                              <input type="checkbox" checked={userSettings.reduceMotion} onChange={(e) => { triggerHaptic(); setReduceMotion(e.target.checked); }} className="sr-only"/>
                              <div className={clsx("absolute top-1 left-1 w-4 h-4 rounded-full shadow-sm transition-transform", userSettings.reduceMotion ? "bg-black translate-x-5" : "bg-white/50 translate-x-0")} />
                          </div>
                      </label>
                      <label className="flex items-center justify-between p-5 bg-white/[0.02] rounded-[1.5rem] border border-white/5 cursor-pointer hover:bg-white/[0.04] transition-colors">
                          <span className="text-sm font-light text-white/80">{t.settings.showTimer}</span>
                          <div className={clsx("w-11 h-6 rounded-full relative transition-colors border border-white/10", userSettings.showTimer ? "bg-white" : "bg-white/10")}>
                              <input type="checkbox" checked={userSettings.showTimer} onChange={(e) => { triggerHaptic(); toggleTimer(); }} className="sr-only"/>
                              <div className={clsx("absolute top-1 left-1 w-4 h-4 rounded-full shadow-sm transition-transform", userSettings.showTimer ? "bg-black translate-x-5" : "bg-white/50 translate-x-0")} />
                          </div>
                      </label>
                      <label className="flex items-center justify-between p-5 bg-white/[0.02] rounded-[1.5rem] border border-white/5 cursor-pointer hover:bg-white/[0.04] transition-colors">
                        <div>
                          <div className="text-sm font-light text-white/80">Bio-Sensors (Camera)</div>
                          <div className="text-xs text-white/40 mt-1">Experimental</div>
                        </div>
                        <div className={clsx("w-11 h-6 rounded-full relative transition-colors border border-white/10", userSettings.cameraVitalsEnabled ? "bg-white" : "bg-white/10")}>
                          <input 
                            type="checkbox" 
                            checked={userSettings.cameraVitalsEnabled} 
                            onChange={() => { triggerHaptic(); toggleCameraVitals(); }} 
                            className="sr-only"
                          />
                          <div className={clsx("absolute top-1 left-1 w-4 h-4 rounded-full shadow-sm transition-transform", userSettings.cameraVitalsEnabled ? "bg-black translate-x-5" : "bg-white/50 translate-x-0")} />
                        </div>
                      </label>
                  </div>
              </section>

              <section>
                   <div className="text-white/30 font-caps text-[9px] tracking-[0.2em] mb-4 flex items-center gap-2 pl-1">
                    Developers
                  </div>
                  <button onClick={() => { triggerHaptic(); toggleKernelMonitor(); setSettingsOpen(false); }} className="w-full p-4 bg-white/5 rounded-xl flex items-center gap-3 text-white/60 hover:text-white hover:bg-white/10 transition-colors">
                    <Terminal size={16} />
                    <span className="text-xs font-mono">Toggle Kernel Monitor</span>
                  </button>
              </section>
              
              <div className="pt-8 text-center">
                 <div className="text-[10px] text-white/20 font-mono">ZenB Kernel v3.4.1</div>
              </div>
           </div>
        </div>
      </div>
  );
}
