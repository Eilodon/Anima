
import React from 'react';
import { Flame, RotateCcw } from 'lucide-react';
import clsx from 'clsx';
import { SessionStats } from '../../types';
import { TRANSLATIONS } from '../../translations';

type Props = {
  stats: SessionStats;
  onClose: () => void;
  t: typeof TRANSLATIONS['en'];
  streak: number;
  language: 'en' | 'vi';
};

export const SummaryModal = ({ stats, onClose, t, streak, language }: Props) => {
  const localizedLabel = TRANSLATIONS[language].patterns[stats.patternId]?.label || stats.patternId;

  return (
    <div className="fixed inset-0 z-[100] flex flex-col items-center justify-center p-8 bg-black/80 backdrop-blur-3xl animate-in fade-in duration-1000" role="dialog" aria-modal="true">
      
      <div className="flex-1 flex flex-col items-center justify-center w-full max-w-md">
        <div className="mb-8 text-white/30 font-caps tracking-widest text-[10px]">{t.ui.sessionComplete}</div>
        <h2 className="text-5xl font-serif text-white text-center mb-4 tracking-tight leading-tight">{localizedLabel}</h2>
        <p className="text-white/50 text-sm mb-14 text-center font-sans font-light max-w-xs leading-relaxed">{t.ui.mindClear}</p>
        
        <div className="grid grid-cols-2 gap-4 w-full mb-14">
          <div className="flex flex-col items-center p-8 rounded-[2rem] bg-white/[0.02] border border-white/5 backdrop-blur-md">
            <div className="text-3xl font-light mb-1 font-sans tracking-tight">
              {Math.floor(stats.durationSec / 60)}<span className="text-lg opacity-40">:</span>{(stats.durationSec % 60).toString().padStart(2, '0')}
            </div>
            <div className="text-white/30 font-caps text-[10px] mt-2">{t.ui.timeBreathed}</div>
          </div>
          <div className="flex flex-col items-center p-8 rounded-[2rem] bg-white/[0.02] border border-white/5 backdrop-blur-md relative overflow-hidden">
             {/* Streak Shine Effect */}
             {streak > 1 && <div className="absolute inset-0 bg-orange-500/5 animate-pulse" />}
            <div className={clsx("text-3xl font-light mb-1 font-sans flex items-center gap-2", streak > 1 ? "text-orange-200" : "text-white")}>
                {streak} <Flame size={18} className={streak > 1 ? "fill-orange-500 text-orange-500" : "text-white/20"} />
            </div>
            <div className="text-white/30 font-caps text-[10px] mt-2">{t.ui.streak}</div>
          </div>
        </div>

        <button 
          onClick={onClose} 
          className="group relative w-full py-4 bg-white text-black font-medium rounded-xl overflow-hidden active:scale-95 transition-all"
          aria-label="Finish session"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/50 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700" />
          <span className="relative flex items-center justify-center gap-2 font-sans text-sm tracking-widest uppercase"><RotateCcw size={14} /> {t.ui.finish}</span>
        </button>
      </div>
    </div>
  );
};
