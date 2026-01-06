
import React, { useState } from 'react';
import { Award, Clock, Play, ArrowRight } from 'lucide-react';
import clsx from 'clsx';
import { unlockAudio } from '../../services/audio';
import { TRANSLATIONS } from '../../translations';

type Props = {
  onComplete: () => void;
  t: typeof TRANSLATIONS['en'];
};

export const OnboardingModal = ({ onComplete, t }: Props) => {
  const [step, setStep] = useState(0);

  const steps = [
    { title: t.ui.welcome, text: t.ui.welcomeDesc, icon: <Award size={32} className="text-white/80" /> },
    { title: t.ui.findRhythm, text: t.ui.findRhythmDesc, icon: <Clock size={32} className="text-white/80" /> },
    { title: t.ui.breatheLight, text: t.ui.breatheLightDesc, icon: <Play size={32} className="text-white/80" /> }
  ];

  const handleNext = () => {
    unlockAudio(); 
    if (step < steps.length - 1) setStep(step + 1);
    else onComplete();
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-6 bg-black/80 backdrop-blur-3xl animate-in fade-in duration-1000" role="dialog" aria-modal="true">
      <div className="w-full max-w-sm flex flex-col items-center text-center">
        <div className="mb-10 p-8 rounded-full bg-white/[0.03] border border-white/5 shadow-2xl shadow-white/5 scale-100 transition-transform duration-500 ring-1 ring-white/5">{steps[step].icon}</div>
        <h2 className="text-3xl font-serif font-medium mb-4 tracking-wide text-white">{steps[step].title}</h2>
        <p className="text-white/60 mb-12 leading-relaxed max-w-[280px] font-sans font-light tracking-wide">{steps[step].text}</p>
        
        <div className="flex gap-2.5 mb-12">
          {steps.map((_, i) => (
            <div key={i} className={clsx("h-0.5 rounded-full transition-all duration-700", i === step ? "w-8 bg-white" : "w-1.5 bg-white/10")} />
          ))}
        </div>

        <button 
          onClick={handleNext}
          className="w-full py-4 bg-white text-black font-sans font-medium text-sm tracking-widest uppercase rounded-xl active:scale-95 transition-transform flex items-center justify-center gap-2 hover:bg-white/90"
          aria-label="Next step"
        >
          {step === steps.length - 1 ? t.ui.beginJourney : t.ui.continue} <ArrowRight size={14} />
        </button>
      </div>
    </div>
  );
};
