
import React, { useEffect, useState, useRef } from 'react';
import { Activity, Terminal, Shield, Cpu, X, Zap, Gauge } from 'lucide-react';
import { kernel, RuntimeState } from '../services/PureZenBKernel';
import { KernelEvent } from '../types';

export function KernelMonitor({ onClose }: { onClose: () => void }) {
  const [state, setState] = useState<RuntimeState>(kernel.getState());
  const [log, setLog] = useState<KernelEvent[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const unsub = kernel.subscribe((s) => {
      setState({ ...s });
      setLog(kernel.getLogBuffer());
    });
    return unsub;
  }, []);

  const fepColor = state.belief.prediction_error > 0.8 ? 'text-red-500' : 
                   state.belief.prediction_error > 0.4 ? 'text-yellow-500' : 'text-green-500';
                   
  const anomalyColor = state.belief.mahalanobis_distance > 3.0 ? 'text-red-500' : 'text-emerald-500';

  return (
    <div className="fixed inset-0 z-[100] bg-black/90 backdrop-blur-xl font-mono text-xs p-4 overflow-hidden flex flex-col animate-in fade-in slide-in-from-bottom-10 duration-300">
      
      {/* Header */}
      <div className="flex justify-between items-center border-b border-white/10 pb-4 mb-4">
        <div className="flex items-center gap-2">
            <Cpu size={16} className="text-emerald-500" />
            <span className="text-emerald-500 font-bold tracking-widest uppercase">ZenB Runtime // v3.5.0 (Active Control)</span>
        </div>
        <button onClick={onClose} className="p-2 hover:bg-white/10 rounded-full"><X size={16} className="text-white/50" /></button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 flex-1 overflow-hidden">
        
        {/* Left Col: State Inspection */}
        <div className="flex flex-col gap-4">
            
            {/* Status Card */}
            <div className="bg-white/5 border border-white/10 rounded-lg p-4">
                <div className="text-white/40 uppercase tracking-widest mb-2 flex items-center gap-2"><Activity size={12}/> Kernel Status</div>
                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <div className="text-[10px] text-white/30">STATUS</div>
                        <div className="text-lg text-white font-bold">{state.status}</div>
                    </div>
                    <div>
                        <div className="text-[10px] text-white/30">TEMPO SCALE</div>
                        <div className={`text-lg font-bold flex items-center gap-2 ${state.tempoScale > 1.05 ? 'text-yellow-400' : 'text-white'}`}>
                            {state.tempoScale.toFixed(2)}x
                            {state.tempoScale > 1.05 && <Gauge size={14} className="animate-pulse"/>}
                        </div>
                    </div>
                    <div>
                        <div className="text-[10px] text-white/30">PHASE</div>
                        <div className="text-white">{state.phase} <span className="text-white/30">({state.phaseElapsed.toFixed(1)}s / {state.phaseDuration.toFixed(1)}s)</span></div>
                    </div>
                </div>
            </div>

            {/* Active Inference Model */}
            <div className="bg-white/5 border border-white/10 rounded-lg p-4 flex-1 overflow-y-auto">
                <div className="text-white/40 uppercase tracking-widest mb-4 flex items-center gap-2"><Shield size={12}/> Adaptive State Estimator v2.0</div>
                
                <div className="space-y-4">
                    <div>
                        <div className="flex justify-between mb-1">
                            <span className="text-white/60">Free Energy (Surprisal)</span>
                            <span className={fepColor}>{state.belief.prediction_error.toFixed(4)}</span>
                        </div>
                        <div className="h-1 w-full bg-white/10 rounded-full overflow-hidden">
                            <div className={`h-full ${state.belief.prediction_error > 0.5 ? 'bg-red-500' : 'bg-emerald-500'}`} style={{ width: `${Math.min(100, state.belief.prediction_error * 100)}%` }} />
                        </div>
                    </div>

                    <div>
                         <div className="flex justify-between mb-1">
                            <span className="text-white/60">Rhythm Alignment</span>
                            <span className="text-blue-400">
                                {(state.belief.rhythm_alignment * 100).toFixed(0)}%
                            </span>
                        </div>
                        <div className="h-1 w-full bg-white/10 rounded-full overflow-hidden">
                             <div className="h-full bg-blue-500" style={{ width: `${state.belief.rhythm_alignment * 100}%` }} />
                        </div>
                    </div>
                    
                    <div>
                         <div className="flex justify-between mb-1">
                            <span className="text-white/60">Anomaly (Mahalanobis)</span>
                            <span className={anomalyColor}>
                                {state.belief.mahalanobis_distance.toFixed(2)} σ
                            </span>
                        </div>
                        {state.belief.mahalanobis_distance > 3.0 && (
                            <div className="text-[9px] text-red-400 mt-1 flex items-center gap-1">
                                <Zap size={8} /> OUTLIER DETECTED - REJECTING
                            </div>
                        )}
                    </div>
                </div>

                <div className="mt-6 border-t border-white/10 pt-4">
                     <div className="text-[10px] text-white/30 mb-2">SENSOR FUSION</div>
                     <div className="grid grid-cols-2 gap-2 text-white/60">
                        <div>HR: {state.lastObservation?.heart_rate ? Math.round(state.lastObservation.heart_rate) : 'N/A'}</div>
                        <div>Conf: {state.lastObservation?.hr_confidence?.toFixed(2) || '0.00'}</div>
                     </div>
                </div>
            </div>
        </div>

        {/* Right Col: Event Log */}
        <div className="bg-black border border-white/10 rounded-lg p-4 flex flex-col font-mono">
             <div className="text-white/40 uppercase tracking-widest mb-2 flex items-center gap-2"><Terminal size={12}/> Event Stream</div>
             <div className="flex-1 overflow-y-auto space-y-1 scrollbar-hide opacity-80" ref={scrollRef}>
                {log.slice().reverse().map((e, i) => (
                    <div key={i} className="flex gap-2 text-[10px] border-b border-white/5 pb-1 mb-1 last:border-0">
                        <span className="text-white/30 shrink-0">{new Date(e.timestamp).toISOString().split('T')[1].slice(0, -1)}</span>
                        <span className={e.type === 'SAFETY_INTERDICTION' ? 'text-red-500 font-bold' : e.type === 'ADJUST_TEMPO' ? 'text-yellow-400' : 'text-emerald-500'}>{e.type}</span>
                        <span className="text-white/60 truncate">{JSON.stringify(e).slice(0, 50)}...</span>
                    </div>
                ))}
             </div>
        </div>

      </div>
    </div>
  );
}
