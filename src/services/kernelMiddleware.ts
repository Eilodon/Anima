
import { KernelEvent } from '../types';
import { RuntimeState, kernel } from './PureZenBKernel';
import { playCue } from './audio';
import { hapticPhase } from './haptics';
import { useSettingsStore } from '../stores/settingsStore';

export type Middleware = (
  event: KernelEvent,
  beforeState: RuntimeState,
  afterState: RuntimeState
) => void;

function phaseToCueType(phase: string): 'inhale' | 'exhale' | 'hold' {
  if (phase === 'holdIn' || phase === 'holdOut') return 'hold';
  return phase as 'inhale' | 'exhale';
}

/**
 * Middleware to handle audio cues on phase transitions
 */
export const audioMiddleware: Middleware = (event, before, after) => {
  if (event.type === 'PHASE_TRANSITION' && after.status === 'RUNNING') {
    const cueType = phaseToCueType(after.phase);
    const settings = useSettingsStore.getState().userSettings;
    
    playCue(
      cueType,
      settings.soundEnabled,
      settings.soundPack,
      after.phaseDuration,
      settings.language
    );
  }
};

/**
 * Middleware to handle haptic feedback
 */
export const hapticMiddleware: Middleware = (event, before, after) => {
  if (event.type === 'PHASE_TRANSITION' && after.status === 'RUNNING') {
    const settings = useSettingsStore.getState().userSettings;
    const cueType = phaseToCueType(after.phase);
    
    hapticPhase(settings.hapticEnabled, settings.hapticStrength, cueType);
  }
};

/**
 * ACTIVE INFERENCE CONTROLLER
 * This closes the biological loop. It adjusts the 'tempoScale' (breathing speed)
 * based on the user's 'rhythm_alignment' (Free Energy proxy).
 * 
 * Logic:
 * - If alignment < 0.3 (Struggling/Stress): Slow down by 10% (scale -> 1.1)
 * - If alignment > 0.8 (Resonance): Speed up slightly towards 1.0 (Natural)
 */
export const biofeedbackMiddleware: Middleware = (event, before, after) => {
    // Only run this logic on belief updates, and throttle it (e.g., only on cycle boundaries or periodic checks)
    // But for responsiveness, we check every update but act slowly using a "Control Deadband".
    
    if (event.type === 'BELIEF_UPDATE' && after.status === 'RUNNING' && after.sessionDuration > 10) {
        
        const alignment = after.belief.rhythm_alignment;
        const currentScale = after.tempoScale;
        let newScale = currentScale;

        // --- CONTROL LAW ---
        
        // CASE 1: User is struggling (Low alignment) -> Co-regulation: Slow down (increase scale)
        if (alignment < 0.35) {
             // Slowly drift slower, max 1.3x slowdown
             newScale = Math.min(1.3, currentScale + 0.002); 
        } 
        // CASE 2: User is locked in (High alignment) -> Return to Baseline (1.0)
        else if (alignment > 0.8) {
             // Slowly drift back to 1.0
             if (currentScale > 1.0) {
                 newScale = Math.max(1.0, currentScale - 0.001);
             }
        }

        // Only dispatch if change is significant to avoid event spam
        if (Math.abs(newScale - currentScale) > 0.01) {
            // We need to dispatch a new event, but we are inside a middleware loop.
            // To avoid infinite loops or blocking, we queue this for the next tick using setTimeout(0)
            setTimeout(() => {
                kernel.dispatch({
                    type: 'ADJUST_TEMPO',
                    scale: newScale,
                    reason: alignment < 0.35 ? 'low_alignment' : 'resonance_restore',
                    timestamp: Date.now()
                });
            }, 0);
        }
    }
};
