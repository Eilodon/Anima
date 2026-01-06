
import { BreathPattern, BreathPhase } from '../types';

export function isPatternValid(pattern: BreathPattern): boolean {
  const t = pattern.timings;
  const sum = t.inhale + t.holdIn + t.exhale + t.holdOut;
  return Number.isFinite(sum) && sum > 0;
}

export function nextPhaseRaw(current: BreathPhase, pattern: BreathPattern): BreathPhase {
  const t = pattern.timings;
  if (current === 'inhale') return t.holdIn > 0 ? 'holdIn' : 'exhale';
  if (current === 'holdIn') return 'exhale';
  if (current === 'exhale') return t.holdOut > 0 ? 'holdOut' : 'inhale';
  return 'inhale'; // default loop
}

/**
 * Skip phases with duration 0s. Safeguard prevents infinite loops.
 */
export function nextPhaseSkipZero(current: BreathPhase, pattern: BreathPattern): BreathPhase {
  let next = nextPhaseRaw(current, pattern);
  let guard = 0;
  
  // Skip phases that have 0 duration
  while (pattern.timings[next] === 0 && guard < 4) {
    next = nextPhaseRaw(next, pattern);
    guard++;
  }
  return next;
}

export function isCycleBoundary(next: BreathPhase): boolean {
  return next === 'inhale';
}
