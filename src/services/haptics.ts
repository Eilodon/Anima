
type Strength = 'light' | 'medium' | 'heavy';

let lastVibeAt = 0;

function strengthMs(s: Strength): number {
  if (s === 'light') return 12;
  if (s === 'medium') return 22;
  return 35;
}

export function hapticTick(enabled: boolean, strength: Strength): void {
  if (!enabled) return;

  const now = performance.now();
  if (now - lastVibeAt < 180) return; // throttle to avoid spamming hardware
  lastVibeAt = now;

  if (typeof navigator !== 'undefined' && 'vibrate' in navigator) {
    navigator.vibrate(strengthMs(strength));
  }
}

export function hapticPhase(
  enabled: boolean,
  strength: Strength,
  kind: 'inhale' | 'exhale' | 'hold'
): void {
  if (!enabled) return;

  if (kind === 'hold') {
    hapticTick(true, 'light');
    return;
  }
  hapticTick(true, strength);
}
