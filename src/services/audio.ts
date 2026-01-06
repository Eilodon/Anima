
import * as Tone from 'tone';
import { SoundPack, CueType, Language } from '../types';
import { TRANSLATIONS } from '../translations';

// ---------------------------------------------------------------------------
// ZenB Audio Driver (robust, mobile-friendly)
// ---------------------------------------------------------------------------
// Goals:
// - No double-trigger crackle (cue debounce guard)
// - No click/pop (always ramp gains; never hard-stop sources)
// - Headroom-first mixing (limiter as safety-net, not a compressor substitute)
// - Path to "real-zen" sample-based pack (AAA realism)
// ---------------------------------------------------------------------------

let isUnlocked = false;
let isSettingUp = false;

// Master graph
let masterGain: Tone.Gain | null = null;
let compressor: Tone.Compressor | null = null;
let limiter: Tone.Limiter | null = null;
let reverb: Tone.Reverb | null = null;

// Instruments (synth fallback)
let bowl: Tone.FMSynth | null = null;
let bell: Tone.MetalSynth | null = null;
let pad: Tone.PolySynth | null = null;

// Breath noise (organic-ish)
let noise: Tone.Noise | null = null;
let noiseFilter: Tone.Filter | null = null;
let noiseGain: Tone.Gain | null = null;

// Sample-based pack: real-zen
type SampleBank = {
  inhale: Tone.Player[];
  exhale: Tone.Player[];
  hold: Tone.Player[];
  finish: Tone.Player[];
  ambience?: Tone.Player;
};
let realZenBank: SampleBank | null = null;

// Cue guard (prevents accidental duplicates)
let lastCueKey: string | null = null;
let lastCueAt = 0;

function clamp(n: number, a: number, b: number) {
  return Math.max(a, Math.min(b, n));
}

function rand(min: number, max: number) {
  return min + Math.random() * (max - min);
}

function safeDispose(node: any) {
  try { node?.dispose?.(); } catch {}
}

function buildMasterGraph() {
  if (masterGain) return;
  masterGain = new Tone.Gain(0.75);
  compressor = new Tone.Compressor({
    threshold: -24,
    ratio: 2,
    attack: 0.01,
    release: 0.25,
    knee: 6
  });
  limiter = new Tone.Limiter(-1);
  reverb = new Tone.Reverb({ decay: 1.6, wet: 0.18 });

  reverb.connect(compressor);
  compressor.connect(limiter);
  limiter.connect(masterGain);
  masterGain.toDestination();
}

async function buildInstruments() {
  if (!masterGain || !reverb) buildMasterGraph();
  if (!reverb) return;

  try { await reverb.generate(); } catch {}

  bowl = new Tone.FMSynth({
    harmonicity: 1.1,
    modulationIndex: 12,
    oscillator: { type: 'sine' },
    envelope: { attack: 0.005, decay: 0.2, sustain: 0.0, release: 2.2 },
    modulation: { type: 'sine' },
    modulationEnvelope: { attack: 0.01, decay: 0.2, sustain: 0.0, release: 1.8 }
  });
  bowl.volume.value = -14;
  bowl.connect(reverb);

  bell = new Tone.MetalSynth({
    frequency: 220,
    envelope: { attack: 0.001, decay: 1.2, release: 0.4 },
    harmonicity: 5.1,
    modulationIndex: 32,
    resonance: 6500,
    octaves: 1.5
  });
  bell.volume.value = -18;
  bell.connect(reverb);

  pad = new Tone.PolySynth(Tone.Synth, {
    oscillator: { type: 'triangle' },
    envelope: { attack: 0.03, decay: 0.1, sustain: 0.25, release: 1.2 }
  });
  pad.volume.value = -26;
  pad.connect(reverb);

  noise = new Tone.Noise('brown');
  noiseFilter = new Tone.Filter({ type: 'bandpass', frequency: 900, Q: 0.8 });
  noiseGain = new Tone.Gain(0.0);

  noise.connect(noiseFilter);
  noiseFilter.connect(noiseGain);
  noiseGain.connect(reverb);

  try { noise.start(); } catch {}
}

function shouldDebounce(cue: CueType, pack: SoundPack, duration: number) {
  const now = performance.now();
  const bucket = Math.floor(now / 100);
  const key = `${pack}:${cue}:${bucket}:${Math.round(duration * 10)}`;
  if (lastCueKey === key && (now - lastCueAt) < 140) return true;
  lastCueKey = key;
  lastCueAt = now;
  return false;
}

function speak(text: string, lang: Language) {
  if (typeof window === 'undefined' || !window.speechSynthesis) return;
  try {
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = lang === 'vi' ? 'vi-VN' : 'en-US';
    utter.rate = lang === 'vi' ? 0.92 : 0.95;
    utter.pitch = 1.0;
    utter.volume = 0.9;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utter);
  } catch {}
}

async function ensureReady() {
  if (isSettingUp) return;
  if (isUnlocked && masterGain && bowl && pad) return;
  isSettingUp = true;
  try {
    buildMasterGraph();
    await buildInstruments();
    isUnlocked = true;
  } finally {
    isSettingUp = false;
  }
}

// -------- REAL-ZEN SAMPLE PACK (AAA realism) --------
const REAL_ZEN_URLS = {
  inhale: [
    '/audio/real-zen/inhale_01.wav',
    '/audio/real-zen/inhale_02.wav',
    '/audio/real-zen/inhale_03.wav'
  ],
  exhale: [
    '/audio/real-zen/exhale_01.wav',
    '/audio/real-zen/exhale_02.wav',
    '/audio/real-zen/exhale_03.wav'
  ],
  hold: [
    '/audio/real-zen/hold_01.wav',
    '/audio/real-zen/hold_02.wav'
  ],
  finish: [
    '/audio/real-zen/finish_01.wav'
  ],
  ambience: '/audio/real-zen/ambience_loop.wav'
} as const;

async function ensureRealZenBank() {
  if (realZenBank) return realZenBank;
  await ensureReady();
  if (!reverb) return null;

  const mkPlayers = (urls: readonly string[], volDb: number) => 
    urls.map((url) => {
      const p = new Tone.Player({
        url,
        autostart: false,
        fadeIn: 0.01,
        fadeOut: 0.02
      });
      p.volume.value = volDb;
      p.connect(reverb!);
      return p;
    });

  try {
    const bank: SampleBank = {
      inhale: mkPlayers(REAL_ZEN_URLS.inhale, -14),
      exhale: mkPlayers(REAL_ZEN_URLS.exhale, -14),
      hold: mkPlayers(REAL_ZEN_URLS.hold, -16),
      finish: mkPlayers(REAL_ZEN_URLS.finish, -12),
    };

    if (REAL_ZEN_URLS.ambience) {
      const amb = new Tone.Player({
        url: REAL_ZEN_URLS.ambience,
        loop: true,
        autostart: false,
        fadeIn: 0.8,
        fadeOut: 1.0
      });
      amb.volume.value = -30;
      amb.connect(reverb);
      bank.ambience = amb;
    }

    realZenBank = bank;
    return realZenBank;
  } catch {
    realZenBank = null;
    return null;
  }
}

function pick<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

function playBreathNoise(shape: 'inhale' | 'exhale', durationSec: number) {
  if (!noiseGain || !noiseFilter) return;
  const now = Tone.now();
  const dur = clamp(durationSec, 0.2, 12);
  const targetFreq = shape === 'inhale' ? 1100 : 750;

  noiseFilter.frequency.setTargetAtTime(targetFreq, now, 0.05);
  
  noiseGain.gain.cancelScheduledValues(now);
  noiseGain.gain.setValueAtTime(noiseGain.gain.value, now);
  noiseGain.gain.linearRampToValueAtTime(0.12, now + 0.03);
  noiseGain.gain.linearRampToValueAtTime(0.08, now + dur * 0.7);
  noiseGain.gain.linearRampToValueAtTime(0.0, now + dur);
}

function playSynthCue(cue: CueType, durationSec: number) {
  if (!bowl || !bell || !pad) return;
  
  const now = Tone.now();
  const dur = clamp(durationSec, 0.2, 12);
  const cents = rand(-8, 8);

  if (cue === 'inhale') {
    pad.triggerAttackRelease(['C4', 'E4'], Math.min(0.6, dur * 0.15), now, 0.2);
    bowl.detune.value = cents;
    bowl.triggerAttackRelease('C4', Math.min(0.8, dur * 0.2), now);
  } else if (cue === 'exhale') {
    pad.triggerAttackRelease(['A3', 'C4'], Math.min(0.6, dur * 0.15), now, 0.18);
    bowl.detune.value = cents;
    bowl.triggerAttackRelease('A3', Math.min(0.8, dur * 0.2), now);
  } else if (cue === 'hold') {
    bell.frequency.value = 220 + rand(-10, 10);
    bell.triggerAttackRelease(0.02, now);
  } else if (cue === 'finish') {
    bell.frequency.value = 330;
    bell.triggerAttackRelease(0.08, now);
    bowl.detune.value = 0;
    bowl.triggerAttackRelease('C4', 1.0, now + 0.03);
  }
}

export const unlockAudio = async () => {
  if (isUnlocked || isSettingUp) return;
  try { await Tone.start(); } catch {}
  await ensureReady();
};

export async function playCue(
  cue: CueType,
  enabled: boolean,
  pack: SoundPack,
  duration: number,
  lang: Language = 'en'
): Promise<void> {
  if (!enabled) return;
  if (isSettingUp) return;
  if (shouldDebounce(cue, pack, duration)) return;

  try {
    if (Tone.context.state !== 'running') await Tone.context.resume();
  } catch {}

  if (pack.startsWith('voice')) {
    const t = TRANSLATIONS[lang] || TRANSLATIONS.en;
    let text = '';
    if (pack === 'voice-12') {
      if (cue === 'inhale') text = lang === 'vi' ? 'Một' : 'One';
      if (cue === 'exhale') text = lang === 'vi' ? 'Hai' : 'Two';
      if (cue === 'hold') text = lang === 'vi' ? 'Giữ' : 'Hold';
    } else {
      if (cue === 'inhale') text = t.phases.inhale;
      if (cue === 'exhale') text = t.phases.exhale;
      if (cue === 'hold') text = t.phases.hold;
      if (cue === 'finish') text = t.ui.finish;
    }
    if (text) speak(text.toLowerCase(), lang);
    return;
  }

  await ensureReady();
  const dur = clamp(duration, 0.2, 12);

  if (pack === 'real-zen') {
    const bank = await ensureRealZenBank();
    if (bank) {
      if (bank.ambience) {
        try {
          if (cue === 'inhale' && bank.ambience.state !== 'started') bank.ambience.start();
          if (cue === 'finish' && bank.ambience.state === 'started') bank.ambience.stop('+0.1');
        } catch {}
      }
      
      const pool = 
        cue === 'inhale' ? bank.inhale :
        cue === 'exhale' ? bank.exhale :
        cue === 'hold' ? bank.hold :
        bank.finish;
      
      if (pool.length) {
        const p = pick(pool);
        p.volume.value += rand(-1.2, 0.8);
        try { p.start(); } catch {}
      }
      
      // Layer subtle noise
      if (cue === 'inhale' || cue === 'exhale') playBreathNoise(cue, dur);
      return;
    }
    // Fallback to synth
    playSynthCue(cue, dur);
    if (cue === 'inhale' || cue === 'exhale') playBreathNoise(cue, dur);
    return;
  }

  // Legacy/Synth Packs
  playSynthCue(cue, dur);
  if (pack === 'breath' && (cue === 'inhale' || cue === 'exhale')) {
    playBreathNoise(cue, dur);
  }
  
  if (pack === 'bells' && bell) {
    const now = Tone.now();
    bell.frequency.value = cue === 'inhale' ? 260 : cue === 'exhale' ? 210 : 320;
    bell.triggerAttackRelease(cue === 'hold' ? 0.03 : 0.06, now);
  }
}

export function cleanupAudio() {
  try { realZenBank?.ambience?.stop(); } catch {}
  
  if (realZenBank) {
    for (const p of [...realZenBank.inhale, ...realZenBank.exhale, ...realZenBank.hold, ...realZenBank.finish]) {
      safeDispose(p);
    }
    safeDispose(realZenBank.ambience);
  }
  realZenBank = null;

  safeDispose(bowl); bowl = null;
  safeDispose(bell); bell = null;
  safeDispose(pad); pad = null;
  
  try { noiseGain?.gain.setValueAtTime(0, Tone.now()); } catch {}
  safeDispose(noise); noise = null;
  safeDispose(noiseFilter); noiseFilter = null;
  safeDispose(noiseGain); noiseGain = null;
  
  safeDispose(reverb); reverb = null;
  safeDispose(compressor); compressor = null;
  safeDispose(limiter); limiter = null;
  safeDispose(masterGain); masterGain = null;

  if (typeof window !== 'undefined' && window.speechSynthesis) {
    try { window.speechSynthesis.cancel(); } catch {}
  }
  
  isUnlocked = false;
  isSettingUp = false;
}
