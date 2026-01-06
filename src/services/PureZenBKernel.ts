
/**
*    ZENB KERNEL V2 - PURE EVENT SOURCING
* ========================================
* ARCHITECTURAL GUARANTEES:
* 1. ✅ Immutable State
* 2. ✅ Pure Reducer
* 3. ✅ Event Log
* 4. ✅ Time Travel
* 5. ✅ Testable
*/
import { BreathPattern, BreathPhase, KernelEvent, BeliefState, BREATHING_PATTERNS, Observation, SafetyProfile } from '../types';
import { AdaptiveStateEstimator } from './AdaptiveStateEstimator';
import { nextPhaseSkipZero, isCycleBoundary } from './phaseMachine';
import { audioMiddleware, hapticMiddleware, biofeedbackMiddleware, Middleware } from './kernelMiddleware';

// --- TYPES ---

export type RuntimeStatus = 'IDLE' | 'RUNNING' | 'PAUSED' | 'HALTED' | 'SAFETY_LOCK';

export interface RuntimeState {
  readonly version: number;
  readonly status: RuntimeStatus;
  readonly bootTimestamp: number;
  readonly lastUpdateTimestamp: number;
  
  // Protocol
  readonly pattern: BreathPattern | null;
  readonly tempoScale: number; // 1.0 = Normal, 1.2 = Slower (User struggling)
  
  // Phase Machine
  readonly phase: BreathPhase;
  readonly phaseStartTime: number;
  readonly phaseDuration: number;
  readonly cycleCount: number;
  readonly sessionStartTime: number;
  
  // Belief State
  readonly belief: BeliefState;
  
  // Safety Registry
  readonly safetyRegistry: Readonly<Record<string, SafetyProfile>>;
  
  // UI Cache (Computed/Ephemeral)
  readonly phaseElapsed: number;
  readonly sessionDuration: number;
  readonly lastObservation: Observation | null;
}

export type SafetyGuard = (event: KernelEvent, state: RuntimeState) => KernelEvent | null;

// --- INITIAL STATE ---

function createInitialState(): RuntimeState {
  return {
    version: 3,
    status: 'IDLE',
    bootTimestamp: Date.now(),
    lastUpdateTimestamp: Date.now(),
    pattern: null,
    tempoScale: 1.0,
    phase: 'inhale',
    phaseStartTime: 0,
    phaseDuration: 0,
    cycleCount: 0,
    sessionStartTime: 0,
    belief: {
        arousal: 0.5,
        attention: 0.5,
        rhythm_alignment: 0.0,
        arousal_variance: 0.2,
        attention_variance: 0.2,
        rhythm_variance: 0.3,
        prediction_error: 0.0,
        innovation: 0.0,
        mahalanobis_distance: 0.0,
        confidence: 0.0
    },
    safetyRegistry: {},
    phaseElapsed: 0,
    sessionDuration: 0,
    lastObservation: null
  };
}

// --- PURE REDUCER ---

function reduce(state: RuntimeState, event: KernelEvent): RuntimeState {
  switch (event.type) {
    case 'BOOT':
      return { ...state, status: 'IDLE', lastUpdateTimestamp: event.timestamp };
      
    case 'LOAD_PROTOCOL': {
      if (state.status === 'SAFETY_LOCK') return state;
      const pattern = BREATHING_PATTERNS[event.patternId];
      if (!pattern) return state;
      return {
        ...state,
        pattern,
        tempoScale: 1.0, // Reset scale on new pattern
        phase: 'inhale',
        phaseStartTime: event.timestamp,
        phaseDuration: pattern.timings.inhale,
        cycleCount: 0,
        sessionStartTime: 0,
        belief: { ...state.belief, rhythm_alignment: 0, prediction_error: 0, confidence: 0 },
        lastUpdateTimestamp: event.timestamp
      };
    }
    
    case 'START_SESSION':
      if (!state.pattern) return state;
      return {
        ...state,
        status: 'RUNNING',
        sessionStartTime: event.timestamp,
        phaseStartTime: event.timestamp,
        lastUpdateTimestamp: event.timestamp
      };
      
    case 'INTERRUPTION':
      if (state.status !== 'RUNNING') return state;
      return { ...state, status: 'PAUSED', lastUpdateTimestamp: event.timestamp };
      
    case 'RESUME':
      if (state.status !== 'PAUSED') return state;
      const pauseDuration = event.timestamp - state.lastUpdateTimestamp;
      return {
        ...state,
        status: 'RUNNING',
        phaseStartTime: state.phaseStartTime + pauseDuration,
        lastUpdateTimestamp: event.timestamp
      };
      
    case 'HALT':
      return { ...state, status: 'HALTED', lastUpdateTimestamp: event.timestamp };
      
    case 'SAFETY_INTERDICTION':
      return { ...state, status: 'SAFETY_LOCK', lastUpdateTimestamp: event.timestamp };
      
    case 'PHASE_TRANSITION': {
      if (!state.pattern) return state;
      // Apply tempoScale to the NEXT phase duration
      // tempoScale > 1 means SLOWER (Duration * Scale)
      const baseDuration = state.pattern.timings[event.to];
      const scaledDuration = baseDuration * state.tempoScale;
      
      return {
        ...state,
        phase: event.to,
        phaseStartTime: event.timestamp,
        phaseDuration: scaledDuration,
        lastUpdateTimestamp: event.timestamp
      };
    }

    case 'ADJUST_TEMPO':
        return {
            ...state,
            tempoScale: event.scale,
            lastUpdateTimestamp: event.timestamp
        };
    
    case 'CYCLE_COMPLETE':
      return { ...state, cycleCount: event.count, lastUpdateTimestamp: event.timestamp };
      
    case 'BELIEF_UPDATE':
      return { ...state, belief: { ...event.belief }, lastUpdateTimestamp: event.timestamp };
      
    case 'TICK':
       return { 
           ...state, 
           lastUpdateTimestamp: event.timestamp, 
           lastObservation: event.observation 
       };

    case 'LOAD_SAFETY_REGISTRY':
        return { ...state, safetyRegistry: { ...event.registry }, lastUpdateTimestamp: event.timestamp };

    default:
      return state;
  }
}

// --- SAFETY GUARD ---

function defaultSafetyGuard(event: KernelEvent, state: RuntimeState): KernelEvent | null {
  // Rule 1: Block START if locked
  if (state.status === 'SAFETY_LOCK' && event.type === 'START_SESSION') {
     return { type: 'SAFETY_INTERDICTION', riskLevel: 1.0, action: 'REJECT_START', timestamp: Date.now() };
  }
  
  // Rule 2: Emergency stop on critical prediction error
  if (event.type === 'BELIEF_UPDATE' && event.belief.prediction_error > 0.95 && state.sessionDuration > 15) {
      return { type: 'SAFETY_INTERDICTION', riskLevel: 0.95, action: 'EMERGENCY_HALT', timestamp: Date.now() };
  }
  
  // Rule 3: Check pattern safety registry
  if (event.type === 'LOAD_PROTOCOL') {
      const profile = state.safetyRegistry[event.patternId];
      if (profile && profile.safety_lock_until > Date.now()) {
          return { type: 'SAFETY_INTERDICTION', riskLevel: 0.8, action: 'PATTERN_LOCKED', timestamp: Date.now() };
      }
  }
  return event;
}

// --- KERNEL CLASS ---

export class PureZenBKernel {
  private state: RuntimeState;
  private estimator: AdaptiveStateEstimator;
  private eventLog: KernelEvent[] = [];
  private readonly MAX_LOG_SIZE = 1000;
  private subscribers = new Set<(state: RuntimeState) => void>();
  private middlewares: Middleware[] = [];
  private safetyGuard: SafetyGuard = defaultSafetyGuard;
  
  constructor() {
    this.state = createInitialState();
    
    this.estimator = new AdaptiveStateEstimator({
        alpha: 1.5e-3,
        adaptive_r: true,
        q_base: 0.015,
        r_adaptation_rate: 0.2
    });

    this.use(audioMiddleware);
    this.use(hapticMiddleware);
    this.use(biofeedbackMiddleware); // Active Inference Controller
    
    this.dispatch({ type: 'BOOT', timestamp: Date.now() });
  }
  
  // --- PUBLIC API ---
  
  public dispatch(event: KernelEvent): void {
    const beforeState = this.state;
    
    // 1. Safety Guard
    const guardedEvent = this.safetyGuard(event, beforeState);
    if (!guardedEvent) return;
    
    // 2. Append Log
    this.eventLog.push(guardedEvent);
    if (this.eventLog.length > this.MAX_LOG_SIZE) this.eventLog.shift();
    
    // 3. Reduce
    const reducedState = reduce(beforeState, guardedEvent);
    
    // 4. Compute Derived
    const enrichedState = this.computeDerivedFields(reducedState);
    
    // 5. Update
    this.state = enrichedState;
    
    // 6. Middleware
    this.middlewares.forEach(mw => mw(guardedEvent, beforeState, enrichedState));
    
    // 7. Notify
    this.notify();
  }
  
  public tick(dt: number, observation: Observation): void {
      const now = Date.now();
      
      this.estimator.setProtocol(this.state.pattern); 
      const newBelief = this.estimator.update(observation, dt);
      
      this.dispatch({ type: 'BELIEF_UPDATE', belief: newBelief, timestamp: now });
      
      // Phase Transition Logic
      if (this.state.status === 'RUNNING' && this.state.pattern) {
          const elapsed = (now - this.state.phaseStartTime) / 1000;
          
          if (elapsed >= this.state.phaseDuration) {
              const nextPhase = nextPhaseSkipZero(this.state.phase, this.state.pattern);
              this.dispatch({ 
                  type: 'PHASE_TRANSITION', 
                  from: this.state.phase, 
                  to: nextPhase, 
                  timestamp: now 
              });
              
              if (isCycleBoundary(nextPhase)) {
                  this.dispatch({ 
                      type: 'CYCLE_COMPLETE', 
                      count: this.state.cycleCount + 1, 
                      timestamp: now 
                  });
              }
          }
      }
      
      this.dispatch({ type: 'TICK', dt, observation, timestamp: now });
  }
  
  public getState(): RuntimeState { return this.state; }
  
  public subscribe(callback: (state: RuntimeState) => void): () => void {
      this.subscribers.add(callback);
      callback(this.state);
      return () => this.subscribers.delete(callback);
  }
  
  public use(middleware: Middleware): void { this.middlewares.push(middleware); }
  
  public loadSafetyRegistry(registry: Record<string, SafetyProfile>): void {
      this.dispatch({ type: 'LOAD_SAFETY_REGISTRY', registry, timestamp: Date.now() });
  }
  
  public getLogBuffer(): KernelEvent[] { return [...this.eventLog]; }
  
  // --- INTERNAL ---
  
  private computeDerivedFields(state: RuntimeState): RuntimeState {
      const now = Date.now();
      return {
          ...state,
          phaseElapsed: state.status === 'RUNNING' ? Math.max(0, (now - state.phaseStartTime) / 1000) : 0,
          sessionDuration: state.sessionStartTime > 0 ? Math.max(0, (now - state.sessionStartTime) / 1000) : 0
      };
  }
  
  private notify(): void {
      this.subscribers.forEach(cb => {
          try { cb(this.state); } catch(e) { console.error('Subscriber error', e); }
      });
  }
}

export const kernel = new PureZenBKernel();
