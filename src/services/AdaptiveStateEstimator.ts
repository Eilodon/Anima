
import { Observation, BreathPattern, BeliefState, BREATHING_PATTERNS } from '../types';

/**
 * ADAPTIVE STATE ESTIMATOR v2.0
 * ============================
 * 
 * An Adaptive Linear Kalman Filter (LKF) optimized for physiological state estimation.
 * 
 * UPGRADES v2.0:
 * - Configurable Process/Measurement noise
 * - Adaptive Measurement Noise (R) based on sensor confidence
 * - Outlier Detection (Mahalanobis Distance)
 * - Innovation tracking
 */

export interface EstimatorConfig {
    alpha?: number;             // Process Noise Scale (System volatility)
    adaptive_r?: boolean;       // Enable Adaptive Measurement Noise
    r_adaptation_rate?: number; // How fast we mistrust noisy sensors
    q_base?: number;            // Base Process Noise
    r_base?: number;            // Base Measurement Noise
}

interface TargetState {
  arousal: number;
  attention: number;
  rhythm_alignment: number;
}

const PROTOCOL_TARGETS: Record<string, TargetState> = {
    // Parasympathetic dominance (sleep, anxiety reduction)
    'parasympathetic': {
        arousal: 0.2, 
        attention: 0.5, 
        rhythm_alignment: 0.8
    },
    // Balanced autonomic state (HRV coherence)
    'balanced': {
        arousal: 0.4,
        attention: 0.7,
        rhythm_alignment: 0.9
    },
    // Sympathetic activation (energizing)
    'sympathetic': {
        arousal: 0.7,
        attention: 0.8,
        rhythm_alignment: 0.6
    },
    'default': {
        arousal: 0.5,
        attention: 0.6,
        rhythm_alignment: 0.7
    }
};

const PATTERN_TO_TARGET: Record<string, keyof typeof PROTOCOL_TARGETS> = {
    '4-7-8': 'parasympathetic',
    'deep-relax': 'parasympathetic',
    '7-11': 'parasympathetic',
    'coherence': 'balanced',
    'calm': 'balanced',
    'box': 'balanced',
    'triangle': 'balanced',
    'tactical': 'balanced',
    'awake': 'sympathetic',
    'wim-hof': 'sympathetic',
    'buteyko': 'parasympathetic',
};

export class AdaptiveStateEstimator {
    private belief: BeliefState;
    private target: TargetState;
    private config: Required<EstimatorConfig>;

    // Time constants for state evolution (seconds)
    private readonly TAU_AROUSAL = 15.0; 
    private readonly TAU_ATTENTION = 5.0; 
    private readonly TAU_RHYTHM = 10.0; 

    constructor(config: EstimatorConfig = {}) {
        this.config = {
            alpha: config.alpha ?? 1e-3,
            adaptive_r: config.adaptive_r ?? true,
            r_adaptation_rate: config.r_adaptation_rate ?? 0.2,
            q_base: config.q_base ?? 0.01,
            r_base: config.r_base ?? 0.15
        };

        this.belief = {
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
        };
        this.target = PROTOCOL_TARGETS.default;
    }

    public setProtocol(pattern: BreathPattern | null): void {
        if (!pattern) {
            this.target = PROTOCOL_TARGETS.default;
            return;
        }
        const targetKey = PATTERN_TO_TARGET[pattern.id] || 'default';
        this.target = PROTOCOL_TARGETS[targetKey];
    }

    public update(obs: Observation, dt: number): BeliefState {
        // Step 1: PREDICTION (where we expect the state to be)
        const predicted = this.predict(dt);

        // Step 2: CORRECTION (incorporate new observation)
        const corrected = this.correct(predicted, obs, dt);

        // Step 3: DIAGNOSTICS
        corrected.prediction_error = this.computePredictionError(corrected);
        corrected.confidence = this.computeConfidence(corrected, obs);

        this.belief = corrected;
        return { ...this.belief };
    }

    private predict(dt: number): BeliefState {
        const { arousal, attention, rhythm_alignment } = this.belief;
        const { arousal_variance, attention_variance, rhythm_variance } = this.belief;

        // Compute decay factors
        const alpha_arousal = 1 - Math.exp(-dt / this.TAU_AROUSAL);
        const alpha_attention = 1 - Math.exp(-dt / this.TAU_ATTENTION);
        const alpha_rhythm = 1 - Math.exp(-dt / this.TAU_RHYTHM);

        // Predict state (settle toward target)
        const predicted_arousal = arousal + alpha_arousal * (this.target.arousal - arousal);
        const predicted_attention = attention + alpha_attention * (this.target.attention - attention);
        const predicted_rhythm = rhythm_alignment + alpha_rhythm * (this.target.rhythm_alignment - rhythm_alignment);

        // Predict uncertainty (increases due to Process Noise Q)
        // Q grows with time step
        const Q = this.config.q_base * dt;
        
        const predicted_arousal_var = arousal_variance + Q;
        const predicted_attention_var = attention_variance + Q;
        const predicted_rhythm_var = rhythm_variance + Q;

        return {
            arousal: this.clamp(predicted_arousal),
            attention: this.clamp(predicted_attention),
            rhythm_alignment: this.clamp(predicted_rhythm),
            arousal_variance: predicted_arousal_var,
            attention_variance: predicted_attention_var,
            rhythm_variance: predicted_rhythm_var,
            prediction_error: 0,
            innovation: 0,
            mahalanobis_distance: 0,
            confidence: 0
        };
    }

    private correct(predicted: BeliefState, obs: Observation, dt: number): BeliefState {
        let corrected = { ...predicted };
        let currentInnovation = 0;
        let mahalanobis = 0;

        // ---- AROUSAL CORRECTION (from Heart Rate) ----
        if (obs.heart_rate !== undefined && obs.hr_confidence !== undefined && obs.hr_confidence > 0.3) {
            // Normalize HR (Resting 50-70 -> ~0.2, Active 90-120 -> ~0.7)
            const normalized_hr = this.clamp((obs.heart_rate - 50) / 70);
            
            // Adaptive R (Measurement Noise)
            // If sensor confidence is low, R increases (we trust it less)
            let R = this.config.r_base;
            if (this.config.adaptive_r) {
                const confidencePenalty = (1 - obs.hr_confidence) * this.config.r_adaptation_rate;
                R += confidencePenalty;
            }

            // Kalman Gain
            const S = predicted.arousal_variance + R; // Innovation Covariance
            const K_arousal = predicted.arousal_variance / S;
            
            const innovation = normalized_hr - predicted.arousal;
            
            // Outlier Detection
            // Mahalanobis distance = sqrt(innovation^2 / S)
            // If > 3.0, it's likely a sensor glitch
            mahalanobis = Math.sqrt((innovation * innovation) / S);

            if (mahalanobis < 3.0) {
                corrected.arousal = predicted.arousal + K_arousal * innovation;
                corrected.arousal_variance = (1 - K_arousal) * predicted.arousal_variance;
                currentInnovation = innovation;
            } else {
                // Reject outlier: do not update, but increase uncertainty slightly
                // console.warn("Outlier detected, rejecting measurement");
                corrected.arousal_variance += 0.01; 
            }
        }

        // ---- ATTENTION CORRECTION (from Interaction) ----
        const isDistracted = obs.user_interaction === 'pause' || obs.visibilty_state === 'hidden';
        
        if (isDistracted) {
            // Context measurements are usually cleaner, use smaller fixed R
            const R_ctx = 0.05;
            const K_attention = predicted.attention_variance / (predicted.attention_variance + R_ctx);
            
            const target_attention = 0.1; 
            const innovation = target_attention - predicted.attention;
            
            corrected.attention = predicted.attention + K_attention * innovation;
            corrected.attention_variance = (1 - K_attention) * predicted.attention_variance;
            
            // Rhythm breaks
            corrected.rhythm_alignment = Math.max(0, corrected.rhythm_alignment - 0.5 * dt);
        } else {
            // No direct measurement of "Attention" when active, 
            // so we model it as a slow recovery process + process noise
            corrected.attention = Math.min(1, corrected.attention + 0.15 * dt);
            corrected.attention_variance = Math.max(0.05, corrected.attention_variance - 0.02 * dt);
            
            // Rhythm builds
            corrected.rhythm_alignment = Math.min(1, corrected.rhythm_alignment + 0.1 * dt);
            corrected.rhythm_variance = Math.max(0.05, corrected.rhythm_variance - 0.01 * dt);
        }

        return {
            ...corrected,
            innovation: currentInnovation,
            mahalanobis_distance: mahalanobis,
            arousal: this.clamp(corrected.arousal),
            attention: this.clamp(corrected.attention),
            rhythm_alignment: this.clamp(corrected.rhythm_alignment)
        };
    }

    private computePredictionError(state: BeliefState): number {
        // PE = sqrt(sum of squared errors from target) - "Free Energy" proxy
        const error_arousal = Math.pow(state.arousal - this.target.arousal, 2);
        const error_attention = Math.pow(state.attention - this.target.attention, 2);
        const error_rhythm = Math.pow(state.rhythm_alignment - this.target.rhythm_alignment, 2);
        const mse = 0.4 * error_arousal + 0.3 * error_attention + 0.3 * error_rhythm;
        return Math.sqrt(mse);
    }

    private computeConfidence(state: BeliefState, obs: Observation): number {
        const certainty = 1 - Math.min(1, (state.arousal_variance + state.attention_variance + state.rhythm_variance) / 3);
        const sensor_quality = obs.hr_confidence ?? 0.5; // Default to 0.5 if no sensor
        const attention_stability = state.attention;
        
        // Weighted geometric mean
        // If Mahalanobis distance is high (outlier), confidence drops
        const anomalyPenalty = Math.max(0, 1 - (state.mahalanobis_distance / 5.0));
        
        const confidence = Math.pow(certainty * sensor_quality * attention_stability * anomalyPenalty, 1/4);
        return this.clamp(confidence);
    }

    private clamp(value: number, min = 0, max = 1): number {
        return Math.max(min, Math.min(max, value));
    }
}
