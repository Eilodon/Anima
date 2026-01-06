
import * as tf from '@tensorflow/tfjs';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import type { FFTRequest, FFTResponse } from './fft.worker';
import { SignalQuality, VitalSigns } from '../types';

/**
 * ZENB CAMERA VITALS ENGINE v2.0
 * =================================
 * The Sensor Driver layer for the ZenB Biological OS.
 * 
 * Features:
 * - rPPG extraction (Green channel)
 * - Ring Buffer (8s window)
 * - Motion Compensation
 * - Web Worker Offloading
 */

interface ROI {
  x: number; y: number; width: number; height: number;
}

// Ring Buffer: High performance circular buffer for continuous signal streaming
class RingBuffer {
  private buffer: Float32Array;
  private head: number = 0;
  private size: number = 0;

  constructor(capacity: number) {
    this.buffer = new Float32Array(capacity);
  }
  
  push(value: number): void {
    this.buffer[this.head] = value;
    this.head = (this.head + 1) % this.buffer.length;
    if (this.size < this.buffer.length) this.size++;
  }

  toArray(): number[] {
    const result = new Array(this.size);
    for (let i = 0; i < this.size; i++) {
      const idx = (this.head - this.size + i + this.buffer.length) % this.buffer.length;
      result[i] = this.buffer[idx];
    }
    return result;
  }
  
  get length(): number { return this.size; }
  clear(): void { this.head = 0; this.size = 0; }
}

// Butterworth Bandpass Filter (0.75Hz - 3.0Hz => 45 BPM - 180 BPM)
class BandpassFilter {
  private readonly b: number[];
  private readonly a: number[];
  private x: number[] = [0, 0, 0];
  private y: number[] = [0, 0, 0];

  constructor(sampleRate: number, lowCutoff: number, highCutoff: number) {
    const nyquist = sampleRate / 2;
    const low = lowCutoff / nyquist;
    const high = highCutoff / nyquist;
    
    const bw = high - low;
    const center = (high + low) / 2;
    const Q = center / bw;
    const K = Math.tan(Math.PI * center);
    const norm = 1 / (1 + K / Q + K * K);
    this.b = [bw * K * norm, 0, -bw * K * norm];
    this.a = [1, 2 * (K * K - 1) * norm, (1 - K / Q + K * K) * norm];
  }

  filter(input: number): number {
    this.x[2] = this.x[1]; this.x[1] = this.x[0]; this.x[0] = input;
    const output = this.b[0]*this.x[0] + this.b[1]*this.x[1] + this.b[2]*this.x[2] 
                 - this.a[1]*this.y[0] - this.a[2]*this.y[1]; // y[0] is prev output
    this.y[2] = this.y[1]; this.y[1] = this.y[0]; this.y[0] = output;
    return output;
  }
  
  reset() { this.x = [0,0,0]; this.y = [0,0,0]; }
}

export class CameraVitalsEngine {
  private detector: faceLandmarksDetection.FaceLandmarksDetector | null = null;
  private canvas: OffscreenCanvas;
  private ctx: OffscreenCanvasRenderingContext2D;
  
  private signalBuffer: RingBuffer;
  private filteredBuffer: RingBuffer;
  private bandpassFilter: BandpassFilter;
  
  private readonly BUFFER_DURATION = 8; // seconds window for FFT
  private readonly SAMPLE_RATE = 30; // target fps
  private readonly MIN_SAMPLES = 128; // Min samples to run FFT

  private fftWorker: Worker | null = null;
  private pendingFFT = false;
  
  private lastFacePosition: { x: number; y: number } | null = null;
  private motionAccumulator = 0;

  constructor() {
    this.canvas = new OffscreenCanvas(640, 480);
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true })!;
    
    const bufferSize = Math.ceil(this.BUFFER_DURATION * this.SAMPLE_RATE);
    this.signalBuffer = new RingBuffer(bufferSize);
    this.filteredBuffer = new RingBuffer(bufferSize);
    this.bandpassFilter = new BandpassFilter(this.SAMPLE_RATE, 0.75, 3.0);
  }

  async init(): Promise<void> {
    try {
      await tf.ready();
      await tf.setBackend('webgl');
      
      this.detector = await faceLandmarksDetection.createDetector(
        faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
        { runtime: 'tfjs', maxFaces: 1, refineLandmarks: false }
      );
      
      this.fftWorker = new Worker(new URL('./fft.worker.ts', import.meta.url), { type: 'module' });
      console.log('[ZenB] CameraVitalsEngine v2.0 initialized (Welch-Ready)');
    } catch (error) {
      console.error('[ZenB] Engine init failed', error);
      throw error;
    }
  }

  async processFrame(video: HTMLVideoElement): Promise<VitalSigns> {
    if (!this.detector) throw new Error('Engine not initialized');
    
    const faces = await this.detector.estimateFaces(video, { flipHorizontal: false });
    if (faces.length === 0) return this.getDefaultVitals();

    const face = faces[0];
    const roi = this.extractForeheadROI(face.keypoints, video.videoWidth, video.videoHeight);
    const greenSignal = this.extractGreenWithSkinDetection(video, roi);
    const motion = this.detectMotion(face.keypoints);

    // Pipeline: Raw -> RingBuffer -> Bandpass -> FilteredBuffer
    this.signalBuffer.push(greenSignal);
    const filtered = this.bandpassFilter.filter(greenSignal);
    this.filteredBuffer.push(filtered);

    // Compute HR if we have enough data
    if (this.filteredBuffer.length >= this.MIN_SAMPLES && !this.pendingFFT) {
       return await this.computeHeartRate(motion);
    }

    return { ...this.getDefaultVitals(), motionLevel: motion };
  }

  private async computeHeartRate(motion: number): Promise<VitalSigns> {
    if (!this.fftWorker) return this.getDefaultVitals();
    
    this.pendingFFT = true;
    try {
        const signal = this.filteredBuffer.toArray();
        const request: FFTRequest = {
            type: 'compute_fft',
            signal,
            sampleRate: this.SAMPLE_RATE,
            minFreq: 0.75,
            maxFreq: 3.0,
            method: 'welch',     // ZENB MANIFESTO COMPLIANCE
            window: 'blackman'   // Best for noise suppression
        };
        
        this.fftWorker.postMessage(request);
        
        const response = await new Promise<FFTResponse>((resolve, reject) => {
             const timeout = setTimeout(() => reject(new Error('FFT Timeout')), 1000);
             const handler = (e: MessageEvent) => {
                 clearTimeout(timeout);
                 this.fftWorker?.removeEventListener('message', handler);
                 if (e.data.type === 'error') reject(e.data.message);
                 else resolve(e.data);
             };
             this.fftWorker?.addEventListener('message', handler);
        });

        // Motion Penalty: If user moves, confidence drops significantly
        const motionPenalty = Math.max(0, 1 - motion * 3.0);
        const adjustedConfidence = response.confidence * motionPenalty;
        
        return {
            heartRate: response.heartRate,
            confidence: adjustedConfidence,
            signalQuality: this.classifySignalQuality(adjustedConfidence, motion, response.snr),
            snr: response.snr,
            motionLevel: motion
        };

    } catch (e) {
        return this.getDefaultVitals();
    } finally {
        this.pendingFFT = false;
    }
  }

  // ROI Extraction: Focus on Forehead (Less muscle movement than cheeks)
  private extractForeheadROI(keypoints: any[], w: number, h: number): ROI {
    // MediaPipe Mesh Indices for Forehead
    const indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288];
    const pts = indices.map(i => keypoints[i]);
    const xs = pts.map(p => p.x);
    const ys = pts.map(p => p.y);
    const minX = Math.max(0, Math.min(...xs));
    const maxX = Math.min(w, Math.max(...xs));
    const minY = Math.max(0, Math.min(...ys));
    const maxY = Math.min(h, Math.max(...ys));
    
    const width = maxX - minX;
    const height = maxY - minY;
    // Slight padding
    return {
        x: Math.max(0, minX - width * 0.1),
        y: Math.max(0, minY - height * 0.1),
        width: width * 1.2,
        height: height * 1.2
    };
  }

  private extractGreenWithSkinDetection(video: HTMLVideoElement, roi: ROI): number {
    this.canvas.width = roi.width;
    this.canvas.height = roi.height;
    this.ctx.drawImage(video, roi.x, roi.y, roi.width, roi.height, 0, 0, roi.width, roi.height);
    
    const imageData = this.ctx.getImageData(0, 0, roi.width, roi.height);
    const pixels = imageData.data;
    let greenSum = 0;
    let weightSum = 0;

    for (let i = 0; i < pixels.length; i += 4) {
        const r = pixels[i], g = pixels[i+1], b = pixels[i+2];
        // Simple skin filter to ignore hair/background in the ROI box
        if (this.isSkinPixel(r,g,b)) {
            greenSum += g;
            weightSum += 1;
        }
    }
    return weightSum > 0 ? greenSum / weightSum : 0;
  }

  // Basic HSV/RGB based skin detection rule
  private isSkinPixel(r: number, g: number, b: number): boolean {
    const sum = r + g + b;
    if (sum === 0) return false;
    const rn = r/sum, gn = g/sum;
    return r > 95 && g > 40 && b > 20 && r > g && r > b && Math.abs(r-g) > 15 && rn > 0.35 && gn > 0.28;
  }

  private detectMotion(keypoints: any[]): number {
    const nose = keypoints[1];
    const current = { x: nose.x, y: nose.y };
    if (!this.lastFacePosition) {
        this.lastFacePosition = current;
        return 0;
    }
    const dx = current.x - this.lastFacePosition.x;
    const dy = current.y - this.lastFacePosition.y;
    const disp = Math.sqrt(dx*dx + dy*dy);
    this.lastFacePosition = current;
    
    // Accumulate motion over time to prevent sudden spikes
    this.motionAccumulator = 0.7 * this.motionAccumulator + 0.3 * disp;
    // Normalize (heuristic)
    return Math.min(1, this.motionAccumulator / 5);
  }

  private classifySignalQuality(conf: number, motion: number, snr: number): SignalQuality {
    if (conf > 0.8 && motion < 0.2 && snr > 5) return 'excellent';
    if (conf > 0.6 && motion < 0.4 && snr > 2) return 'good';
    if (conf > 0.4) return 'fair';
    return 'poor';
  }

  private getDefaultVitals(): VitalSigns {
      return { heartRate: 0, confidence: 0, signalQuality: 'poor', snr: 0, motionLevel: 0 };
  }

  dispose(): void {
    this.detector?.dispose();
    this.fftWorker?.terminate();
    this.signalBuffer.clear();
    this.filteredBuffer.clear();
    this.bandpassFilter.reset();
  }
}
