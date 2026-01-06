
import { useEffect, useState, useRef } from 'react';
import { CameraVitalsEngine } from '../services/CameraVitalsEngine.v2';
import { VitalSigns } from '../types';

export function useCameraVitals(enabled: boolean) {
  const [vitals, setVitals] = useState<VitalSigns>({
    heartRate: 0,
    confidence: 0,
    signalQuality: 'poor',
    snr: 0,
    motionLevel: 0
  });
  
  const [error, setError] = useState<string | null>(null);
  const [isReady, setIsReady] = useState(false);
  
  const engineRef = useRef<CameraVitalsEngine | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  
  useEffect(() => {
    if (!enabled) {
      cleanup();
      return;
    }
    
    let mounted = true;
    
    const init = async () => {
      try {
        // Initialize engine
        const engine = new CameraVitalsEngine();
        await engine.init();
        engineRef.current = engine;
        
        // Request camera
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'user',
            width: { ideal: 640 },
            height: { ideal: 480 },
            frameRate: { ideal: 30 }
          }
        });
        streamRef.current = stream;
        
        // Create video element
        const video = document.createElement('video');
        video.srcObject = stream;
        video.playsInline = true;
        video.muted = true;
        await video.play();
        videoRef.current = video;
        
        if (!mounted) {
          cleanup();
          return;
        }
        
        setIsReady(true);
        
        // Processing loop (30fps target)
        let lastProcessTime = 0;
        const targetFrameTime = 1000 / 30;
        
        const processLoop = async (now: number) => {
           if (!engineRef.current || !videoRef.current || !mounted) return;
           
           if (now - lastProcessTime >= targetFrameTime) {
               try {
                   const result = await engineRef.current.processFrame(videoRef.current);
                   if (mounted) setVitals(result);
                   lastProcessTime = now;
               } catch (err) {
                   console.error('[rPPG] Processing error:', err);
               }
           }
           rafRef.current = requestAnimationFrame(processLoop);
        };
        
        rafRef.current = requestAnimationFrame(processLoop);
        
      } catch (err: any) {
        console.error('[rPPG] Initialization failed:', err);
        setError(err.message || 'Camera access denied');
      }
    };
    
    init();
    
    return () => {
      mounted = false;
      cleanup();
    };
  }, [enabled]);
  
  const cleanup = () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    if (streamRef.current) streamRef.current.getTracks().forEach(track => track.stop());
    if (engineRef.current) engineRef.current.dispose();
    
    setIsReady(false);
    setVitals({
        heartRate: 0,
        confidence: 0,
        signalQuality: 'poor',
        snr: 0,
        motionLevel: 0
    });
  };
  
  return { vitals, isReady, error };
}
