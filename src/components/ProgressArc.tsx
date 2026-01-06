
import React from 'react';

type ProgressArcProps = {
  size?: number;
  stroke?: number;
  circleRef: React.RefObject<SVGCircleElement | null>;
};

// A minimal, glowing progress arc - Zen Masterpiece Version
export const ProgressArc = ({ size = 200, stroke = 1, circleRef }: ProgressArcProps) => {
  const radius = (size - stroke) / 2;
  const circumference = radius * 2 * Math.PI;

  return (
    <div className="relative flex items-center justify-center transition-all duration-1000 ease-out opacity-80 mix-blend-screen" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="absolute inset-0 rotate-[-90deg]">
        {/* Background Trace */}
        <circle
          stroke="rgba(255,255,255,0.05)"
          fill="transparent"
          strokeWidth={1}
          r={radius}
          cx={size / 2}
          cy={size / 2}
        />
        {/* Active Glow - Thinner, brighter, purer */}
        <circle
          ref={circleRef}
          stroke="url(#gradient)"
          fill="transparent"
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference} 
          r={radius}
          cx={size / 2}
          cy={size / 2}
          className="transition-all duration-0 ease-linear"
          filter="url(#glow)"
        />
        <defs>
          <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="white" stopOpacity="0.4" />
            <stop offset="50%" stopColor="white" stopOpacity="0.8" />
            <stop offset="100%" stopColor="white" stopOpacity="1.0" />
          </linearGradient>
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            {/* Double blur for 'Bloom' effect on SVG */}
            <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
            <feGaussianBlur stdDeviation="4" result="coloredBlur2"/>
            <feMerge>
                <feMergeNode in="coloredBlur2"/>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>
      </svg>
    </div>
  );
};
