
import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children?: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null
  };

  public props!: Readonly<Props>;

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ZenB Critical Error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className="flex items-center justify-center min-h-screen bg-[#050508] text-white p-6 font-sans">
          <div className="text-center max-w-sm">
            <div className="text-6xl mb-6 opacity-80">🍃</div>
            <h2 className="text-2xl font-serif mb-3 text-white">Pause for a moment.</h2>
            <p className="text-sm text-white/60 mb-8 leading-relaxed">
              We encountered an unexpected disturbance in the flow. 
            </p>
            <button 
              onClick={() => window.location.reload()} 
              className="px-8 py-4 bg-white text-black rounded-2xl font-medium active:scale-95 transition-transform"
            >
              Refresh ZenB
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
