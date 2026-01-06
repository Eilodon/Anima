
import { create } from 'zustand';

type UIState = {
  isSettingsOpen: boolean;
  isHistoryOpen: boolean;
  showSummary: boolean;

  setSettingsOpen: (isOpen: boolean) => void;
  setHistoryOpen: (isOpen: boolean) => void;
  setShowSummary: (show: boolean) => void;
};

export const useUIStore = create<UIState>((set) => ({
  isSettingsOpen: false,
  isHistoryOpen: false,
  showSummary: false,

  setSettingsOpen: (isOpen) => set({ isSettingsOpen: isOpen }),
  setHistoryOpen: (isOpen) => set({ isHistoryOpen: isOpen }),
  setShowSummary: (show) => set({ showSummary: show }),
}));
