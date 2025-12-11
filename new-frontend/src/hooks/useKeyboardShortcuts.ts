import { useEffect, useCallback } from 'react';

type KeyboardShortcut = {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  meta?: boolean;
  action: () => void;
  description?: string;
};

export function useKeyboardShortcuts(shortcuts: KeyboardShortcut[], enabled = true) {
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (!enabled) return;

      // Ignore if user is typing in an input
      const target = event.target as HTMLElement;
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.isContentEditable
      ) {
        return;
      }

      for (const shortcut of shortcuts) {
        const keyMatch = event.key.toLowerCase() === shortcut.key.toLowerCase();
        const ctrlMatch = shortcut.ctrl ? event.ctrlKey || event.metaKey : !event.ctrlKey && !event.metaKey;
        const shiftMatch = shortcut.shift ? event.shiftKey : !event.shiftKey;
        const altMatch = shortcut.alt ? event.altKey : !event.altKey;

        if (keyMatch && ctrlMatch && shiftMatch && altMatch) {
          event.preventDefault();
          shortcut.action();
          return;
        }
      }
    },
    [shortcuts, enabled]
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);
}

// DICOM viewer specific shortcuts
export function useDicomViewerShortcuts({
  onPan,
  onZoom,
  onWindowLevel,
  onLength,
  onEllipse,
  onRectangle,
  onCrosshair,
  onReset,
  onPlayPause,
  onNextSlice,
  onPrevSlice,
  onFirstSlice,
  onLastSlice,
  onToggleFullscreen,
}: {
  onPan?: () => void;
  onZoom?: () => void;
  onWindowLevel?: () => void;
  onLength?: () => void;
  onEllipse?: () => void;
  onRectangle?: () => void;
  onCrosshair?: () => void;
  onReset?: () => void;
  onPlayPause?: () => void;
  onNextSlice?: () => void;
  onPrevSlice?: () => void;
  onFirstSlice?: () => void;
  onLastSlice?: () => void;
  onToggleFullscreen?: () => void;
}) {
  const shortcuts: KeyboardShortcut[] = [];

  if (onPan) shortcuts.push({ key: 'h', action: onPan, description: 'Pan tool' });
  if (onZoom) shortcuts.push({ key: 'z', action: onZoom, description: 'Zoom tool' });
  if (onWindowLevel) shortcuts.push({ key: 'w', action: onWindowLevel, description: 'Window/Level' });
  if (onLength) shortcuts.push({ key: 'l', action: onLength, description: 'Length measurement' });
  if (onEllipse) shortcuts.push({ key: 'e', action: onEllipse, description: 'Ellipse tool' });
  if (onRectangle) shortcuts.push({ key: 'r', action: onRectangle, description: 'Rectangle tool' });
  if (onCrosshair) shortcuts.push({ key: 'c', action: onCrosshair, description: 'Crosshair' });
  if (onReset) shortcuts.push({ key: 'r', ctrl: true, action: onReset, description: 'Reset viewport' });
  if (onPlayPause) shortcuts.push({ key: ' ', action: onPlayPause, description: 'Play/Pause' });
  if (onNextSlice) shortcuts.push({ key: 'ArrowDown', action: onNextSlice, description: 'Next slice' });
  if (onPrevSlice) shortcuts.push({ key: 'ArrowUp', action: onPrevSlice, description: 'Previous slice' });
  if (onFirstSlice) shortcuts.push({ key: 'Home', action: onFirstSlice, description: 'First slice' });
  if (onLastSlice) shortcuts.push({ key: 'End', action: onLastSlice, description: 'Last slice' });
  if (onToggleFullscreen) shortcuts.push({ key: 'f', action: onToggleFullscreen, description: 'Fullscreen' });

  useKeyboardShortcuts(shortcuts);
}
