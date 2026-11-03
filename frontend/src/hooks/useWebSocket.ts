import { useEffect, useRef, useState, useCallback } from 'react';
import type { WSSecurityAlert } from '../types';

const WS_URL = (window.location.protocol === 'https:' ? 'wss://' : 'ws://') + window.location.host + '/ws/security-alerts';

// ── Alert sound on/off ───────────────────────────────────────────────────
// Persisted so the preference survives reloads/navigation; defaults to on.
const SOUND_PREF_KEY = 'sentinel_sound_enabled';
let soundEnabled = localStorage.getItem(SOUND_PREF_KEY) !== 'false';
let soundListeners = new Set<(enabled: boolean) => void>();

export function isSoundEnabled(): boolean {
  return soundEnabled;
}

export function setSoundEnabled(enabled: boolean): void {
  soundEnabled = enabled;
  try {
    localStorage.setItem(SOUND_PREF_KEY, String(enabled));
  } catch (e) {
    // Ignore storage errors (e.g. private browsing quota)
  }
  soundListeners.forEach(l => l(enabled));
}

// Web Audio API Synthesized Chime sound
function playAlertChime(severity: string = 'info') {
  if (!soundEnabled) return;
  try {
    const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();

    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = severity === 'critical' ? 'sawtooth' : 'sine';
    osc.frequency.setValueAtTime(severity === 'critical' ? 880 : 587.33, ctx.currentTime); // A5 or D5
    osc.frequency.exponentialRampToValueAtTime(severity === 'critical' ? 440 : 880, ctx.currentTime + 0.3);

    gain.gain.setValueAtTime(0.3, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.35);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start();
    osc.stop(ctx.currentTime + 0.35);
  } catch (e) {
    // Ignore audio permission errors
  }
}

// Request Browser Desktop Notification permission
function triggerDesktopNotification(alert: WSSecurityAlert) {
  if (!('Notification' in window)) return;

  const severityText = alert.severity ? alert.severity.toUpperCase() : 'INFO';

  if (Notification.permission === 'granted') {
    new Notification(`⚠️ Security Alert: ${alert.activity || 'Incident Detected'}`, {
      body: `Camera: ${alert.camera_name || 'Camera'} (${alert.camera_location || 'Location'})\nSeverity: ${severityText}`,
      icon: '/favicon.png',
    });
  } else if (Notification.permission !== 'denied') {
    Notification.requestPermission();
  }
}

// ── Module-level singleton connection ───────────────────────────────────
// Header (always mounted, inside Layout) and Dashboard both call
// useWebSocket() independently. With a per-hook-instance WebSocket, that
// meant two live sockets to the same endpoint whenever the Dashboard was
// open, each independently playing the alert chime / firing the desktop
// notification for the SAME incoming alert. A single shared connection
// (subscriber list instead of per-component sockets) fixes both the
// duplicate-connection resource leak and the duplicate side effects — the
// chime/notification fire exactly once per real message, from here, not
// once per mounted consumer.
//
// This also fixes a real reconnect bug: the old per-hook `disconnect()`
// closed the socket, but the shared `onclose` handler couldn't tell an
// intentional disconnect from a dropped connection and always scheduled a
// reconnect — so logging out (unmounting Header) still reconnected 5s
// later. Now reconnection is only scheduled while at least one subscriber
// still wants the connection (tracked via `desiredSubscriberCount`).
type Listener = {
  onOpen: () => void;
  onClose: () => void;
  onAlert: (alert: WSSecurityAlert) => void;
};

let sharedWs: WebSocket | null = null;
let reconnectTimer: number | null = null;
let subscribers = new Set<Listener>();
let intentionallyClosed = false;

function connectShared() {
  const token = localStorage.getItem('sentinel_token');
  if (!token) return;
  if (sharedWs && (sharedWs.readyState === WebSocket.OPEN || sharedWs.readyState === WebSocket.CONNECTING)) {
    return; // already connected/connecting — nothing to do
  }

  intentionallyClosed = false;
  try {
    const socket = new WebSocket(`${WS_URL}?token=${token}`);
    sharedWs = socket;

    socket.onopen = () => {
      if (sharedWs !== socket) return; // a newer socket has since replaced this one
      console.log('WebSocket connected to RESIDE Security Feed');
      subscribers.forEach(l => l.onOpen());
    };

    socket.onmessage = (event) => {
      if (sharedWs !== socket) return;
      try {
        const data: WSSecurityAlert = JSON.parse(event.data);
        if (data.type === 'security_alert') {
          playAlertChime(data.severity);
          triggerDesktopNotification(data);
          subscribers.forEach(l => l.onAlert(data));
        }
      } catch (e) {
        console.error('WS parse error:', e);
      }
    };

    socket.onclose = () => {
      if (sharedWs !== socket) return;
      subscribers.forEach(l => l.onClose());
      if (!intentionallyClosed && subscribers.size > 0) {
        reconnectTimer = window.setTimeout(connectShared, 5000);
      }
    };

    socket.onerror = () => {
      socket.close();
    };
  } catch (e) {
    console.error('WS connection error:', e);
  }
}

function releaseShared() {
  if (subscribers.size > 0) return; // other components still need the connection
  intentionallyClosed = true;
  if (reconnectTimer !== null) {
    window.clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  sharedWs?.close();
  sharedWs = null;
}

export function useSoundEnabled(): [boolean, (enabled: boolean) => void] {
  const [enabled, setEnabled] = useState(soundEnabled);

  useEffect(() => {
    const listener = (v: boolean) => setEnabled(v);
    soundListeners.add(listener);
    return () => {
      soundListeners.delete(listener);
    };
  }, []);

  return [enabled, setSoundEnabled];
}

export function useWebSocket() {
  const [isConnected, setIsConnected] = useState(sharedWs?.readyState === WebSocket.OPEN);
  const [lastAlert, setLastAlert] = useState<WSSecurityAlert | null>(null);
  const [alerts, setAlerts] = useState<WSSecurityAlert[]>([]);
  const listenerRef = useRef<Listener | null>(null);

  useEffect(() => {
    const listener: Listener = {
      onOpen: () => setIsConnected(true),
      onClose: () => setIsConnected(false),
      onAlert: (data) => {
        setLastAlert(data);
        setAlerts(prev => [data, ...prev.slice(0, 49)]);
      },
    };
    listenerRef.current = listener;
    subscribers.add(listener);
    connectShared();

    return () => {
      subscribers.delete(listener);
      releaseShared();
    };
  }, []);

  const clearLastAlert = useCallback(() => setLastAlert(null), []);
  const reconnect = useCallback(() => connectShared(), []);

  return { isConnected, lastAlert, alerts, clearLastAlert, reconnect };
}
