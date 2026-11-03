import { useEffect, useRef, useState } from 'react';

const WS_BASE = (window.location.protocol === 'https:' ? 'wss://' : 'ws://') + window.location.host + '/api/stream';

/**
 * Live camera frame delivery over a WebSocket instead of either a
 * persistent MJPEG connection (see useSnapshotStream.ts's doc comment —
 * held open forever, so it permanently occupies one of the browser's ~6
 * concurrent HTTP/1.1-per-host connection slots, SHARED across every open
 * tab) or a polled single-JPEG snapshot (each request completes so it
 * doesn't hit that cap, but a poll interval loose enough to not hammer the
 * server, 300-500ms, reads as visibly choppy and delayed).
 *
 * A WebSocket sidesteps the tradeoff: browsers don't count it against the
 * HTTP/1.1 per-host limit (far more concurrent WS connections are allowed
 * per origin), so it scales the same way polling did, while the backend
 * (see backend/app/api/stream.py's stream_ws) pushes frames as fast as
 * they're actually captured (capped at 20fps for bandwidth, not to work
 * around a connection limit) — no polling delay, no choppiness.
 *
 * Returns a blob: URL string usable as an <img src>, updated as each new
 * frame arrives, or null before the first frame / while reconnecting.
 */
export function useWebSocketStream(
  cameraId: number | string,
  options?: { variant?: 'full' | 'fire_smoke'; enabled?: boolean },
): string | null {
  const { variant = 'full', enabled = true } = options || {};
  const [frameUrl, setFrameUrl] = useState<string | null>(null);
  const blobUrlRef = useRef<string | null>(null);

  useEffect(() => {
    if (!enabled || !cameraId) {
      setFrameUrl(null);
      return;
    }

    let closedByUs = false;
    let ws: WebSocket | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      const token = localStorage.getItem('sentinel_token') || '';
      const variantParam = variant === 'fire_smoke' ? '&variant=fire_smoke' : '';
      ws = new WebSocket(`${WS_BASE}/${cameraId}/ws?token=${token}${variantParam}`);
      ws.binaryType = 'blob';

      ws.onmessage = (event) => {
        const blob = event.data as Blob;
        const newUrl = URL.createObjectURL(blob);
        setFrameUrl(newUrl);
        if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = newUrl;
      };

      ws.onclose = () => {
        if (!closedByUs) {
          retryTimer = setTimeout(connect, 2000);
        }
      };

      ws.onerror = () => {
        ws?.close();
      };
    };

    connect();

    return () => {
      closedByUs = true;
      if (retryTimer) clearTimeout(retryTimer);
      ws?.close();
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = null;
      }
    };
  }, [cameraId, variant, enabled]);

  return frameUrl;
}
