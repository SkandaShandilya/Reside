import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Camera, VideoOff, X, RefreshCw } from 'lucide-react';
import api from '../api/client';
import { useWebSocketStream } from '../hooks/useWebSocketStream';

interface Props {
  title: string;
  hint?: string;
  facingMode?: 'user' | 'environment';
  guide?: 'face' | 'plate';
  onCapture: (file: File, fullFrame: File) => void;
  onClose: () => void;
}

type CamState = 'requesting' | 'live' | 'server' | 'denied' | 'unavailable';

// Prompts for the device camera (only when opened), shows a live preview
// with a positioning guide, and hands back the captured frame as a JPEG
// File. The camera is always released on capture / cancel / unmount.
// The on-screen guide is an oval centred in a 4:3 box whose media is shown
// with object-fit: cover. Map it back onto the source frame, crop to the
// oval and blank everything outside it, so only the face INSIDE the guide
// is ever sent for verification (a bystander or background outside the
// oval never reaches the face model). A neutral margin is kept around the
// oval so the face detector still sees a normal, un-clipped face.
const FACE_OVAL_W = 0.46;
const FACE_OVAL_H = 0.72;
const OVAL_MARGIN = 0.2;

function drawFaceOval(src: CanvasImageSource, w: number, h: number): HTMLCanvasElement {
  const boxAspect = 4 / 3;
  let visW = w;
  let visH = h;
  if (w / h > boxAspect) visW = h * boxAspect; else visH = w / boxAspect;
  const cx = w / 2;
  const cy = h / 2;
  const ovalW = visW * FACE_OVAL_W;
  const ovalH = visH * FACE_OVAL_H;

  const outW = Math.round(ovalW * (1 + OVAL_MARGIN * 2));
  const outH = Math.round(ovalH * (1 + OVAL_MARGIN * 2));
  const out = document.createElement('canvas');
  out.width = outW;
  out.height = outH;
  const ctx = out.getContext('2d')!;
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, outW, outH);

  ctx.save();
  ctx.beginPath();
  ctx.ellipse(outW / 2, outH / 2, ovalW / 2, ovalH / 2, 0, 0, Math.PI * 2);
  ctx.clip();
  ctx.drawImage(
    src,
    cx - ovalW / 2 - ovalW * OVAL_MARGIN, cy - ovalH / 2 - ovalH * OVAL_MARGIN,
    outW, outH,
    0, 0, outW, outH,
  );
  ctx.restore();
  return out;
}

const CameraCaptureModal: React.FC<Props> = ({ title, hint, facingMode = 'user', guide = 'face', onCapture, onClose }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [camState, setCamState] = useState<CamState>('requesting');
  const [error, setError] = useState<string | null>(null);
  // When the browser can't open the camera because another program (here:
  // the RESIDE server's own webcam camera) already holds it, fall back to
  // that camera's live feed instead of failing.
  const [serverCamId, setServerCamId] = useState<number | null>(null);
  const serverFrame = useWebSocketStream(serverCamId ?? 0, { enabled: serverCamId !== null });
  const serverImgRef = useRef<HTMLImageElement>(null);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  const startCamera = useCallback(async () => {
    setCamState('requesting');
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setCamState('unavailable');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode }, audio: false });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setCamState('live');
    } catch (err: any) {
      if (err?.name === 'NotAllowedError' || err?.name === 'PermissionDeniedError') {
        setCamState('denied');
        return;
      }
      // Busy (NotReadableError) / missing device: try the server's webcam feed.
      try {
        const cams = await api.getCameras();
        const webcam = cams.find((c: any) => c.source_type === 'webcam' && c.is_active);
        if (webcam) {
          setServerCamId(webcam.id);
          setCamState('server');
          return;
        }
      } catch { /* fall through */ }
      setCamState('unavailable');
    }
  }, [facingMode]);

  useEffect(() => {
    startCamera();
    return stopCamera;
  }, [startCamera, stopCamera]);

  const toFile = (c: HTMLCanvasElement, name: string) => new Promise<File | null>(resolve => {
    c.toBlob(blob => resolve(blob ? new File([blob], name, { type: 'image/jpeg' }) : null), 'image/jpeg', 0.92);
  });

  // Always grabs the whole frame; for the face guide it also derives the
  // oval-only crop. `file` is what gets verified, `fullFrame` is the
  // untouched shot (e.g. what a resident should see in a visitor request).
  const finishCapture = async (source: CanvasImageSource, w: number, h: number, stopAfter: boolean) => {
    const full = document.createElement('canvas');
    full.width = w;
    full.height = h;
    full.getContext('2d')!.drawImage(source, 0, 0);
    const stamp = Date.now();
    const fullFile = await toFile(full, `capture-${stamp}.jpg`);
    const file = guide === 'face' ? await toFile(drawFaceOval(full, w, h), `capture-${stamp}-oval.jpg`) : fullFile;
    if (!file || !fullFile) { setError('Unable to capture the image. Please try again.'); return; }
    if (stopAfter) stopCamera();
    onCapture(file, fullFile);
  };

  const handleCapture = () => {
    if (camState === 'server') {
      const img = serverImgRef.current;
      if (!img || !img.naturalWidth) return;
      void finishCapture(img, img.naturalWidth, img.naturalHeight, false);
      return;
    }
    const video = videoRef.current;
    if (!video || camState !== 'live' || !video.videoWidth) return;
    void finishCapture(video, video.videoWidth, video.videoHeight, true);
  };

  const handleClose = () => { stopCamera(); onClose(); };

  return (
    <div className="modal-overlay" onClick={handleClose}>
      <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 520 }}>
        <div className="modal-header">
          <h3 className="modal-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Camera size={18} style={{ color: 'var(--color-info)' }} /> {title}
          </h3>
          <button className="btn btn-ghost btn-sm" onClick={handleClose} aria-label="Close"><X size={16} /></button>
        </div>
        <div className="modal-body">
          <div style={{
            position: 'relative', width: '100%', aspectRatio: '4/3', background: '#000',
            borderRadius: 'var(--radius-md)', overflow: 'hidden',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              style={{ width: '100%', height: '100%', objectFit: 'cover', display: camState === 'live' ? 'block' : 'none' }}
            />
            {camState === 'server' && (
              serverFrame ? (
                <img ref={serverImgRef} src={serverFrame} alt="Server webcam" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              ) : (
                <div style={{ textAlign: 'center', color: 'var(--text-dim)' }}>
                  <Camera size={36} style={{ marginBottom: 8 }} />
                  <p style={{ fontSize: 13, margin: 0 }}>Connecting to the server camera…</p>
                </div>
              )
            )}
            {(camState === 'live' || (camState === 'server' && serverFrame)) && (
              <div style={{
                position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
                width: guide === 'plate' ? '70%' : '46%', height: guide === 'plate' ? '28%' : '72%',
                border: '2px dashed rgba(255,255,255,0.85)',
                borderRadius: guide === 'plate' ? 8 : '50%',
                pointerEvents: 'none', boxShadow: '0 0 0 9999px rgba(0,0,0,0.28)',
              }} />
            )}
            {camState === 'requesting' && (
              <div style={{ textAlign: 'center', color: 'var(--text-dim)' }}>
                <Camera size={36} style={{ marginBottom: 8 }} />
                <p style={{ fontSize: 13, margin: 0 }}>Requesting camera access…</p>
              </div>
            )}
            {(camState === 'denied' || camState === 'unavailable') && (
              <div style={{ textAlign: 'center', color: 'var(--color-critical)', padding: 20 }}>
                <VideoOff size={36} style={{ marginBottom: 8 }} />
                <p style={{ fontSize: 13, margin: 0, fontWeight: 600 }}>
                  {camState === 'denied'
                    ? 'Camera access is required to capture the image.'
                    : 'Unable to access the camera. Please check your camera connection.'}
                </p>
                <button className="btn btn-sm btn-ghost" style={{ marginTop: 10 }} onClick={startCamera}>
                  <RefreshCw size={13} /> Try Again
                </button>
              </div>
            )}
          </div>
          {camState === 'server' && <p style={{ fontSize: 12, color: 'var(--color-warning)', margin: '8px 0 0' }}>Your browser's camera is in use by the RESIDE server, so this is the server webcam's live feed.</p>}
          {hint && <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: '8px 0 0' }}>{hint}</p>}
          {error && <p style={{ fontSize: 12, color: 'var(--color-critical)', margin: '8px 0 0' }}>{error}</p>}
        </div>
        <div className="modal-footer">
          <button className="btn btn-ghost" onClick={handleClose}>Cancel</button>
          <button className="btn btn-primary" onClick={handleCapture} disabled={camState !== 'live' && !(camState === 'server' && !!serverFrame)}>Capture</button>
        </div>
      </div>
    </div>
  );
};

export default CameraCaptureModal;
