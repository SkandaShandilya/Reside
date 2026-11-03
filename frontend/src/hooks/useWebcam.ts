/**
 * useWebcam — Browser webcam capture hook
 *
 * Provides a MediaStream for browser webcam sources.
 * Uses navigator.mediaDevices.getUserMedia with configurable constraints.
 *
 * Usage:
 *   const { videoRef, stream, error, start, stop, isActive } = useWebcam();
 *   <video ref={videoRef} autoPlay muted playsInline />
 */
import { useRef, useState, useCallback, useEffect } from 'react';

interface WebcamOptions {
  width?: number;
  height?: number;
  facingMode?: 'user' | 'environment';
  frameRate?: number;
}

interface WebcamState {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  stream: MediaStream | null;
  error: string | null;
  isActive: boolean;
  start: () => Promise<void>;
  stop: () => void;
  snapshot: () => string | null; // Returns base64 JPEG data URL
}

export function useWebcam(options: WebcamOptions = {}): WebcamState {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isActive, setIsActive] = useState(false);

  const start = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera access is not supported in this browser or context (requires HTTPS).');
      return;
    }
    try {
      const constraints: MediaStreamConstraints = {
        video: {
          width: { ideal: options.width || 1280 },
          height: { ideal: options.height || 720 },
          facingMode: options.facingMode || 'environment',
          frameRate: { ideal: options.frameRate || 30 },
        },
        audio: false,
      };
      const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      setStream(mediaStream);
      setError(null);
      setIsActive(true);

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
    } catch (err: any) {
      const msg =
        err.name === 'NotAllowedError' ? 'Camera permission denied. Please allow camera access.'
        : err.name === 'NotFoundError' ? 'No camera device found.'
        : err.name === 'NotReadableError' ? 'Camera is in use by another application.'
        : `Camera error: ${err.message}`;
      setError(msg);
      setIsActive(false);
    }
  }, [options.width, options.height, options.facingMode, options.frameRate]);

  const stop = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsActive(false);
  }, [stream]);

  const snapshot = useCallback((): string | null => {
    if (!videoRef.current || !isActive) return null;
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    canvas.getContext('2d')?.drawImage(videoRef.current, 0, 0);
    return canvas.toDataURL('image/jpeg', 0.85);
  }, [isActive]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]);

  return { videoRef, stream, error, isActive, start, stop, snapshot };
}
