import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Camera, VideoOff, X, CheckCircle, XCircle, Clock, UserPlus, RefreshCw } from 'lucide-react';
import api from '../api/client';
import type { ResidentSummary, VisitorFaceRequest } from '../types';

type Step = 'prompt' | 'confirm' | 'capture' | 'sending' | 'waiting';
type CamState = 'requesting' | 'live' | 'denied' | 'unavailable';

interface Props {
  onClose: () => void;
  // The photo the guard already captured/uploaded for face verification —
  // sent to the resident as-is instead of asking for a second capture.
  photo?: File | null;
}

// Unknown-face workflow: prompt -> live camera capture (validated for a
// usable face by the existing face-service) -> request sent to a resident
// -> status polling. Entirely additive; the registered-face flow that
// renders this never touches it.
const VisitorRequestFlow: React.FC<Props> = ({ onClose, photo }) => {
  const [step, setStep] = useState<Step>('prompt');
  const [residents, setResidents] = useState<ResidentSummary[]>([]);
  const [residentId, setResidentId] = useState<number | ''>('');
  const [location, setLocation] = useState('Main Gate');
  const [visitorName, setVisitorName] = useState('');
  const [camState, setCamState] = useState<CamState>('requesting');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [request, setRequest] = useState<VisitorFaceRequest | null>(null);

  const [photoUrl, setPhotoUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!photo) { setPhotoUrl(null); return; }
    const url = URL.createObjectURL(photo);
    setPhotoUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [photo]);

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

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
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setCamState('live');
    } catch (err: any) {
      setCamState(err?.name === 'NotAllowedError' || err?.name === 'PermissionDeniedError' ? 'denied' : 'unavailable');
    }
  }, []);

  useEffect(() => {
    api.getActiveResidents().then(res => setResidents(res.residents)).catch(() => {});
  }, []);

  // Camera is requested only once the guard chooses to capture, and always
  // released when leaving the capture step or closing the flow.
  useEffect(() => {
    if (step === 'capture') startCamera();
    else stopCamera();
    return stopCamera;
  }, [step, startCamera, stopCamera]);

  // Poll the request until the resident decides (or it expires).
  useEffect(() => {
    if (step !== 'waiting' || !request || request.status !== 'pending') return;
    const timer = setInterval(async () => {
      try {
        const latest = await api.getVisitorRequest(request.id);
        setRequest(latest);
      } catch {
        // transient network error — keep polling
      }
    }, 3000);
    return () => clearInterval(timer);
  }, [step, request]);

  const handleCapture = async () => {
    const video = videoRef.current;
    if (!video || camState !== 'live' || !video.videoWidth) return;
    if (!residentId) { setError('Select the resident being visited.'); return; }
    setBusy(true);
    setError(null);
    try {
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d')!.drawImage(video, 0, 0);
      const blob: Blob | null = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.9));
      if (!blob) throw new Error('Unable to capture the image.');

      const check = await api.validateVisitorFace(blob);
      if (!check.face_detected) {
        setError('No face detected. Please position the visitor inside the frame and try again.');
        return;
      }

      stopCamera();
      setStep('sending');
      try {
        setRequest(await api.createVisitorRequest(blob, Number(residentId), location.trim() || 'Main Gate', visitorName.trim()));
        setStep('waiting');
      } catch (err: any) {
        setError(err.message || 'Unable to send the visitor request. Please try again.');
        setStep('capture');
      }
    } catch (err: any) {
      setError(err.message || 'Unable to send the visitor request. Please try again.');
    } finally {
      setBusy(false);
    }
  };

  const handleSendExisting = async () => {
    if (!photo) return;
    if (!residentId) { setError('Select the resident being visited.'); return; }
    setBusy(true);
    setError(null);
    try {
      setRequest(await api.createVisitorRequest(photo, Number(residentId), location.trim() || 'Main Gate', visitorName.trim()));
      setStep('waiting');
    } catch (err: any) {
      setError(err.message || 'Unable to send the visitor request. Please try again.');
    } finally {
      setBusy(false);
    }
  };

  const handleCancelRequest = async () => {
    if (request && request.status === 'pending') {
      try { await api.cancelVisitorRequest(request.id); } catch { /* best effort */ }
    }
    onClose();
  };

  const statusView = () => {
    if (!request) return null;
    if (request.status === 'approved') return { color: 'success', icon: <CheckCircle size={18} />, title: 'Approved ✓', text: 'Visitor request approved.' };
    if (request.status === 'rejected') return { color: 'critical', icon: <XCircle size={18} />, title: 'Rejected', text: 'Visitor request rejected by resident.' };
    if (request.status === 'expired') return { color: 'gray', icon: <Clock size={18} />, title: 'Expired', text: 'The request expired or was cancelled.' };
    return { color: 'warning', icon: <Clock size={18} />, title: 'Pending', text: 'Request sent to the resident. Waiting for resident response...' };
  };

  const view = statusView();
  const residentName = residents.find(r => r.id === residentId)?.full_name;

  return (
    <div className="modal-overlay" onClick={step === 'capture' || step === 'prompt' ? onClose : undefined}>
      <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 520 }}>
        <div className="modal-header">
          <h3 className="modal-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <UserPlus size={18} style={{ color: 'var(--color-info)' }} /> Visitor Verification
          </h3>
          {step !== 'sending' && (
            <button className="btn btn-ghost btn-sm" onClick={onClose} aria-label="Close">
              <X size={16} />
            </button>
          )}
        </div>

        {step === 'prompt' && (
          <>
            <div className="modal-body">
              <p style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>
                Face isn't registered. Would you like to send a request to the resident?
              </p>
              <p style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 8 }}>
                The visitor is not approved unless the resident accepts.
              </p>
            </div>
            <div className="modal-footer">
              <button className="btn btn-ghost" onClick={onClose}>Cancel</button>
              <button className="btn btn-primary" onClick={() => setStep(photo ? 'confirm' : 'capture')}>Send Request</button>
            </div>
          </>
        )}

        {step === 'confirm' && (
          <>
            <div className="modal-body">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div className="form-group" style={{ marginBottom: 12 }}>
                  <label className="form-label">Visiting resident</label>
                  <select className="select" value={residentId} onChange={e => setResidentId(e.target.value ? Number(e.target.value) : '')}>
                    <option value="">Select resident…</option>
                    {residents.map(r => (
                      <option key={r.id} value={r.id}>{r.full_name} — {r.apartment_number}</option>
                    ))}
                  </select>
                </div>
                <div className="form-group" style={{ marginBottom: 12 }}>
                  <label className="form-label">Visitor Name (optional)</label>
                  <input className="input" value={visitorName} maxLength={120} placeholder="e.g., Ravi Kumar" onChange={e => setVisitorName(e.target.value)} />
                </div>
                <div className="form-group" style={{ marginBottom: 12 }}>
                  <label className="form-label">Location</label>
                  <input className="input" value={location} onChange={e => setLocation(e.target.value)} />
                </div>
              </div>
              {photoUrl && (
                <img src={photoUrl} alt="Visitor photo to send" style={{ width: '100%', maxHeight: 320, objectFit: 'cover', borderRadius: 'var(--radius-md)', background: '#000' }} />
              )}
              <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: '8px 0 0' }}>
                This is the same photo used for face verification. It will be sent to the resident.
              </p>
              {error && (
                <div style={{ marginTop: 12, padding: '10px 14px', borderRadius: 8, fontSize: 13, background: 'var(--color-critical-bg)', border: '1px solid var(--color-critical-border)', color: 'var(--color-critical)' }}>
                  {error}
                </div>
              )}
            </div>
            <div className="modal-footer">
              <button className="btn btn-ghost" onClick={onClose} disabled={busy}>Cancel</button>
              <button className="btn btn-primary" onClick={handleSendExisting} disabled={busy}>{busy ? 'Sending…' : 'Send Request'}</button>
            </div>
          </>
        )}

        {(step === 'capture' || step === 'sending') && (
          <>
            <div className="modal-body">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div className="form-group" style={{ marginBottom: 12 }}>
                  <label className="form-label">Visiting resident</label>
                  <select className="select" value={residentId} onChange={e => setResidentId(e.target.value ? Number(e.target.value) : '')} disabled={step === 'sending'}>
                    <option value="">Select resident…</option>
                    {residents.map(r => (
                      <option key={r.id} value={r.id}>{r.full_name} — {r.apartment_number}</option>
                    ))}
                  </select>
                </div>
                <div className="form-group" style={{ marginBottom: 12 }}>
                  <label className="form-label">Visitor Name (optional)</label>
                  <input className="input" value={visitorName} maxLength={120} placeholder="e.g., Ravi Kumar" onChange={e => setVisitorName(e.target.value)} disabled={step === 'sending'} />
                </div>
                <div className="form-group" style={{ marginBottom: 12 }}>
                  <label className="form-label">Location</label>
                  <input className="input" value={location} onChange={e => setLocation(e.target.value)} disabled={step === 'sending'} />
                </div>
              </div>

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
                {camState === 'live' && (
                  <div style={{
                    position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
                    width: '46%', height: '72%', border: '2px dashed rgba(255,255,255,0.85)',
                    borderRadius: '50%', pointerEvents: 'none', boxShadow: '0 0 0 9999px rgba(0,0,0,0.28)',
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
                        ? "Camera access is required to capture the visitor's face."
                        : 'Unable to access the camera. Please check your camera connection.'}
                    </p>
                    <button className="btn btn-sm btn-ghost" style={{ marginTop: 10 }} onClick={startCamera}>
                      <RefreshCw size={13} /> Try Again
                    </button>
                  </div>
                )}
              </div>
              <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: '8px 0 0' }}>
                Position the visitor's face inside the guide, then capture.
              </p>

              {error && (
                <div style={{
                  marginTop: 12, padding: '10px 14px', borderRadius: 8, fontSize: 13,
                  background: 'var(--color-critical-bg)', border: '1px solid var(--color-critical-border)', color: 'var(--color-critical)',
                }}>
                  {error}
                </div>
              )}
            </div>
            <div className="modal-footer">
              <button className="btn btn-ghost" onClick={onClose} disabled={step === 'sending'}>Cancel</button>
              <button className="btn btn-primary" onClick={handleCapture} disabled={camState !== 'live' || busy || step === 'sending'}>
                {step === 'sending' ? 'Sending…' : busy ? 'Checking…' : error?.startsWith('No face') ? 'Try Again' : 'Capture'}
              </button>
            </div>
          </>
        )}

        {step === 'waiting' && view && (
          <>
            <div className="modal-body">
              <div style={{
                padding: 16, borderRadius: 'var(--radius-md)',
                background: `var(--color-${view.color}-bg)`, border: `1px solid var(--color-${view.color}-border)`,
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700, color: `var(--color-${view.color})` }}>
                  {view.icon} Status: {view.title}
                </div>
                <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginTop: 6 }}>{view.text}</div>
                {residentName && (
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 6 }}>Sent to {residentName}</div>
                )}
              </div>
              {request?.status === 'approved' && (
                <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 10 }}>
                  You may continue the normal entry flow for this visitor.
                </p>
              )}
            </div>
            <div className="modal-footer">
              {request?.status === 'pending' ? (
                <button className="btn btn-danger" onClick={handleCancelRequest}>Cancel Request</button>
              ) : (
                <button className="btn btn-primary" onClick={onClose}>Done</button>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default VisitorRequestFlow;
