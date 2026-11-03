import React, { useEffect, useRef, useState } from 'react';
import { ScanFace, CheckCircle, Camera as CameraIcon, Image as ImageIcon, X, Users, Plus, Trash2 } from 'lucide-react';
import api from '../api/client';
import VisitorFaceRequests from '../components/VisitorFaceRequests';
import type { RegisteredPerson } from '../types';

const MAX_PHOTO_BYTES = 8 * 1024 * 1024; // 8MB — generous for a phone-camera face photo, small enough to reject anything obviously wrong

function validatePhoto(file: Blob): string | null {
  if (!file || file.size === 0) return 'Please provide a face photo.';
  if (!file.type.startsWith('image/')) return 'The selected file is not a supported image type.';
  if (file.size > MAX_PHOTO_BYTES) return 'That image is too large. Please use a photo under 8MB.';
  return null;
}

function cameraErrorMessage(err: any): string {
  const name = err?.name;
  if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
    return 'Camera access was denied. You can allow camera access in your browser settings or choose a photo from your device.';
  }
  if (name === 'NotFoundError' || name === 'DevicesNotFoundError' || name === 'OverconstrainedError') {
    return 'No camera was detected. Please choose a photo from your device.';
  }
  return 'Unable to access the camera. Please try again or choose a photo from your device.';
}

/** In-browser live camera capture — opens getUserMedia, lets the resident
 * capture a frame, review it, retake, or use it. The MediaStream is
 * always stopped: right after capture (so it's never left running while
 * the resident reviews the shot), on retake-restart, and on close/unmount. */
const CameraCaptureModal: React.FC<{ onUsePhoto: (blob: Blob) => void; onClose: () => void }> = ({ onUsePhoto, onClose }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  // Bumped on every startStream() call (mount, retake, StrictMode's
  // dev-only double-invoke). An in-flight attempt checks its own
  // snapshot against the current value before touching state/refs, so
  // an abandoned attempt (superseded by a retake, or unmounted before
  // getUserMedia/play() resolves) can never clobber a newer one or
  // surface a stray AbortError from a video element that's already
  // gone — without this guard, StrictMode's mount→cleanup→mount in dev
  // reliably aborts the first play() call and falsely reports "unable
  // to access the camera" even though the camera access itself succeeded.
  const requestIdRef = useRef(0);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(true);
  const [capturedUrl, setCapturedUrl] = useState<string | null>(null);
  const [capturedBlob, setCapturedBlob] = useState<Blob | null>(null);

  const stopStream = () => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
  };

  const startStream = async () => {
    const myRequestId = ++requestIdRef.current;
    setError(null);
    setStarting(true);
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera is not supported in this browser. Please choose a photo from your device.');
      setStarting(false);
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'user' } },
        audio: false,
      });
      if (myRequestId !== requestIdRef.current) {
        // Superseded (retaken, or the effect that started this was
        // already cleaned up) — don't touch a stale video element,
        // just release the camera we were granted.
        stream.getTracks().forEach(t => t.stop());
        return;
      }
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        try {
          await videoRef.current.play();
        } catch (playErr: any) {
          if (myRequestId !== requestIdRef.current) return; // superseded mid-play(), not a real error
          throw playErr;
        }
      }
      if (myRequestId === requestIdRef.current) setStarting(false);
    } catch (err: any) {
      if (myRequestId !== requestIdRef.current) return;
      setError(cameraErrorMessage(err));
      setStarting(false);
    }
  };

  useEffect(() => {
    startStream();
    // Belt-and-suspenders: stop on unmount no matter how the modal closes.
    return () => { requestIdRef.current++; stopStream(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleCapture = () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    stopStream(); // never keep the camera running once a frame is captured
    canvas.toBlob(blob => {
      if (!blob) return;
      setCapturedBlob(blob);
      setCapturedUrl(URL.createObjectURL(blob));
    }, 'image/jpeg', 0.92);
  };

  const handleRetake = () => {
    setCapturedUrl(null);
    setCapturedBlob(null);
    startStream();
  };

  const handleClose = () => {
    stopStream();
    onClose();
  };

  return (
    <div className="modal-overlay" onClick={handleClose}>
      <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 480 }}>
        <div className="modal-header">
          <h3 className="modal-title">Live Camera</h3>
          <button className="btn btn-ghost btn-sm" onClick={handleClose}><X size={16} /></button>
        </div>
        <div className="modal-body" style={{ textAlign: 'center' }}>
          {error ? (
            <div style={{ padding: '20px 8px' }}>
              <div style={{ color: 'var(--color-critical)', fontSize: 13, marginBottom: 16 }}>{error}</div>
              <button className="btn btn-ghost" onClick={handleClose}>Close</button>
            </div>
          ) : capturedUrl ? (
            <>
              <img
                src={capturedUrl}
                alt="Captured face"
                style={{ width: '100%', maxHeight: 320, objectFit: 'contain', borderRadius: 'var(--radius-md)', background: '#000' }}
              />
              <div style={{ display: 'flex', gap: 8, marginTop: 16, justifyContent: 'center' }}>
                <button className="btn btn-ghost" onClick={handleRetake}>Retake</button>
                <button className="btn btn-primary" onClick={() => capturedBlob && onUsePhoto(capturedBlob)}>Use This Photo</button>
              </div>
            </>
          ) : (
            <>
              <video
                ref={videoRef}
                muted
                playsInline
                autoPlay
                style={{ width: '100%', maxHeight: 320, borderRadius: 'var(--radius-md)', background: '#000', transform: 'scaleX(-1)' }}
              />
              {starting && <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8 }}>Starting camera…</div>}
              <div style={{ display: 'flex', gap: 8, marginTop: 16, justifyContent: 'center' }}>
                <button className="btn btn-ghost" onClick={handleClose}>Cancel</button>
                <button className="btn btn-primary" disabled={starting} onClick={handleCapture}>Capture Photo</button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

const RELATIONSHIP_LABELS: Record<string, string> = {
  self: 'Self', spouse: 'Spouse', child: 'Child', parent: 'Parent', other: 'Other',
};

const FaceRegistration: React.FC = () => {
  const [people, setPeople] = useState<RegisteredPerson[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [removeTarget, setRemoveTarget] = useState<RegisteredPerson | null>(null);
  const [removing, setRemoving] = useState(false);

  const [fullName, setFullName] = useState('');
  const [phone, setPhone] = useState('');
  const [relationship, setRelationship] = useState('self');

  const [photoBlob, setPhotoBlob] = useState<Blob | null>(null);
  const [photoPreview, setPhotoPreview] = useState<string | null>(null);
  const [showCamera, setShowCamera] = useState(false);

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadHousehold = () => {
    api.getHousehold()
      .then(res => setPeople(res.people))
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(loadHousehold, []);

  useEffect(() => {
    return () => { if (photoPreview) URL.revokeObjectURL(photoPreview); };
  }, [photoPreview]);

  const setPhoto = (blob: Blob) => {
    if (photoPreview) URL.revokeObjectURL(photoPreview);
    setPhotoBlob(blob);
    setPhotoPreview(URL.createObjectURL(blob));
    setError(null);
  };

  const openForm = () => {
    setFullName('');
    setPhone('');
    setRelationship(people.length === 0 ? 'self' : 'other');
    setPhotoBlob(null);
    setPhotoPreview(null);
    setError(null);
    setShowForm(true);
  };

  const closeForm = () => {
    setShowForm(false);
    if (photoPreview) URL.revokeObjectURL(photoPreview);
    setPhotoBlob(null);
    setPhotoPreview(null);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    e.target.value = ''; // allow reselecting the same file later
    if (!f) return;
    const validationError = validatePhoto(f);
    if (validationError) {
      setError(validationError);
      return;
    }
    setPhoto(f);
  };

  const handleCameraCapture = (blob: Blob) => {
    const validationError = validatePhoto(blob);
    if (validationError) {
      setError(validationError);
      setShowCamera(false);
      return;
    }
    setPhoto(blob);
    setShowCamera(false);
  };

  const handleSubmit = async () => {
    if (!photoBlob) {
      setError('Please provide a face photo.');
      return;
    }
    const photoError = validatePhoto(photoBlob);
    if (photoError) {
      setError(photoError);
      return;
    }
    if (!fullName.trim()) {
      setError('Full name is required.');
      return;
    }
    if (!phone.trim()) {
      setError('Phone number is required.');
      return;
    }

    setSubmitting(true);
    setError(null);
    try {
      const res = await api.registerPerson(photoBlob, fullName.trim(), phone.trim(), relationship);
      setPeople(prev => [...prev, res.person]);
      closeForm();
    } catch (err: any) {
      setError(err.message || 'Registration failed.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleRemove = async () => {
    if (!removeTarget) return;
    setRemoving(true);
    try {
      await api.deleteHouseholdMember(removeTarget.id);
      setPeople(prev => prev.filter(p => p.id !== removeTarget.id));
      setRemoveTarget(null);
    } catch (err: any) {
      setError(err.message || 'Failed to remove household member.');
      setRemoveTarget(null);
    } finally {
      setRemoving(false);
    }
  };

  return (
    <div style={{ maxWidth: 560, margin: '0 auto' }}>
      <div style={{ textAlign: 'center', marginBottom: 24 }}>
        <div style={{
          width: 60, height: 60, background: 'var(--accent-gradient)',
          borderRadius: 'var(--radius-lg)', display: 'inline-flex',
          alignItems: 'center', justifyContent: 'center', marginBottom: 16,
        }}>
          <ScanFace size={28} color="white" />
        </div>
        <h1 className="page-title" style={{ marginBottom: 4 }}>Face & Family Registration</h1>
        <p style={{ color: 'var(--text-muted)', fontSize: 14 }}>
          Register yourself and family members living in your apartment for faster identity verification.
        </p>
      </div>

      <div className="card" style={{ marginBottom: 16, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontWeight: 700, color: 'var(--text-primary)' }}>
          Registered People: {loading ? '…' : people.length}
        </span>
        <Users size={18} style={{ color: 'var(--text-muted)' }} />
      </div>

      {people.map(person => (
        <div key={person.id} className="card" style={{ marginBottom: 12 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', gap: 12 }}>
            <div>
              <div style={{ fontWeight: 700, color: 'var(--text-primary)' }}>{person.full_name}</div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
                {RELATIONSHIP_LABELS[person.relationship] || person.relationship}
                {person.phone && ` · ${person.phone}`}
              </div>
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 12, fontWeight: 600, color: 'var(--color-success)', marginTop: 6 }}>
                <CheckCircle size={13} /> Face Registered
              </div>
            </div>
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              title={`Remove ${person.full_name}`}
              onClick={() => setRemoveTarget(person)}
            >
              <Trash2 size={14} />
            </button>
          </div>
        </div>
      ))}

      {!loading && people.length === 0 && (
        <div className="empty-state" style={{ marginBottom: 16 }}>
          <div className="icon"><Users size={28} /></div>
          <h3>No one registered yet</h3>
          <p>Register yourself first, then add family members living with you.</p>
        </div>
      )}

      {!showForm && (
        <button className="btn btn-primary" style={{ width: '100%', justifyContent: 'center' }} onClick={openForm}>
          <Plus size={16} /> Register Another Person
        </button>
      )}

      {showForm && (
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <h3 className="section-title" style={{ margin: 0 }}>Register Family Member</h3>
            <button type="button" className="btn btn-ghost btn-sm" onClick={closeForm}><X size={16} /></button>
          </div>

          <div className="form-group">
            <label className="form-label">Full Name</label>
            <input
              className="input"
              style={{ width: '100%' }}
              value={fullName}
              onChange={e => setFullName(e.target.value)}
              placeholder="Full name"
              required
            />
          </div>

          <div className="form-group" style={{ marginTop: 12 }}>
            <label className="form-label">Phone Number</label>
            <input
              className="input"
              style={{ width: '100%' }}
              type="tel"
              value={phone}
              onChange={e => setPhone(e.target.value)}
              placeholder="e.g. +91 98765 43210"
              required
            />
          </div>

          <div className="form-group" style={{ marginTop: 12 }}>
            <label className="form-label">Relationship</label>
            <select className="select" value={relationship} onChange={e => setRelationship(e.target.value)}>
              <option value="self">Self</option>
              <option value="spouse">Spouse</option>
              <option value="child">Child</option>
              <option value="parent">Parent</option>
              <option value="other">Other</option>
            </select>
          </div>

          <div className="form-group" style={{ marginTop: 16 }}>
            <label className="form-label">Face Photo</label>

            <div style={{
              border: '1px dashed var(--border-default)', borderRadius: 'var(--radius-md)',
              padding: photoPreview ? 8 : 24, textAlign: 'center', marginBottom: 12,
              background: 'var(--bg-tertiary)',
            }}>
              {photoPreview ? (
                <img src={photoPreview} alt="Face preview" style={{ maxWidth: '100%', maxHeight: 260, borderRadius: 'var(--radius-sm)' }} />
              ) : (
                <span style={{ color: 'var(--text-dim)', fontSize: 13 }}>No photo yet</span>
              )}
            </div>

            <div style={{ display: 'flex', gap: 8 }}>
              <button type="button" className="btn btn-ghost" style={{ flex: 1, justifyContent: 'center' }} onClick={() => setShowCamera(true)}>
                <CameraIcon size={16} /> Use Camera
              </button>
              <button type="button" className="btn btn-ghost" style={{ flex: 1, justifyContent: 'center' }} onClick={() => fileInputRef.current?.click()}>
                <ImageIcon size={16} /> Choose Photo
              </button>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />

            {photoPreview && (
              <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
                <button type="button" className="btn btn-ghost btn-sm" style={{ flex: 1, justifyContent: 'center' }} onClick={() => setShowCamera(true)}>
                  Retake
                </button>
                <button type="button" className="btn btn-ghost btn-sm" style={{ flex: 1, justifyContent: 'center' }} onClick={() => fileInputRef.current?.click()}>
                  Change Photo
                </button>
              </div>
            )}

            <p style={{ fontSize: 12, color: 'var(--text-dim)', marginTop: 8 }}>
              Exactly one face must be visible. Well-lit, front-facing photos work best.
            </p>
          </div>

          {error && <div style={{ color: 'var(--color-critical)', fontSize: 12, marginBottom: 12 }}>{error}</div>}

          <button
            className="btn btn-primary"
            style={{ width: '100%', justifyContent: 'center' }}
            disabled={!photoBlob || submitting}
            onClick={handleSubmit}
          >
            {submitting ? 'Registering...' : 'Register Person'}
          </button>
        </div>
      )}

      <VisitorFaceRequests />

      {showCamera && (
        <CameraCaptureModal
          onUsePhoto={handleCameraCapture}
          onClose={() => setShowCamera(false)}
        />
      )}

      {removeTarget && (
        <div className="modal-overlay" onClick={() => !removing && setRemoveTarget(null)}>
          <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 420 }}>
            <div className="modal-header">
              <h3 className="modal-title">Remove Household Member</h3>
              <button className="btn btn-ghost btn-sm" onClick={() => setRemoveTarget(null)} disabled={removing}><X size={16} /></button>
            </div>
            <div className="modal-body">
              <p style={{ fontSize: 14, color: 'var(--text-secondary)' }}>
                Remove {removeTarget.full_name} from registered household members?
              </p>
            </div>
            <div className="modal-footer">
              <button className="btn btn-ghost" onClick={() => setRemoveTarget(null)} disabled={removing}>Cancel</button>
              <button className="btn btn-danger" onClick={handleRemove} disabled={removing}>
                {removing ? 'Removing...' : 'Remove'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FaceRegistration;
