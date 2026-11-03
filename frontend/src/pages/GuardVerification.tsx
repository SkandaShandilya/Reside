import React, { useEffect, useState } from 'react';
import { ScanFace, CarFront, CheckCircle, XCircle, AlertTriangle, Camera } from 'lucide-react';
import PageHeader from '../components/ui/PageHeader';
import api from '../api/client';
import { useAuth } from '../contexts/AuthContext';
import AdminResidentDirectory from './AdminResidentDirectory';
import VisitorRequestFlow from '../components/VisitorRequestFlow';
import CameraCaptureModal from '../components/CameraCaptureModal';
import type { FaceVerifyResult, PlateVerifyResult, ResidentSummary } from '../types';

// Admin gets the Resident & Visitor Directory here instead of the
// Face/Plate scanning tools — those stay guard-only, completely
// unchanged, in GuardVerificationTools below. Kept as one route
// (rather than a second duplicate page) since the split is purely
// presentational and the route is already isGuard-gated (covers admin
// too) in App.tsx. The role check lives in this outer wrapper, not in
// the middle of GuardVerificationTools's own hooks, so neither
// component ever calls hooks conditionally.
const GuardVerification: React.FC = () => {
  const { isAdmin } = useAuth();
  return isAdmin ? <AdminResidentDirectory /> : <GuardVerificationTools />;
};

const GuardVerificationTools: React.FC = () => {
  const [faceFile, setFaceFile] = useState<File | null>(null);
  // Untouched camera frame, kept for visitor requests (faceFile may be the oval-only crop).
  const [faceFullFrame, setFaceFullFrame] = useState<File | null>(null);
  const [faceResult, setFaceResult] = useState<FaceVerifyResult | null>(null);
  const [faceBusy, setFaceBusy] = useState(false);
  const [showVisitorFlow, setShowVisitorFlow] = useState(false);
  const [cameraFor, setCameraFor] = useState<'face' | 'plate' | null>(null);
  const [faceError, setFaceError] = useState<string | null>(null);

  const [plateFile, setPlateFile] = useState<File | null>(null);
  const [plateResult, setPlateResult] = useState<PlateVerifyResult | null>(null);
  const [plateBusy, setPlateBusy] = useState(false);
  const [plateError, setPlateError] = useState<string | null>(null);

  const [residents, setResidents] = useState<ResidentSummary[]>([]);
  const [requestResidentId, setRequestResidentId] = useState<number | ''>('');
  const [visitorName, setVisitorName] = useState('');
  const [purpose, setPurpose] = useState('');
  const [requestSubmitting, setRequestSubmitting] = useState(false);
  const [requestSent, setRequestSent] = useState(false);

  useEffect(() => {
    api.getActiveResidents().then(res => setResidents(res.residents)).catch(() => {});
  }, []);

  const handleVerifyFace = async (fileOverride?: File) => {
    const file = fileOverride ?? faceFile;
    if (!file) return;
    setFaceBusy(true);
    setFaceError(null);
    setFaceResult(null);
    try {
      setFaceResult(await api.verifyFace(file));
    } catch (err: any) {
      setFaceError(err.message || 'Face verification failed.');
    } finally {
      setFaceBusy(false);
    }
  };

  const handleVerifyPlate = async (fileOverride?: File) => {
    const file = fileOverride ?? plateFile;
    if (!file) return;
    setPlateBusy(true);
    setPlateError(null);
    setPlateResult(null);
    setRequestSent(false);
    try {
      setPlateResult(await api.verifyPlate(file));
    } catch (err: any) {
      setPlateError(err.message || 'Plate verification failed.');
    } finally {
      setPlateBusy(false);
    }
  };

  const handleCreateRequest = async () => {
    if (!plateResult?.detected_number || !requestResidentId) return;
    setRequestSubmitting(true);
    try {
      await api.createVehicleRequest({
        plate_number: plateResult.detected_number,
        resident_id: Number(requestResidentId),
        visitor_name: visitorName,
        purpose,
      });
      setRequestSent(true);
    } catch (err: any) {
      setPlateError(err.message || 'Failed to create visitor request.');
    } finally {
      setRequestSubmitting(false);
    }
  };

  return (
    <div>
      <PageHeader
        title="Guard Verification"
        description="Verify a visitor's face against registered residents, or scan a vehicle plate for entry."
      />

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        {/* Face verification */}
        <div className="card">
          <div className="section-header" style={{ marginBottom: 12 }}>
            <h3 className="section-title"><ScanFace size={16} style={{ verticalAlign: 'middle', marginRight: 6 }} />Face Verification</h3>
          </div>
          <button className="btn btn-secondary" style={{ width: '100%', justifyContent: 'center', marginBottom: 10 }} onClick={() => setCameraFor('face')}>
            <Camera size={15} /> Use Camera
          </button>
          <input
            type="file"
            accept="image/*"
            capture="environment"
            className="input"
            style={{ width: '100%', marginBottom: 12 }}
            onChange={e => { setFaceFile(e.target.files?.[0] || null); setFaceFullFrame(null); setFaceResult(null); setFaceError(null); }}
          />
          <button className="btn btn-primary" style={{ width: '100%', justifyContent: 'center' }} disabled={!faceFile || faceBusy} onClick={() => handleVerifyFace()}>
            {faceBusy ? 'Verifying...' : 'Verify Face'}
          </button>

          {faceError && <div style={{ color: 'var(--color-critical)', fontSize: 12, marginTop: 12 }}>{faceError}</div>}

          {faceResult && (
            <div style={{
              marginTop: 16, padding: 12, borderRadius: 'var(--radius-md)',
              border: `1px solid ${faceResult.matched ? 'var(--color-success-border)' : 'var(--color-critical-border, var(--color-warning-border))'}`,
              background: faceResult.matched ? 'var(--color-success-bg)' : 'var(--color-warning-bg)',
            }}>
              {faceResult.matched ? (
                <>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700, color: 'var(--color-success)' }}>
                    <CheckCircle size={16} /> Match: {faceResult.person?.full_name}
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
                    {faceResult.person?.relationship && (
                      <>Relationship: <span style={{ textTransform: 'capitalize' }}>{faceResult.person.relationship}</span><br /></>
                    )}
                    Apartment {faceResult.resident?.apartment_number} · Resident: {faceResult.resident?.full_name}
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
                    Match: {(faceResult.similarity * 100).toFixed(0)}%
                  </div>
                </>
              ) : (
                <>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700, color: 'var(--color-warning)' }}>
                    <XCircle size={16} /> No matching household member ({(faceResult.similarity * 100).toFixed(0)}% best similarity)
                  </div>
                  <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginTop: 8 }}>
                    {faceResult.message && faceResult.message !== 'No registered household member matched.'
                      ? faceResult.message
                      : "Face isn't registered. Would you like to send a request to the resident?"}
                  </div>
                  <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
                    <button className="btn btn-primary btn-sm" onClick={() => setShowVisitorFlow(true)}>Send Request</button>
                    <button className="btn btn-ghost btn-sm" onClick={() => setFaceResult(null)}>Cancel</button>
                  </div>
                </>
              )}
            </div>
          )}
        </div>

        {/* Plate verification */}
        <div className="card">
          <div className="section-header" style={{ marginBottom: 12 }}>
            <h3 className="section-title"><CarFront size={16} style={{ verticalAlign: 'middle', marginRight: 6 }} />Vehicle Plate Scan</h3>
          </div>
          <button className="btn btn-secondary" style={{ width: '100%', justifyContent: 'center', marginBottom: 10 }} onClick={() => setCameraFor('plate')}>
            <Camera size={15} /> Use Camera
          </button>
          <input
            type="file"
            accept="image/*"
            capture="environment"
            className="input"
            style={{ width: '100%', marginBottom: 12 }}
            onChange={e => { setPlateFile(e.target.files?.[0] || null); setPlateResult(null); setPlateError(null); setRequestSent(false); }}
          />
          <button className="btn btn-primary" style={{ width: '100%', justifyContent: 'center' }} disabled={!plateFile || plateBusy} onClick={() => handleVerifyPlate()}>
            {plateBusy ? 'Scanning...' : 'Scan Plate'}
          </button>

          {plateError && <div style={{ color: 'var(--color-critical)', fontSize: 12, marginTop: 12 }}>{plateError}</div>}

          {plateResult && (
            <div style={{ marginTop: 16 }}>
              {plateResult.entry_status === 'ALLOWED' && (
                <div style={{ padding: 12, borderRadius: 'var(--radius-md)', border: '1px solid var(--color-success-border)', background: 'var(--color-success-bg)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700, color: 'var(--color-success)' }}>
                    <CheckCircle size={16} /> ALLOWED — {plateResult.detected_number}
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 6 }}>
                    <span
                      style={{
                        fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 0.3,
                        padding: '2px 8px', borderRadius: 999,
                        background: plateResult.vehicle?.source === 'visitor' ? 'var(--color-warning-bg)' : 'var(--color-info-bg, var(--color-success-bg))',
                        color: plateResult.vehicle?.source === 'visitor' ? 'var(--color-warning)' : 'var(--color-success)',
                      }}
                    >
                      {plateResult.vehicle?.source === 'visitor' ? 'Visitor Parking' : 'Resident Parking'}
                    </span>
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
                    {plateResult.vehicle?.owner_name} · Apt {plateResult.vehicle?.apartment_number}
                  </div>
                </div>
              )}

              {plateResult.entry_status === 'OCR_FAILED' && (
                <div style={{ padding: 12, borderRadius: 'var(--radius-md)', border: '1px solid var(--color-warning-border)', background: 'var(--color-warning-bg)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700, color: 'var(--color-warning)' }}>
                    <AlertTriangle size={16} /> {plateResult.message}
                  </div>
                </div>
              )}

              {plateResult.entry_status === 'REQUIRES_APPROVAL' && (
                <div style={{ padding: 12, borderRadius: 'var(--radius-md)', border: '1px solid var(--color-warning-border)', background: 'var(--color-warning-bg)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700, color: 'var(--color-warning)', marginBottom: 4 }}>
                    <AlertTriangle size={16} /> Not registered — {plateResult.detected_number}
                  </div>
                  {plateResult.message && (
                    <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 10 }}>{plateResult.message}</div>
                  )}

                  {requestSent ? (
                    <div style={{ fontSize: 13, color: 'var(--color-success)' }}>Approval request sent to resident.</div>
                  ) : (
                    <>
                      <div className="form-group" style={{ marginBottom: 8 }}>
                        <label className="form-label">Visiting resident</label>
                        <select className="select" value={requestResidentId} onChange={e => setRequestResidentId(e.target.value ? Number(e.target.value) : '')}>
                          <option value="">Select resident…</option>
                          {residents.map(r => (
                            <option key={r.id} value={r.id}>{r.full_name} — {r.apartment_number}</option>
                          ))}
                        </select>
                      </div>
                      <div className="form-group" style={{ marginBottom: 8 }}>
                        <label className="form-label">Visitor name</label>
                        <input className="input" style={{ width: '100%' }} value={visitorName} onChange={e => setVisitorName(e.target.value)} />
                      </div>
                      <div className="form-group" style={{ marginBottom: 8 }}>
                        <label className="form-label">Purpose</label>
                        <input className="input" style={{ width: '100%' }} value={purpose} onChange={e => setPurpose(e.target.value)} placeholder="e.g., Delivery" />
                      </div>
                      <button
                        className="btn btn-warning btn-sm"
                        style={{ width: '100%', justifyContent: 'center' }}
                        disabled={!requestResidentId || requestSubmitting}
                        onClick={handleCreateRequest}
                      >
                        {requestSubmitting ? 'Sending...' : 'Request Resident Approval'}
                      </button>
                    </>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      {cameraFor && (
        <CameraCaptureModal
          title={cameraFor === 'face' ? 'Capture Face' : 'Capture Number Plate'}
          hint={cameraFor === 'face' ? "Position the visitor's face inside the oval, then capture — only what is inside the oval is verified." : 'Fit the number plate inside the box, then capture.'}
          facingMode={cameraFor === 'face' ? 'user' : 'environment'}
          guide={cameraFor === 'face' ? 'face' : 'plate'}
          onClose={() => setCameraFor(null)}
          onCapture={(file, fullFrame) => {
            const target = cameraFor;
            setCameraFor(null);
            if (target === 'face') { setFaceFile(file); setFaceFullFrame(fullFrame); handleVerifyFace(file); }
            else { setPlateFile(file); handleVerifyPlate(file); }
          }}
        />
      )}
      {showVisitorFlow && <VisitorRequestFlow photo={faceFullFrame ?? faceFile} onClose={() => setShowVisitorFlow(false)} />}
    </div>
  );
};

export default GuardVerification;
