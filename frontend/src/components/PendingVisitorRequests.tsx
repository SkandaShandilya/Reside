import React, { useEffect, useState } from 'react';
import { CheckCircle, XCircle, UserPlus, CarFront, Maximize2, X } from 'lucide-react';
import api from '../api/client';
import type { VisitorFaceRequest, VehicleRequest } from '../types';

// Pending visitor-verification requests addressed to the logged-in
// resident, with Accept / Reject. Polls so a new request appears without a
// reload. Renders nothing when there is nothing waiting.
const PendingVisitorRequests: React.FC = () => {
  const [requests, setRequests] = useState<VisitorFaceRequest[]>([]);
  const [vehicleRequests, setVehicleRequests] = useState<VehicleRequest[]>([]);
  const [busyId, setBusyId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const token = api.getToken() || '';
  const [enlarged, setEnlarged] = useState<string | null>(null);

  useEffect(() => {
    if (!enlarged) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setEnlarged(null); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [enlarged]);

  // Everything addressed to THIS resident (the API filters by the logged-in
  // resident's id), refreshed continuously so new requests just appear.
  const load = () => Promise.all([
    api.getResidentVisitorRequests()
      .then(res => setRequests(res.requests.filter(r => r.status === 'pending')))
      .catch(() => {}),
    api.getResidentVehicleRequests()
      .then(res => setVehicleRequests(res.requests.filter(r => r.status === 'pending')))
      .catch(() => {}),
  ]);

  useEffect(() => {
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);

  const respond = async (id: number, action: 'approve' | 'reject') => {
    setBusyId(id);
    setError(null);
    try {
      await api.respondVisitorRequest(id, action);
      await load();
    } catch (err: any) {
      setError(err.message || 'Unable to send your response. Please try again.');
    } finally {
      setBusyId(null);
    }
  };

  const respondVehicle = async (id: number, action: 'approve' | 'reject') => {
    setBusyId(-id);
    setError(null);
    try {
      await api.respondVehicleRequest(id, action);
      await load();
    } catch (err: any) {
      setError(err.message || 'Unable to send your response. Please try again.');
    } finally {
      setBusyId(null);
    }
  };

  if (requests.length === 0 && vehicleRequests.length === 0) return null;

  return (
    <div style={{ marginBottom: 20 }}>
      {error && (
        <div style={{ marginBottom: 12, padding: '10px 14px', borderRadius: 8, fontSize: 13, background: 'var(--color-critical-bg)', border: '1px solid var(--color-critical-border)', color: 'var(--color-critical)' }}>
          {error}
        </div>
      )}
      {vehicleRequests.map(req => (
        <div key={`v${req.id}`} className="card" style={{ marginBottom: 12, borderColor: 'var(--color-warning-border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700, color: 'var(--color-warning)' }}>
            <CarFront size={16} /> Visitor Vehicle Request
          </div>
          <div style={{ margin: '8px 0 4px', fontWeight: 700 }}>
            {req.visitor_name || 'Visitor'} · <span className="mono">{req.plate_number}</span>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            {req.purpose || 'No purpose given'} · Guard: {req.guard.full_name} · {new Date(req.created_at).toLocaleTimeString()}
          </div>
          <div style={{ fontWeight: 600, margin: '10px 0 8px' }}>Allow this vehicle?</div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn btn-success" style={{ flex: 1, justifyContent: 'center' }} disabled={busyId === -req.id} onClick={() => respondVehicle(req.id, 'approve')}>
              <CheckCircle size={14} /> Accept
            </button>
            <button className="btn btn-danger" style={{ flex: 1, justifyContent: 'center' }} disabled={busyId === -req.id} onClick={() => respondVehicle(req.id, 'reject')}>
              <XCircle size={14} /> Reject
            </button>
          </div>
        </div>
      ))}
      {requests.map(req => (
        <div key={req.id} className="card" style={{ marginBottom: 12, borderColor: 'var(--color-warning-border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700, color: 'var(--color-warning)' }}>
            <UserPlus size={16} /> Visitor Verification Request
          </div>
          <div style={{ fontSize: 13, color: 'var(--text-muted)', margin: '4px 0 10px' }}>{req.visitor_name ? <><strong>{req.visitor_name}</strong> is requesting access.</> : 'A visitor is requesting access.'}</div>
          <div style={{ position: 'relative' }}>
            <img
              src={`${req.image_url}?token=${token}`}
              alt="Captured visitor"
              onClick={() => setEnlarged(`${req.image_url}?token=${token}`)}
              style={{ width: '100%', maxHeight: 320, objectFit: 'cover', borderRadius: 'var(--radius-md)', background: '#000', cursor: 'zoom-in', display: 'block' }}
            />
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              title="Enlarge photo"
              onClick={() => setEnlarged(`${req.image_url}?token=${token}`)}
              style={{ position: 'absolute', top: 8, right: 8, background: 'rgba(0,0,0,0.6)', color: '#fff' }}
            >
              <Maximize2 size={14} /> Enlarge
            </button>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', margin: '10px 0' }}>
            Guard: {req.guard.full_name} · Location: {req.location} · {new Date(req.created_at).toLocaleTimeString()}
          </div>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>Allow this visitor?</div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn btn-success" style={{ flex: 1, justifyContent: 'center' }} disabled={busyId === req.id} onClick={() => respond(req.id, 'approve')}>
              <CheckCircle size={14} /> Accept
            </button>
            <button className="btn btn-danger" style={{ flex: 1, justifyContent: 'center' }} disabled={busyId === req.id} onClick={() => respond(req.id, 'reject')}>
              <XCircle size={14} /> Reject
            </button>
          </div>
        </div>
      ))}
      {enlarged && (
        <div
          onClick={() => setEnlarged(null)}
          style={{ position: 'fixed', inset: 0, zIndex: 3000, background: 'rgba(0,0,0,0.88)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16, cursor: 'zoom-out' }}
        >
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            aria-label="Close"
            onClick={() => setEnlarged(null)}
            style={{ position: 'absolute', top: 16, right: 16, color: '#fff' }}
          >
            <X size={20} />
          </button>
          <img
            src={enlarged}
            alt="Captured visitor enlarged"
            onClick={e => e.stopPropagation()}
            style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain', borderRadius: 'var(--radius-md)', cursor: 'default' }}
          />
        </div>
      )}
    </div>
  );
};

export default PendingVisitorRequests;
