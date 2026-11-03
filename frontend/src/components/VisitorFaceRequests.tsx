import React, { useEffect, useState } from 'react';
import { CheckCircle, XCircle } from 'lucide-react';
import api from '../api/client';
import type { VisitorFaceRequest } from '../types';

// Visitor face-verification requests addressed to this resident
// (pending with Accept / Reject, plus recent history). Polls so a new
// request appears without a reload.
const VisitorFaceRequests: React.FC = () => {
  const [faceRequests, setFaceRequests] = useState<VisitorFaceRequest[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = () => api.getResidentVisitorRequests()
      .then(res => setFaceRequests(res.requests))
      .catch(() => {})
      .finally(() => setLoading(false));
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);

  const handleRespondFace = async (id: number, action: 'approve' | 'reject') => {
    try {
      await api.respondVisitorRequest(id, action);
      const res = await api.getResidentVisitorRequests();
      setFaceRequests(res.requests);
    } catch (err) { console.error(err); }
  };

  const token = api.getToken() || '';
  const pendingFace = faceRequests.filter(r => r.status === 'pending');
  const resolvedFace = faceRequests.filter(r => r.status !== 'pending');

  return (
    <div style={{ marginTop: 24 }}>
      <h3 className="section-title" style={{ marginBottom: 12 }}>Visitor Verification Requests</h3>
      {pendingFace.length === 0 && !loading && (
        <div className="empty-state" style={{ marginBottom: 24 }}>
          <p>No visitors waiting for your approval.</p>
        </div>
      )}
      {pendingFace.map(req => (
        <div key={req.id} className="card" style={{ marginBottom: 12 }}>
          <div style={{ fontWeight: 700, marginBottom: 2 }}>Visitor Request</div>
          <div style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 10 }}>{req.visitor_name ? <><strong>{req.visitor_name}</strong> is requesting access.</> : 'A visitor is requesting access.'}</div>
          <img
            src={`${req.image_url}?token=${token}`}
            alt="Captured visitor"
            style={{ width: '100%', maxHeight: 320, objectFit: 'cover', borderRadius: 'var(--radius-md)', background: '#000' }}
          />
          <div style={{ fontSize: 12, color: 'var(--text-muted)', margin: '10px 0' }}>
            Guard: {req.guard.full_name} · Location: {req.location} · {new Date(req.created_at).toLocaleTimeString()}
          </div>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>Allow this visitor?</div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn btn-success" style={{ flex: 1, justifyContent: 'center' }} onClick={() => handleRespondFace(req.id, 'approve')}>
              <CheckCircle size={14} /> Accept
            </button>
            <button className="btn btn-danger" style={{ flex: 1, justifyContent: 'center' }} onClick={() => handleRespondFace(req.id, 'reject')}>
              <XCircle size={14} /> Reject
            </button>
          </div>
        </div>
      ))}
      {resolvedFace.slice(0, 5).map(req => (
        <div key={req.id} className="card" style={{ marginBottom: 12, opacity: 0.75, display: 'flex', alignItems: 'center', gap: 12 }}>
          <img src={`${req.image_url}?token=${token}`} alt="Visitor" style={{ width: 48, height: 48, objectFit: 'cover', borderRadius: 8 }} />
          <div style={{ flex: 1, fontSize: 12, color: 'var(--text-muted)' }}>
            {req.location} · {new Date(req.created_at).toLocaleString()}
          </div>
          <span className={`status-badge ${req.status === 'approved' ? 'resolved' : req.status === 'used' ? 'used' : req.status === 'rejected' || req.status === 'expired' ? 'unresolved' : ''}`}>{req.status}</span>
        </div>
      ))}

    </div>
  );
};

export default VisitorFaceRequests;
