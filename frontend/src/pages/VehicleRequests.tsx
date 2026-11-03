import React, { useEffect, useState } from 'react';
import { CarFront, CheckCircle, XCircle, Trash2 } from 'lucide-react';
import PageHeader from '../components/ui/PageHeader';
import api from '../api/client';
import type { Vehicle, VehicleRequest } from '../types';

const VehicleRequests: React.FC = () => {
  const [vehicles, setVehicles] = useState<Vehicle[]>([]);
  const [requests, setRequests] = useState<VehicleRequest[]>([]);
  const [loading, setLoading] = useState(true);

  const [plateNumber, setPlateNumber] = useState('');
  const [vehicleType, setVehicleType] = useState('car');
  const [registering, setRegistering] = useState(false);
  const [registerError, setRegisterError] = useState<string | null>(null);

  const load = () => {
    Promise.all([
      api.getMyVehicles().then(res => setVehicles(res.vehicles)).catch(() => {}),
      api.getResidentVehicleRequests().then(res => setRequests(res.requests)).catch(() => {}),
    ]).finally(() => setLoading(false));
  };

  useEffect(load, []);

  const handleRegisterVehicle = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!plateNumber.trim()) return;
    setRegistering(true);
    setRegisterError(null);
    try {
      await api.registerVehicle({ plate_number: plateNumber.trim(), vehicle_type: vehicleType });
      setPlateNumber('');
      load();
    } catch (err: any) {
      setRegisterError(err.message || 'Failed to register vehicle.');
    } finally {
      setRegistering(false);
    }
  };

  const handleDeleteVehicle = async (id: number) => {
    try {
      await api.deleteVehicle(id);
      setVehicles(prev => prev.filter(v => v.id !== id));
    } catch (err) { console.error(err); }
  };

  const handleRespond = async (id: number, action: 'approve' | 'reject') => {
    try {
      await api.respondVehicleRequest(id, action);
      load();
    } catch (err) { console.error(err); }
  };

  const pending = requests.filter(r => r.status === 'pending');
  const resolved = requests.filter(r => r.status !== 'pending');

  return (
    <div style={{ maxWidth: 640, margin: '0 auto' }}>
      <PageHeader title="My Vehicles & Visitor Requests" />

      {/* Register a vehicle */}
      <div className="card" style={{ marginBottom: 24 }}>
        <h3 className="section-title" style={{ marginBottom: 12 }}>
          <CarFront size={16} style={{ verticalAlign: 'middle', marginRight: 6 }} />Register a Vehicle
        </h3>
        <form onSubmit={handleRegisterVehicle} style={{ display: 'flex', gap: 8, alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div className="form-group" style={{ flex: 1, minWidth: 160 }}>
            <label className="form-label">Plate number</label>
            <input className="input" style={{ width: '100%' }} value={plateNumber} onChange={e => setPlateNumber(e.target.value.toUpperCase())} placeholder="KA01AB1234" />
          </div>
          <div className="form-group">
            <label className="form-label">Type</label>
            <select className="select" value={vehicleType} onChange={e => setVehicleType(e.target.value)}>
              <option value="car">Car</option>
              <option value="bike">Bike</option>
              <option value="other">Other</option>
            </select>
          </div>
          <button type="submit" className="btn btn-primary" disabled={registering}>
            {registering ? 'Adding...' : 'Add Vehicle'}
          </button>
        </form>
        {registerError && <div style={{ color: 'var(--color-critical)', fontSize: 12, marginTop: 8 }}>{registerError}</div>}

        {vehicles.length > 0 && (
          <div style={{ marginTop: 16 }}>
            {vehicles.map(v => (
              <div key={v.id} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 0', borderTop: '1px solid var(--border-default)' }}>
                <div>
                  <span className="mono" style={{ fontWeight: 700 }}>{v.plate_number}</span>
                  <span style={{ fontSize: 12, color: 'var(--text-muted)', marginLeft: 8 }}>{v.vehicle_type}</span>
                </div>
                <button className="btn btn-ghost btn-sm" onClick={() => handleDeleteVehicle(v.id)} title="Remove vehicle">
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Pending visitor requests */}
      <h3 className="section-title" style={{ marginBottom: 12 }}>Pending Visitor Approvals</h3>
      {pending.length === 0 && !loading && (
        <div className="empty-state">
          <div className="icon">✅</div>
          <p>No pending visitor requests.</p>
        </div>
      )}
      {pending.map(req => (
        <div key={req.id} className="card" style={{ marginBottom: 12 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', gap: 12 }}>
            <div>
              <div style={{ fontWeight: 700 }}>{req.visitor_name || 'Visitor'} · <span className="mono">{req.plate_number}</span></div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>{req.purpose || 'No purpose given'}</div>
              <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 4 }}>Requested by guard {req.guard.full_name} · {new Date(req.created_at).toLocaleString()}</div>
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              <button className="btn btn-primary btn-sm" onClick={() => handleRespond(req.id, 'approve')}>
                <CheckCircle size={14} /> Approve
              </button>
              <button className="btn btn-danger btn-sm" onClick={() => handleRespond(req.id, 'reject')}>
                <XCircle size={14} /> Reject
              </button>
            </div>
          </div>
        </div>
      ))}

      {resolved.length > 0 && (
        <>
          <h3 className="section-title" style={{ marginTop: 24, marginBottom: 12 }}>Past Requests</h3>
          {resolved.map(req => (
            <div key={req.id} className="card" style={{ marginBottom: 12, opacity: 0.75 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
                <div>
                  <div style={{ fontWeight: 700 }}>{req.visitor_name || 'Visitor'} · <span className="mono">{req.plate_number}</span></div>
                  <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 2 }}>{req.responded_at ? new Date(req.responded_at).toLocaleString() : ''}</div>
                </div>
                <span className={`status-badge ${req.status}`}>{req.status}</span>
              </div>
            </div>
          ))}
        </>
      )}
    </div>
  );
};

export default VehicleRequests;
