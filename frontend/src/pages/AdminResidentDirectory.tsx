import React, { useEffect, useMemo, useState } from 'react';
import { Users, Search, CarFront, UserCheck, ScanFace } from 'lucide-react';
import PageHeader from '../components/ui/PageHeader';
import api from '../api/client';
import type { AdminResidentSummary } from '../types';

const RELATIONSHIP_LABELS: Record<string, string> = {
  self: 'Self', spouse: 'Spouse', child: 'Child', parent: 'Parent', other: 'Other',
};

const AdminResidentDirectory: React.FC = () => {
  const [residents, setResidents] = useState<AdminResidentSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState('');

  useEffect(() => {
    api.getResidentDirectory()
      .then(res => setResidents(res.residents))
      .catch(err => setError(err.message || 'Failed to load resident directory.'))
      .finally(() => setLoading(false));
  }, []);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return residents;
    return residents.filter(r =>
      r.full_name.toLowerCase().includes(q) ||
      r.email.toLowerCase().includes(q) ||
      (r.apartment_number || '').toLowerCase().includes(q) ||
      r.household_members.some(m => m.full_name.toLowerCase().includes(q)) ||
      r.vehicles.some(v => v.plate_number.toLowerCase().includes(q)) ||
      r.visitors.some(v => v.visitor_name.toLowerCase().includes(q) || v.plate_number.toLowerCase().includes(q))
    );
  }, [residents, query]);

  return (
    <div>
      <PageHeader
        title="Resident & Visitor Directory"
        description="View registered residents, their vehicles, and saved visitor information."
      />

      <div className="card" style={{ marginBottom: 20, padding: '10px 14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <Search size={16} style={{ color: 'var(--text-muted)' }} />
          <input
            className="input"
            style={{ width: '100%', border: 'none', background: 'none' }}
            placeholder="Search by name, email, apartment, or vehicle/visitor plate…"
            value={query}
            onChange={e => setQuery(e.target.value)}
          />
        </div>
      </div>

      {error && <div style={{ color: 'var(--color-critical)', marginBottom: 16 }}>{error}</div>}

      {!loading && filtered.length === 0 && (
        <div className="empty-state">
          <div className="icon"><Users size={28} /></div>
          <h3>No residents found</h3>
          <p>{query ? 'Try a different search term.' : 'No registered residents yet.'}</p>
        </div>
      )}

      {filtered.map(resident => (
        <div key={resident.id} className="card" style={{ marginBottom: 16 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', flexWrap: 'wrap', gap: 12, marginBottom: 16 }}>
            <div>
              <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>{resident.full_name}</div>
              <div style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 2 }}>
                {resident.apartment_number ? `Apartment ${resident.apartment_number}` : 'No apartment on file'}
              </div>
              <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>{resident.email}</div>
              {resident.phone && <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>{resident.phone}</div>}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 6 }}>
              <span className={`status-badge ${resident.is_active ? 'resolved' : 'unresolved'}`}>
                {resident.is_active ? 'Active' : 'Inactive'}
              </span>
            </div>
          </div>

          <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-secondary)', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
            <ScanFace size={14} /> REGISTERED HOUSEHOLD MEMBERS ({resident.household_members.length})
          </div>
          {resident.household_members.length === 0 ? (
            <div style={{ fontSize: 13, color: 'var(--text-dim)', marginBottom: 16 }}>No household members registered</div>
          ) : (
            <table className="data-table" style={{ marginBottom: 16 }}>
              <thead>
                <tr><th>Name</th><th>Relationship</th><th>Phone</th><th>Status</th></tr>
              </thead>
              <tbody>
                {resident.household_members.map(m => (
                  <tr key={m.id}>
                    <td>{m.full_name}</td>
                    <td>{RELATIONSHIP_LABELS[m.relationship] || m.relationship}</td>
                    <td className="mono">{m.phone || '—'}</td>
                    <td>
                      <span className={`status-badge ${m.face_registered ? 'resolved' : 'unresolved'}`}>
                        {m.face_registered ? 'Face Registered' : 'Not Registered'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-secondary)', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
            <CarFront size={14} /> REGISTERED VEHICLES ({resident.vehicles.length})
          </div>
          {resident.vehicles.length === 0 ? (
            <div style={{ fontSize: 13, color: 'var(--text-dim)', marginBottom: 16 }}>No vehicles registered</div>
          ) : (
            <table className="data-table" style={{ marginBottom: 16 }}>
              <thead>
                <tr><th>Plate Number</th><th>Type</th><th>Status</th></tr>
              </thead>
              <tbody>
                {resident.vehicles.map(v => (
                  <tr key={v.id}>
                    <td className="mono">{v.plate_number}</td>
                    <td style={{ textTransform: 'capitalize' }}>{v.vehicle_type}</td>
                    <td><span className={`status-badge ${v.is_active ? 'resolved' : 'unresolved'}`}>{v.is_active ? 'Active' : 'Inactive'}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-secondary)', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
            <UserCheck size={14} /> SAVED VISITORS ({resident.visitors.length})
          </div>
          {resident.visitors.length === 0 ? (
            <div style={{ fontSize: 13, color: 'var(--text-dim)' }}>No visitors saved</div>
          ) : (
            <table className="data-table">
              <thead>
                <tr><th>Visitor</th><th>Vehicle Plate</th><th>Purpose</th><th>Status</th><th>Requested By</th></tr>
              </thead>
              <tbody>
                {resident.visitors.map(v => (
                  <tr key={v.id}>
                    <td>{v.visitor_name}</td>
                    <td className="mono">{v.plate_number || 'No vehicle registered'}</td>
                    <td>{v.purpose || '—'}</td>
                    <td><span className={`status-badge ${v.status}`}>{v.status}</span></td>
                    <td style={{ color: 'var(--text-muted)', fontSize: 12 }}>{v.guard_name}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      ))}
    </div>
  );
};

export default AdminResidentDirectory;
