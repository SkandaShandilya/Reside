import React, { useEffect, useState } from 'react';
import { Eye, Search, Download } from 'lucide-react';
import PageHeader from '../components/ui/PageHeader';
import api from '../api/client';

import type { SecurityEvent } from '../types';

const IncidentHistory: React.FC = () => {
  const [incidents, setIncidents] = useState<SecurityEvent[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState('');
  const [severity, setSeverity] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fire/Smoke incidents have their own dedicated history on the Fire &
    // Smoke tab — this page shows behaviour incidents only.
    const params: Record<string, string> = { page: String(page), page_size: '20', category: 'behaviour' };
    if (search) params.search = search;
    if (severity) params.severity = severity;
    api.getIncidents(params).then(res => {
      setIncidents(res.incidents);
      setTotal(res.total);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, [page, search, severity]);

  const formatDate = (ts: string) => new Date(ts).toLocaleString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit', hour12: true,
  });

  const handleExportCSV = async () => {
    try {
      const params: Record<string, string> = {};
      if (search) params.search = search;
      if (severity) params.severity = severity;
      await api.downloadIncidentsCSV(params);
    } catch (err) {
      console.error('Export error:', err);
    }
  };

  return (
    <div>
      <PageHeader
        title="Incident History"
        action={{ label: 'Export CSV', icon: Download, onClick: handleExportCSV, variant: 'secondary' }}
      />

      <div style={{ display: 'flex', gap: 12, marginBottom: 20, flexWrap: 'wrap' }}>
        <div style={{ position: 'relative', flex: 1, minWidth: 250 }}>
          <Search size={16} style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-dim)' }} />
          <input className="input" style={{ paddingLeft: 36 }} placeholder="Search incidents..." value={search} onChange={e => { setSearch(e.target.value); setPage(1); }} />
        </div>
        <select className="select" style={{ width: 160 }} value={severity} onChange={e => { setSeverity(e.target.value); setPage(1); }}>
          <option value="">All Severity</option>
          <option value="critical">Critical</option>
          <option value="warning">Warning</option>
          <option value="info">Info</option>
        </select>
      </div>


      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <table className="data-table">
          <thead><tr><th>Time</th><th>Camera</th><th>Activity</th><th>Severity</th><th>Persons</th><th>Duration</th></tr></thead>
          <tbody>
            {incidents.map(inc => (
              <tr key={inc.id}>
                <td className="mono">{formatDate(inc.timestamp)}</td>
                <td>Camera #{inc.camera_id}</td>
                <td style={{ fontWeight: 600 }}>{inc.activity_class}</td>
                <td><span className={`severity-badge ${inc.severity}`}>{inc.severity}</span></td>
                <td>{inc.persons_detected}</td>
                <td className="mono">{inc.duration_seconds ? `${inc.duration_seconds.toFixed(1)}s` : '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {total > 20 && (
        <div style={{ display: 'flex', justifyContent: 'center', gap: 8, marginTop: 16 }}>
          <button className="btn btn-ghost btn-sm" disabled={page <= 1} onClick={() => setPage(p => p - 1)}>Previous</button>
          <span style={{ alignSelf: 'center', fontSize: 13, color: 'var(--text-muted)' }}>Page {page} of {Math.ceil(total / 20)}</span>
          <button className="btn btn-ghost btn-sm" disabled={page >= Math.ceil(total / 20)} onClick={() => setPage(p => p + 1)}>Next</button>
        </div>
      )}
    </div>
  );
};

export default IncidentHistory;
