import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Eye, CheckCircle, ArrowUpRight, Download, X, ShieldAlert, AlertTriangle, CheckSquare } from 'lucide-react';
import api from '../api/client';
import type { Alert } from '../types';
import { useWebSocketStream } from '../hooks/useWebSocketStream';
import PageHeader from '../components/ui/PageHeader';

const Alerts: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [severityFilter, setSeverityFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);
  // Frames pushed over a WebSocket instead of a persistent MJPEG
  // connection or a polled snapshot — see hooks/useWebSocketStream.ts.
  // Camera id 0 while no alert is selected is never actually rendered
  // (the <img> below only mounts when the detail modal is open).
  const selectedAlertStreamUrl = useWebSocketStream(selectedAlert?.camera_id ?? 0, { enabled: !!selectedAlert });

  const handleExportCSV = async () => {
    try {
      const params: Record<string, string> = {};
      if (severityFilter) params.severity = severityFilter;
      if (statusFilter) params.status = statusFilter;
      await api.downloadAlertsCSV(params);
    } catch (err) {
      console.error(err);
    }
  };

  const fetchAlerts = async () => {
    setLoading(true);
    try {
      // Fire/Smoke alerts have their own dedicated panel on the Fire &
      // Smoke tab (spec: "do not insert Fire/Smoke into the behaviour
      // alert list") — this page shows behaviour alerts only.
      const params: Record<string, string> = { page: String(page), page_size: '15', category: 'behaviour' };
      if (severityFilter) params.severity = severityFilter;
      if (statusFilter) params.status = statusFilter;
      const res = await api.getAlerts(params);
      setAlerts(res.alerts);
      setTotal(res.total);
      if (id) {
        const found = res.alerts.find((a: Alert) => a.id === Number(id));
        if (found) setSelectedAlert(found);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchAlerts(); }, [page, severityFilter, statusFilter, id]);

  const handleAction = async (alertId: number, action: string) => {
    try {
      if (action === 'acknowledge') await api.acknowledgeAlert(alertId);
      else if (action === 'resolve') await api.resolveAlert(alertId);
      else if (action === 'escalate') await api.escalateAlert(alertId);
      else if (action === 'false_positive') await api.markFalsePositive(alertId);
      fetchAlerts();
      if (selectedAlert && selectedAlert.id === alertId) {
        setSelectedAlert(prev => prev ? { ...prev, status: action as any } : null);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const severityIcon = (s: string) => {
    if (s === 'critical') return '🚨';
    if (s === 'warning') return '⚠️';
    return 'ℹ️';
  };

  const formatTime = (ts: string) => {
    const d = new Date(ts);
    return d.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', hour12: true });
  };

  return (
    <div>
      <PageHeader
        title="Security Alerts"
        description="Behaviour-detection alerts across all cameras"
        action={{ label: 'Export CSV', icon: Download, onClick: handleExportCSV, variant: 'secondary' }}
      />

      {/* Filters */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 20, flexWrap: 'wrap' }}>
        <select className="select" style={{ width: 160 }} value={severityFilter} onChange={e => { setSeverityFilter(e.target.value); setPage(1); }}>
          <option value="">All Severity</option>
          <option value="critical">🚨 Critical</option>
          <option value="warning">⚠️ Warning</option>
          <option value="info">ℹ️ Info</option>
        </select>
        <select className="select" style={{ width: 160 }} value={statusFilter} onChange={e => { setStatusFilter(e.target.value); setPage(1); }}>
          <option value="">All Status</option>
          <option value="unresolved">Unresolved</option>
          <option value="acknowledged">Acknowledged</option>
          <option value="resolved">Resolved</option>
          <option value="escalated">Escalated</option>
          <option value="false_positive">False Positive</option>
        </select>
        <span style={{ color: 'var(--text-muted)', fontSize: 13, alignSelf: 'center' }}>
          {total} total alerts
        </span>
      </div>

      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Camera</th>
              <th>Activity</th>
              <th>Severity</th>
              <th>Status</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {alerts.map(alert => (
              <tr key={alert.id}>
                <td className="mono">{formatTime(alert.created_at)}</td>
                <td>
                  <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{alert.camera_name}</div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{alert.camera_location}</div>
                </td>
                <td style={{ fontWeight: 600 }}>{alert.activity}</td>
                <td><span className={`severity-badge ${alert.severity}`}>{severityIcon(alert.severity)} {alert.severity}</span></td>
                <td><span className={`status-badge ${alert.status}`}>{alert.status.replace('_', ' ')}</span></td>
                <td>
                  <div style={{ display: 'flex', gap: 4 }}>
                    {alert.status === 'unresolved' && (
                      <>
                        <button className="btn btn-warning btn-sm" onClick={() => handleAction(alert.id, 'acknowledge')} title="Acknowledge">
                          <CheckCircle size={13} />
                        </button>
                        <button className="btn btn-danger btn-sm" onClick={() => handleAction(alert.id, 'escalate')} title="Escalate">
                          <ArrowUpRight size={13} />
                        </button>
                      </>
                    )}
                    {(alert.status === 'unresolved' || alert.status === 'acknowledged') && (
                      <button className="btn btn-success btn-sm" onClick={() => handleAction(alert.id, 'resolve')} title="Resolve">
                        <CheckCircle size={13} />
                      </button>
                    )}
                    <button className="btn btn-ghost btn-sm" onClick={() => setSelectedAlert(alert)} title="View Details">
                      <Eye size={13} />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {total > 15 && (
        <div style={{ display: 'flex', justifyContent: 'center', gap: 8, marginTop: 16 }}>
          <button className="btn btn-ghost btn-sm" disabled={page <= 1} onClick={() => setPage(p => p - 1)}>Previous</button>
          <span style={{ alignSelf: 'center', fontSize: 13, color: 'var(--text-muted)' }}>Page {page} of {Math.ceil(total / 15)}</span>
          <button className="btn btn-ghost btn-sm" disabled={page >= Math.ceil(total / 15)} onClick={() => setPage(p => p + 1)}>Next</button>
        </div>
      )}

      {/* Alert Detail Modal */}
      {selectedAlert && (
        <div className="modal-overlay" onClick={() => { setSelectedAlert(null); navigate('/alerts'); }}>
          <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 640 }}>
            <div className="modal-header">
              <h3 className="modal-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <ShieldAlert size={18} style={{ color: selectedAlert.severity === 'critical' ? 'var(--color-critical)' : 'var(--color-warning)' }} />
                Alert Details — {selectedAlert.activity}
              </h3>
              <button className="btn btn-ghost btn-sm" onClick={() => { setSelectedAlert(null); navigate('/alerts'); }}>
                <X size={16} />
              </button>
            </div>
            <div className="modal-body">
              <div style={{ aspectRatio: '16/9', background: '#000', borderRadius: 8, overflow: 'hidden', marginBottom: 16, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                {selectedAlertStreamUrl ? (
                  <img
                    src={selectedAlertStreamUrl}
                    alt={selectedAlert.camera_name}
                    style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                  />
                ) : (
                  <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>Connecting...</span>
                )}
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
                <div className="card" style={{ padding: 12, background: 'var(--bg-tertiary)' }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Camera Source</div>
                  <div style={{ fontWeight: 700, fontSize: 14, marginTop: 2 }}>{selectedAlert.camera_name}</div>
                  <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>{selectedAlert.camera_location}</div>
                </div>
                <div className="card" style={{ padding: 12, background: 'var(--bg-tertiary)' }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Status</div>
                  <div style={{ fontWeight: 800, fontSize: 18, marginTop: 2, color: 'var(--color-warning)' }}>
                    {selectedAlert.status?.toUpperCase() || 'UNKNOWN'}
                  </div>
                </div>
              </div>

              <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                Detected on {formatTime(selectedAlert.created_at)}
              </div>
            </div>

            <div className="modal-footer" style={{ justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', gap: 8 }}>
                <button className="btn btn-warning btn-sm" onClick={() => handleAction(selectedAlert.id, 'acknowledge')}>
                  Acknowledge
                </button>
                <button className="btn btn-success btn-sm" onClick={() => handleAction(selectedAlert.id, 'resolve')}>
                  Mark Resolved
                </button>
              </div>
              <button className="btn btn-ghost" onClick={() => { setSelectedAlert(null); navigate('/alerts'); }}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Alerts;
