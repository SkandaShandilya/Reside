import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Camera, AlertTriangle, Eye, CheckCircle, MonitorPlay } from 'lucide-react';
import PageHeader from '../components/ui/PageHeader';
import FireSmokeAlertBanner from '../components/FireSmokeAlertBanner';
import api from '../api/client';
import type { Alert } from '../types';

const GuardDashboard: React.FC = () => {
  const navigate = useNavigate();
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getAlerts({ page_size: '10', status: 'unresolved' }).then(res => {
      setAlerts(res.alerts);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  const handleAcknowledge = async (id: number) => {
    try {
      await api.acknowledgeAlert(id);
      setAlerts(prev => prev.map(a => a.id === id ? { ...a, status: 'acknowledged' as const } : a));
    } catch (err) { console.error(err); }
  };

  return (
    <div>
      <PageHeader
        title="Guard Dashboard"
        action={{ label: 'Open Live Monitors', icon: MonitorPlay, onClick: () => navigate('/live-monitoring') }}
      />

      <FireSmokeAlertBanner />

      {/* Critical Alerts */}
      {alerts.filter(a => a.severity === 'critical' && !a.activity.includes('Fire')).map(alert => (
        <div key={alert.id} className="alert-banner" style={{ marginBottom: 12 }}>
          <span className="alert-icon">🚨</span>
          <div className="alert-content">
            <div className="alert-title">CRITICAL — {alert.activity}</div>
            <div className="alert-desc">{alert.camera_name} · {alert.camera_location}</div>
          </div>
          <div className="alert-actions">
            <button className="btn btn-danger btn-sm" onClick={() => handleAcknowledge(alert.id)}>
              <CheckCircle size={14} /> Acknowledge
            </button>
            <button className="btn btn-ghost btn-sm" onClick={() => navigate(`/live-monitoring`)}>
              <Eye size={14} /> View Live
            </button>
          </div>
        </div>
      ))}

      <div className="section-header" style={{ marginTop: 24 }}>
        <h3 className="section-title">Active Security Alerts</h3>
      </div>

      {alerts.map(alert => (
        <div key={alert.id} className="card" style={{ marginBottom: 12, display: 'flex', alignItems: 'center', gap: 16 }}>
          <span className={`severity-badge ${alert.severity}`} style={{ minWidth: 80, textAlign: 'center' }}>
            {alert.severity === 'critical' ? '🚨' : '⚠️'} {alert.severity}
          </span>
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 700, color: 'var(--text-primary)' }}>{alert.activity}</div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{alert.camera_name} · {new Date(alert.created_at).toLocaleTimeString()}</div>
          </div>
          <div style={{ display: 'flex', gap: 4 }}>
            <button className="btn btn-warning btn-sm" onClick={() => handleAcknowledge(alert.id)}>ACK</button>
            <button className="btn btn-ghost btn-sm" onClick={() => navigate(`/alerts`)} title="View alert details">
              <Eye size={14} />
            </button>
          </div>
        </div>
      ))}

      {alerts.length === 0 && !loading && (
        <div className="empty-state">
          <div className="icon">✅</div>
          <h3>All Clear</h3>
          <p>No active security alerts at this time.</p>
        </div>
      )}
    </div>
  );
};

export default GuardDashboard;
