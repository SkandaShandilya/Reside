import React, { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Flame, CheckCircle, Eye } from 'lucide-react';
import api from '../api/client';
import { useWebSocket } from '../hooks/useWebSocket';
import type { Alert } from '../types';

// Prominent banner for active Fire/Smoke alerts, shown on the admin and
// guard dashboards. Loads unresolved fire/smoke alerts and refreshes
// instantly when a new one arrives over the live alert WebSocket (plus a
// slow poll as a safety net).
const FireSmokeAlertBanner: React.FC = () => {
  const navigate = useNavigate();
  const { lastAlert } = useWebSocket();
  const [alerts, setAlerts] = useState<Alert[]>([]);

  const load = useCallback(() => {
    api.getAlerts({ category: 'fire_smoke', status: 'unresolved', page_size: '5' })
      .then(res => setAlerts(res.alerts || []))
      .catch(() => {});
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 10000);
    return () => clearInterval(t);
  }, [load]);

  useEffect(() => {
    if (lastAlert?.type === 'security_alert' && (lastAlert.activity || '').includes('Fire')) load();
  }, [lastAlert, load]);

  const acknowledge = async (id: number) => {
    try {
      await api.acknowledgeAlert(id);
      setAlerts(prev => prev.filter(a => a.id !== id));
    } catch (err) {
      console.error(err);
    }
  };

  if (alerts.length === 0) return null;

  return (
    <>
      {alerts.map(alert => (
        <div key={alert.id} className="alert-banner" style={{ marginBottom: 12 }}>
          <span className="alert-icon"><Flame size={26} /></span>
          <div className="alert-content">
            <div className="alert-title">FIRE / SMOKE DETECTED</div>
            <div className="alert-desc">{alert.camera_name} · {alert.camera_location}</div>
            <div className="alert-meta">
              {new Date(alert.created_at).toLocaleTimeString()}
            </div>
          </div>
          <div className="alert-actions">
            <button className="btn btn-danger btn-sm" onClick={() => navigate('/fire-smoke')}>
              <Eye size={14} /> View Fire &amp; Smoke
            </button>
            <button className="btn btn-ghost btn-sm" onClick={() => acknowledge(alert.id)}>
              <CheckCircle size={14} /> Acknowledge
            </button>
          </div>
        </div>
      ))}
    </>
  );
};

export default FireSmokeAlertBanner;
