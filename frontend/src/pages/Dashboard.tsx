import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  Camera, AlertTriangle, AlertOctagon, ClipboardCheck,
  Wifi, WifiOff, Activity, Clock, ChevronRight, Eye
} from 'lucide-react';
import api from '../api/client';
import { useWebSocket } from '../hooks/useWebSocket';
import type { Alert, Camera as CameraType, AnalyticsOverview } from '../types';
import PageHeader from '../components/ui/PageHeader';
import FireSmokeAlertBanner from '../components/FireSmokeAlertBanner';

const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const [overview, setOverview] = useState<AnalyticsOverview | null>(null);
  const [recentAlerts, setRecentAlerts] = useState<Alert[]>([]);
  const [cameras, setCameras] = useState<CameraType[]>([]);
  const [loading, setLoading] = useState(true);
  const [acknowledgedAlertIds, setAcknowledgedAlertIds] = useState<number[]>([]);
  const { lastAlert } = useWebSocket();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [analyticsRes, alertsRes, camerasRes] = await Promise.all([
          api.getAnalytics(),
          api.getAlerts({ page_size: '5' }),
          api.getCameras(),
        ]);
        setOverview(analyticsRes.overview);
        setRecentAlerts(alertsRes.alerts);
        setCameras(camerasRes);
      } catch (err) {
        console.error('Dashboard fetch error:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const onlineCams = cameras.filter(c => c.is_online);
  const offlineCams = cameras.filter(c => !c.is_online);

  const formatTime = (ts: string) => {
    const d = new Date(ts);
    return d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });
  };

  const severityIcon = (sev: string) => {
    if (sev === 'critical') return '🚨';
    if (sev === 'warning') return '⚠️';
    return 'ℹ️';
  };

  // `?? ` not `||`: an alert_id of 0 is a valid (if unusual) id and must
  // not collapse into the same "no id" sentinel bucket as every other
  // id-less alert — the old `||` check did exactly that, which meant
  // acknowledging one such alert silently acknowledged all of them.
  const lastAlertId = lastAlert?.alert_id ?? (lastAlert ? 9999 : 0);

  return (
    <div>
      <PageHeader title="Dashboard" description="Real-time security monitoring and system overview" />

      <FireSmokeAlertBanner />

      {/* Live Alert Banner (fire/smoke has its own banner above) */}
      {lastAlert && lastAlert.severity === 'critical' && !(lastAlert.activity || '').includes('Fire') && !acknowledgedAlertIds.includes(lastAlertId) && (
        <div className="alert-banner">
          <span className="alert-icon">🚨</span>
          <div className="alert-content">
            <div className="alert-title">Critical Security Alert</div>
            <div className="alert-desc">
              Possible {lastAlert.activity} detected
            </div>
            <div className="alert-meta">
              Camera: {lastAlert.camera_name}
            </div>
          </div>
          <div className="alert-actions">
            <button className="btn btn-danger btn-sm" onClick={() => navigate('/live-monitoring')}>
              View Live
            </button>
            <button
              className="btn btn-ghost btn-sm"
              onClick={async () => {
                try {
                  if (lastAlert.alert_id) await api.acknowledgeAlert(lastAlert.alert_id);
                  setAcknowledgedAlertIds(prev => [...prev, lastAlertId]);
                } catch (err) {
                  console.error(err);
                }
              }}
            >
              Acknowledge
            </button>
          </div>
        </div>
      )}

      {/* Stats Cards */}
      <div className="stats-grid">
        <Link to="/live-monitoring" style={{ textDecoration: 'none' }}>
          <div className="card card-clickable stat-card cameras">
            <div className="stat-icon cameras"><Camera size={22} /></div>
            <div className="stat-value">{cameras.length}</div>
            <div className="stat-label">Active Cameras</div>
            <div className="stat-sub" style={{ color: 'var(--color-success)' }}>
              <Wifi size={12} style={{ verticalAlign: 'middle' }} /> {onlineCams.length} Online
              {offlineCams.length > 0 && (
                <span style={{ color: 'var(--color-critical)', marginLeft: 8 }}>
                  <WifiOff size={12} style={{ verticalAlign: 'middle' }} /> {offlineCams.length} Offline
                </span>
              )}
            </div>
          </div>
        </Link>

        <Link to="/alerts" style={{ textDecoration: 'none' }}>
          <div className="card card-clickable stat-card alerts">
            <div className="stat-icon alerts"><AlertTriangle size={22} /></div>
            <div className="stat-value">{overview?.alerts_today || 0}</div>
            <div className="stat-label">Alerts Today</div>
            <div className="stat-sub">Last 24 hours</div>
          </div>
        </Link>

        <Link to="/alerts?severity=critical" style={{ textDecoration: 'none' }}>
          <div className="card card-clickable stat-card critical">
            <div className="stat-icon critical"><AlertOctagon size={22} /></div>
            <div className="stat-value">{overview?.critical_alerts_today || 0}</div>
            <div className="stat-label">Critical Alerts</div>
            <div className="stat-sub" style={{ color: 'var(--color-critical)' }}>Requires immediate action</div>
          </div>
        </Link>

        <Link to="/review" style={{ textDecoration: 'none' }}>
          <div className="card card-clickable stat-card review">
            <div className="stat-icon review"><ClipboardCheck size={22} /></div>
            <div className="stat-value">{overview?.events_under_review || 0}</div>
            <div className="stat-label">Under Review</div>
            <div className="stat-sub">Awaiting verification</div>
          </div>
        </Link>
      </div>

      {/* Recent Security Events */}
      <div className="section-header">
        <h3 className="section-title">Recent Security Events</h3>
        <Link to="/alerts" className="btn btn-ghost btn-sm">
          View All <ChevronRight size={14} />
        </Link>
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
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {recentAlerts.map(alert => (
              <tr key={alert.id}>
                <td className="mono">{formatTime(alert.created_at)}</td>
                <td>
                  <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{alert.camera_name}</div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{alert.camera_location}</div>
                </td>
                <td style={{ fontWeight: 600 }}>{alert.activity}</td>
                <td>
                  <span className={`severity-badge ${alert.severity}`}>
                    {severityIcon(alert.severity)} {alert.severity}
                  </span>
                </td>
                <td>
                  <span className={`status-badge ${alert.status}`}>
                    {alert.status.replace('_', ' ')}
                  </span>
                </td>
                <td>
                  <Link to={`/alerts/${alert.id}`} className="btn btn-ghost btn-sm">
                    <Eye size={14} /> View
                  </Link>
                </td>
              </tr>
            ))}
            {recentAlerts.length === 0 && !loading && (
              <tr>
                <td colSpan={7} style={{ textAlign: 'center', padding: 40, color: 'var(--text-muted)' }}>
                  No recent alerts
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default Dashboard;
