import React, { useEffect, useState } from 'react';
import { NavLink, useNavigate, Link } from 'react-router-dom';
import {
  LayoutDashboard, ScanFace, CarFront, MonitorPlay, Brain, AlertTriangle,
  Bell, X, ChevronRight, Volume2, VolumeX, Menu,
} from 'lucide-react';
import { useAuth } from '../../contexts/AuthContext';
import { useWebSocket, useSoundEnabled } from '../../hooks/useWebSocket';
import api from '../../api/client';
import type { Alert } from '../../types';
import ProfileMenu from './ProfileMenu';
import ThemeToggle from './ThemeToggle';
import resideLogo from '../../assets/reside-logo.png';

// Primary navbar items only — everything else (Residents/Vehicles/
// Analytics/Settings/etc.) lives in ProfileMenu instead of being
// crammed into the navbar, per the RESIDE navigation spec.
const TopNavbar: React.FC = () => {
  const { user, isAdmin, isGuard } = useAuth();
  const { lastAlert } = useWebSocket();
  const [soundEnabled, setSoundEnabled] = useSoundEnabled();
  const navigate = useNavigate();

  const [showNotif, setShowNotif] = useState(false);
  const [showProfile, setShowProfile] = useState(false);
  const [showMobile, setShowMobile] = useState(false);
  const [recentAlerts, setRecentAlerts] = useState<Alert[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [alertCount, setAlertCount] = useState(0);
  const [reviewCount, setReviewCount] = useState(0);

  const isResident = user?.role === 'resident';
  const dashboardPath = user?.role === 'resident' ? '/resident' : '/';

  useEffect(() => {
    // GET /api/alerts is guard/admin-scoped — Residents get a 403, so this
    // whole polling block is skipped entirely for them (same rule the
    // original Header/Sidebar enforced).
    if (!isGuard) return;
    const fetchCounts = () => {
      api.getAlerts({ page_size: '5', status: 'unresolved' })
        .then(res => {
          setRecentAlerts(res.alerts || []);
          if (res.total != null) { setUnreadCount(res.total); setAlertCount(res.total); }
        })
        .catch(() => {});
      api.getReviewItems({ page_size: '1' })
        .then(res => setReviewCount(res.total ?? 0))
        .catch(() => {});
    };
    fetchCounts();
    const interval = setInterval(fetchCounts, 15000);
    return () => clearInterval(interval);
  }, [isGuard]);

  useEffect(() => {
    if (isGuard && lastAlert && lastAlert.type === 'security_alert') {
      const convertedAlert: Alert = {
        id: lastAlert.alert_id || Date.now(),
        camera_id: lastAlert.camera_id || 1,
        camera_name: lastAlert.camera_name || 'Camera Feed',
        camera_location: lastAlert.camera_location || 'Security Zone',
        activity: lastAlert.activity || 'Activity Detected',
        confidence: lastAlert.confidence || 0.9,
        severity: (lastAlert.severity as any) || 'critical',
        status: 'unresolved',
        persons_detected: 1,
        created_at: lastAlert.timestamp || new Date().toISOString(),
        updated_at: lastAlert.timestamp || new Date().toISOString(),
      };
      setRecentAlerts(prev => [convertedAlert, ...prev.slice(0, 4)]);
      setUnreadCount(cnt => cnt + 1);
      setAlertCount(cnt => cnt + 1);
    }
  }, [lastAlert]);

  const primaryNav = [
    { path: dashboardPath, label: 'Dashboard', icon: LayoutDashboard, show: true, end: dashboardPath === '/' },
    { path: '/face-registration', label: 'Face Registration', icon: ScanFace, show: isResident, end: false },
    { path: '/vehicle-requests', label: 'My Vehicles', icon: CarFront, show: isResident, end: false },
    { path: '/live-monitoring', label: 'Live Cameras', icon: MonitorPlay, show: isGuard, end: false },
    { path: '/guard-verification', label: 'Verification', icon: Brain, show: isGuard, end: false },
    { path: '/alerts', label: 'Alerts', icon: AlertTriangle, show: isGuard, end: false, badge: alertCount },
  ].filter(item => item.show);

  return (
    <nav className="top-navbar" aria-label="Primary">
      <Link to={dashboardPath} className="navbar-brand">
        <span className="brand-badge"><img src={resideLogo} alt="RESIDE" /></span>
        <span className="brand-word">RESIDE</span>
      </Link>

      <div className="navbar-primary-nav">
        {primaryNav.map(item => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.end}
            className={({ isActive }) => `navbar-link ${isActive ? 'active' : ''}`}
          >
            <item.icon size={18} />
            {item.label}
            {item.badge != null && item.badge > 0 && (
              <span style={{
                marginLeft: 2, fontSize: 10, fontWeight: 700, background: 'var(--ink-5)',
                padding: '1px 6px', borderRadius: 8,
              }}>{item.badge}</span>
            )}
          </NavLink>
        ))}
      </div>

      <button className="navbar-mobile-toggle" onClick={() => setShowMobile(s => !s)} aria-label="Menu">
        <Menu size={18} />
      </button>

      <div className="navbar-actions">

        <button
          className="navbar-icon-btn"
          onClick={() => setSoundEnabled(!soundEnabled)}
          title={soundEnabled ? 'Mute alert sounds' : 'Unmute alert sounds'}
        >
          {soundEnabled ? <Volume2 size={16} /> : <VolumeX size={16} />}
        </button>

        <ThemeToggle />


        {isGuard && (
          <div style={{ position: 'relative' }}>
            <button
              className="navbar-icon-btn"
              onClick={() => { setShowNotif(p => !p); setUnreadCount(0); setShowProfile(false); }}
              title="Notifications"
            >
              <Bell size={16} />
              {unreadCount > 0 && (
                <span className="notification-badge" style={{
                  position: 'absolute', top: -2, right: -2, width: 16, height: 16, background: 'var(--color-critical)',
                  color: '#fff', fontSize: 9, fontWeight: 700, borderRadius: '50%', display: 'flex',
                  alignItems: 'center', justifyContent: 'center', border: '2px solid var(--navbar-solid-tint)',
                }}>{unreadCount}</span>
              )}
            </button>
            {showNotif && (
              <div className="glass-panel" style={{ width: 340, right: 0 }}>
                <div style={{
                  padding: '12px 16px', borderBottom: '1px solid var(--border-subtle)',
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                }}>
                  <span style={{ fontWeight: 700, fontSize: 13, color: 'var(--text-primary)' }}>Security Notifications</span>
                  <button onClick={() => setShowNotif(false)} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}>
                    <X size={14} />
                  </button>
                </div>
                <div style={{ maxHeight: 300, overflowY: 'auto' }}>
                  {recentAlerts.length === 0 ? (
                    <div style={{ padding: 24, textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
                      No recent security notifications
                    </div>
                  ) : (
                    recentAlerts.map(alert => (
                      <div
                        key={alert.id}
                        onClick={() => { setShowNotif(false); navigate('/alerts'); }}
                        style={{
                          padding: '12px 16px', borderBottom: '1px solid var(--border-subtle)',
                          cursor: 'pointer', display: 'flex', gap: 10, alignItems: 'start',
                        }}
                      >
                        <span style={{ fontSize: 16 }}>{alert.severity === 'critical' ? '🚨' : '⚠️'}</span>
                        <div style={{ flex: 1 }}>
                          <div style={{ fontWeight: 700, fontSize: 13, color: 'var(--text-primary)' }}>{alert.activity} Detected</div>
                          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
                            {alert.camera_name}
                          </div>
                        </div>
                      </div>
                    ))
                  )}
                </div>
                <div style={{ padding: 10, textAlign: 'center', borderTop: '1px solid var(--border-subtle)' }}>
                  <Link to="/alerts" onClick={() => setShowNotif(false)} style={{ fontSize: 12, fontWeight: 700, color: 'var(--color-info)', textDecoration: 'none' }}>
                    View All Security Alerts <ChevronRight size={12} style={{ verticalAlign: 'middle' }} />
                  </Link>
                </div>
              </div>
            )}
          </div>
        )}

        <div style={{ position: 'relative' }}>
          <div
            className="navbar-avatar"
            onClick={() => { setShowProfile(p => !p); setShowNotif(false); }}
            title={user?.full_name || 'Account'}
          >
            {user?.avatar_url ? (
              <img
                src={`${user.avatar_url}&token=${api.getToken() || ''}`}
                alt=""
                style={{ width: '100%', height: '100%', borderRadius: '50%', objectFit: 'cover' }}
              />
            ) : (
              user?.full_name?.charAt(0) || 'U'
            )}
          </div>
          {showProfile && (
            <ProfileMenu reviewCount={reviewCount} onClose={() => setShowProfile(false)} />
          )}
        </div>
      </div>

      {showMobile && (
        <div className="navbar-mobile-panel">
          {primaryNav.map(item => (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.end}
              className={({ isActive }) => `navbar-link ${isActive ? 'active' : ''}`}
              onClick={() => setShowMobile(false)}
            >
              <item.icon size={16} />
              {item.label}
            </NavLink>
          ))}
        </div>
      )}

    </nav>
  );
};

export default TopNavbar;
