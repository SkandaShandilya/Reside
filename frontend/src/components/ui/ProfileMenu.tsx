import React, { useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import {
  ScanFace, CarFront, Flame, History, ClipboardCheck, Brain, Database,
  BarChart3, Bell, Settings, Users, LogOut, UserCircle,
} from 'lucide-react';
import { useAuth } from '../../contexts/AuthContext';

interface ProfileMenuProps {
  reviewCount: number;
  onClose: () => void;
}

// Every non-primary route from the old permanent sidebar lives here now,
// grouped by role — nothing dropped, just relocated out of the navbar
// per the "don't overcrowd the navbar" requirement.
const ProfileMenu: React.FC<ProfileMenuProps> = ({ reviewCount, onClose }) => {
  const { user, isAdmin, isGuard, logout } = useAuth();
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
  }, [onClose]);

  return (
    <div className="glass-panel profile-menu" ref={ref}>
      <div className="profile-menu-header">
        <div className="pm-name">{user?.full_name || 'User'}</div>
        <div className="pm-role">{user?.role?.replace('_', ' ') || 'Role'}</div>
      </div>

      <Link to="/profile" className="profile-menu-item" onClick={onClose}>
        <UserCircle size={15} /> My Profile
      </Link>

      {isGuard && (
        <>
          <div className="profile-menu-section-label">Monitoring</div>
          <Link to="/fire-smoke" className="profile-menu-item" onClick={onClose}>
            <Flame size={15} /> Fire &amp; Smoke
          </Link>
          {!isAdmin && (
            <>
              <Link to="/incidents" className="profile-menu-item" onClick={onClose}>
              <History size={15} /> Incident History
            </Link>
            <Link to="/review" className="profile-menu-item" onClick={onClose}>
              <ClipboardCheck size={15} /> Review Queue
              {reviewCount > 0 && <span className="pm-badge">{reviewCount}</span>}
            </Link>
            <Link to="/guard-verification" className="profile-menu-item" onClick={onClose}>
              <ScanFace size={15} /> Guard Verification
            </Link>
            </>
          )}
        </>
      )}

      {(isAdmin || user?.role === 'guard') && (
        <>
          <div className="profile-menu-section-label">AI &amp; Analytics</div>
          <Link to="/ai-model" className="profile-menu-item" onClick={onClose}>
            <Brain size={15} /> AI Model
          </Link>
          <Link to="/training-data" className="profile-menu-item" onClick={onClose}>
            <Database size={15} /> Training Data
          </Link>
          <Link to="/analytics" className="profile-menu-item" onClick={onClose}>
            <BarChart3 size={15} /> Analytics
          </Link>
        </>
      )}

      {isAdmin && (
        <>
          <div className="profile-menu-section-label">Administration</div>
          <Link to="/notification-settings" className="profile-menu-item" onClick={onClose}>
            <Bell size={15} /> Notifications
          </Link>
          <Link to="/settings" className="profile-menu-item" onClick={onClose}>
            <Settings size={15} /> System Settings
          </Link>
          <Link to="/users" className="profile-menu-item" onClick={onClose}>
            <Users size={15} /> User Management
          </Link>
        </>
      )}

      <div className="profile-menu-section-label" style={{ paddingTop: 10 }} />
      <button className="profile-menu-item danger" onClick={logout}>
        <LogOut size={15} /> Logout
      </button>
    </div>
  );
};

export default ProfileMenu;
