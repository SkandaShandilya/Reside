import React from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import TopNavbar from '../ui/TopNavbar';

const pageTitles: Record<string, string> = {
  '/': 'Security Dashboard',
  '/live-monitoring': 'Live Monitoring',
  '/guard-verification': 'Verification',
  '/fire-smoke': 'Fire & Smoke Detection',
  '/cameras': 'Camera Management',
  '/alerts': 'Security Alerts',
  '/incidents': 'Incident History',
  '/review': 'AI Review Queue',
  '/ai-model': 'AI Model Management',
  '/training-data': 'Training Data',
  '/analytics': 'Security Analytics',
  '/notification-settings': 'Notification Settings',
  '/settings': 'System Settings',
  '/users': 'User Management',
  '/guard': 'Guard Dashboard',
  '/resident': 'Community Security',
  '/profile': 'My Profile',
};

const Layout: React.FC = () => {
  const location = useLocation();
  // Reserved for future per-page <title> wiring — pageTitles kept as the
  // single source of truth for page names (used nowhere else yet).
  void (pageTitles[location.pathname] || 'RESIDE');

  return (
    <div className="app-layout">
      <TopNavbar />
      <main className="main-content">
        <div className="page-content route-fade" key={location.pathname}>
          <Outlet />
        </div>
      </main>
    </div>
  );
};

export default Layout;
