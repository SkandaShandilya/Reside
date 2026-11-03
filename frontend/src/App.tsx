import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { ThemeProvider } from './contexts/ThemeContext';
import Layout from './components/layout/Layout';
import LoadingSpinner from './components/ui/LoadingSpinner';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import LiveMonitoring from './pages/LiveMonitoring';
import FireSmoke from './pages/FireSmoke';
import CameraDetail from './pages/CameraDetail';
import Alerts from './pages/Alerts';
import IncidentHistory from './pages/IncidentHistory';
import ReviewQueue from './pages/ReviewQueue';
import AIModel from './pages/AIModel';
import TrainingData from './pages/TrainingData';
import Analytics from './pages/Analytics';
import NotificationSettings from './pages/NotificationSettings';
import SystemSettings from './pages/SystemSettings';
import GuardDashboard from './pages/GuardDashboard';
import ResidentFeed from './pages/ResidentFeed';
import Users from './pages/Users';
import FaceRegistration from './pages/FaceRegistration';
import GuardVerification from './pages/GuardVerification';
import VehicleRequests from './pages/VehicleRequests';
import Profile from './pages/Profile';
import './index.css';

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth();
  if (isLoading) {
    return <LoadingSpinner label="Initializing RESIDE..." fullscreen />;
  }
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" />;
}

// Sidebar.tsx already HIDES nav links a role shouldn't see (isAdmin/isGuard
// checks), but that's cosmetic only — the underlying page components (e.g.
// Users.tsx) perform no role check of their own, so a Resident typing
// /users directly into the address bar previously got the fully-rendered
// admin page, including calling admin-only API endpoints from the browser
// before any backend 403 could stop it. This wraps route elements so
// navigation itself is blocked for the wrong role, matching exactly what
// Sidebar.tsx already decides is visible to each role (isAdmin: settings/
// users pages; isAdmin-or-guard: AI/analytics pages).
//
// A Resident is not a security operator (see the strict role-separation
// requirement this app enforces): every security-monitoring route — Live
// Monitoring, Fire & Smoke, camera detail, Alerts, Incident History,
// Review Queue, Guard Verification, the main Dashboard — is a RoleRoute
// gated on isGuard (which already covers admin/super_admin too). The
// matching backend endpoints are independently locked to require_guard/
// require_admin (see app/api/*.py) — this frontend gate is a UX
// convenience, never the only enforcement.
function homeFor(auth: ReturnType<typeof useAuth>) {
  return auth.user?.role === 'resident' ? '/resident' : '/';
}

function RoleRoute({ allow, children }: { allow: (ctx: ReturnType<typeof useAuth>) => boolean; children: React.ReactNode }) {
  const auth = useAuth();
  return allow(auth) ? <>{children}</> : <Navigate to={homeFor(auth)} replace />;
}

// "/" is the full security-operations Dashboard (camera overview, recent
// alerts, live WebSocket alert feed) — never appropriate for a Resident.
// Rather than gating it like every other guard-only route (which would
// bounce a Resident back to "/" and loop), it redirects them straight to
// their own home page instead.
function HomeRoute() {
  const auth = useAuth();
  return auth.user?.role === 'resident' ? <Navigate to="/resident" replace /> : <Dashboard />;
}

function AppRoutes() {
  const { isAuthenticated } = useAuth();

  return (
    <Routes>
      <Route path="/login" element={isAuthenticated ? <Navigate to="/" /> : <Login />} />
      <Route element={<ProtectedRoute><Layout /></ProtectedRoute>}>
        <Route path="/" element={<HomeRoute />} />
        <Route path="/live-monitoring" element={<RoleRoute allow={a => a.isGuard}><LiveMonitoring /></RoleRoute>} />
        <Route path="/ai-monitoring" element={<Navigate to="/guard-verification" replace />} />
        <Route path="/fire-smoke" element={<RoleRoute allow={a => a.isGuard}><FireSmoke /></RoleRoute>} />
        {/* No standalone camera-list page — Configure links from Live
            Monitoring / Fire & Smoke go straight to a camera's detail/
            settings view; adding/removing a source happens in those tabs
            themselves. */}
        <Route path="/cameras/:id" element={<RoleRoute allow={a => a.isGuard}><CameraDetail /></RoleRoute>} />
        <Route path="/alerts" element={<RoleRoute allow={a => a.isGuard}><Alerts /></RoleRoute>} />
        <Route path="/alerts/:id" element={<RoleRoute allow={a => a.isGuard}><Alerts /></RoleRoute>} />
        <Route path="/incidents" element={<RoleRoute allow={a => a.isGuard}><IncidentHistory /></RoleRoute>} />
        <Route path="/review" element={<RoleRoute allow={a => a.isGuard}><ReviewQueue /></RoleRoute>} />
        <Route path="/ai-model" element={<RoleRoute allow={a => a.isGuard}><AIModel /></RoleRoute>} />
        <Route path="/training-data" element={<RoleRoute allow={a => a.isGuard}><TrainingData /></RoleRoute>} />
        <Route path="/analytics" element={<RoleRoute allow={a => a.isGuard}><Analytics /></RoleRoute>} />
        <Route path="/notification-settings" element={<RoleRoute allow={a => a.isAdmin}><NotificationSettings /></RoleRoute>} />
        <Route path="/settings" element={<RoleRoute allow={a => a.isAdmin}><SystemSettings /></RoleRoute>} />
        <Route path="/guard" element={<RoleRoute allow={a => a.isGuard}><GuardDashboard /></RoleRoute>} />
        <Route path="/resident" element={<ResidentFeed />} />
        <Route path="/profile" element={<Profile />} />
        <Route path="/users" element={<RoleRoute allow={a => a.isAdmin}><Users /></RoleRoute>} />
        <Route path="/face-registration" element={<RoleRoute allow={a => a.user?.role === 'resident'}><FaceRegistration /></RoleRoute>} />
        <Route path="/vehicle-requests" element={<RoleRoute allow={a => a.user?.role === 'resident'}><VehicleRequests /></RoleRoute>} />
        <Route path="/guard-verification" element={<RoleRoute allow={a => a.isGuard}><GuardVerification /></RoleRoute>} />
      </Route>
      <Route path="*" element={<Navigate to="/" />} />
    </Routes>
  );
}

function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <AuthProvider>
          <AppRoutes />
        </AuthProvider>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
