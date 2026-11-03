import React, { useEffect, useState } from 'react';
import { Shield, Bell, AlertTriangle, X, CheckCircle } from 'lucide-react';
import api from '../api/client';
import { useAuth } from '../contexts/AuthContext';
import type { Alert } from '../types';
import PendingVisitorRequests from '../components/PendingVisitorRequests';

const ResidentFeed: React.FC = () => {
  const { user } = useAuth();
  const [myReports, setMyReports] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(true);
  const [showReportModal, setShowReportModal] = useState(false);
  const [reportLocation, setReportLocation] = useState('');
  const [reportDescription, setReportDescription] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [reportError, setReportError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  // A Resident is not a security operator and must never see AI-detected
  // security alerts (Behaviour/Fire-Smoke), at any severity — only the
  // reports they personally submitted. GET /api/alerts now enforces this
  // server-side too (403 for anything but reported_by=self), so this is
  // the only query this page is allowed to make against that endpoint.
  const loadMyReports = () => {
    if (!user) { setLoading(false); return; }
    api.getAlerts({ reported_by: String(user.id), page_size: '10' })
      .then(res => setMyReports(res.alerts))
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadMyReports();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.id]);

  const handleSubmitReport = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!reportLocation.trim()) {
      setReportError('Please enter a location.');
      return;
    }
    if (!reportDescription.trim()) {
      setReportError('Please describe what you observed.');
      return;
    }
    setSubmitting(true);
    setReportError(null);
    try {
      await api.reportSuspiciousActivity(reportLocation.trim(), reportDescription.trim());
      setShowReportModal(false);
      setReportDescription('');
      setReportLocation('');
      setToast('Report submitted. Security has been notified.');
      setTimeout(() => setToast(null), 4000);
      loadMyReports();
    } catch (err: any) {
      setReportError(err.message || 'Failed to submit report.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div style={{ maxWidth: 600, margin: '0 auto' }}>
      <PendingVisitorRequests />
      {toast && (
        <div style={{
          position: 'fixed', top: 20, right: 20, zIndex: 2000,
          padding: '12px 18px', borderRadius: 10, fontWeight: 600, fontSize: 13,
          background: 'var(--color-success-bg)', color: 'var(--color-success)',
          border: '1px solid var(--color-success-border)',
          display: 'flex', alignItems: 'center', gap: 8, boxShadow: '0 4px 24px rgba(0,0,0,0.3)',
        }}>
          <CheckCircle size={15} /> {toast}
        </div>
      )}

      <div style={{ textAlign: 'center', marginBottom: 32 }}>
        <div style={{
          width: 60, height: 60, background: 'var(--accent-gradient)',
          borderRadius: 'var(--radius-lg)', display: 'inline-flex',
          alignItems: 'center', justifyContent: 'center', marginBottom: 16,
        }}>
          <Shield size={28} color="white" />
        </div>
        <h1 className="page-title" style={{ marginBottom: 4 }}>Community Security</h1>
        <p style={{ color: 'var(--text-muted)', fontSize: 14 }}>Report suspicious activity and track your reports</p>
      </div>

      {/* Report suspicious activity */}
      <button
        className="btn btn-primary"
        style={{ width: '100%', justifyContent: 'center', marginBottom: 24 }}
        onClick={() => setShowReportModal(true)}
      >
        <AlertTriangle size={16} /> Report Suspicious Activity
      </button>

      {/* Status — deliberately static, not sourced from the security-alert
          feed: a Resident does not receive Behaviour/Fire-Smoke alerts
          (see App.tsx/Sidebar.tsx/api/alerts.py for the enforced role
          separation), so this page only reflects what a resident IS
          entitled to see — their own submitted reports below. */}
      <div className="card" style={{ textAlign: 'center', marginBottom: 24 }}>
        <div style={{ fontSize: 32, marginBottom: 8 }}>🏠</div>
        <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-primary)' }}>
          Welcome{user?.full_name ? `, ${user.full_name}` : ''}
        </div>
        <div style={{ color: 'var(--text-muted)', fontSize: 13, marginTop: 4 }}>
          Security personnel monitor the property around the clock. Use the button above to report anything unusual you observe.
        </div>
      </div>

      {!loading && myReports.length === 0 && (
        <div className="card" style={{ textAlign: 'center', padding: 40 }}>
          <Bell size={32} style={{ color: 'var(--text-dim)', marginBottom: 12 }} />
          <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>You haven't submitted any reports</p>
        </div>
      )}

      {/* My submitted reports */}
      {myReports.length > 0 && (
        <div style={{ marginTop: 32 }}>
          <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 12, color: 'var(--text-secondary)' }}>
            My Reports
          </h3>
          {myReports.map(report => (
            <div key={report.id} className="card" style={{ marginBottom: 12 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', gap: 8 }}>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{report.notes}</div>
                  <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 6 }}>
                    {report.camera_location} · {new Date(report.created_at).toLocaleString()}
                  </div>
                </div>
                <span className={`status-badge ${report.status}`} style={{ whiteSpace: 'nowrap' }}>
                  {report.status.replace('_', ' ')}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Report modal */}
      {showReportModal && (
        <div className="modal-overlay" onClick={() => setShowReportModal(false)}>
          <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 480 }}>
            <div className="modal-header">
              <h3 className="modal-title">Report Suspicious Activity</h3>
              <button className="btn btn-ghost btn-sm" onClick={() => setShowReportModal(false)}>
                <X size={16} />
              </button>
            </div>
            <form onSubmit={handleSubmitReport}>
              <div className="modal-body">
                <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 16 }}>
                  Describe what you observed. Security personnel will review it — this does not
                  need to be a confirmed incident, just what you saw.
                </p>
                <div className="form-group">
                  <label className="form-label">Location</label>
                  <input
                    className="input"
                    value={reportLocation}
                    maxLength={255}
                    onChange={e => setReportLocation(e.target.value)}
                    placeholder="e.g., Near Block B parking entrance"
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">What did you observe?</label>
                  <textarea
                    className="input"
                    rows={4}
                    value={reportDescription}
                    onChange={e => setReportDescription(e.target.value)}
                    placeholder="e.g., An unfamiliar person has been standing near the main gate for over 15 minutes."
                    style={{ resize: 'vertical', width: '100%' }}
                  />
                </div>
                {reportError && (
                  <div style={{ color: 'var(--color-critical)', fontSize: 12, marginTop: 4 }}>{reportError}</div>
                )}
              </div>
              <div className="modal-footer">
                <button type="button" className="btn btn-ghost" onClick={() => setShowReportModal(false)}>Cancel</button>
                <button type="submit" className="btn btn-primary" disabled={submitting}>
                  {submitting ? 'Submitting...' : 'Submit Report'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default ResidentFeed;
