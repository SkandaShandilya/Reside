import React, { useEffect, useState } from 'react';
import { Shield, Database, Cpu, Save, CheckCircle } from 'lucide-react';
import PageHeader from '../components/ui/PageHeader';
import api from '../api/client';

const SystemSettings: React.FC = () => {
  // These three are the only settings genuinely wired into the live
  // detection pipeline (see backend/app/api/settings.py) — changing them
  // here takes effect on already-running camera streams immediately.
  const [processingFps, setProcessingFps] = useState(6);
  const [confidenceThreshold, setConfidenceThreshold] = useState(80);
  const [alertCooldown, setAlertCooldown] = useState(30);

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [toast, setToast] = useState<string | null>(null);

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3500);
  };

  useEffect(() => {
    api.getSystemSettings().then(s => {
      setProcessingFps(s.ai_analysis_fps);
      setConfidenceThreshold(Math.round(s.confidence_threshold * 100));
      setAlertCooldown(s.alert_cooldown_seconds);
    }).catch(() => {}).finally(() => setLoading(false));
  }, []);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    try {
      await api.updateSystemSettings({
        ai_analysis_fps: processingFps,
        confidence_threshold: confidenceThreshold / 100,
        alert_cooldown_seconds: alertCooldown,
      });
      showToast('Settings saved and applied to all running camera streams.');
    } catch (err: any) {
      showToast(err.message || 'Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div>
      {/* Toast */}
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

      <PageHeader
        title="System Settings"
        description="These settings apply live to every currently-running camera stream — no restart needed."
      />

      <form onSubmit={handleSave}>
        <div className="two-col">
          <div>
            <div className="card" style={{ marginBottom: 20 }}>
              <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
                <Cpu size={18} style={{ color: 'var(--color-info)' }} /> AI Processing
              </h3>
              <div className="form-group">
                <label className="form-label">AI Analysis Rate ({processingFps}/sec)</label>
                <input
                  type="range" min={1} max={30} value={processingFps}
                  onChange={e => setProcessingFps(Number(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--accent-primary)' }}
                  disabled={loading}
                />
                <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                  How often (per second) each camera runs a fresh YOLO detection pass. The video
                  itself always streams at full native FPS — this only controls how often the GPU
                  is asked to re-detect. Lower = less GPU load with several cameras open.
                </p>
              </div>
              <div className="form-group">
                <label className="form-label">Confidence Threshold ({confidenceThreshold}%)</label>
                <input
                  type="range" min={50} max={99} value={confidenceThreshold}
                  onChange={e => setConfidenceThreshold(Number(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--accent-primary)' }}
                  disabled={loading}
                />
                <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                  Detections below this confidence are routed to the review queue instead of
                  raising a direct alert.
                </p>
              </div>
              <div className="form-group">
                <label className="form-label">Alert Cooldown</label>
                <input
                  className="input"
                  type="number"
                  min={5} max={600}
                  value={alertCooldown}
                  onChange={e => setAlertCooldown(Number(e.target.value))}
                  style={{ width: 120 }}
                  disabled={loading}
                />
                <span style={{ fontSize: 12, color: 'var(--text-muted)', marginLeft: 8 }}>seconds</span>
                <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                  Minimum time between repeat alerts for the same activity on the same camera.
                </p>
              </div>
            </div>
          </div>

          <div>
            <div className="card" style={{ marginBottom: 20 }}>
              <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
                <Shield size={18} style={{ color: 'var(--color-info)' }} /> Security &amp; Privacy (fixed)
              </h3>
              <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12 }}>
                These reflect how the system is actually built, not adjustable toggles —
                shown here for transparency rather than as settings that could silently do nothing.
              </p>
              <ul style={{ fontSize: 13, color: 'var(--text-secondary)', paddingLeft: 18, lineHeight: 1.8 }}>
                <li>Every API endpoint requires a valid JWT and enforces role-based access on the backend</li>
                <li>CCTV access is restricted by role (Resident/Guard/Admin) at the API layer, not just hidden in the UI</li>
                <li>No facial recognition is implemented anywhere in this system</li>
                <li>Resident-facing feeds omit camera names and identifying detail by design</li>
              </ul>
            </div>

            <div className="card" style={{ marginBottom: 20 }}>
              <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
                <Database size={18} style={{ color: 'var(--color-info)' }} /> Storage
              </h3>
              <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                Recordings and snapshots are written to local disk under <code>./storage/</code>.
                Per-camera retention period is configured on each camera's own Settings tab.
                Cloud storage (S3) and automatic retention cleanup are not implemented yet.
              </p>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 8 }}>
          <button type="submit" className="btn btn-primary btn-lg" disabled={saving || loading}>
            <Save size={16} /> {saving ? 'Saving...' : 'Save Settings'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default SystemSettings;
