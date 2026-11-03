import React, { useEffect, useState, useRef, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { Flame, ShieldAlert, WifiOff, CheckCircle, Eye, Clock, Plus, Settings, Trash2 } from 'lucide-react';
import api from '../api/client';
import type { Camera as CameraType, Alert, SecurityEvent } from '../types';
import { useWebSocketStream } from '../hooks/useWebSocketStream';
import { useAuth } from '../contexts/AuthContext';
import AddCameraModal from '../components/AddCameraModal';
import ConfirmDialog from '../components/ConfirmDialog';

// Same underlying camera feed the Live Monitoring / Cameras pages use — a
// SEPARATE server-side render pass draws only fire/smoke results onto it
// (see stream.py: /fire-smoke, inference_engine.render_fire_smoke_only()),
// so this page can never show person boxes/behaviour labels even though
// the same camera capture loop computes both results every frame.
const BASE_URL = '';
const spinnerCSS = `@keyframes fs-spin { to { transform: rotate(360deg); } }`;

interface Telemetry {
  fire_state?: string;
  smoke_state?: string;
  fire_smoothed_confidence?: number;
  smoke_smoothed_confidence?: number;
}

// Builds a servable URL for a relative storage path (e.g. "clips/x.mp4",
// as returned by ai_service/video_processor.py) against backend/app/main.py's
// GET /storage/{file_path:path}. Uses ?token=, not an Authorization header,
// for the same reason the fire/smoke MJPEG stream above does — a plain
// <video>/<img> tag cannot attach custom headers (see app/api/stream.py's
// get_stream_user, which this route now shares).
function storageUrl(relativePath: string, token: string | null): string {
  return `${BASE_URL}/storage/${relativePath}?token=${token || ''}`;
}

// ------------------------------------------------------------------
// Clip/snapshot viewer modal
// ------------------------------------------------------------------
function ClipViewerModal({ clipPath, thumbnailPath, activity, onClose }: {
  clipPath?: string; thumbnailPath?: string; activity: string; onClose: () => void;
}) {
  const token = localStorage.getItem('sentinel_token');
  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 1000, background: 'rgba(0,0,0,0.75)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24,
      }}
    >
      <div onClick={e => e.stopPropagation()} style={{
        background: 'var(--bg-secondary)', borderRadius: 10, padding: 16,
        maxWidth: '90vw', maxHeight: '90vh', display: 'flex', flexDirection: 'column', gap: 10,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <strong style={{ fontSize: 14 }}>{activity} — incident clip</strong>
          <button className="btn btn-sm" onClick={onClose}>Close</button>
        </div>
        {clipPath ? (
          <video
            controls
            autoPlay
            src={storageUrl(clipPath, token)}
            poster={thumbnailPath ? storageUrl(thumbnailPath, token) : undefined}
            style={{ maxWidth: '80vw', maxHeight: '75vh', background: '#000' }}
          />
        ) : thumbnailPath ? (
          <img
            src={storageUrl(thumbnailPath, token)}
            alt={`${activity} snapshot`}
            style={{ maxWidth: '80vw', maxHeight: '75vh' }}
          />
        ) : (
          <div style={{ color: 'var(--text-muted)', padding: 20 }}>No clip or snapshot available for this incident.</div>
        )}
      </div>
    </div>
  );
}

const STATE_COLOR: Record<string, string> = {
  NORMAL: 'var(--text-muted)',
  POSSIBLE: 'var(--color-warning)',
  CONFIRMED: 'var(--color-critical)',
  RECOVERY: 'var(--color-warning)',
};

const STATE_LABEL: Record<string, string> = {
  NORMAL: 'Normal',
  POSSIBLE: 'Possible',
  CONFIRMED: 'CONFIRMED',
  RECOVERY: 'Fading',
};

function StatusPill({ icon, label, state, confidence }: { icon: React.ReactNode; label: string; state: string; confidence: number }) {
  const color = STATE_COLOR[state] || 'var(--text-muted)';
  const isActive = state && state !== 'NORMAL';
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 8,
      background: isActive ? `${color}18` : 'var(--bg-tertiary)',
      border: `1px solid ${isActive ? color : 'var(--border-default)'}`,
      borderRadius: 8, padding: '8px 12px', flex: 1,
    }}>
      <span style={{ color, display: 'flex' }}>{icon}</span>
      <div style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <span style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, letterSpacing: 0.5 }}>{label}</span>
        <span style={{ fontSize: 13, fontWeight: 700, color }}>
          {isActive ? `${STATE_LABEL[state]} — ${Math.round((confidence || 0) * 100)}%` : '—'}
        </span>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------
// Per-camera feed tile
// ------------------------------------------------------------------
interface FireSmokeFeedProps {
  cam: CameraType;
  telemetry: Telemetry;
  isAdmin?: boolean;
  onDelete?: (id: number) => void;
}

const FireSmokeFeed: React.FC<FireSmokeFeedProps> = ({ cam, telemetry, isAdmin, onDelete }) => {
  // Frames pushed over a WebSocket (fire/smoke-overlay variant) instead of
  // either a persistent MJPEG connection or a polled snapshot — see
  // hooks/useWebSocketStream.ts. Reconnects automatically on drop, so
  // there's no separate error/retry state to manage here.
  const streamUrl = useWebSocketStream(cam.id, { variant: 'fire_smoke' });
  const connected = !!streamUrl;
  const isStreamable = cam.is_active === true || cam.is_online === true || cam.source_type?.toLowerCase() === 'upload';

  const fireState = telemetry.fire_state || 'NORMAL';
  const smokeState = telemetry.smoke_state || 'NORMAL';
  const isHazard = fireState === 'CONFIRMED' || smokeState === 'CONFIRMED';

  // Fire and smoke are shown as ONE combined FIRE/SMOKE indicator — the
  // backend already tracks them independently (fireState/smokeState
  // above, still read from the same telemetry the backend provides, not
  // invented client-side) and that per-class detail is preserved in the
  // Alert's metadata; this is purely a display combination: whichever
  // sub-state is currently "most active" wins for the single pill.
  const STATE_RANK: Record<string, number> = { NORMAL: 0, POSSIBLE: 1, RECOVERY: 2, CONFIRMED: 3 };
  const combinedState = (STATE_RANK[fireState] ?? 0) >= (STATE_RANK[smokeState] ?? 0) ? fireState : smokeState;
  const combinedConfidence = Math.max(telemetry.fire_smoothed_confidence || 0, telemetry.smoke_smoothed_confidence || 0);

  return (
    <div className="camera-card" style={{ position: 'relative', border: isHazard ? '1px solid var(--color-critical)' : undefined }}>
      <div className="camera-feed" style={{ position: 'relative', background: '#060a12', overflow: 'hidden' }}>
        {isStreamable && connected && (
          <img
            src={streamUrl!}
            alt={`${cam.name} fire/smoke feed`}
            style={{
              position: 'absolute', inset: 0, width: '100%', height: '100%',
              objectFit: 'cover', zIndex: 1,
            }}
          />
        )}

        {isStreamable && !connected && (
          <div style={{
            position: 'absolute', inset: 0, zIndex: 2, background: 'rgba(6,10,18,0.7)',
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 10,
          }}>
            <div style={{
              width: 24, height: 24, border: '2px solid rgba(239,68,68,0.25)',
              borderTopColor: 'rgba(239,68,68,0.85)', borderRadius: '50%',
              animation: 'fs-spin 0.75s linear infinite',
            }} />
            <span style={{ color: '#667788', fontSize: 11 }}>Connecting...</span>
          </div>
        )}

        {!isStreamable && (
          <div style={{
            position: 'absolute', inset: 0, zIndex: 3, background: 'rgba(6,10,18,0.93)',
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 10,
          }}>
            <WifiOff size={30} style={{ color: '#445', opacity: 0.5 }} />
            <span style={{ color: '#556', fontSize: 11 }}>Camera offline</span>
          </div>
        )}

        {isHazard && (
          <div style={{
            position: 'absolute', top: 0, left: 0, right: 0, zIndex: 10,
            background: 'var(--color-critical)', color: '#fff', textAlign: 'center',
            fontSize: 12, fontWeight: 700, letterSpacing: 1, padding: '4px 0',
          }}>
            🔥 FIRE/SMOKE DETECTED
          </div>
        )}
      </div>

      <div className="camera-info">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: 10 }}>
          <div>
            <div className="camera-name">{cam.name}</div>
            <div className="camera-location">{cam.location}</div>
          </div>
          <span style={{
            fontSize: 10, fontFamily: "'JetBrains Mono', monospace", color: 'var(--text-dim)',
            background: 'var(--bg-tertiary)', padding: '2px 6px', borderRadius: 4,
          }}>
            CAM-{String(cam.id).padStart(2, '0')}
          </span>
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
          <StatusPill icon={<Flame size={16} />} label="FIRE/SMOKE" state={combinedState} confidence={combinedConfidence} />
        </div>

        {isAdmin && (
          <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
            <Link to={`/cameras/${cam.id}`} className="btn btn-ghost btn-sm" style={{ flex: 1, justifyContent: 'center' }}>
              <Settings size={14} /> Configure
            </Link>
            <button className="btn btn-danger btn-sm" onClick={() => onDelete && onDelete(cam.id)}>
              <Trash2 size={14} />
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

// ------------------------------------------------------------------
// Page-level aggregate status banner
// ------------------------------------------------------------------
function AggregateStatusBanner({ telemetryByCam }: { telemetryByCam: Record<number, Telemetry> }) {
  const values = Object.values(telemetryByCam);
  // Fire OR smoke (or both) on ANY camera -> one combined "detected" state
  // — no separate fire-only/smoke-only/both wording.
  const confirmed = values.some(t => t.fire_state === 'CONFIRMED' || t.smoke_state === 'CONFIRMED');
  const possible = values.some(t => t.fire_state === 'POSSIBLE' || t.smoke_state === 'POSSIBLE');

  let icon = '🟢', text = 'NO FIRE/SMOKE DETECTED', color = 'var(--color-success)', bg = 'var(--color-success-bg)', border = 'var(--color-success-border)';
  if (confirmed) { icon = '🔴'; text = 'FIRE/SMOKE DETECTED'; color = 'var(--color-critical)'; bg = 'var(--color-critical-bg)'; border = 'var(--color-critical-border)'; }
  else if (possible) { icon = '🟠'; text = 'POSSIBLE FIRE/SMOKE'; color = 'var(--color-warning)'; bg = 'var(--color-warning-bg)'; border = 'var(--color-warning-border)'; }

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 10, padding: '10px 16px',
      borderRadius: 8, border: `1px solid ${border}`, background: bg, marginBottom: 16,
    }}>
      <span style={{ fontSize: 18 }}>{icon}</span>
      <span style={{ fontWeight: 700, fontSize: 14, color }}>{text}</span>
      <span style={{ fontSize: 12, color: 'var(--text-muted)', marginLeft: 4 }}>
        Behaviour activity does not affect this status.
      </span>
    </div>
  );
}

// ------------------------------------------------------------------
// Dedicated Fire/Smoke alert panel
// ------------------------------------------------------------------
function FireSmokeAlertPanel({ alerts, onAcknowledge, onView }: { alerts: Alert[]; onAcknowledge: (id: number) => void; onView: (a: Alert) => void }) {
  if (alerts.length === 0) {
    return (
      <div style={{ color: 'var(--text-muted)', fontSize: 13, padding: '16px 0' }}>
        No active fire/smoke alerts.
      </div>
    );
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {alerts.map(a => (
        <div key={a.id} style={{
          border: '1px solid var(--color-critical-border)', background: 'var(--color-critical-bg)', borderRadius: 8, padding: 14,
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
            <div style={{ fontWeight: 700, fontSize: 14, color: 'var(--color-critical)', display: 'flex', alignItems: 'center', gap: 6 }}>
              {a.activity.includes('Smoke') && !a.activity.includes('Fire') ? '💨' : '🔥'} {a.activity.toUpperCase()}
            </div>
            <span style={{
              fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 4,
              background: a.severity === 'critical' ? 'var(--color-critical)' : 'var(--color-warning)', color: '#fff',
            }}>{a.severity.toUpperCase()}</span>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 6, display: 'flex', gap: 14, flexWrap: 'wrap' }}>
            <span>Camera: {a.camera_name}</span>
            {a.duration_seconds != null && <span>Duration: {a.duration_seconds.toFixed(1)}s</span>}
            <span><Clock size={11} style={{ verticalAlign: 'middle', marginRight: 2 }} />{new Date(a.created_at).toLocaleTimeString()}</span>
          </div>
          <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
            {a.status === 'unresolved' && (
              <button className="btn btn-sm" onClick={() => onAcknowledge(a.id)}>
                <CheckCircle size={13} /> Acknowledge
              </button>
            )}
            {a.status !== 'unresolved' && (
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Status: {a.status}</span>
            )}
            {(a.video_clip_path || a.thumbnail_path) && (
              <button className="btn btn-sm" onClick={() => onView(a)}>
                <Eye size={13} /> View Clip
              </button>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

// ------------------------------------------------------------------
// Dedicated Fire/Smoke incident history
// ------------------------------------------------------------------
function FireSmokeHistory({ incidents, onView }: { incidents: SecurityEvent[]; onView: (e: SecurityEvent) => void }) {
  if (incidents.length === 0) {
    return <div style={{ color: 'var(--text-muted)', fontSize: 13, padding: '16px 0' }}>No fire/smoke incidents recorded yet.</div>;
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {incidents.map(e => (
        <div key={e.id} style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          padding: '10px 12px', borderRadius: 6, background: 'var(--bg-tertiary)', fontSize: 12,
        }}>
          <div>
            <div style={{ color: 'var(--text-dim)', fontSize: 11 }}>{new Date(e.timestamp).toLocaleString()}</div>
            <div style={{ fontWeight: 600, marginTop: 2 }}>
              {e.activity_class.includes('Smoke') && !e.activity_class.includes('Fire') ? '💨' : '🔥'} {e.activity_class}
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            {(e.video_clip_path || e.thumbnail_path) && (
              <a href="#" onClick={ev => { ev.preventDefault(); onView(e); }} style={{ fontSize: 11, color: 'var(--color-info)', display: 'flex', alignItems: 'center', gap: 3, cursor: 'pointer' }}>
                <Eye size={11} /> View
              </a>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

// ------------------------------------------------------------------
// Page
// ------------------------------------------------------------------
const FireSmoke: React.FC = () => {
  const { isAdmin } = useAuth();
  const [cameras, setCameras] = useState<CameraType[]>([]);
  const [loading, setLoading] = useState(true);
  const [telemetryByCam, setTelemetryByCam] = useState<Record<number, Telemetry>>({});
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [incidents, setIncidents] = useState<SecurityEvent[]>([]);
  const [viewing, setViewing] = useState<{ clipPath?: string; thumbnailPath?: string; activity: string } | null>(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const pollTimer = useRef<any>(null);

  useEffect(() => {
    api.getCameras().then(data => {
      setCameras(data);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  const pollAll = useCallback(async () => {
    if (cameras.length === 0) return;
    const entries = await Promise.all(cameras.map(async c => {
      try {
        const t = await api.getCameraTelemetry(c.id);
        return [c.id, t] as const;
      } catch {
        return [c.id, {}] as const;
      }
    }));
    setTelemetryByCam(Object.fromEntries(entries));
  }, [cameras]);

  const fetchAlertsAndHistory = useCallback(async () => {
    try {
      const [alertRes, incidentRes] = await Promise.all([
        api.getAlerts({ category: 'fire_smoke', page_size: '10' }),
        api.getIncidents({ category: 'fire_smoke', page_size: '15' }),
      ]);
      setAlerts(alertRes.alerts || []);
      setIncidents(incidentRes.incidents || []);
    } catch {
      // best-effort — page still works without these panels populated
    }
  }, []);

  useEffect(() => {
    pollAll();
    fetchAlertsAndHistory();
    pollTimer.current = setInterval(() => {
      pollAll();
      fetchAlertsAndHistory();
    }, 4000);
    return () => { if (pollTimer.current) clearInterval(pollTimer.current); };
  }, [pollAll, fetchAlertsAndHistory]);

  const handleAcknowledge = async (id: number) => {
    try {
      await api.acknowledgeAlert(id);
      fetchAlertsAndHistory();
    } catch (e) {
      console.error(e);
    }
  };

  const [deleteTargetId, setDeleteTargetId] = useState<number | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const confirmDelete = async () => {
    if (deleteTargetId == null) return;
    const id = deleteTargetId;
    setDeleteTargetId(null);
    try {
      await api.deleteCamera(id);
      setCameras(prev => prev.filter(c => c.id !== id));
    } catch (err: any) {
      console.error(err);
      setDeleteError(err.message || 'Failed to delete camera.');
    }
  };

  return (
    <div>
      <style>{spinnerCSS}</style>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, flexWrap: 'wrap', gap: 12 }}>
        <div>
          <h1 className="page-title" style={{ marginBottom: 4, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Flame size={22} style={{ color: 'var(--color-critical)' }} /> Fire & Smoke Monitoring
          </h1>
          <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>
            Same live camera feeds as Live Monitoring — a separate, dedicated fire/smoke overlay and alert system.
            <span style={{ marginLeft: 8, color: 'var(--text-dim)' }}>
              <ShieldAlert size={12} style={{ verticalAlign: 'middle', marginRight: 4 }} />
              Detection runs independently of which tab is open.
            </span>
          </p>
        </div>

        {isAdmin && (
          <button className="btn btn-primary btn-sm" onClick={() => setShowAddModal(true)}>
            <Plus size={14} /> Add Video Source
          </button>
        )}
      </div>

      {showAddModal && (
        <AddCameraModal
          onClose={() => setShowAddModal(false)}
          onAdded={(cam) => {
            setCameras(prev => [cam, ...prev]);
            setShowAddModal(false);
          }}
        />
      )}

      {deleteTargetId != null && (
        <ConfirmDialog
          title="Delete Camera"
          message="Delete this camera? This cannot be undone."
          confirmLabel="Delete"
          onConfirm={confirmDelete}
          onCancel={() => setDeleteTargetId(null)}
        />
      )}

      {deleteError && (
        <div className="modal-overlay" onClick={() => setDeleteError(null)}>
          <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 400 }}>
            <div className="modal-header"><h3 className="modal-title">Delete Failed</h3></div>
            <div className="modal-body"><p style={{ color: 'var(--color-critical)', fontSize: 14 }}>{deleteError}</p></div>
            <div className="modal-footer">
              <button className="btn btn-primary" onClick={() => setDeleteError(null)}>OK</button>
            </div>
          </div>
        </div>
      )}

      <AggregateStatusBanner telemetryByCam={telemetryByCam} />

      {loading ? (
        <div style={{ textAlign: 'center', padding: 60, color: 'var(--text-muted)' }}>Loading cameras...</div>
      ) : cameras.length === 0 ? (
        <div className="empty-state">
          <div className="icon">🔥</div>
          <h3>No cameras configured</h3>
          <p>Add a camera to start fire/smoke monitoring.</p>
          {isAdmin && (
            <button className="btn btn-primary" style={{ marginTop: 16 }} onClick={() => setShowAddModal(true)}>
              <Plus size={16} /> Add Video Source
            </button>
          )}
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: '2.2fr 1fr', gap: 20, alignItems: 'start' }}>
          <div className="camera-grid" style={{ gridTemplateColumns: 'repeat(2, 1fr)' }}>
            {cameras.map(cam => (
              <FireSmokeFeed
                key={cam.id}
                cam={cam}
                telemetry={telemetryByCam[cam.id] || {}}
                isAdmin={isAdmin}
                onDelete={setDeleteTargetId}
              />
            ))}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            <div className="card" style={{ padding: 16 }}>
              <h3 style={{ fontSize: 14, marginBottom: 10, display: 'flex', alignItems: 'center', gap: 6 }}>
                🔥 Fire/Smoke Alerts
              </h3>
              <FireSmokeAlertPanel
                alerts={alerts}
                onAcknowledge={handleAcknowledge}
                onView={a => setViewing({ clipPath: a.video_clip_path, thumbnailPath: a.thumbnail_path, activity: a.activity })}
              />
            </div>

            <div className="card" style={{ padding: 16 }}>
              <h3 style={{ fontSize: 14, marginBottom: 10 }}>Fire & Smoke Incidents</h3>
              <FireSmokeHistory
                incidents={incidents}
                onView={e => setViewing({ clipPath: e.video_clip_path, thumbnailPath: e.thumbnail_path, activity: e.activity_class })}
              />
            </div>
          </div>
        </div>
      )}

      {viewing && (
        <ClipViewerModal
          clipPath={viewing.clipPath}
          thumbnailPath={viewing.thumbnailPath}
          activity={viewing.activity}
          onClose={() => setViewing(null)}
        />
      )}
    </div>
  );
};

export default FireSmoke;
