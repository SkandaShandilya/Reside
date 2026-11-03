import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Plus, Settings, Brain, WifiOff, Maximize2, Trash2 } from 'lucide-react';
import api from '../api/client';
import type { Camera as CameraType, Alert } from '../types';
import { useWebSocketStream } from '../hooks/useWebSocketStream';
import { useAuth } from '../contexts/AuthContext';
import AddCameraModal from '../components/AddCameraModal';
import ConfirmDialog from '../components/ConfirmDialog';
import AlertRow from '../components/ui/AlertRow';

const spinnerCSS = `@keyframes cam-spin { to { transform: rotate(360deg); } }`;

const sourceTypeLabels: Record<string, string> = {
  rtsp: 'IP Camera',
  webcam: 'Webcam',
  upload: 'Video File',
  network: 'Network Cam',
  nvr: 'NVR',
  RTSP: 'IP Camera',
  UPLOAD: 'Video File',
};

// ------------------------------------------------------------------
// CameraFeed — single camera tile with MJPEG stream
// ------------------------------------------------------------------
interface CameraFeedProps {
  cam: CameraType;
  onFullscreen?: (cam: CameraType) => void;
  paused?: boolean;
  isAdmin?: boolean;
  onDelete?: (id: number) => void;
}

const CameraFeed: React.FC<CameraFeedProps> = ({ cam, onFullscreen, paused, isAdmin, onDelete }) => {
  const isStreamable = cam.is_active === true || cam.is_online === true || cam.source_type?.toLowerCase() === 'upload';
  const src = sourceTypeLabels[cam.source_type] || cam.source_type;
  // Frames pushed over a WebSocket instead of either a persistent MJPEG
  // connection (holds one of the browser's shared ~6-per-host HTTP/1.1
  // slots forever) or a polled snapshot (choppy/delayed at a safe
  // interval) — see hooks/useWebSocketStream.ts. Reconnects automatically
  // on drop, so there's no separate error/retry state to manage here:
  // `streamUrl` is simply null while connecting/reconnecting.
  const streamUrl = useWebSocketStream(cam.id, { variant: 'full', enabled: isStreamable && !paused });
  const connected = !!streamUrl;

  return (
    <div className="camera-card" style={{ position: 'relative' }}>
      <div className="camera-feed" style={{ position: 'relative', background: '#060a12', overflow: 'hidden' }}>

        {/* Live badge */}
        <div className="live-indicator">
          <span style={{
            display: 'flex', alignItems: 'center', gap: 5,
            color: !isStreamable ? '#445' : !connected ? '#f59e0b' : 'var(--color-critical)',
            fontSize: 10, fontWeight: 700, letterSpacing: 1,
          }}>
            {isStreamable && (
              <span style={{
                width: 6, height: 6,
                background: !connected ? '#f59e0b' : 'var(--color-critical)',
                borderRadius: '50%', display: 'inline-block',
                animation: connected ? 'pulse-glow 1.5s infinite' : 'none',
              }} />
            )}
            {!isStreamable ? 'OFFLINE' : !connected ? 'CONNECTING' : 'LIVE'}
          </span>
        </div>

        {/* Source chip */}
        <div style={{
          position: 'absolute', bottom: 38, left: 8, zIndex: 10,
          background: 'rgba(0,0,0,0.65)', color: '#667788', fontSize: 9,
          padding: '2px 6px', borderRadius: 3, letterSpacing: 1, fontFamily: 'monospace',
        }}>{src}</div>

        {/* Frames pushed over a WebSocket (see hooks/useWebSocketStream.ts)
            — smooth, low-latency, and doesn't hold open one of the
            browser's shared per-host connection slots the way a
            persistent MJPEG <img> would, so many tiles across many open
            tabs can render at once. Still skipped while shown fullscreen
            (via `paused`) purely to avoid opening the same camera's
            connection twice at once. */}
        {isStreamable && !paused && connected && (
          <img
            src={streamUrl!}
            alt={`${cam.name} live`}
            style={{
              position: 'absolute', inset: 0,
              width: '100%', height: '100%',
              objectFit: 'cover',
              zIndex: 1,
            }}
          />
        )}

        {/* Spinner overlay — shown until the first frame arrives (or again
            if the connection drops and is silently reconnecting) */}
        {isStreamable && !connected && (
          <div style={{
            position: 'absolute', inset: 0, zIndex: 2,
            background: 'rgba(6,10,18,0.7)',
            display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center', gap: 10,
            pointerEvents: 'none',
          }}>
            <div style={{
              width: 24, height: 24,
              border: '2px solid rgba(0,220,140,0.25)',
              borderTopColor: 'rgba(0,220,140,0.85)',
              borderRadius: '50%',
              animation: 'cam-spin 0.75s linear infinite',
            }} />
            <span style={{ color: '#667788', fontSize: 11 }}>
              {cam.source_type?.toLowerCase() === 'upload' ? 'Loading video...' : 'Connecting...'}
            </span>
          </div>
        )}

        {/* Offline overlay */}
        {!isStreamable && (
          <div style={{
            position: 'absolute', inset: 0, zIndex: 3,
            background: 'rgba(6,10,18,0.93)',
            display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center', gap: 10,
          }}>
            <WifiOff size={30} style={{ color: '#445', opacity: 0.5 }} />
            <span style={{ color: '#556', fontSize: 11 }}>Camera offline</span>
          </div>
        )}

        {/* AI badge */}
        {isStreamable && connected && (
          <div className="camera-ai-overlay" style={{ zIndex: 10 }}>
            <span className="camera-ai-status" style={{ color: cam.ai_enabled ? 'var(--color-success)' : 'var(--text-muted)' }}>
              <Brain size={11} style={{ marginRight: 3, verticalAlign: 'middle' }} />
              AI {cam.ai_enabled ? 'ON' : 'OFF'}
            </span>
            <span className="camera-ai-status">{cam.fps || 30} FPS</span>
          </div>
        )}

        {/* Fullscreen button — floating circular glass control */}
        {onFullscreen && isStreamable && (
          <div className="camera-float-controls">
            <button
              className="camera-float-btn"
              onClick={e => { e.preventDefault(); onFullscreen(cam); }}
              title="Fullscreen"
            >
              <Maximize2 size={13} />
            </button>
          </div>
        )}
      </div>

      {/* Camera Info Bar */}
      <div className="camera-info">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
          <div>
            <div className="camera-name">{cam.name}</div>
            <div className="camera-location">{cam.location}</div>
          </div>
          <span style={{
            fontSize: 10, fontFamily: "'JetBrains Mono', monospace",
            color: 'var(--text-dim)', background: 'var(--bg-tertiary)',
            padding: '2px 6px', borderRadius: 4,
          }}>
            CAM-{String(cam.id).padStart(2, '0')}
          </span>
        </div>
        <div className="camera-meta">
          <span>{src}</span>
          <span>AI: {cam.ai_processing_fps} FPS</span>
          {cam.recording_enabled && <span style={{ color: 'var(--color-critical)' }}>● REC</span>}
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
// Fullscreen Modal
// ------------------------------------------------------------------
const FullscreenModal: React.FC<{ cam: CameraType; onClose: () => void }> = ({ cam, onClose }) => {
  const streamUrl = useWebSocketStream(cam.id, { variant: 'full' });

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [onClose]);

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.92)',
        zIndex: 9999, display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
      }}
    >
      <div onClick={e => e.stopPropagation()} style={{ width: '90vw', maxWidth: 1280 }}>
        {/* Header */}
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          marginBottom: 12, padding: '0 4px',
        }}>
          <div>
            <span style={{ fontWeight: 700, fontSize: 16, color: '#fff' }}>{cam.name}</span>
            <span style={{ color: 'var(--text-muted)', marginLeft: 12, fontSize: 13 }}>{cam.location}</span>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'var(--bg-tertiary)', border: '1px solid var(--border-default)',
              borderRadius: 6, padding: '6px 14px', color: 'var(--text-secondary)',
              cursor: 'pointer', fontSize: 13,
            }}
          >
            ESC / Close
          </button>
        </div>

        {/* Stream */}
        <div style={{
          background: '#000', borderRadius: 8, overflow: 'hidden',
          aspectRatio: '16/9', border: '1px solid var(--border-default)',
          position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          {!streamUrl ? (
            <div style={{ color: '#667788', fontSize: 13, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
              <div style={{
                width: 28, height: 28, border: '2px solid rgba(0,220,140,0.25)',
                borderTopColor: 'rgba(0,220,140,0.85)', borderRadius: '50%',
                animation: 'cam-spin 0.75s linear infinite',
              }} />
              <span>Connecting...</span>
            </div>
          ) : (
            <img
              src={streamUrl}
              alt={`${cam.name} fullscreen`}
              style={{ width: '100%', height: '100%', objectFit: 'contain' }}
            />
          )}
        </div>

        {/* Footer stats */}
        <div style={{
          display: 'flex', gap: 16, marginTop: 12, fontSize: 12,
          color: 'var(--text-muted)', justifyContent: 'center',
        }}>
          <span>● {cam.resolution}</span>
          <span>● {cam.fps} FPS</span>
          <span style={{ color: cam.ai_enabled ? 'var(--color-success)' : 'var(--text-dim)' }}>
            ● AI {cam.ai_enabled ? 'Active' : 'Off'}
          </span>
          {cam.recording_enabled && <span style={{ color: 'var(--color-critical)' }}>● RECORDING</span>}
        </div>
      </div>
    </div>
  );
};

// ------------------------------------------------------------------
// LiveMonitoring Page
// ------------------------------------------------------------------
const LiveMonitoring: React.FC = () => {
  const navigate = useNavigate();
  const { isAdmin, isGuard } = useAuth();
  const [cameras, setCameras] = useState<CameraType[]>([]);
  const [loading, setLoading] = useState(true);
  const [gridCols, setGridCols] = useState<2 | 3 | 4>(3);
  const [fullscreenCam, setFullscreenCam] = useState<CameraType | null>(null);
  const [filter, setFilter] = useState<'all' | 'online' | 'uploads' | 'offline'>('all');
  const [showAddModal, setShowAddModal] = useState(false);
  const [recentAlerts, setRecentAlerts] = useState<Alert[]>([]);

  useEffect(() => {
    api.getCameras().then(data => {
      setCameras(data);
      setLoading(false);
    }).catch(() => setLoading(false));
    // Same GET /api/alerts already used by the navbar/notification popover
    // — no new backend call invented, just a second, contextual place to
    // surface it next to the live feeds.
    if (isGuard) {
      api.getAlerts({ page_size: '6', status: 'unresolved' }).then(res => setRecentAlerts(res.alerts || [])).catch(() => {});
    }
  }, [isGuard]);

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

  const filteredCameras = cameras.filter(c => {
    const st = c.source_type?.toLowerCase();
    if (filter === 'online') return c.is_online && st !== 'upload';
    if (filter === 'uploads') return st === 'upload';
    if (filter === 'offline') return !c.is_online && !c.is_active;
    return true;
  });

  const activeCams = cameras.filter(c => c.is_online || c.is_active);
  const uploadCams = cameras.filter(c => c.source_type?.toLowerCase() === 'upload');

  return (
    <div>
      {/* Fullscreen modal */}
      {fullscreenCam && (
        <FullscreenModal cam={fullscreenCam} onClose={() => setFullscreenCam(null)} />
      )}

      <style>{spinnerCSS}</style>

      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20, flexWrap: 'wrap', gap: 12 }}>
        <div>
          <h1 className="page-title" style={{ marginBottom: 4 }}>Live Monitoring</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>
            {activeCams.length} of {cameras.length} cameras active
            {uploadCams.length > 0 && (
              <span style={{ color: 'var(--color-info)', marginLeft: 10 }}>● {uploadCams.length} uploaded video{uploadCams.length !== 1 ? 's' : ''}</span>
            )}
          </p>
        </div>

        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
          {/* Filter */}
          <div style={{ display: 'flex', gap: 4 }}>
            {([['all','All'], ['online','Online'], ['uploads','📹 Uploads'], ['offline','Offline']] as const).map(([f, label]) => (
              <button
                key={f}
                onClick={() => setFilter(f as any)}
                className={`btn btn-sm ${filter === f ? 'btn-primary' : ''}`}
                style={{ fontSize: 11 }}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Grid layout */}
          <div style={{ display: 'flex', gap: 4 }}>
            {([2, 3, 4] as const).map(cols => (
              <button
                key={cols}
                onClick={() => setGridCols(cols)}
                className={`btn btn-sm ${gridCols === cols ? 'btn-primary' : ''}`}
                style={{ fontSize: 12, minWidth: 32 }}
              >
                {cols}×
              </button>
            ))}
          </div>

          {isAdmin && (
            <button className="btn btn-primary btn-sm" onClick={() => setShowAddModal(true)}>
              <Plus size={14} /> Add Video Source
            </button>
          )}
        </div>
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

      <div className={isGuard ? 'live-split' : undefined}>
        {/* Camera Grid */}
        <div>
          {loading ? (
            <div style={{ textAlign: 'center', padding: 60, color: 'var(--text-muted)' }}>
              Loading cameras...
            </div>
          ) : filteredCameras.length === 0 ? (
            <div className="empty-state">
              <div className="icon">📹</div>
              <h3>{filter !== 'all' ? `No ${filter} cameras` : 'No cameras configured'}</h3>
              <p>
                {filter !== 'all'
                  ? 'Try changing the filter.'
                  : 'Add your first camera to start monitoring.'}
              </p>
              {filter === 'all' && isAdmin && (
                <button className="btn btn-primary" style={{ marginTop: 16 }} onClick={() => setShowAddModal(true)}>
                  <Plus size={16} /> Add Camera
                </button>
              )}
            </div>
          ) : (
            <div
              className="camera-grid"
              style={{
                gridTemplateColumns: `repeat(${gridCols}, 1fr)`,
              }}
            >
              {filteredCameras.map(cam => (
                <CameraFeed
                  key={cam.id}
                  cam={cam}
                  onFullscreen={setFullscreenCam}
                  paused={fullscreenCam?.id === cam.id}
                  isAdmin={isAdmin}
                  onDelete={setDeleteTargetId}
                />
              ))}
            </div>
          )}
        </div>

        {/* Recent Alerts side panel */}
        {isGuard && (
          <div className="card" style={{ padding: 16, alignSelf: 'start' }}>
            <div className="section-header" style={{ marginBottom: 12 }}>
              <h3 className="section-title" style={{ fontSize: 14 }}>Recent Alerts</h3>
              <Link to="/alerts" className="btn btn-ghost btn-sm">View All</Link>
            </div>
            {recentAlerts.length === 0 ? (
              <p style={{ color: 'var(--text-muted)', fontSize: 12.5, padding: '8px 0' }}>No unresolved alerts</p>
            ) : (
              <div>
                {recentAlerts.map(alert => (
                  <AlertRow key={alert.id} alert={alert} onClick={() => navigate(`/alerts/${alert.id}`)} />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default LiveMonitoring;
