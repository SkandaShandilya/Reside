import React, { useEffect, useRef, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Camera, Wifi, WifiOff, Brain, Settings, Eye, History, AlertTriangle, ArrowLeft, Save, RefreshCw, CheckCircle, Clock, MapPin, Trash2, X } from 'lucide-react';
import api from '../api/client';
import type { Camera as CameraType, Alert as AlertType, CameraZone } from '../types';
import { useWebSocketStream } from '../hooks/useWebSocketStream';
import ConfirmDialog from '../components/ConfirmDialog';

const DEFAULT_DETECTION_RULES = {
  smoke: true,
  fire: true,
  fighting: true,
  falling: true,
  snatching: true,
  loitering: true,
};

const CameraDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [camera, setCamera] = useState<CameraType | null>(null);
  const [activeTab, setActiveTab] = useState('live');
  const [loading, setLoading] = useState(true);
  const [retrying, setRetrying] = useState(false);
  const [savingSettings, setSavingSettings] = useState(false);
  const [cameraAlerts, setCameraAlerts] = useState<AlertType[]>([]);
  const [toast, setToast] = useState<{ msg: string; type: 'success' | 'error' } | null>(null);

  // Settings editable state
  const [detectionRules, setDetectionRules] = useState<Record<string, boolean>>(DEFAULT_DETECTION_RULES);
  const [thresholdPct, setThresholdPct] = useState(75);
  const [cooldownSec, setCooldownSec] = useState(30);
  const [aiEnabled, setAiEnabled] = useState(true);

  // Zone editor state
  const [zones, setZones] = useState<CameraZone[]>([]);
  const [zonesLoading, setZonesLoading] = useState(true);
  const [drawingPoints, setDrawingPoints] = useState<{ x: number; y: number }[]>([]);
  const [zoneName, setZoneName] = useState('');
  const [zoneType, setZoneType] = useState<'restricted' | 'monitoring' | 'normal' | 'asset'>('restricted');
  const [savingZone, setSavingZone] = useState(false);
  // Zone points are stored as fractions of the FRAME; the drawing canvas must
  // match the frame's true aspect ratio or points land in the wrong place.
  const [zoneAspect, setZoneAspect] = useState(16 / 9);

  const token = localStorage.getItem('sentinel_token');
  const isStreamable = camera ? (camera.is_online || camera.is_active || camera.source_type?.toLowerCase() === 'upload') : false;
  // Frames pushed over a WebSocket instead of a persistent MJPEG
  // connection or a polled snapshot — see hooks/useWebSocketStream.ts.
  // Placeholder camera id (0) while `camera` hasn't loaded yet is never
  // actually rendered (early "Loading..." return below).
  const liveStreamUrl = useWebSocketStream(camera?.id ?? 0, { enabled: !!camera });

  const showToast = (msg: string, type: 'success' | 'error' = 'success') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3500);
  };

  const loadCamera = async () => {
    if (!id) return;
    try {
      const data = await api.getCamera(Number(id));
      setCamera(data);
      setAiEnabled(data.ai_enabled ?? true);
      setThresholdPct(Math.round((data.alert_confidence_threshold || 0.75) * 100));
      setCooldownSec(data.cooldown_seconds || 30);
      if (data.detection_config && Object.keys(data.detection_config).length > 0) {
        setDetectionRules(prev => ({ ...prev, ...data.detection_config }));
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const loadCameraAlerts = async () => {
    if (!id) return;
    try {
      const res = await api.getAlerts({ camera_id: id, page_size: '20' });
      setCameraAlerts(res.alerts || []);
    } catch (err) {
      console.error(err);
    }
  };

  const loadZones = async () => {
    if (!id) return;
    setZonesLoading(true);
    try {
      const data = await api.getCameraZones(Number(id));
      setZones(data);
    } catch (err) {
      console.error(err);
    } finally {
      setZonesLoading(false);
    }
  };

  useEffect(() => {
    loadCamera();
    loadCameraAlerts();
    loadZones();
  }, [id]);

  const handleZoneCanvasClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    setDrawingPoints(prev => [...prev, { x, y }]);
  };

  const undoLastPoint = () => setDrawingPoints(prev => prev.slice(0, -1));
  const clearDrawing = () => {
    setDrawingPoints([]);
    setZoneName('');
  };

  const handleSaveZone = async () => {
    if (!id) return;
    if (drawingPoints.length < 3) {
      showToast('Click at least 3 points to draw a zone.', 'error');
      return;
    }
    if (!zoneName.trim()) {
      showToast('Please name this zone.', 'error');
      return;
    }
    setSavingZone(true);
    try {
      await api.createCameraZone(Number(id), {
        name: zoneName.trim(),
        zone_type: zoneType,
        polygon_points: drawingPoints,
        alert_severity: zoneType === 'restricted' || zoneType === 'asset' ? 'critical' : 'warning',
      });
      clearDrawing();
      await loadZones();
      showToast('Zone saved — now active on the live feed.');
    } catch (err: any) {
      showToast(err.message || 'Failed to save zone', 'error');
    } finally {
      setSavingZone(false);
    }
  };

  const [deleteZoneId, setDeleteZoneId] = useState<number | null>(null);

  const confirmDeleteZone = async () => {
    if (!id || deleteZoneId == null) return;
    const zoneId = deleteZoneId;
    setDeleteZoneId(null);
    try {
      await api.deleteCameraZone(Number(id), zoneId);
      await loadZones();
      showToast('Zone deleted.');
    } catch (err: any) {
      showToast(err.message || 'Failed to delete zone', 'error');
    }
  };

  const handleRetryConnection = async () => {
    if (!camera) return;
    setRetrying(true);
    try {
      await api.testCamera(camera.id);
      await loadCamera();
      showToast('Camera status refreshed!');
    } catch (err: any) {
      showToast(err.message || 'Retry connection failed', 'error');
    } finally {
      setRetrying(false);
    }
  };

  const handleSaveSettings = async () => {
    if (!camera) return;
    setSavingSettings(true);
    try {
      const updated = await api.updateCamera(camera.id, {
        ai_enabled: aiEnabled,
        alert_confidence_threshold: thresholdPct / 100,
        cooldown_seconds: Number(cooldownSec),
        detection_config: detectionRules,
      });
      setCamera(updated);
      showToast('Camera settings saved successfully!');
    } catch (err: any) {
      showToast(err.message || 'Failed to save settings', 'error');
    } finally {
      setSavingSettings(false);
    }
  };

  if (!camera && !loading) return <div className="empty-state"><h3>Camera not found</h3></div>;
  if (!camera) return <div style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>Loading camera details...</div>;

  return (
    <div>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>

      {/* Toast */}
      {toast && (
        <div style={{
          position: 'fixed', top: 20, right: 20, zIndex: 2000,
          padding: '12px 18px', borderRadius: 10, fontWeight: 600, fontSize: 13,
          background: toast.type === 'success' ? 'var(--color-success-bg)' : 'var(--color-danger-bg)',
          color: toast.type === 'success' ? 'var(--color-success)' : 'var(--color-danger)',
          border: `1px solid ${toast.type === 'success' ? 'var(--color-success-border)' : 'var(--color-danger-border)'}`,
          display: 'flex', alignItems: 'center', gap: 8, boxShadow: '0 4px 24px rgba(0,0,0,0.3)',
        }}>
          <CheckCircle size={15} />
          {toast.msg}
        </div>
      )}

      <button
        onClick={() => navigate(-1)}
        style={{
          color: 'var(--text-muted)', fontSize: 13, background: 'none', border: 'none', cursor: 'pointer',
          display: 'inline-flex', alignItems: 'center', gap: 4, marginBottom: 16, padding: 0,
        }}
      >
        <ArrowLeft size={14} /> Back
      </button>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: 24 }}>
        <div>
          <h1 className="page-title" style={{ marginBottom: 4 }}>{camera.name}</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>{camera.location} · CAM-{String(camera.id).padStart(2, '0')}</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '6px 14px', borderRadius: 20,
            background: camera.is_online ? 'var(--color-success-bg)' : 'var(--color-critical-bg)',
            border: `1px solid ${camera.is_online ? 'var(--color-success-border)' : 'var(--color-critical-border)'}`,
            color: camera.is_online ? 'var(--color-success)' : 'var(--color-critical)',
            fontWeight: 700, fontSize: 12,
          }}>
            {camera.is_online ? <Wifi size={14} /> : <WifiOff size={14} />}
            {camera.is_online ? 'Online' : 'Offline'}
          </span>
        </div>
      </div>

      {/* Tabs */}
      <div className="tabs">
        {[
          { id: 'live', label: 'Live View', icon: Eye },
          { id: 'zones', label: 'Zones', icon: MapPin },
          { id: 'recordings', label: 'Recordings', icon: History },
          { id: 'events', label: 'Events', icon: AlertTriangle },
          { id: 'settings', label: 'Settings', icon: Settings },
        ].map(tab => (
          <button key={tab.id} className={`tab ${activeTab === tab.id ? 'active' : ''}`} onClick={() => setActiveTab(tab.id)}>
            <tab.icon size={14} style={{ marginRight: 6, verticalAlign: 'middle' }} />
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === 'live' && (
        <div>
          {/* Video Feed */}
          <div className="card" style={{ aspectRatio: '16/9', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--bg-primary)', marginBottom: 20, position: 'relative', overflow: 'hidden' }}>
            {isStreamable && liveStreamUrl ? (
              <img
                src={liveStreamUrl}
                alt={camera.name}
                style={{ width: '100%', height: '100%', objectFit: 'contain', borderRadius: 8, display: 'block' }}
              />
            ) : isStreamable ? (
              <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
                <div style={{
                  width: 32, height: 32, border: '2px solid rgba(0,220,140,0.25)',
                  borderTopColor: 'rgba(0,220,140,0.85)', borderRadius: '50%',
                  animation: 'spin 0.75s linear infinite', margin: '0 auto 12px',
                }} />
                <p>Connecting...</p>
              </div>
            ) : (
              <div style={{ textAlign: 'center', color: 'var(--color-critical)' }}>
                <WifiOff size={64} style={{ opacity: 0.5, marginBottom: 12 }} />
                <p>Camera Offline</p>
                <button className="btn btn-warning btn-sm" style={{ marginTop: 8 }} onClick={handleRetryConnection} disabled={retrying}>
                  <RefreshCw size={14} className={retrying ? 'animate-spin' : ''} /> {retrying ? 'Retrying...' : 'Retry Connection'}
                </button>
              </div>
            )}

            {/* AI Overlay */}
            {camera.is_online && camera.ai_enabled && (
              <div style={{
                position: 'absolute', bottom: 12, left: 12, right: 12,
                display: 'flex', justifyContent: 'space-between',
              }}>
                <span style={{
                  background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(8px)',
                  padding: '4px 10px', borderRadius: 6, fontSize: 12, fontWeight: 600,
                  color: 'var(--color-success)', fontFamily: "'JetBrains Mono', monospace",
                }}>AI: Active Monitoring</span>
                <span style={{
                  background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(8px)',
                  padding: '4px 10px', borderRadius: 6, fontSize: 12,
                  fontFamily: "'JetBrains Mono', monospace", color: 'var(--text-secondary)',
                }}>{camera.resolution} · {camera.fps} FPS</span>
              </div>
            )}
          </div>

          {/* Camera Info Grid */}
          <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))' }}>
            <div className="card" style={{ padding: 14, textAlign: 'center' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>Resolution</div>
              <div className="mono" style={{ fontSize: 16, fontWeight: 700, marginTop: 4 }}>{camera.resolution}</div>
            </div>
            <div className="card" style={{ padding: 14, textAlign: 'center' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>FPS</div>
              <div className="mono" style={{ fontSize: 16, fontWeight: 700, marginTop: 4 }}>{camera.fps}</div>
            </div>
            <div className="card" style={{ padding: 14, textAlign: 'center' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>AI Status</div>
              <div style={{ fontSize: 14, fontWeight: 700, marginTop: 4, color: camera.ai_enabled ? 'var(--color-success)' : 'var(--text-muted)' }}>
                {camera.ai_enabled ? 'Active' : 'Disabled'}
              </div>
            </div>
            <div className="card" style={{ padding: 14, textAlign: 'center' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>Recording</div>
              <div style={{ fontSize: 14, fontWeight: 700, marginTop: 4, color: camera.recording_enabled ? 'var(--color-critical)' : 'var(--text-muted)' }}>
                {camera.recording_enabled ? '● REC' : 'Off'}
              </div>
            </div>
            <div className="card" style={{ padding: 14, textAlign: 'center' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>AI FPS</div>
              <div className="mono" style={{ fontSize: 16, fontWeight: 700, marginTop: 4 }}>{camera.ai_processing_fps}</div>
            </div>
            <div className="card" style={{ padding: 14, textAlign: 'center' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>Retention</div>
              <div className="mono" style={{ fontSize: 16, fontWeight: 700, marginTop: 4 }}>{camera.retention_days}d</div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'zones' && (
        <div>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 16 }}>
            Click on the image below to place points and draw a zone polygon (3+ points).
            A <strong>restricted</strong> zone starts a 20-second timer for anyone who steps inside it; if they
            stay that long they are flagged as an "Unauthorized Entry" alert (leaving resets the timer). An <strong>asset</strong> zone instead
            requires a person to dwell inside it continuously for a few seconds before triggering a
            "Vandalism" alert — for equipment, storage, or valuables where brief passage is normal
            but lingering isn't. Both are checked live on the video feed.
          </p>

          <div className="card" style={{ marginBottom: 20, padding: 12 }}>
            <div
              onClick={handleZoneCanvasClick}
              style={{
                position: 'relative', width: '100%', aspectRatio: String(zoneAspect),
                borderRadius: 8, overflow: 'hidden', cursor: 'crosshair',
                background: '#000',
              }}
            >
              {isStreamable ? (
                <img
                  src={`/api/stream/${camera.id}/snapshot?token=${token || ''}`}
                  alt="Zone reference"
                  draggable={false}
                  onLoad={e => {
                    const im = e.currentTarget;
                    if (im.naturalWidth && im.naturalHeight) setZoneAspect(im.naturalWidth / im.naturalHeight);
                  }}
                  style={{ width: '100%', height: '100%', objectFit: 'fill', pointerEvents: 'none' }}
                />
              ) : (
                <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-dim)' }}>
                  Camera offline — start the stream to draw zones against a live reference.
                </div>
              )}

              {/* SVG overlay: existing saved zones + the polygon currently being drawn */}
              <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }} viewBox="0 0 100 100" preserveAspectRatio="none">
                {zones.map(z => {
                  const pts = z.polygon_points.map(p => `${p.x * 100},${p.y * 100}`).join(' ');
                  const stroke = z.zone_type === 'restricted' ? '#ef4444' : z.zone_type === 'asset' ? '#a855f7' : z.zone_type === 'monitoring' ? '#eab308' : '#64748b';
                  return <polygon key={z.id} points={pts} fill={stroke} fillOpacity={0.15} stroke={stroke} strokeWidth={0.4} vectorEffect="non-scaling-stroke" />;
                })}
                {drawingPoints.length > 0 && (
                  <polygon
                    points={drawingPoints.map(p => `${p.x * 100},${p.y * 100}`).join(' ')}
                    fill="#00dc8c" fillOpacity={0.2} stroke="#00dc8c" strokeWidth={0.4} vectorEffect="non-scaling-stroke"
                  />
                )}
                {drawingPoints.map((p, i) => (
                  <circle key={i} cx={p.x * 100} cy={p.y * 100} r={0.8} fill="#00dc8c" vectorEffect="non-scaling-stroke" />
                ))}
              </svg>
            </div>

            <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end', marginTop: 14, flexWrap: 'wrap' }}>
              <div className="form-group" style={{ marginBottom: 0, flex: 1, minWidth: 160 }}>
                <label className="form-label">Zone Name</label>
                <input className="input" value={zoneName} onChange={e => setZoneName(e.target.value)} placeholder="e.g., Back Gate" />
              </div>
              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">Type</label>
                <select className="select" value={zoneType} onChange={e => setZoneType(e.target.value as any)}>
                  <option value="restricted">Restricted</option>
                  <option value="asset">Asset</option>
                  <option value="monitoring">Monitoring</option>
                  <option value="normal">Normal</option>
                </select>
              </div>
              <button className="btn btn-ghost" onClick={undoLastPoint} disabled={drawingPoints.length === 0}>Undo Point</button>
              <button className="btn btn-ghost" onClick={clearDrawing} disabled={drawingPoints.length === 0}>Clear</button>
              <button className="btn btn-primary" onClick={handleSaveZone} disabled={savingZone || drawingPoints.length < 3}>
                {savingZone ? 'Saving...' : 'Save Zone'}
              </button>
            </div>
          </div>

          <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 12, color: 'var(--text-primary)' }}>Configured Zones</h3>
          {zonesLoading ? (
            <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>Loading zones...</p>
          ) : zones.length === 0 ? (
            <div className="card" style={{ textAlign: 'center', padding: 30, color: 'var(--text-muted)', fontSize: 13 }}>
              No zones configured for this camera yet.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {zones.map(z => (
                <div key={z.id} className="card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: 12 }}>
                  <div>
                    <span style={{ fontWeight: 700, color: 'var(--text-primary)' }}>{z.name}</span>
                    <span
                      className={`severity-badge ${z.zone_type === 'restricted' || z.zone_type === 'asset' ? 'critical' : z.zone_type === 'monitoring' ? 'warning' : 'info'}`}
                      style={{ marginLeft: 10, fontSize: 10 }}
                    >
                      {z.zone_type.toUpperCase()}
                    </span>
                    <span style={{ marginLeft: 10, fontSize: 12, color: 'var(--text-muted)' }}>{z.polygon_points.length} points</span>
                  </div>
                  <button className="btn btn-danger btn-sm" onClick={() => setDeleteZoneId(z.id)}>
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {activeTab === 'settings' && (
        <div>
          <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 16, color: 'var(--text-primary)' }}>Detection Rules</h3>
          <div className="card" style={{ marginBottom: 20 }}>
            <label className="checkbox-label" style={{ marginBottom: 12, fontWeight: 700, color: 'var(--color-info)' }}>
              <input
                type="checkbox"
                checked={aiEnabled}
                onChange={e => setAiEnabled(e.target.checked)}
              />
              Enable AI Detection Processing
            </label>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 10, marginTop: 8 }}>
              {Object.entries(detectionRules).map(([key, enabled]) => (
                <label key={key} className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={enabled}
                    onChange={e => setDetectionRules({ ...detectionRules, [key]: e.target.checked })}
                  />
                  {key.charAt(0).toUpperCase() + key.slice(1)} Detection
                </label>
              ))}
            </div>
          </div>

          <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 16, color: 'var(--text-primary)' }}>Alert Thresholds</h3>
          <div className="card" style={{ marginBottom: 20 }}>
            <div className="form-group">
              <label className="form-label">Confidence Threshold ({thresholdPct}%)</label>
              <input
                type="range" min={50} max={95} value={thresholdPct}
                onChange={e => setThresholdPct(Number(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--accent-primary)' }}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Alert Cooldown (Seconds)</label>
              <input
                className="input" type="number" value={cooldownSec}
                onChange={e => setCooldownSec(Number(e.target.value))}
                style={{ width: 140 }}
              />
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
            <button className="btn btn-primary" onClick={handleSaveSettings} disabled={savingSettings}>
              <Save size={16} /> {savingSettings ? 'Saving...' : 'Save Camera Settings'}
            </button>
          </div>
        </div>
      )}

      {(activeTab === 'recordings' || activeTab === 'events') && (
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Activity</th>
                <th>Severity</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {cameraAlerts.map(alert => (
                <tr key={alert.id}>
                  <td className="mono">{new Date(alert.created_at).toLocaleTimeString()}</td>
                  <td style={{ fontWeight: 600 }}>{alert.activity}</td>
                  <td><span className={`severity-badge ${alert.severity}`}>{alert.severity}</span></td>
                  <td><span className={`status-badge ${alert.status}`}>{alert.status}</span></td>
                </tr>
              ))}
              {cameraAlerts.length === 0 && (
                <tr>
                  <td colSpan={5} style={{ textAlign: 'center', padding: 40, color: 'var(--text-muted)' }}>
                    No recorded events for this camera
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {deleteZoneId != null && (
        <ConfirmDialog
          title="Delete Zone"
          message="Delete this zone? This cannot be undone."
          confirmLabel="Delete"
          onConfirm={confirmDeleteZone}
          onCancel={() => setDeleteZoneId(null)}
        />
      )}
    </div>
  );
};

export default CameraDetail;
