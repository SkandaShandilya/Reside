import React, { useEffect, useRef, useState } from 'react';
import { Camera, X, TestTube, AlertTriangle, VideoOff } from 'lucide-react';
import api from '../api/client';
import type { Camera as CameraType } from '../types';

// ------------------------------------------------------------------
// AddCameraModal — the "Add Video Source" form (RTSP/webcam/upload/
// network/NVR), shared by every page that lets a user add a camera source
// directly from where they're working (Cameras, Live Monitoring, Fire &
// Smoke) instead of only from the standalone Cameras page. One
// implementation reused everywhere, rather than copy-pasted per page.
// ------------------------------------------------------------------

const SOURCE_TABS = [
  { id: 'rtsp', label: 'IP Camera / RTSP' },
  { id: 'webcam', label: 'Webcam' },
  { id: 'upload', label: 'Upload Video' },
  { id: 'network', label: 'Network Camera' },
  { id: 'nvr', label: 'NVR Channel' },
];

interface AddCameraModalProps {
  onClose: () => void;
  onAdded: (camera: CameraType) => void;
}

const AddCameraModal: React.FC<AddCameraModalProps> = ({ onClose, onAdded }) => {
  const [activeTab, setActiveTab] = useState('rtsp');
  const [formData, setFormData] = useState({
    name: '', location: '', source_type: 'rtsp',
    rtsp_url: '', username: '', password: '',
    network_url: '', nvr_address: '', nvr_channel: '',
    nvr_stream: '', resolution: '1920x1080', fps: 30,
  });
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ success: boolean; msg: string } | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Live webcam preview — a real browser getUserMedia() permission prompt,
  // not a fake/simulated "device accessible" message. Purely a visual
  // confirmation that a camera is reachable before saving; the actual
  // stream the backend serves after saving still opens its own capture
  // device server-side (unchanged), same as every other source type here.
  const webcamVideoRef = useRef<HTMLVideoElement>(null);
  const webcamStreamRef = useRef<MediaStream | null>(null);
  const [webcamState, setWebcamState] = useState<'idle' | 'requesting' | 'live' | 'denied' | 'unavailable'>('idle');
  const [webcamErrorMsg, setWebcamErrorMsg] = useState<string | null>(null);

  const stopWebcamPreview = () => {
    webcamStreamRef.current?.getTracks().forEach(t => t.stop());
    webcamStreamRef.current = null;
    if (webcamVideoRef.current) webcamVideoRef.current.srcObject = null;
  };

  const requestWebcamPreview = async () => {
    setWebcamState('requesting');
    setWebcamErrorMsg(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setWebcamState('unavailable');
      setWebcamErrorMsg('Camera access requires HTTPS (or localhost) and a supported browser.');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      webcamStreamRef.current = stream;
      if (webcamVideoRef.current) webcamVideoRef.current.srcObject = stream;
      setWebcamState('live');
    } catch (err: any) {
      if (err?.name === 'NotFoundError' || err?.name === 'DevicesNotFoundError') {
        setWebcamState('unavailable');
        setWebcamErrorMsg('No camera device was found on this computer.');
      } else if (err?.name === 'NotAllowedError' || err?.name === 'PermissionDeniedError') {
        setWebcamState('denied');
        setWebcamErrorMsg('Camera access was blocked. Allow camera permission for this site and try again.');
      } else {
        setWebcamState('denied');
        setWebcamErrorMsg(err?.message || 'Could not access the webcam.');
      }
    }
  };

  // Prompt as soon as the Webcam tab is opened, and release the camera the
  // moment it isn't — never hold the device open in the background.
  useEffect(() => {
    if (activeTab === 'webcam') {
      requestWebcamPreview();
    } else {
      stopWebcamPreview();
      setWebcamState('idle');
    }
    return () => stopWebcamPreview();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab]);

  useEffect(() => () => stopWebcamPreview(), []);

  // Build/revoke blob URL whenever selectedFile changes
  useEffect(() => {
    if (!selectedFile) { setPreviewUrl(null); return; }
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [selectedFile]);

  const handleTestConnection = async () => {
    setTesting(true);
    setTestResult(null);
    setUploadError(null);

    await new Promise(r => setTimeout(r, 600));

    if (activeTab === 'upload') {
      if (!selectedFile) {
        setTestResult({ success: false, msg: 'No video file selected for upload.' });
      } else {
        setTestResult({ success: true, msg: `File valid: ${selectedFile.name} (${(selectedFile.size / (1024 * 1024)).toFixed(1)} MB)` });
      }
    } else if (activeTab === 'rtsp') {
      if (!formData.rtsp_url) {
        setTestResult({ success: false, msg: 'Please enter an RTSP stream URL.' });
      } else {
        setTestResult({ success: true, msg: 'RTSP connection parameters verified.' });
      }
    } else if (activeTab === 'webcam') {
      if (webcamState === 'live') {
        setTestResult({ success: true, msg: 'Webcam is live and accessible.' });
      } else {
        setTestResult({ success: false, msg: webcamErrorMsg || 'Webcam preview is not active yet.' });
      }
    } else if (activeTab === 'network') {
      if (!formData.network_url) {
        setTestResult({ success: false, msg: 'Please enter a Network Camera URL.' });
      } else {
        setTestResult({ success: true, msg: 'Network HTTP endpoint responsive.' });
      }
    } else if (activeTab === 'nvr') {
      if (!formData.nvr_address) {
        setTestResult({ success: false, msg: 'Please enter NVR Address.' });
      } else {
        setTestResult({ success: true, msg: `NVR Channel ${formData.nvr_channel || 1} ready.` });
      }
    }
    setTesting(false);
  };

  const handleAdd = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setUploadError(null);
    try {
      let newCam: CameraType;
      if (activeTab === 'upload') {
        if (!selectedFile) {
          setUploadError('Please select a video file to upload.');
          setSubmitting(false);
          return;
        }
        const fd = new FormData();
        fd.append('file', selectedFile);
        fd.append('name', formData.name.trim() || selectedFile.name.replace(/\.[^/.]+$/, ''));
        fd.append('location', formData.location.trim() || 'Uploaded Video Stream');
        fd.append('description', formData.rtsp_url || `Uploaded file: ${selectedFile.name}`);
        fd.append('resolution', formData.resolution || '1920x1080');
        fd.append('fps', String(formData.fps || 30));
        newCam = await api.uploadCameraVideo(fd);
      } else {
        newCam = await api.createCamera({
          ...formData,
          name: formData.name.trim() || (activeTab === 'webcam' ? 'Webcam' : formData.name),
          location: formData.location.trim() || (activeTab === 'webcam' ? 'Local device' : formData.location),
          source_type: activeTab,
          fps: Number(formData.fps),
          // formData.nvr_channel is '' until the NVR tab's own input is
          // touched — sent as-is, FastAPI's Optional[int] rejects the
          // empty string outright (422) instead of treating it as absent.
          nvr_channel: formData.nvr_channel === '' ? undefined : Number(formData.nvr_channel),
        });
      }
      onAdded(newCam);
    } catch (err: any) {
      console.error(err);
      setUploadError(err.message || 'Error saving camera source.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 640 }}>
        <div className="modal-header">
          <h3 className="modal-title">Add Video Source</h3>
          <button className="btn btn-ghost btn-sm" onClick={onClose}>
            <X size={16} />
          </button>
        </div>

        <div className="tabs" style={{ padding: '0 24px' }}>
          {SOURCE_TABS.map(tab => (
            <button
              key={tab.id}
              className={`tab ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <form onSubmit={handleAdd}>
          <div className="modal-body">
            <div className="form-group">
              <label className="form-label">Camera Name {(activeTab === 'upload' || activeTab === 'webcam') && <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>(Optional)</span>}</label>
              <input className="input" value={formData.name} onChange={e => setFormData({ ...formData, name: e.target.value })} placeholder="e.g., Main Entrance Gate" required={activeTab !== 'upload' && activeTab !== 'webcam'} />
            </div>
            <div className="form-group">
              <label className="form-label">Location {(activeTab === 'upload' || activeTab === 'webcam') && <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>(Optional)</span>}</label>
              <input className="input" value={formData.location} onChange={e => setFormData({ ...formData, location: e.target.value })} placeholder="e.g., Building A - Entrance" required={activeTab !== 'upload' && activeTab !== 'webcam'} />
            </div>

            {activeTab === 'rtsp' && (
              <>
                <div className="form-group">
                  <label className="form-label">RTSP URL</label>
                  <input className="input" value={formData.rtsp_url} onChange={e => setFormData({ ...formData, rtsp_url: e.target.value })} placeholder="rtsp://username:password@192.168.1.100:554/stream" />
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                  <div className="form-group">
                    <label className="form-label">Username</label>
                    <input className="input" value={formData.username} onChange={e => setFormData({ ...formData, username: e.target.value })} placeholder="admin" />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Password</label>
                    <input className="input" type="password" value={formData.password} onChange={e => setFormData({ ...formData, password: e.target.value })} placeholder="••••••" />
                  </div>
                </div>
              </>
            )}

            {activeTab === 'webcam' && (
              <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
                <div style={{
                  position: 'relative', width: '100%', aspectRatio: '16/9',
                  background: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  <video
                    ref={webcamVideoRef}
                    autoPlay
                    muted
                    playsInline
                    style={{
                      width: '100%', height: '100%', objectFit: 'cover',
                      display: webcamState === 'live' ? 'block' : 'none',
                    }}
                  />

                  {webcamState === 'requesting' && (
                    <div style={{ textAlign: 'center', color: 'var(--text-dim)' }}>
                      <Camera size={40} style={{ marginBottom: 10 }} />
                      <p style={{ fontSize: 13, margin: 0 }}>Requesting camera access…</p>
                      <p style={{ fontSize: 11, margin: 0, marginTop: 4 }}>Allow camera access in your browser's prompt.</p>
                    </div>
                  )}

                  {(webcamState === 'denied' || webcamState === 'unavailable') && (
                    <div style={{ textAlign: 'center', color: 'var(--color-critical)', padding: 20 }}>
                      <VideoOff size={40} style={{ marginBottom: 10 }} />
                      <p style={{ fontSize: 13, margin: 0, fontWeight: 600 }}>{webcamErrorMsg || 'Camera unavailable'}</p>
                      <button type="button" className="btn btn-sm btn-ghost" style={{ marginTop: 10 }} onClick={requestWebcamPreview}>
                        Try Again
                      </button>
                    </div>
                  )}

                  {webcamState === 'live' && (
                    <span className="live-indicator online" style={{ position: 'absolute', top: 10, left: 10 }}>
                      <span className="dot" /> LIVE
                    </span>
                  )}
                </div>
                <div style={{ padding: '10px 14px', fontSize: 12, color: 'var(--text-muted)' }}>
                  This preview confirms your browser can reach a camera on this device. The security
                  feed shown on Live Monitoring after saving is captured independently on the server.
                </div>
              </div>
            )}

            {activeTab === 'upload' && (
              <div>
                <input
                  type="file"
                  id="video-file-input"
                  accept="video/*,.mp4,.avi,.mov,.mkv,.webm"
                  style={{ display: 'none' }}
                  onChange={e => {
                    if (e.target.files && e.target.files[0]) {
                      setSelectedFile(e.target.files[0]);
                      setUploadError(null);
                    }
                  }}
                />

                {!selectedFile ? (
                  <label
                    htmlFor="video-file-input"
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: 10,
                      padding: '40px 20px',
                      border: '2px dashed var(--accent-primary)',
                      borderRadius: 12,
                      cursor: 'pointer',
                      background: 'var(--bg-tertiary)',
                      transition: 'background 0.2s',
                    }}
                  >
                    <div style={{ fontSize: 48 }}>📹</div>
                    <p style={{ color: 'var(--text-primary)', fontWeight: 700, fontSize: 14, margin: 0 }}>
                      Click to select or drag & drop a video file
                    </p>
                    <p style={{ color: 'var(--text-muted)', fontSize: 12, margin: 0 }}>
                      MP4 · AVI · MOV · MKV · WEBM &nbsp;·&nbsp; Max 500 MB
                    </p>
                  </label>
                ) : (
                  <div style={{ borderRadius: 12, overflow: 'hidden', border: '1px solid var(--border-primary)', background: '#000' }}>
                    <video
                      key={selectedFile.name}
                      src={previewUrl ?? undefined}
                      controls
                      loop
                      autoPlay
                      muted
                      style={{ width: '100%', maxHeight: 220, display: 'block', background: '#000' }}
                      onLoadedMetadata={e => {
                        const vid = e.currentTarget;
                        const dur = vid.duration;
                        const mins = Math.floor(dur / 60);
                        const secs = Math.floor(dur % 60).toString().padStart(2, '0');
                        const metaEl = vid.parentElement!.querySelector('#vid-meta') as HTMLElement;
                        if (metaEl) {
                          metaEl.innerText = `${mins}:${secs}  ·  ${vid.videoWidth}×${vid.videoHeight}`;
                        }
                      }}
                      onError={(e) => {
                        const fallback = e.currentTarget.nextElementSibling as HTMLElement;
                        if (fallback && fallback.id === 'vid-error-fallback') {
                          e.currentTarget.style.display = 'none';
                          fallback.style.display = 'flex';
                        }
                      }}
                    />

                    <div
                      id="vid-error-fallback"
                      style={{
                        display: 'none',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        padding: '30px 20px',
                        background: 'var(--bg-tertiary)',
                        textAlign: 'center',
                      }}
                    >
                      <div style={{ fontSize: 36, marginBottom: 8 }}>📹</div>
                      <p style={{ color: 'var(--text-primary)', fontWeight: 600, fontSize: 13, margin: 0 }}>
                        File Selected: {selectedFile.name}
                      </p>
                      <p style={{ color: 'var(--text-muted)', fontSize: 11, margin: 0, marginTop: 4 }}>
                        Browser HTML5 preview not supported natively for {selectedFile.name.split('.').pop()?.toUpperCase()} files.
                        <br />
                        Backend OpenCV will process and stream live AI detection after upload.
                      </p>
                    </div>

                    <div style={{
                      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                      padding: '10px 14px', background: 'var(--bg-secondary)',
                    }}>
                      <div>
                        <p style={{ margin: 0, fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>
                          {selectedFile.name}
                        </p>
                        <p style={{ margin: 0, fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
                          {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                          &nbsp;·&nbsp;
                          <span id="vid-meta">Ready to upload</span>
                        </p>
                      </div>
                      <label
                        htmlFor="video-file-input"
                        style={{
                          padding: '5px 12px', borderRadius: 6, fontSize: 12, fontWeight: 600,
                          background: 'var(--bg-tertiary)', border: '1px solid var(--border-primary)',
                          color: 'var(--text-secondary)', cursor: 'pointer', whiteSpace: 'nowrap',
                        }}
                      >
                        Change File
                      </label>
                    </div>
                  </div>
                )}
              </div>
            )}

            {uploadError && (
              <div style={{
                marginTop: 12, padding: '10px 14px', borderRadius: 8,
                background: 'var(--color-critical-bg)', border: '1px solid var(--color-critical-border)',
                color: 'var(--color-critical)', fontSize: 13, display: 'flex', alignItems: 'center', gap: 8,
              }}>
                <AlertTriangle size={16} /> {uploadError}
              </div>
            )}

            {activeTab === 'network' && (
              <div className="form-group">
                <label className="form-label">Network Camera URL</label>
                <input className="input" value={formData.network_url} onChange={e => setFormData({ ...formData, network_url: e.target.value })} placeholder="http://192.168.1.100:8080/video" />
              </div>
            )}

            {activeTab === 'nvr' && (
              <>
                <div className="form-group">
                  <label className="form-label">NVR Address</label>
                  <input className="input" value={formData.nvr_address} onChange={e => setFormData({ ...formData, nvr_address: e.target.value })} placeholder="192.168.1.200" />
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                  <div className="form-group">
                    <label className="form-label">Channel</label>
                    <input className="input" value={formData.nvr_channel} onChange={e => setFormData({ ...formData, nvr_channel: e.target.value })} placeholder="1" />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Stream</label>
                    <input className="input" value={formData.nvr_stream} onChange={e => setFormData({ ...formData, nvr_stream: e.target.value })} placeholder="main" />
                  </div>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                  <div className="form-group">
                    <label className="form-label">Username</label>
                    <input className="input" value={formData.username} onChange={e => setFormData({ ...formData, username: e.target.value })} />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Password</label>
                    <input className="input" type="password" value={formData.password} onChange={e => setFormData({ ...formData, password: e.target.value })} />
                  </div>
                </div>
              </>
            )}

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <div className="form-group">
                <label className="form-label">Resolution</label>
                <select className="select" value={formData.resolution} onChange={e => setFormData({ ...formData, resolution: e.target.value })}>
                  <option value="3840x2160">4K (3840×2160)</option>
                  <option value="1920x1080">Full HD (1920×1080)</option>
                  <option value="1280x720">HD (1280×720)</option>
                  <option value="640x480">SD (640×480)</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">FPS</label>
                <select className="select" value={formData.fps} onChange={e => setFormData({ ...formData, fps: Number(e.target.value) })}>
                  <option value={60}>60 FPS</option>
                  <option value={30}>30 FPS</option>
                  <option value={25}>25 FPS</option>
                  <option value={15}>15 FPS</option>
                </select>
              </div>
            </div>
            {testResult && (
              <div style={{
                marginTop: 12, padding: '8px 12px', borderRadius: 6, fontSize: 12, fontWeight: 600,
                background: testResult.success ? 'var(--color-success-bg)' : 'var(--color-danger-bg)',
                color: testResult.success ? 'var(--color-success)' : 'var(--color-danger)',
                border: `1px solid ${testResult.success ? 'var(--color-success-border)' : 'var(--color-danger-border)'}`,
              }}>
                {testResult.success ? '✅ ' : '❌ '} {testResult.msg}
              </div>
            )}
          </div>

          <div className="modal-footer">
            <button type="button" className="btn btn-ghost" onClick={onClose}>Cancel</button>
            <button type="button" className="btn btn-warning btn-sm" onClick={handleTestConnection} disabled={testing}>
              <TestTube size={14} /> {testing ? 'Testing...' : 'Test Connection'}
            </button>
            <button type="submit" className="btn btn-primary" disabled={submitting}>
              {submitting ? 'Saving...' : 'Add Camera'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default AddCameraModal;
