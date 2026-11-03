import React, { useEffect, useState } from 'react';
import { CheckCircle, PlayCircle, Clock, Tag, X, Video } from 'lucide-react';
import PageHeader from '../components/ui/PageHeader';
import api from '../api/client';
import type { ReviewItem } from '../types';

const STREAM_BASE = '/api/stream';
const ACTIVITY_LABELS = ['Fighting', 'Running', 'Loitering', 'Crowd Formation', 'Normal', 'Other'];

const ReviewQueue: React.FC = () => {
  const [items, setItems] = useState<ReviewItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [selectedLabels, setSelectedLabels] = useState<Record<number, string>>({});
  const [previewClip, setPreviewClip] = useState<ReviewItem | null>(null);
  const token = localStorage.getItem('sentinel_token');

  useEffect(() => {
    api.getReviewItems({ reviewed: 'false' }).then(res => {
      setItems(res.items);
      setTotal(res.total);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  const handleVerify = async (id: number) => {
    const label = selectedLabels[id];
    if (!label) return;
    try {
      await api.verifyReview(id, { verified_label: label, add_to_training: true });
      setItems(prev => prev.filter(i => i.id !== id));
      setTotal(t => t - 1);
    } catch (err) {
      console.error(err);
    }
  };

  const formatTime = (ts: string) => new Date(ts).toLocaleString('en-US', {
    month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', hour12: true,
  });

  return (
    <div>
      <PageHeader
        title="AI Review Queue"
        description={`${total} events awaiting human verification`}
      />

      {items.length === 0 && !loading && (
        <div className="empty-state">
          <div className="icon">✅</div>
          <h3>All caught up!</h3>
          <p>No events currently need review.</p>
        </div>
      )}

      {items.map(item => (
        <div key={item.id} className="review-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: 12 }}>
            <div>
              <div style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: 15 }}>
                {item.camera_name}
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: 12, display: 'flex', alignItems: 'center', gap: 6, marginTop: 4 }}>
                <Clock size={12} /> {formatTime(item.timestamp)}
                {item.model_version && (
                  <span style={{ marginLeft: 8 }}>
                    <Tag size={12} style={{ verticalAlign: 'middle' }} /> Model v{item.model_version}
                  </span>
                )}
              </div>
            </div>
            <button className="btn btn-ghost btn-sm" onClick={() => setPreviewClip(item)}>
              <PlayCircle size={14} /> View Clip
            </button>
          </div>

          <div className="card" style={{ background: 'var(--bg-tertiary)', padding: 14, marginBottom: 12 }}>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 6, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              AI Prediction
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <span style={{ fontSize: 16, fontWeight: 700, color: 'var(--color-warning)' }}>
                {item.original_prediction}
              </span>
              <span className="mono" style={{ fontSize: 20, fontWeight: 800, color: 'var(--text-primary)' }}>
                {Math.round(item.original_confidence * 100)}%
              </span>
              <span className="confidence-bar" style={{ width: 80 }}>
                <span
                  className={`fill ${item.original_confidence >= 0.6 ? 'medium' : 'low'}`}
                  style={{ width: `${item.original_confidence * 100}%` }}
                />
              </span>
            </div>
          </div>

          <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8 }}>
            What actually happened?
          </div>

          <div className="prediction-grid">
            {ACTIVITY_LABELS.map(label => (
              <button
                key={label}
                className={`prediction-chip ${selectedLabels[item.id] === label ? 'selected' : ''}`}
                onClick={() => setSelectedLabels(prev => ({ ...prev, [item.id]: label }))}
              >
                {label}
              </button>
            ))}
          </div>

          {selectedLabels[item.id] && (
            <div style={{ marginTop: 16, display: 'flex', gap: 8, alignItems: 'center' }}>
              <button className="btn btn-primary" onClick={() => handleVerify(item.id)}>
                <CheckCircle size={14} /> Verify as "{selectedLabels[item.id]}" & Add to Training
              </button>
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                This sample will be added to the training dataset
              </span>
            </div>
          )}
        </div>
      ))}

      {/* Video Clip Preview Modal */}
      {previewClip && (
        <div className="modal-overlay" onClick={() => setPreviewClip(null)}>
          <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 640 }}>
            <div className="modal-header">
              <h3 className="modal-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Video size={18} style={{ color: 'var(--color-info)' }} /> Clip Preview — {previewClip.camera_name}
              </h3>
              <button className="btn btn-ghost btn-sm" onClick={() => setPreviewClip(null)}>
                <X size={16} />
              </button>
            </div>
            <div className="modal-body">
              <div style={{
                aspectRatio: '16/9', background: '#000', borderRadius: 8,
                overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center',
                position: 'relative',
              }}>
                {previewClip.video_path ? (
                  <video
                    src={`/${previewClip.video_path.replace(/^\.?\/?/, '')}?token=${token || ''}`}
                    controls
                    autoPlay
                    loop
                    style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                  />
                ) : (
                  <img
                    src={`${STREAM_BASE}/${previewClip.camera_id || 1}?token=${token}`}
                    alt="Review Clip"
                    style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                    onError={e => {
                      (e.target as HTMLElement).style.display = 'none';
                    }}
                  />
                )}
                <div style={{
                  position: 'absolute', bottom: 12, left: 12,
                  background: 'rgba(0,0,0,0.7)', padding: '4px 10px', borderRadius: 6,
                  fontSize: 12, color: 'var(--color-warning)', fontWeight: 700,
                }}>
                  Prediction: {previewClip.original_prediction} ({Math.round(previewClip.original_confidence * 100)}%)
                </div>
              </div>
              <div style={{ marginTop: 12, fontSize: 12, color: 'var(--text-muted)' }}>
                Recorded at {formatTime(previewClip.timestamp)} · Model v{previewClip.model_version || '1.0'}
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-ghost" onClick={() => setPreviewClip(null)}>Close</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ReviewQueue;
