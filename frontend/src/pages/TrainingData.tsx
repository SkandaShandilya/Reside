import React, { useEffect, useState } from 'react';
import { Database, FolderOpen, Tag, Plus, X, UploadCloud, CheckCircle } from 'lucide-react';
import PageHeader from '../components/ui/PageHeader';
import api from '../api/client';

const TrainingData: React.FC = () => {
  const [showImportModal, setShowImportModal] = useState(false);
  const [datasetName, setDatasetName] = useState('');
  const [datasetSource, setDatasetSource] = useState('upload');
  const [importing, setImporting] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [data, setData] = useState<Awaited<ReturnType<typeof api.getTrainingData>> | null>(null);
  const [loadError, setLoadError] = useState(false);

  useEffect(() => {
    api.getTrainingData().then(setData).catch(() => setLoadError(true));
  }, []);

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3500);
  };

  const handleImport = async (e: React.FormEvent) => {
    e.preventDefault();
    setImporting(true);
    await new Promise(r => setTimeout(r, 800));
    setImporting(false);
    setShowImportModal(false);
    showToast(`Dataset "${datasetName || 'Custom CCTV Dataset'}" queued for preprocessing!`);
    setDatasetName('');
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
        title="Training Data"
        action={{ label: 'Import Dataset', icon: Plus, onClick: () => setShowImportModal(true) }}
      />

      <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))' }}>
        <div className="card metric-card">
          <div className="metric-value" style={{ color: 'var(--color-info)' }}>{data ? data.total.toLocaleString() : '—'}</div>
          <div className="metric-label">Total Samples</div>
        </div>
        <div className="card metric-card">
          <div className="metric-value">{data ? data.original.toLocaleString() : '—'}</div>
          <div className="metric-label">Original Dataset</div>
        </div>
        <div className="card metric-card">
          <div className="metric-value" style={{ color: 'var(--color-success)' }}>{data ? data.verified_cctv.toLocaleString() : '—'}</div>
          <div className="metric-label">Verified CCTV</div>
        </div>
        <div className="card metric-card">
          <div className="metric-value">{data ? data.class_count : '—'}</div>
          <div className="metric-label">Activity Classes</div>
        </div>
      </div>

      <div className="section-header">
        <h3 className="section-title">Dataset Breakdown</h3>
      </div>

      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <table className="data-table">
          <thead><tr><th>Activity Class</th><th>Original</th><th>Verified CCTV</th><th>Total</th><th>Split</th></tr></thead>
          <tbody>
            {(data?.classes ?? []).map(row => (
              <tr key={row.name}>
                <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}><Tag size={12} style={{ marginRight: 6, verticalAlign: 'middle', color: 'var(--color-info)' }} />{row.name}</td>
                <td className="mono">{row.original.toLocaleString()}</td>
                <td className="mono" style={{ color: 'var(--color-success)' }}>{row.verified_cctv.toLocaleString()}</td>
                <td className="mono" style={{ fontWeight: 700 }}>{row.total.toLocaleString()}</td>
                <td>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    <div style={{ width: 80, height: 6, background: 'var(--bg-tertiary)', borderRadius: 3, overflow: 'hidden' }}>
                      <div style={{ width: `${row.total ? (row.original / row.total) * 100 : 0}%`, height: '100%', background: 'var(--color-info)', borderRadius: 3 }} />
                    </div>
                    <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{row.total ? Math.round((row.verified_cctv / row.total) * 100) : 0}% CCTV</span>
                  </div>
                </td>
              </tr>
            ))}
            {!data && (
              <tr><td colSpan={5} style={{ textAlign: 'center', padding: 32, color: 'var(--text-muted)' }}>
                {loadError ? 'Failed to load training data.' : 'Loading…'}
              </td></tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="card" style={{ marginTop: 24, padding: 20 }}>
        <h4 style={{ fontWeight: 700, marginBottom: 12, color: 'var(--text-primary)' }}>Training Pipeline</h4>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          {['Original Dataset', 'Verified CCTV Samples', 'Training Dataset', 'Model Training', 'Validation', 'Evaluation', 'Deployment'].map((step, i) => (
            <React.Fragment key={step}>
              <span style={{
                padding: '6px 14px', borderRadius: 20, fontSize: 12, fontWeight: 600,
                background: 'var(--bg-tertiary)', border: '1px solid var(--border-default)',
                color: 'var(--text-secondary)',
              }}>{step}</span>
              {i < 6 && <span style={{ color: 'var(--text-dim)' }}>→</span>}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Import Dataset Modal */}
      {showImportModal && (
        <div className="modal-overlay" onClick={() => setShowImportModal(false)}>
          <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 520 }}>
            <div className="modal-header">
              <h3 className="modal-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <UploadCloud size={18} style={{ color: 'var(--color-info)' }} /> Import Custom Training Dataset
              </h3>
              <button className="btn btn-ghost btn-sm" onClick={() => setShowImportModal(false)}>
                <X size={16} />
              </button>
            </div>
            <form onSubmit={handleImport}>
              <div className="modal-body">
                <div className="form-group">
                  <label className="form-label">Dataset Name</label>
                  <input
                    className="input"
                    placeholder="e.g., Perimeter Security Video Clip Archive"
                    value={datasetName}
                    onChange={e => setDatasetName(e.target.value)}
                    required
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Import Method</label>
                  <select className="select" value={datasetSource} onChange={e => setDatasetSource(e.target.value)}>
                    <option value="upload">Upload Zip / Annotation Folder (.zip)</option>
                    <option value="url">Download from Remote URL / S3 Bucket</option>
                    <option value="roboflow">Import from Roboflow / Kaggle API</option>
                  </select>
                </div>
                {datasetSource === 'upload' ? (
                  <div style={{ padding: 24, border: '2px dashed var(--border-default)', borderRadius: 8, textAlign: 'center', background: 'var(--bg-tertiary)' }}>
                    <FolderOpen size={32} style={{ color: 'var(--color-info)', marginBottom: 8 }} />
                    <p style={{ margin: 0, fontSize: 13, fontWeight: 600 }}>Click or drag dataset ZIP archive here</p>
                    <p style={{ margin: 0, fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>YOLO / COCO format annotations supported</p>
                  </div>
                ) : (
                  <div className="form-group">
                    <label className="form-label">Dataset Remote URL / Endpoint</label>
                    <input className="input" placeholder="https://s3.amazonaws.com/reside-datasets/security_v2.zip" required />
                  </div>
                )}
              </div>
              <div className="modal-footer">
                <button type="button" className="btn btn-ghost" onClick={() => setShowImportModal(false)}>Cancel</button>
                <button type="submit" className="btn btn-primary" disabled={importing}>
                  {importing ? 'Importing Dataset...' : 'Import Dataset'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default TrainingData;
