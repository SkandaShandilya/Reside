import React, { useEffect, useState, useCallback } from 'react';
import {
  Brain, Rocket, RotateCcw, Play, BarChart3, CheckCircle2,
  AlertTriangle, Clock, Cpu, Layers, ChevronDown, ChevronUp,
  Zap, Target, TrendingUp, X, Trash2,
} from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, Legend,
} from 'recharts';
import api from '../api/client';
import type { ModelVersion } from '../types';
import { useAuth } from '../contexts/AuthContext';
import ConfirmDialog from '../components/ConfirmDialog';
import PageHeader from '../components/ui/PageHeader';

/* ─── helpers ─────────────────────────────────────────────────────────────── */
const pct = (v: number | null | undefined) => {
  if (v == null) return '—';
  const val = v <= 1 ? v * 100 : v;
  return `${val.toFixed(1)}%`;
};


const metricColor = (v: number | null | undefined) => {
  if (v == null) return 'var(--text-muted)';
  if (v >= 0.9) return 'var(--color-success)';
  if (v >= 0.75) return 'var(--color-warning)';
  return 'var(--color-danger)';
};

const fmtDuration = (s: number | null | undefined) => {
  if (s == null) return '—';
  if (s < 60) return `${s.toFixed(1)}s`;
  return `${(s / 60).toFixed(1)}m`;
};

/* ─── Fine Tuning Progress Bar Component ─────────────────────────────────── */
export interface FineTuningStatus {
  is_training: boolean;
  version?: string;
  current_epoch: number;
  total_epochs: number;
  progress_percent: number;
  stage: string;
  loss?: number | null;
  dataset_name?: string;
}

const FineTuningProgressBar: React.FC<{ status: FineTuningStatus }> = ({ status }) => {
  const percent = Math.min(100, Math.max(0, status.progress_percent || 0));

  return (
    <div style={{
      background: 'var(--bg-secondary)',
      border: '1px solid var(--accent-primary)',
      borderRadius: 12,
      padding: '16px 20px',
      boxShadow: '0 8px 24px var(--accent-primary-dim)',
      marginBottom: 20,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 10, height: 10, borderRadius: '50%', background: 'var(--accent-primary)',
            boxShadow: '0 0 10px var(--accent-primary)', animation: 'pulse 1.5s infinite'
          }} />
          <span style={{ fontWeight: 800, fontSize: 14, color: 'var(--text-primary)' }}>
            Fine-Tuning In Progress {status.version ? `(v${status.version})` : ''}
          </span>
          {status.total_epochs > 0 && (
            <span style={{
              fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 12,
              background: 'var(--accent-primary-dim)', color: 'var(--color-info)',
              border: '1px solid var(--accent-primary)'
            }}>
              Epoch {status.current_epoch} / {status.total_epochs}
            </span>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {status.loss != null && (
            <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--color-warning)', fontFamily: 'monospace' }}>
              Loss: {status.loss.toFixed(4)}
            </span>
          )}
          <span style={{ fontSize: 18, fontWeight: 900, color: 'var(--color-info)', fontFamily: 'monospace' }}>
            {percent}%
          </span>
        </div>
      </div>

      <div style={{
        height: 12, width: '100%', background: 'var(--bg-tertiary)', borderRadius: 6,
        overflow: 'hidden', border: '1px solid rgba(255,255,255,0.05)',
      }}>
        <div style={{
          height: '100%', width: `${percent}%`,
          background: 'linear-gradient(90deg, var(--color-action) 0%, var(--color-accent) 50%, var(--color-success) 100%)',
          borderRadius: 6, transition: 'width 0.4s ease-out',
          boxShadow: '0 0 12px var(--accent-primary-dim)',
        }} />
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 10, fontSize: 12, color: 'var(--text-secondary)' }}>
        <span style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 6 }}>
          <Cpu size={14} style={{ color: 'var(--color-info)' }} />
          {status.stage || 'Executing PyTorch training pipeline...'}
        </span>
        {status.dataset_name && (
          <span style={{ fontSize: 11, color: 'var(--text-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 220 }}>
            {status.dataset_name}
          </span>
        )}
      </div>

      {status.total_epochs > 1 && (
        <div style={{ display: 'flex', gap: 4, marginTop: 12 }}>
          {Array.from({ length: status.total_epochs }).map((_, idx) => {
            const epNum = idx + 1;
            const isDone = epNum < status.current_epoch || percent >= 100;
            const isCurrent = epNum === status.current_epoch && percent < 100;
            return (
              <div
                key={idx}
                title={`Epoch ${epNum}`}
                style={{
                  flex: 1, height: 4, borderRadius: 2,
                  background: isDone ? 'var(--color-success)' : isCurrent ? 'var(--accent-primary)' : 'var(--bg-tertiary)',
                  opacity: isCurrent ? 1 : isDone ? 0.8 : 0.4, transition: 'all 0.3s ease',
                }}
              />
            );
          })}
        </div>
      )}
    </div>
  );
};

/* ─── Modal ────────────────────────────────────────────────────────────────── */
interface TrainModalProps {
  onClose: () => void;
  onSubmit: (cfg: { epochs: number; name: string; notes: string; selected_datasets: string[] }) => void;
  loading: boolean;
  trainingStatus?: FineTuningStatus;
}

const DATASET_OPTIONS = [
  { id: 'smoke_fire', name: 'Smoke & Fire Detection Dataset (YOLO)', samples: 8200, desc: '2.84 GB CCTV Smoke & Fire detection dataset' },
  { id: 'rwf_2000', name: 'RWF-2000 Violence Detection Database', samples: 2000, desc: '2,000 CCTV surveillance clips (1,000 Violent / Fight & 1,000 Non-Violent)' },
  { id: 'ucf_crime', name: 'UCF Crime Dataset', samples: 10400, desc: '11.0 GB real-world crime clips (Fighting, Snatching, Loitering)' },
  { id: 'cctv_action', name: 'CCTV Action Recognition Dataset', samples: 2300, desc: '2,300 CCTV video samples (Action recognition clips)' },
  { id: 'verified_cctv', name: 'Human-Verified Review Queue', samples: 450, desc: 'Live security staff verified labels' },
  { id: 'sentinel_base', name: 'RESIDE Baseline Synthetic Set', samples: 8500, desc: 'Base synthetic camera simulation clips' },
];

const TrainModal: React.FC<TrainModalProps> = ({ onClose, onSubmit, loading, trainingStatus }) => {
  const [epochs, setEpochs] = useState(5);
  const [name, setName] = useState('');
  const [notes, setNotes] = useState('');
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>(['smoke_fire', 'rwf_2000', 'ucf_crime', 'cctv_action', 'verified_cctv']);

  const toggleDataset = (id: string) => {
    setSelectedDatasets(prev =>
      prev.includes(id) ? prev.filter(d => d !== id) : [...prev, id]
    );
  };

  const totalSamples = DATASET_OPTIONS
    .filter(opt => selectedDatasets.includes(opt.id))
    .reduce((sum, opt) => sum + opt.samples, 0);

  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'var(--scrim)', display: 'flex',
      alignItems: 'center', justifyContent: 'center', zIndex: 1000, backdropFilter: 'blur(6px)',
    }}>
      <div className="card" style={{ width: 540, padding: 32, position: 'relative', maxHeight: '90vh', overflowY: 'auto' }}>
        <button
          onClick={onClose}
          style={{ position: 'absolute', top: 16, right: 16, background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}
        ><X size={18} /></button>

        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
          <div style={{ padding: 10, background: 'var(--accent-primary-dim)', borderRadius: 10 }}>
            <Brain size={22} style={{ color: 'var(--color-info)' }} />
          </div>
          <div>
            <h2 style={{ fontSize: 18, fontWeight: 800, margin: 0 }}>Start Multi-Dataset Fine-Tuning</h2>
            <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: 0, marginTop: 2 }}>
              Select datasets to train model ensemble
            </p>
          </div>
        </div>

        {/* Live Training Progress Bar inside Modal if active */}
        {loading && trainingStatus?.is_training ? (
          <FineTuningProgressBar status={trainingStatus} />
        ) : (
          <div style={{ background: 'var(--color-warning-bg)', border: '1px solid var(--color-warning-border)', borderRadius: 8, padding: '10px 14px', marginBottom: 20, display: 'flex', gap: 8 }}>
            <AlertTriangle size={15} style={{ color: 'var(--color-warning)', flexShrink: 0, marginTop: 1 }} />
            <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              Selected datasets will be combined for fine-tuning. AI predictions are never treated as ground truth during training.
            </span>
          </div>
        )}

        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div>
            <label className="form-label">Model Name <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>(optional)</span></label>
            <input
              className="form-input"
              placeholder="e.g. MultiDataset-Ensemble-v3"
              value={name}
              disabled={loading}
              onChange={e => setName(e.target.value)}
            />
          </div>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <label className="form-label" style={{ margin: 0 }}>Select Training Datasets</label>
              <span style={{ fontSize: 12, color: 'var(--color-info)', fontWeight: 700 }}>
                ~{totalSamples.toLocaleString()} total samples
              </span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {DATASET_OPTIONS.map(ds => {
                const checked = selectedDatasets.includes(ds.id);
                return (
                  <div
                    key={ds.id}
                    onClick={() => !loading && toggleDataset(ds.id)}
                    style={{
                      padding: '10px 14px',
                      background: checked ? 'var(--accent-primary-dim)' : 'var(--bg-tertiary)',
                      border: `1px solid ${checked ? 'var(--accent-primary)' : 'var(--card-border)'}`,
                      borderRadius: 8, cursor: loading ? 'not-allowed' : 'pointer', display: 'flex', alignItems: 'flex-start', gap: 10,
                      opacity: loading ? 0.6 : 1, transition: 'all 0.15s ease',
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={checked}
                      disabled={loading}
                      onChange={() => {}}
                      style={{ marginTop: 3, accentColor: 'var(--accent-primary)', cursor: 'pointer' }}
                    />
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontSize: 13, fontWeight: 700, color: checked ? 'var(--accent-primary)' : 'var(--text-primary)' }}>
                          {ds.name}
                        </span>
                        <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                          {ds.samples.toLocaleString()} clips
                        </span>
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
                        {ds.desc}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div>
            <label className="form-label">Epochs: <strong style={{ color: 'var(--color-info)' }}>{epochs}</strong></label>
            <input
              type="range" min={1} max={20} value={epochs}
              disabled={loading}
              onChange={e => setEpochs(Number(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--accent-primary)', cursor: 'pointer' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
              <span>1 (quick)</span><span>10 (balanced)</span><span>20 (thorough)</span>
            </div>
          </div>

          <div>
            <label className="form-label">Notes <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>(optional)</span></label>
            <textarea
              className="form-input"
              rows={2}
              disabled={loading}
              placeholder="e.g. Fine-tuned on UCF Crime + CCTV Action datasets"
              value={notes}
              onChange={e => setNotes(e.target.value)}
              style={{ resize: 'vertical' }}
            />
          </div>
        </div>

        <div style={{ display: 'flex', gap: 10, marginTop: 24 }}>
          <button className="btn btn-ghost" onClick={onClose} disabled={loading} style={{ flex: 1 }}>
            Cancel
          </button>
          <button
            className="btn btn-primary"
            onClick={() => onSubmit({ epochs, name, notes, selected_datasets: selectedDatasets })}
            disabled={loading || selectedDatasets.length === 0}
            style={{ flex: 2 }}
          >
            {loading ? (
              <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
                <span className="pulse-dot" />
                Fine-Tuning Active…
              </span>
            ) : (
              <><Play size={15} /> Start Fine-Tuning</>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

/* ─── Per-class bar chart ──────────────────────────────────────────────────── */
const PerClassChart: React.FC<{ metrics: Record<string, { precision: number; recall: number; f1: number }> }> = ({ metrics }) => {
  const data = Object.entries(metrics).map(([cls, m]) => ({
    name: cls.length > 10 ? cls.slice(0, 9) + '…' : cls,
    fullName: cls,
    Precision: +(m.precision * 100).toFixed(1),
    Recall: +(m.recall * 100).toFixed(1),
    F1: +(m.f1 * 100).toFixed(1),
  }));

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null;
    const full = data.find(d => d.name === label)?.fullName || label;
    return (
      <div className="card" style={{ padding: '8px 12px', fontSize: 12 }}>
        <div style={{ fontWeight: 700, marginBottom: 6 }}>{full}</div>
        {payload.map((p: any) => (
          <div key={p.name} style={{ color: p.color }}>{p.name}: {p.value}%</div>
        ))}
      </div>
    );
  };

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -12 }}>
        <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
        <YAxis domain={[60, 100]} tick={{ fontSize: 10, fill: 'var(--text-muted)' }} unit="%" />
        <Tooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Bar dataKey="Precision" fill="#6366f1" radius={[3, 3, 0, 0]} />
        <Bar dataKey="Recall" fill="#06b6d4" radius={[3, 3, 0, 0]} />
        <Bar dataKey="F1" fill="#10b981" radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
};

/* ─── Model card ───────────────────────────────────────────────────────────── */
const ModelCard: React.FC<{
  model: ModelVersion;
  isDeployed: boolean;
  isAdmin: boolean;
  onDeploy: (id: number) => void;
  onRollback: (id: number) => void;
  onDelete: (id: number) => void;
  actionLoading: number | null;
}> = ({ model, isDeployed, isAdmin, onDeploy, onRollback, onDelete, actionLoading }) => {
  const [expanded, setExpanded] = useState(false);
  const busy = actionLoading === model.id;

  return (
    <div className="card" style={{
      marginBottom: 12,
      borderColor: isDeployed ? 'var(--color-success-border)' : 'var(--card-border)',
      transition: 'border-color 0.2s',
    }}>
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <div style={{
          padding: '6px 10px', background: isDeployed ? 'var(--color-success-bg)' : 'var(--bg-tertiary)',
          borderRadius: 8, fontFamily: 'monospace', fontWeight: 700, fontSize: 13,
          color: isDeployed ? 'var(--color-success)' : 'var(--text-secondary)',
          flexShrink: 0,
        }}>
          v{model.version}
        </div>

        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontWeight: 700, fontSize: 14, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {model.name}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 6, marginTop: 2 }}>
            <Cpu size={11} /> {model.architecture}
            <span>·</span>
            <Target size={11} style={{ color: 'var(--color-info)' }} /> {model.dataset_name || 'UCF Crime + Verified CCTV Dataset'}
            {model.trained_at && (
              <><span>·</span><Clock size={11} /> {new Date(model.trained_at).toLocaleDateString()}</>
            )}
          </div>

        </div>

        {/* Quick metrics */}
        <div style={{ display: 'flex', gap: 16, fontSize: 13, flexShrink: 0 }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontWeight: 700, color: metricColor(model.accuracy) }}>{pct(model.accuracy)}</div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Accuracy</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontWeight: 700, color: metricColor(model.f1_score) }}>{model.f1_score?.toFixed(3) || '—'}</div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>F1</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontWeight: 700, color: 'var(--text-secondary)' }}>{model.training_samples_count?.toLocaleString() || '—'}</div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Samples</div>
          </div>
        </div>

        {/* Status badge */}
        {isDeployed ? (
          <span className="severity-badge info" style={{ background: 'var(--color-success-bg)', color: 'var(--color-success)', borderColor: 'var(--color-success-border)', flexShrink: 0 }}>
            <CheckCircle2 size={11} /> LIVE
          </span>
        ) : model.is_baseline ? (
          <span className="severity-badge info" style={{ flexShrink: 0 }}>BASELINE</span>
        ) : (
          <span style={{ color: 'var(--text-muted)', fontSize: 12, flexShrink: 0 }}>Available</span>
        )}

        {/* Actions */}
        {isAdmin && !isDeployed && (
          <div style={{ display: 'flex', gap: 6, flexShrink: 0 }}>
            {model.is_baseline ? (
              <button className="btn btn-warning btn-sm" onClick={() => onRollback(model.id)} disabled={busy}>
                <RotateCcw size={12} /> {busy ? '…' : 'Rollback'}
              </button>
            ) : (
              <button className="btn btn-success btn-sm" onClick={() => onDeploy(model.id)} disabled={busy}>
                <Rocket size={12} /> {busy ? '…' : 'Deploy'}
              </button>
            )}
            <button className="btn btn-danger btn-sm" onClick={() => onDelete(model.id)} disabled={busy} title="Delete this version">
              <Trash2 size={12} />
            </button>
          </div>
        )}

        <button
          onClick={() => setExpanded(x => !x)}
          style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', flexShrink: 0 }}
        >
          {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
      </div>

      {/* Expanded per-class */}
      {expanded && (
        <div style={{ marginTop: 20, borderTop: '1px solid var(--card-border)', paddingTop: 16 }}>
          {model.per_class_metrics && typeof model.per_class_metrics === 'object' ? (
            <>
              <h4 style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 12 }}>
                Per-Class Performance
              </h4>
              <PerClassChart metrics={model.per_class_metrics as any} />

              {/* table */}
              <div className="card" style={{ padding: 0, overflow: 'hidden', background: 'var(--bg-tertiary)', marginTop: 14 }}>
                <table className="data-table">
                  <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
                  <tbody>
                    {Object.entries(model.per_class_metrics as Record<string, any>).map(([cls, m]) => (
                      <tr key={cls}>
                        <td style={{ fontWeight: 600 }}>{cls}</td>
                        <td className="mono" style={{ color: metricColor(m.precision) }}>{pct(m.precision)}</td>
                        <td className="mono" style={{ color: metricColor(m.recall) }}>{pct(m.recall)}</td>
                        <td className="mono" style={{ color: metricColor(m.f1) }}>{pct(m.f1)}</td>

                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>No per-class metrics available for this version.</p>
          )}

          {/* extra details */}
          <div style={{ display: 'flex', gap: 20, marginTop: 14, fontSize: 12, color: 'var(--text-muted)', flexWrap: 'wrap' }}>
            {model.epochs && <span><strong>Epochs:</strong> {model.epochs}</span>}
            {model.training_duration_seconds != null && (
              <span><strong>Duration:</strong> {fmtDuration(model.training_duration_seconds)}</span>
            )}
            {model.validation_samples_count != null && (
              <span><strong>Val Samples:</strong> {model.validation_samples_count.toLocaleString()}</span>
            )}
            {model.weights_path && (
              <span><strong>Weights:</strong> <code style={{ fontSize: 11 }}>{model.weights_path.split(/[\\/]/).slice(-1)[0]}</code></span>
            )}
          </div>

          {model.notes && (
            <div style={{ marginTop: 10, fontSize: 12, color: 'var(--text-muted)', background: 'var(--bg-tertiary)', borderRadius: 6, padding: '8px 12px' }}>
              {model.notes}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

/* ─── Main Page ────────────────────────────────────────────────────────────── */
const AIModel: React.FC = () => {
  const { isAdmin } = useAuth();
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [deployed, setDeployed] = useState<ModelVersion | null>(null);
  const [loading, setLoading] = useState(true);
  const [showTrainModal, setShowTrainModal] = useState(false);
  const [trainLoading, setTrainLoading] = useState(false);
  const [trainResult, setTrainResult] = useState<ModelVersion | null>(null);
  const [actionLoading, setActionLoading] = useState<number | null>(null);
  const [toast, setToast] = useState<{ msg: string; type: 'success' | 'error' } | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<FineTuningStatus>({
    is_training: false,
    current_epoch: 0,
    total_epochs: 0,
    progress_percent: 0,
    stage: 'Idle',
  });

  const showToast = (msg: string, type: 'success' | 'error' = 'success') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3500);
  };

  const refresh = useCallback(async () => {
    try {
      const res = await api.getModels();
      setModels(res.models);
      setDeployed(res.deployed_model);
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    refresh().finally(() => setLoading(false));
  }, [refresh]);

  useEffect(() => {
    let interval: any = null;
    const checkStatus = async () => {
      try {
        const st = await api.getTrainingStatus();
        if (st) {
          setTrainingStatus(st);
          // Previously only ever set `true` here, never `false` — a tab
          // that merely OBSERVES a training run started elsewhere (another
          // tab, or already running before this page loaded) would latch
          // trainLoading=true forever once the poll saw it, since nothing
          // in that tab's own handleTrain() would ever run its `finally`
          // to reset it. Mirror the real server-side flag both ways.
          setTrainLoading(st.is_training);
        }
      } catch { /* ignore */ }
    };
    checkStatus();
    interval = setInterval(checkStatus, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleTrain = async (cfg: { epochs: number; name: string; notes: string; selected_datasets: string[] }) => {
    setTrainLoading(true);
    try {
      const result = await api.trainModel({
        epochs: cfg.epochs,
        name: cfg.name || undefined,
        notes: cfg.notes || undefined,
        selected_datasets: cfg.selected_datasets,
      });
      setTrainResult(result);
      await refresh();
      setShowTrainModal(false);
      showToast(`Model v${result.version} trained on ${cfg.selected_datasets.length} dataset(s)!`);
    } catch (err: any) {
      showToast(err?.response?.data?.detail || 'Training failed', 'error');
    } finally {
      setTrainLoading(false);
    }
  };


  const handleDeploy = async (id: number) => {
    setActionLoading(id);
    try {
      await api.deployModel(id);
      await refresh();
      showToast('Model deployed successfully!');
    } catch {
      showToast('Deploy failed', 'error');
    } finally {
      setActionLoading(null);
    }
  };

  const handleRollback = async (id: number) => {
    setActionLoading(id);
    try {
      await api.rollbackModel(id);
      await refresh();
      showToast('Rolled back to baseline!');
    } catch {
      showToast('Rollback failed', 'error');
    } finally {
      setActionLoading(null);
    }
  };

  const [deleteTargetId, setDeleteTargetId] = useState<number | null>(null);

  const confirmDeleteModel = async () => {
    if (deleteTargetId == null) return;
    const id = deleteTargetId;
    setDeleteTargetId(null);
    setActionLoading(id);
    try {
      await api.deleteModel(id);
      await refresh();
      showToast('Model version deleted.');
    } catch (err: any) {
      showToast(err.message || 'Delete failed', 'error');
    } finally {
      setActionLoading(null);
    }
  };

  const deployedMetrics = deployed ? [
    { name: 'Accuracy', value: +(( deployed.accuracy || 0) * 100).toFixed(1), fill: '#6366f1' },
    { name: 'Val Acc', value: +((deployed.val_accuracy || 0) * 100).toFixed(1), fill: '#06b6d4' },
    { name: 'F1', value: +((deployed.f1_score || 0) * 100).toFixed(1), fill: '#10b981' },
    { name: 'Precision', value: +((deployed.precision_score || 0) * 100).toFixed(1), fill: '#f59e0b' },
    { name: 'Recall', value: +((deployed.recall_score || 0) * 100).toFixed(1), fill: '#ef4444' },
  ] : [];

  return (
    <div>
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
          {toast.type === 'success' ? <CheckCircle2 size={15} /> : <AlertTriangle size={15} />}
          {toast.msg}
        </div>
      )}

      {/* Header */}
      <PageHeader
        title="AI Model Management"
        description="Fine-tune, deploy, and monitor activity recognition model versions"
        action={isAdmin ? { label: 'Start Fine-Tuning', icon: Zap, onClick: () => setShowTrainModal(true) } : undefined}
      />

      {/* Live Training Progress Bar Banner if training is active */}
      {trainingStatus.is_training && (
        <FineTuningProgressBar status={trainingStatus} />
      )}

      {/* Last training result banner */}
      {trainResult && (
        <div style={{
          marginBottom: 20, padding: '14px 18px', borderRadius: 10,
          background: 'var(--color-success-bg)', border: '1px solid var(--color-success-border)',
          display: 'flex', alignItems: 'center', gap: 12,
        }}>
          <CheckCircle2 size={18} style={{ color: 'var(--color-success)', flexShrink: 0 }} />
          <div style={{ flex: 1 }}>
            <span style={{ fontWeight: 700, color: 'var(--color-success)' }}>Training complete!</span>
            <span style={{ color: 'var(--text-secondary)', fontSize: 13, marginLeft: 8 }}>
              Model <strong>v{trainResult.version}</strong> created with {pct(trainResult.accuracy)} accuracy
              and F1 {trainResult.f1_score?.toFixed(3)}.
            </span>
          </div>
          <button onClick={() => setTrainResult(null)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}>
            <X size={15} />
          </button>
        </div>
      )}

      {loading ? (
        <div style={{ textAlign: 'center', padding: 80, color: 'var(--text-muted)' }}>
          <Brain size={40} style={{ opacity: 0.3, marginBottom: 12 }} />
          <div>Loading models…</div>
        </div>
      ) : (
        <>
          {/* Everything currently running in production */}
          <h3 style={{ fontSize: 14, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--text-secondary)', margin: '0 0 12px' }}>
            Currently Deployed
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 16, marginBottom: 16 }}>
            {models.filter(m => m.is_deployed).map(m => {
              const classes = m.activity_classes || [];
              const isFire = classes.includes('fire') || classes.includes('smoke');
              return (
                <div key={m.id} className="card" style={{ borderColor: 'var(--color-success-border)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', gap: 8 }}>
                    <div>
                      <div style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 600 }}>
                        {isFire ? 'Fire & Smoke Detection' : 'Behaviour Classification'}
                      </div>
                      <div style={{ fontSize: 16, fontWeight: 700, marginTop: 2 }}>{m.name}</div>
                      <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>v{m.version} · {m.architecture}</div>
                    </div>
                    <span className="status-badge active">Deployed</span>
                  </div>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', margin: '12px 0' }}>
                    {classes.map(c => (
                      <span key={c} style={{ fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 999, background: 'var(--accent-primary-dim)', color: 'var(--text-secondary)' }}>{c}</span>
                    ))}
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 8, fontSize: 12 }}>
                    <div><div style={{ color: 'var(--text-muted)' }}>Accuracy</div><strong>{m.accuracy != null ? `${(m.accuracy * 100).toFixed(1)}%` : '—'}</strong></div>
                    <div><div style={{ color: 'var(--text-muted)' }}>Val Acc</div><strong>{m.val_accuracy != null ? `${(m.val_accuracy * 100).toFixed(1)}%` : '—'}</strong></div>
                    <div><div style={{ color: 'var(--text-muted)' }}>Precision</div><strong>{m.precision_score != null ? `${(m.precision_score * 100).toFixed(1)}%` : '—'}</strong></div>
                    <div><div style={{ color: 'var(--text-muted)' }}>Recall</div><strong>{m.recall_score != null ? `${(m.recall_score * 100).toFixed(1)}%` : '—'}</strong></div>
                    <div><div style={{ color: 'var(--text-muted)' }}>F1</div><strong>{m.f1_score != null ? `${(m.f1_score * 100).toFixed(1)}%` : '—'}</strong></div>
                  </div>
                  {(() => {
                    const overall = [
                      { name: 'Accuracy', value: m.accuracy, fill: 'var(--color-info)' },
                      { name: 'Val Acc', value: m.val_accuracy, fill: 'var(--color-success)' },
                      { name: 'Precision', value: m.precision_score, fill: 'var(--color-warning)' },
                      { name: 'Recall', value: m.recall_score, fill: 'var(--color-critical)' },
                      { name: 'F1', value: m.f1_score, fill: 'var(--color-gray)' },
                    ].filter(d => d.value != null).map(d => ({ ...d, value: +((d.value as number) * 100).toFixed(1) }));
                    const perClass = m.per_class_metrics && typeof m.per_class_metrics === 'object' ? m.per_class_metrics : null;
                    const cm = m.confusion_matrix && typeof m.confusion_matrix === 'object' ? (m.confusion_matrix as Record<string, Record<string, number>>) : null;
                    return (
                      <div style={{ marginTop: 14 }}>
                        <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-secondary)', marginBottom: 4 }}>Overall Metrics (%)</div>
                        <ResponsiveContainer width="100%" height={170}>
                          <BarChart data={overall} margin={{ top: 4, right: 8, bottom: 0, left: -18 }}>
                            <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                            <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                            <Tooltip contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-default)', borderRadius: 8, fontSize: 12, color: 'var(--text-primary)' }} formatter={(v: any) => `${v}%`} />
                            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                              {overall.map(d => <Cell key={d.name} fill={d.fill} />)}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                        {perClass && (
                          <>
                            <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-secondary)', margin: '10px 0 4px' }}>Per-Class Performance (%)</div>
                            <PerClassChart metrics={perClass as any} />
                            <table className="data-table" style={{ fontSize: 12, marginTop: 6 }}>
                              <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
                              <tbody>
                                {Object.entries(perClass as Record<string, any>).map(([cls, pm]) => (
                                  <tr key={cls}>
                                    <td style={{ fontWeight: 600 }}>{cls}</td>
                                    <td className="mono">{(pm.precision * 100).toFixed(1)}%</td>
                                    <td className="mono" style={{ fontWeight: 700 }}>{(pm.recall * 100).toFixed(1)}%</td>
                                    <td className="mono">{(pm.f1 * 100).toFixed(1)}%</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </>
                        )}
                        {cm && (
                          <>
                            <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-secondary)', margin: '10px 0 6px' }}>Confusion Matrix (actual rows, predicted columns)</div>
                            <table className="data-table" style={{ fontSize: 12 }}>
                              <thead><tr><th></th>{Object.keys(cm).map(c => <th key={c}>{c}</th>)}</tr></thead>
                              <tbody>
                                {Object.entries(cm).map(([actual, row]) => (
                                  <tr key={actual}>
                                    <td style={{ fontWeight: 600 }}>{actual}</td>
                                    {Object.keys(cm).map(pred => (
                                      <td key={pred} className="mono" style={{ fontWeight: actual === pred ? 700 : 400, color: actual === pred ? 'var(--color-success)' : 'var(--text-secondary)' }}>{row[pred] ?? 0}</td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </>
                        )}
                      </div>
                    );
                  })()}
                  <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 10 }}>
                    {m.training_samples_count != null && <>{m.training_samples_count.toLocaleString()} train / {(m.validation_samples_count ?? 0).toLocaleString()} val samples · </>}
                    {m.epochs != null && <>{m.epochs} epochs · </>}
                    {m.deployed_at && <>deployed {new Date(m.deployed_at).toLocaleDateString()}</>}
                  </div>
                  {m.dataset_name && (
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>Dataset: {m.dataset_name}</div>
                  )}
                </div>
              );
            })}
          </div>

          <div className="card" style={{ marginBottom: 24 }}>
            <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10 }}>Other Deployed Components</div>
            <table className="data-table">
              <thead><tr><th>Component</th><th>Used for</th><th>Type</th></tr></thead>
              <tbody>
                {[
                  ['YOLOv8n', 'Person detection on every camera frame', 'Pretrained detector'],
                  ['Centroid tracker + re-ID', 'Keeps a stable ID per person (used by zones, loitering, running)', 'Algorithm'],
                  ['Rule engine', 'Loitering, Running, Crowd Formation, Overcrowding, Vandalism, Unauthorized Entry (restricted zones, 20 s timer)', 'Rule-based'],
                  ['InsightFace buffalo_l', 'Face detection + embedding for guard verification (match threshold 0.50)', 'Pretrained model'],
                  ['PaddleOCR', 'Vehicle number-plate reading', 'Pretrained model'],
                ].map(([name, use, type]) => (
                  <tr key={name}><td style={{ fontWeight: 600 }}>{name}</td><td style={{ color: 'var(--text-secondary)' }}>{use}</td><td>{type}</td></tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Deployed model metrics overview */}
          {deployed && (
            <div className="card" style={{ marginBottom: 24, borderColor: 'var(--color-success-border)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: 20 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <div style={{ padding: 10, background: 'var(--color-success-bg)', borderRadius: 10 }}>
                    <Brain size={20} style={{ color: 'var(--color-success)' }} />
                  </div>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ fontSize: 16, fontWeight: 800 }}>Active Model</span>
                      <span className="severity-badge info" style={{ background: 'var(--color-success-bg)', color: 'var(--color-success)', borderColor: 'var(--color-success-border)' }}>
                        LIVE
                      </span>
                    </div>
                    <div style={{ color: 'var(--text-muted)', fontSize: 13 }}>
                      {deployed.name} · v{deployed.version} · <span style={{ color: 'var(--color-info)', fontWeight: 600 }}>Dataset: {deployed.dataset_name || 'UCF Crime + Verified CCTV Dataset'}</span>
                    </div>

                  </div>
                </div>
                <div style={{ display: 'flex', gap: 16, fontSize: 12, color: 'var(--text-muted)' }}>
                  {deployed.epochs && <span><Layers size={12} style={{ verticalAlign: 'middle' }} /> {deployed.epochs} epochs</span>}
                  {deployed.training_duration_seconds != null && <span><Clock size={12} style={{ verticalAlign: 'middle' }} /> {fmtDuration(deployed.training_duration_seconds)}</span>}
                  {deployed.training_samples_count != null && <span><Target size={12} style={{ verticalAlign: 'middle' }} /> {deployed.training_samples_count.toLocaleString()} samples</span>}
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
                {/* metric mini cards */}
                <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                  {[
                    { label: 'Accuracy', value: deployed.accuracy, icon: <TrendingUp size={14} /> },
                    { label: 'Val Accuracy', value: deployed.val_accuracy, icon: <BarChart3 size={14} /> },
                    { label: 'F1 Score', value: deployed.f1_score, icon: <Target size={14} /> },
                    { label: 'Precision', value: deployed.precision_score, icon: <Zap size={14} /> },
                    { label: 'Recall', value: deployed.recall_score, icon: <CheckCircle2 size={14} /> },
                  ].map(m => (
                    <div key={m.label} className="card" style={{ padding: '10px 14px', background: 'var(--bg-tertiary)', minWidth: 100, flex: '1 1 100px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 5, color: 'var(--text-muted)', fontSize: 11, marginBottom: 4 }}>
                        {m.icon} {m.label}
                      </div>
                      <div style={{ fontSize: 18, fontWeight: 800, color: metricColor(m.value) }}>
                        {m.value != null ? `${(m.value * 100).toFixed(1)}%` : '—'}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Radar-style bar */}
                <div>
                  <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 8 }}>
                    Overview Metrics
                  </div>
                  <ResponsiveContainer width="100%" height={140}>
                    <BarChart data={deployedMetrics} margin={{ top: 0, right: 0, bottom: 0, left: -20 }}>
                      <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                      <YAxis domain={[60, 100]} tick={{ fontSize: 9, fill: 'var(--text-muted)' }} unit="%" />
                      <Tooltip formatter={(v: any) => [`${v}%`]} contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--card-border)', fontSize: 11 }} />
                      <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                        {deployedMetrics.map((entry) => <Cell key={entry.name} fill={entry.fill} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* per-class table for deployed */}
              {deployed.per_class_metrics && typeof deployed.per_class_metrics === 'object' && (
                <div style={{ marginTop: 20, borderTop: '1px solid var(--card-border)', paddingTop: 16 }}>
                  <h4 style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 12 }}>
                    Per-Class Performance
                  </h4>
                  <PerClassChart metrics={deployed.per_class_metrics as any} />
                </div>
              )}
            </div>
          )}

          {/* Version list */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <h3 style={{ fontSize: 14, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--text-secondary)', margin: 0 }}>
              All Versions ({models.length})
            </h3>
          </div>

          {models.length === 0 ? (
            <div className="card" style={{ textAlign: 'center', padding: 48, color: 'var(--text-muted)' }}>
              <Brain size={36} style={{ opacity: 0.2, marginBottom: 12 }} />
              <div>No model versions yet. Start training to create the first version.</div>
            </div>
          ) : (
            models.map(m => (
              <ModelCard
                key={m.id}
                model={m}
                isDeployed={m.is_deployed}
                isAdmin={isAdmin}
                onDeploy={handleDeploy}
                onRollback={handleRollback}
                onDelete={setDeleteTargetId}
                actionLoading={actionLoading}
              />
            ))
          )}

          {deleteTargetId != null && (
            <ConfirmDialog
              title="Delete Model Version"
              message="Delete this model version? Its checkpoint file will also be removed. This cannot be undone."
              confirmLabel="Delete"
              onConfirm={confirmDeleteModel}
              onCancel={() => setDeleteTargetId(null)}
            />
          )}
        </>
      )}

      {showTrainModal && (
        <TrainModal
          onClose={() => !trainLoading && setShowTrainModal(false)}
          onSubmit={handleTrain}
          loading={trainLoading}
          trainingStatus={trainingStatus}
        />
      )}
    </div>
  );
};

export default AIModel;
