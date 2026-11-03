import React, { useEffect, useState, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, BarChart, Bar, Legend
} from 'recharts';
import { Download, Calendar, RefreshCw } from 'lucide-react';
import PageHeader from '../components/ui/PageHeader';
import api from '../api/client';
import type { AnalyticsData } from '../types';

// The detection classes this project actually produces (deployed behaviour
// model + rule-based detectors + fire/smoke) — the backend only ever reports
// these, and they are all listed here even when a class has no alerts yet.
const PROJECT_CLASSES: Record<string, string> = {
  'Fighting': '#ef4444',
  'Unauthorized Entry': '#dc2626',
  'Fire/Smoke Detected': '#f97316',
  'Vandalism': '#ec4899',
  'Overcrowding': '#f59e0b',
  'Crowd Formation': '#eab308',
  'Loitering': '#3b82f6',
  'Repeat Loitering': '#8b5cf6',
  'Running': '#14b8a6',
};

const tooltipStyle = {
  contentStyle: { background: 'var(--bg-elevated)', border: '1px solid var(--border-default)', borderRadius: 8, fontSize: 12, color: 'var(--text-primary)' },
  labelStyle: { color: 'var(--text-primary)' },
  itemStyle: { color: 'var(--text-secondary)' },
};

const Analytics: React.FC = () => {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [days, setDays] = useState(30);

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const d = await api.getAnalytics({ days: String(days) });
      setData(d);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [days]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleExportAlertsCSV = async () => {
    try {
      await api.downloadAlertsCSV();
    } catch (err) {
      console.error(err);
    }
  };

  const handleExportIncidentsCSV = async () => {
    try {
      await api.downloadIncidentsCSV();
    } catch (err) {
      console.error(err);
    }
  };

  if (!data && !loading) return <div className="empty-state"><h3>No analytics data available</h3></div>;

  return (
    <div>
      {/* Page Header with Time Window & Export Controls */}
      <PageHeader
        title="Security Analytics"
        description="Real-time incident trends, activity breakdown, and camera performance"
      >
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {/* Day range filter */}
          <div style={{ display: 'flex', background: 'var(--bg-tertiary)', borderRadius: 8, padding: 3, border: '1px solid var(--border-default)' }}>
            {[7, 30, 90].map(d => (
              <button
                key={d}
                onClick={() => setDays(d)}
                style={{
                  padding: '5px 12px',
                  borderRadius: 6,
                  border: 'none',
                  fontSize: 12,
                  fontWeight: 600,
                  cursor: 'pointer',
                  background: days === d ? 'var(--accent-primary)' : 'transparent',
                  color: days === d ? 'var(--text-on-accent)' : 'var(--text-muted)',
                  transition: 'all 0.15s ease',
                }}
              >
                {d} Days
              </button>
            ))}
          </div>

          <button className="btn btn-ghost btn-sm" onClick={loadData} disabled={loading} title="Refresh analytics">
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          </button>

          <button className="btn btn-secondary btn-sm" onClick={handleExportAlertsCSV}>
            <Download size={14} /> Export Alerts CSV
          </button>

          <button className="btn btn-secondary btn-sm" onClick={handleExportIncidentsCSV}>
            <Download size={14} /> Export Incidents CSV
          </button>
        </div>
      </PageHeader>

      {loading && !data ? (
        <div style={{ padding: 60, textAlign: 'center', color: 'var(--text-muted)' }}>Loading analytics...</div>
      ) : data && (
        <>
          {/* Overview Stats */}
          <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))' }}>
            <div className="card metric-card">
              <div className="metric-value">{data.overview.total_events}</div>
              <div className="metric-label">Events ({days}d)</div>
            </div>
            <div className="card metric-card">
              <div className="metric-value">{data.overview.alerts_today}</div>
              <div className="metric-label">Alerts Today</div>
            </div>
            <div className="card metric-card">
              <div className="metric-value" style={{ color: 'var(--color-critical)' }}>{data.overview.critical_alerts_today}</div>
              <div className="metric-label">Critical Today</div>
            </div>
            <div className="card metric-card">
              <div className="metric-value">{data.overview.events_under_review}</div>
              <div className="metric-label">Under Review</div>
            </div>
            <div className="card metric-card">
              <div className="metric-value" style={{ color: data.overview.false_positive_rate && data.overview.false_positive_rate > 25 ? 'var(--color-warning)' : 'var(--color-success)' }}>
                {data.overview.false_positive_rate?.toFixed(1) || '0'}%
              </div>
              <div className="metric-label">False Positive Rate</div>
            </div>
          </div>

          <div className="two-col">
            {/* Alerts per Day */}
            <div className="chart-container">
              <h3 className="chart-title" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <Calendar size={15} style={{ color: 'var(--color-info)' }} />
                Alerts Per Day (Last {days} Days)
              </h3>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={data.alerts_per_day}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)" />
                  <XAxis dataKey="date" tick={{ fill: 'var(--text-muted)', fontSize: 10 }} tickFormatter={d => d.slice(5)} />
                  <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 10 }} />
                  <Tooltip {...tooltipStyle} />
                  <Line type="monotone" dataKey="count" stroke="var(--color-info)" strokeWidth={2} dot={false} name="Total" />
                  <Line type="monotone" dataKey="critical" stroke="var(--color-critical)" strokeWidth={2} dot={false} name="Critical" />
                  <Line type="monotone" dataKey="warning" stroke="var(--color-warning)" strokeWidth={1.5} dot={false} name="Warning" />
                  <Legend wrapperStyle={{ fontSize: 12, color: 'var(--text-secondary)' }} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Activity Distribution */}
            <div className="chart-container">
              <h3 className="chart-title">Activity Distribution</h3>
              {data.activity_distribution.length === 0 && (
                <div style={{ padding: 20, textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>No detections in this period.</div>
              )}
              <ResponsiveContainer width="100%" height={data.activity_distribution.length === 0 ? 1 : 280}>
                <PieChart>
                  <Pie
                    data={data.activity_distribution}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={3}
                    dataKey="count"
                    nameKey="activity"
                  >
                    {data.activity_distribution.map(d => (
                      <Cell key={d.activity} fill={PROJECT_CLASSES[d.activity] || 'var(--color-gray)'} />
                    ))}
                  </Pie>
                  <Tooltip {...tooltipStyle} />
                  <Legend wrapperStyle={{ fontSize: 12, color: 'var(--text-secondary)' }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Every project class with its count (zero when none yet) */}
          <div className="chart-container" style={{ marginTop: 24 }}>
            <h3 className="chart-title">Detection Classes (Last {days} Days)</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(190px, 1fr))', gap: 10 }}>
              {Object.keys(PROJECT_CLASSES).map(cls => {
                const row = data.activity_distribution.find(d => d.activity === cls);
                return (
                  <div key={cls} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '10px 12px', borderRadius: 'var(--radius-md)', background: 'var(--bg-tertiary)' }}>
                    <span style={{ width: 10, height: 10, borderRadius: '50%', background: PROJECT_CLASSES[cls], flexShrink: 0 }} />
                    <span style={{ flex: 1, fontSize: 13, color: 'var(--text-secondary)' }}>{cls}</span>
                    <span className="mono" style={{ fontWeight: 700, color: row ? 'var(--text-primary)' : 'var(--text-dim)' }}>{row?.count ?? 0}</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Camera Activity */}
          <div className="chart-container" style={{ marginTop: 24 }}>
            <h3 className="chart-title">Camera Activity (Top 10)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={data.camera_activity}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)" />
                <XAxis dataKey="camera_name" tick={{ fill: 'var(--text-muted)', fontSize: 10 }} angle={-25} textAnchor="end" height={60} />
                <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 10 }} />
                <Tooltip {...tooltipStyle} />
                <Bar dataKey="event_count" fill="var(--color-info)" radius={[4, 4, 0, 0]} name="Events" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
};

export default Analytics;
