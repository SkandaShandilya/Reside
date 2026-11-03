import React from 'react';

interface LoadingSpinnerProps {
  label?: string;
  fullscreen?: boolean;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ label, fullscreen }) => (
  <div
    style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12,
      ...(fullscreen ? { minHeight: '100vh' } : { padding: 48 }),
    }}
  >
    <div className="spinner" />
    {label && <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{label}</span>}
  </div>
);

export const Skeleton: React.FC<{ width?: number | string; height?: number | string; style?: React.CSSProperties }> = ({ width = '100%', height = 14, style }) => (
  <div className="skeleton" style={{ width, height, ...style }} />
);

export default LoadingSpinner;
