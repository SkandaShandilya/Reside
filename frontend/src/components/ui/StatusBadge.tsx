import React from 'react';

type Status = 'active' | 'warning' | 'critical' | 'disabled' | 'success' | 'info';

interface StatusBadgeProps {
  status: Status;
  label?: string;
  size?: 'sm' | 'md';
}

const DEFAULT_LABEL: Record<Status, string> = {
  active: 'Active',
  warning: 'Warning',
  critical: 'Critical',
  disabled: 'Offline',
  success: 'Verified',
  info: 'Info',
};

// Centralizes the semantic-color-only mapping (green/orange/red/gray) so
// no page hand-rolls a colored badge with its own literal colors.
const StatusBadge: React.FC<StatusBadgeProps> = ({ status, label, size = 'md' }) => {
  const cls = status === 'active' ? 'active' : status === 'disabled' ? 'disabled'
    : status === 'critical' ? 'critical' : status === 'warning' ? 'warning'
    : status === 'success' ? 'active' : 'info';
  return (
    <span
      className={`status-badge ${cls}`}
      style={size === 'sm' ? { fontSize: 10, padding: '2px 8px' } : undefined}
    >
      {label || DEFAULT_LABEL[status]}
    </span>
  );
};

export default StatusBadge;
