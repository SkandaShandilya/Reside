import React from 'react';
import { Flame, AlertTriangle, ShieldAlert } from 'lucide-react';
import { formatDistanceToNowStrict } from 'date-fns';
import type { Alert } from '../../types';

interface AlertRowProps {
  alert: Alert;
  onClick?: () => void;
  compact?: boolean;
}

function iconFor(activity: string) {
  const a = activity.toLowerCase();
  if (a.includes('fire') || a.includes('smoke')) return Flame;
  return AlertTriangle;
}

// Compact "icon + activity + time-ago + camera + View →" row, reused by
// the Alerts list, the navbar notification popover, and Dashboard's
// recent-events panel — avoids duplicating this markup three times.
const AlertRow: React.FC<AlertRowProps> = ({ alert, onClick }) => {
  const Icon = iconFor(alert.activity);
  const colorClass = alert.severity === 'critical' ? 'critical' : alert.severity === 'warning' ? 'warning' : 'info';
  let timeAgo = '';
  try {
    timeAgo = formatDistanceToNowStrict(new Date(alert.created_at), { addSuffix: false });
  } catch {
    timeAgo = '';
  }

  return (
    <div className="alert-row" onClick={onClick} style={onClick ? { cursor: 'pointer' } : undefined}>
      <div className={`alert-row-icon ${colorClass}`}>
        <Icon size={16} />
      </div>
      <div className="alert-row-body">
        <div className="alert-row-title">{alert.activity}</div>
        <div className="alert-row-sub">{alert.camera_name}</div>
      </div>
      <div className="alert-row-time">{timeAgo}</div>
      {onClick && (
        <span className="alert-row-link" onClick={e => { e.stopPropagation(); onClick(); }}>
          View <ShieldAlert size={11} style={{ verticalAlign: 'middle' }} />
        </span>
      )}
    </div>
  );
};

export default AlertRow;
