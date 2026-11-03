import React from 'react';
import { AlertTriangle } from 'lucide-react';

// ------------------------------------------------------------------
// ConfirmDialog — an in-app confirmation modal, used instead of the
// browser-native window.confirm(). Several embedded webviews (IDE preview
// panes, some Electron/webview contexts) silently return false from
// confirm()/alert() without ever showing a prompt — every "Delete this
// camera?" confirm() in the app would then bail out immediately, making
// every delete button look completely dead with no visible error. This
// renders with the same modal-overlay/modal markup already used elsewhere
// (see AddCameraModal), so it behaves identically in every environment.
// ------------------------------------------------------------------

interface ConfirmDialogProps {
  title?: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  danger?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

const ConfirmDialog: React.FC<ConfirmDialogProps> = ({
  title = 'Confirm',
  message,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  danger = true,
  onConfirm,
  onCancel,
}) => {
  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 400 }}>
        <div className="modal-header">
          <h3 className="modal-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <AlertTriangle size={18} style={{ color: danger ? 'var(--color-critical)' : 'var(--color-warning)' }} />
            {title}
          </h3>
        </div>
        <div className="modal-body">
          <p style={{ color: 'var(--text-secondary)', fontSize: 14, margin: 0 }}>{message}</p>
        </div>
        <div className="modal-footer">
          <button type="button" className="btn btn-ghost" onClick={onCancel}>{cancelLabel}</button>
          <button
            type="button"
            className={danger ? 'btn btn-danger' : 'btn btn-primary'}
            onClick={onConfirm}
            autoFocus
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ConfirmDialog;
