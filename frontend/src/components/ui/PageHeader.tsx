import React from 'react';
import type { LucideIcon } from 'lucide-react';
import Button from './Button';

interface PageHeaderProps {
  title: string;
  description?: string;
  action?: {
    label: string;
    icon?: LucideIcon;
    onClick: () => void;
    variant?: 'primary' | 'secondary' | 'danger';
  };
  children?: React.ReactNode; // e.g. a secondary action or filter row appended after
}

// Title + description + top-right primary action — applied consistently
// across every major page per the RESIDE page-header pattern.
const PageHeader: React.FC<PageHeaderProps> = ({ title, description, action, children }) => (
  <div className="page-header-block">
    <div>
      <h1 className="phb-title">{title}</h1>
      {description && <p className="phb-desc">{description}</p>}
    </div>
    <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
      {children}
      {action && (
        <Button variant={action.variant || 'primary'} icon={action.icon} onClick={action.onClick}>
          {action.label}
        </Button>
      )}
    </div>
  </div>
);

export default PageHeader;
