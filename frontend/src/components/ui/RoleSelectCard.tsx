import React from 'react';
import type { LucideIcon } from 'lucide-react';

interface RoleSelectCardProps {
  label: string;
  description?: string;
  icon: LucideIcon;
  selected: boolean;
  onSelect: () => void;
}

const RoleSelectCard: React.FC<RoleSelectCardProps> = ({ label, description, icon: Icon, selected, onSelect }) => (
  <button
    type="button"
    className={`role-card ${selected ? 'selected' : ''}`}
    onClick={onSelect}
    aria-pressed={selected}
  >
    <span className="role-icon"><Icon size={20} /></span>
    <span className="role-name">{label}</span>
    {description && <span className="role-desc">{description}</span>}
  </button>
);

export default RoleSelectCard;
