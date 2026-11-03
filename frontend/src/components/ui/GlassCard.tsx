import React from 'react';

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  hoverable?: boolean;
  style?: React.CSSProperties;
  onClick?: () => void;
}

// Thin semantic wrapper over the existing `.card`/`.card-clickable`
// utility classes (already restyled to the glass look in index.css) —
// avoids inventing a parallel styling system.
const GlassCard: React.FC<GlassCardProps> = ({ children, className = '', hoverable, style, onClick }) => (
  <div
    className={`card ${hoverable ? 'card-clickable' : ''} ${className}`.trim()}
    style={style}
    onClick={onClick}
  >
    {children}
  </div>
);

export default GlassCard;
