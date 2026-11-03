import React from 'react';
import type { LucideIcon } from 'lucide-react';

type Variant = 'primary' | 'secondary' | 'danger' | 'ghost' | 'warning' | 'success';
type Size = 'sm' | 'md' | 'lg';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  icon?: LucideIcon;
  iconPosition?: 'left' | 'right';
  loading?: boolean;
}

// Thin wrapper over the existing `.btn .btn-{variant}` classes — keeps
// the single existing button system as the source of truth instead of
// introducing a second, parallel one.
const Button: React.FC<ButtonProps> = ({
  variant = 'primary', size = 'md', icon: Icon, iconPosition = 'left',
  loading, children, className = '', disabled, ...rest
}) => {
  const sizeClass = size === 'sm' ? 'btn-sm' : size === 'lg' ? 'btn-lg' : '';
  return (
    <button
      className={`btn btn-${variant} ${sizeClass} ${className}`.trim()}
      disabled={disabled || loading}
      {...rest}
    >
      {Icon && iconPosition === 'left' && <Icon size={size === 'sm' ? 13 : 15} />}
      {children}
      {Icon && iconPosition === 'right' && <Icon size={size === 'sm' ? 13 : 15} />}
    </button>
  );
};

export default Button;
