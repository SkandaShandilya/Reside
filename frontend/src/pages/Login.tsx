import React, { useState } from 'react';
import { Eye, EyeOff, ArrowLeft, User, ShieldHalf, Settings2 } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import RoleSelectCard from '../components/ui/RoleSelectCard';
import resideLogo from '../assets/reside-logo.png';

type Role = 'resident' | 'guard' | 'admin';

const ROLE_LABEL: Record<Role, string> = {
  resident: 'Resident',
  guard: 'Guard',
  admin: 'Admin',
};

const Login: React.FC = () => {
  const { login } = useAuth();
  const [step, setStep] = useState<'role' | 'form'>('role');
  const [selectedRole, setSelectedRole] = useState<Role | null>(null);
  const [email, setEmail] = useState('admin@sentinel.local');
  const [password, setPassword] = useState('admin123');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  // Role selection is a purely cosmetic, local-UI-only concept — it is
  // NEVER sent to the backend and never gates/redirects anything. The
  // actual authenticated role always comes from the server's response to
  // this exact same login(email, password) call, unchanged from before.
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await login(email, password);
    } catch (err: any) {
      setError(err.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page">
      <div className="login-card">
        <div className="logo">
          <div className="logo-badge">
            <img src={resideLogo} alt="RESIDE" style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: 'var(--radius-md)' }} />
          </div>
          <h2>RESIDE</h2>
          <p>AI-Powered Smart Surveillance System</p>
        </div>

        {step === 'role' ? (
          <div>
            <div style={{ textAlign: 'center', marginBottom: 18 }}>
              <div style={{ fontSize: 17, fontWeight: 700, color: 'var(--text-primary)' }}>Select your role</div>
              <div style={{ fontSize: 14, color: 'var(--text-muted)', marginTop: 3 }}>Choose how you'd like to sign in</div>
            </div>
            <div className="role-select-grid">
              <RoleSelectCard
                label="Resident"
                description="Household access"
                icon={User}
                selected={selectedRole === 'resident'}
                onSelect={() => setSelectedRole('resident')}
              />
              <RoleSelectCard
                label="Guard"
                description="Security operator"
                icon={ShieldHalf}
                selected={selectedRole === 'guard'}
                onSelect={() => setSelectedRole('guard')}
              />
              <RoleSelectCard
                label="Admin"
                description="Full system access"
                icon={Settings2}
                selected={selectedRole === 'admin'}
                onSelect={() => setSelectedRole('admin')}
              />
            </div>
            <button
              type="button"
              className="btn btn-primary btn-lg"
              style={{ width: '100%', justifyContent: 'center', marginTop: 22 }}
              disabled={!selectedRole}
              onClick={() => setStep('form')}
            >
              Continue
            </button>
          </div>
        ) : (
          <form onSubmit={handleSubmit}>
            <button type="button" className="login-back-btn" onClick={() => setStep('role')}>
              <ArrowLeft size={13} /> Change role
            </button>

            <div style={{ marginBottom: 22 }}>
              <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--text-primary)' }}>Welcome Back</div>
              <div style={{ fontSize: 14, color: 'var(--text-muted)', marginTop: 3 }}>
                Sign in as {selectedRole ? ROLE_LABEL[selectedRole] : ''}
              </div>
            </div>

            {error && (
              <div style={{
                background: 'var(--color-critical-bg)',
                border: '1px solid var(--color-critical-border)',
                borderRadius: 'var(--radius-md)',
                padding: '10px 14px',
                marginBottom: 16,
                fontSize: 13,
                color: 'var(--color-critical)',
              }}>
                {error}
              </div>
            )}

            <div className="form-group">
              <label className="form-label">Email / Username</label>
              <input
                type="email"
                className="input"
                value={email}
                onChange={e => setEmail(e.target.value)}
                placeholder="you@example.com"
                required
              />
            </div>

            <div className="form-group">
              <label className="form-label">Password</label>
              <div style={{ position: 'relative' }}>
                <input
                  type={showPassword ? 'text' : 'password'}
                  className="input"
                  style={{ paddingRight: 38 }}
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  placeholder="Enter password"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(s => !s)}
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                  style={{
                    position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)',
                    background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-dim)',
                    display: 'flex', alignItems: 'center',
                  }}
                >
                  {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>

            <div style={{ textAlign: 'right', marginBottom: 6 }}>
              <a
                href="#"
                onClick={e => e.preventDefault()}
                style={{ fontSize: 14, color: 'var(--color-info)', textDecoration: 'none', fontWeight: 500 }}
              >
                Forgot password?
              </a>
            </div>

            <button
              type="submit"
              className="btn btn-primary btn-lg"
              style={{ width: '100%', justifyContent: 'center', marginTop: 12 }}
              disabled={loading}
            >
              {loading ? 'Authenticating...' : 'Sign In'}
            </button>

            <div style={{
              marginTop: 20, padding: 14,
              background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)',
              fontSize: 12, color: 'var(--text-muted)',
            }}>
              <div style={{ fontWeight: 600, marginBottom: 6, color: 'var(--text-secondary)' }}>Demo Credentials</div>
              <div>Admin: admin@sentinel.local / admin123</div>
              <div>Guard: guard1@sentinel.local / guard123</div>
              <div>Resident: resident1@sentinel.local / res123</div>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};

export default Login;
