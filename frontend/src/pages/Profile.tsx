import React, { useEffect, useRef, useState } from 'react';
import { Camera, Trash2, Save, KeyRound, CheckCircle, AlertTriangle } from 'lucide-react';
import PageHeader from '../components/ui/PageHeader';
import api from '../api/client';
import { useAuth } from '../contexts/AuthContext';

const MAX_AVATAR_BYTES = 5 * 1024 * 1024;

type Notice = { type: 'success' | 'error'; msg: string } | null;

const NoticeBox: React.FC<{ notice: Notice }> = ({ notice }) => {
  if (!notice) return null;
  const ok = notice.type === 'success';
  return (
    <div style={{
      marginTop: 12, padding: '10px 14px', borderRadius: 8, fontSize: 13,
      display: 'flex', alignItems: 'center', gap: 8,
      background: ok ? 'var(--color-success-bg)' : 'var(--color-critical-bg)',
      border: `1px solid ${ok ? 'var(--color-success-border)' : 'var(--color-critical-border)'}`,
      color: ok ? 'var(--color-success)' : 'var(--color-critical)',
    }}>
      {ok ? <CheckCircle size={15} /> : <AlertTriangle size={15} />} {notice.msg}
    </div>
  );
};

const Profile: React.FC = () => {
  const { user, refreshUser } = useAuth();
  const fileRef = useRef<HTMLInputElement>(null);
  const token = api.getToken() || '';

  const [fullName, setFullName] = useState(user?.full_name || '');
  const [phone, setPhone] = useState(user?.phone || '');
  const [savingProfile, setSavingProfile] = useState(false);
  const [profileNotice, setProfileNotice] = useState<Notice>(null);

  const [avatarBusy, setAvatarBusy] = useState(false);
  const [avatarNotice, setAvatarNotice] = useState<Notice>(null);

  const [currentPw, setCurrentPw] = useState('');
  const [newPw, setNewPw] = useState('');
  const [confirmPw, setConfirmPw] = useState('');
  const [savingPw, setSavingPw] = useState(false);
  const [pwNotice, setPwNotice] = useState<Notice>(null);

  useEffect(() => {
    setFullName(user?.full_name || '');
    setPhone(user?.phone || '');
  }, [user?.full_name, user?.phone]);

  const handleSaveProfile = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!fullName.trim()) {
      setProfileNotice({ type: 'error', msg: 'Name cannot be empty.' });
      return;
    }
    setSavingProfile(true);
    setProfileNotice(null);
    try {
      await api.updateProfile({ full_name: fullName.trim(), phone: phone.trim() });
      await refreshUser();
      setProfileNotice({ type: 'success', msg: 'Profile updated.' });
    } catch (err: any) {
      setProfileNotice({ type: 'error', msg: err.message || 'Failed to update profile.' });
    } finally {
      setSavingProfile(false);
    }
  };

  const handleAvatarPicked = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setAvatarNotice({ type: 'error', msg: 'Please choose an image file.' });
      return;
    }
    if (file.size > MAX_AVATAR_BYTES) {
      setAvatarNotice({ type: 'error', msg: 'Image is too large. Please use a photo under 5MB.' });
      return;
    }
    setAvatarBusy(true);
    setAvatarNotice(null);
    try {
      await api.uploadAvatar(file);
      await refreshUser();
      setAvatarNotice({ type: 'success', msg: 'Profile picture updated.' });
    } catch (err: any) {
      setAvatarNotice({ type: 'error', msg: err.message || 'Failed to upload picture.' });
    } finally {
      setAvatarBusy(false);
    }
  };

  const handleRemoveAvatar = async () => {
    setAvatarBusy(true);
    setAvatarNotice(null);
    try {
      await api.removeAvatar();
      await refreshUser();
      setAvatarNotice({ type: 'success', msg: 'Profile picture removed.' });
    } catch (err: any) {
      setAvatarNotice({ type: 'error', msg: err.message || 'Failed to remove picture.' });
    } finally {
      setAvatarBusy(false);
    }
  };

  const handleChangePassword = async (e: React.FormEvent) => {
    e.preventDefault();
    setPwNotice(null);
    if (newPw.length < 6) {
      setPwNotice({ type: 'error', msg: 'New password must be at least 6 characters.' });
      return;
    }
    if (newPw !== confirmPw) {
      setPwNotice({ type: 'error', msg: 'New passwords do not match.' });
      return;
    }
    setSavingPw(true);
    try {
      await api.changePassword(currentPw, newPw);
      setCurrentPw(''); setNewPw(''); setConfirmPw('');
      setPwNotice({ type: 'success', msg: 'Password changed.' });
    } catch (err: any) {
      setPwNotice({ type: 'error', msg: err.message || 'Failed to change password.' });
    } finally {
      setSavingPw(false);
    }
  };

  if (!user) return null;

  return (
    <div style={{ maxWidth: 640, margin: '0 auto' }}>
      <PageHeader title="My Profile" description="Manage your picture, details and password" />

      {/* Picture */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 20, flexWrap: 'wrap' }}>
          {user.avatar_url ? (
            <img
              src={`${user.avatar_url}&token=${token}`}
              alt="Profile"
              style={{ width: 96, height: 96, borderRadius: '50%', objectFit: 'cover', border: '2px solid var(--border-default)' }}
            />
          ) : (
            <div style={{
              width: 96, height: 96, borderRadius: '50%', background: 'var(--color-accent)',
              color: 'var(--text-on-accent)', display: 'flex', alignItems: 'center',
              justifyContent: 'center', fontSize: 38, fontWeight: 700,
            }}>
              {user.full_name?.charAt(0)?.toUpperCase() || 'U'}
            </div>
          )}
          <div>
            <div style={{ fontWeight: 700, fontSize: 16 }}>{user.full_name}</div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10 }}>
              JPG or PNG, up to 5MB. It is cropped to a square.
            </div>
            <input ref={fileRef} type="file" accept="image/*" hidden onChange={handleAvatarPicked} />
            <div style={{ display: 'flex', gap: 8 }}>
              <button type="button" className="btn btn-primary btn-sm" disabled={avatarBusy} onClick={() => fileRef.current?.click()}>
                <Camera size={14} /> {avatarBusy ? 'Working...' : user.avatar_url ? 'Change Picture' : 'Upload Picture'}
              </button>
              {user.avatar_url && (
                <button type="button" className="btn btn-ghost btn-sm" disabled={avatarBusy} onClick={handleRemoveAvatar}>
                  <Trash2 size={14} /> Remove
                </button>
              )}
            </div>
          </div>
        </div>
        <NoticeBox notice={avatarNotice} />
      </div>

      {/* Details */}
      <form className="card" style={{ marginBottom: 24 }} onSubmit={handleSaveProfile}>
        <h3 className="section-title" style={{ marginBottom: 12 }}>Personal Details</h3>
        <div className="form-group">
          <label className="form-label">Full Name</label>
          <input className="input" value={fullName} maxLength={255} onChange={e => setFullName(e.target.value)} required />
        </div>
        <div className="form-group">
          <label className="form-label">Phone Number</label>
          <input className="input" value={phone} maxLength={20} onChange={e => setPhone(e.target.value)} placeholder="e.g., +91 98765 43210" />
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <div className="form-group">
            <label className="form-label">Email</label>
            <input className="input" value={user.email} disabled />
          </div>
          <div className="form-group">
            <label className="form-label">Apartment</label>
            <input className="input" value={user.apartment_number || '—'} disabled />
          </div>
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 12 }}>
          Email and apartment are managed by the administrator.
        </div>
        <button type="submit" className="btn btn-primary" disabled={savingProfile}>
          <Save size={14} /> {savingProfile ? 'Saving...' : 'Save Changes'}
        </button>
        <NoticeBox notice={profileNotice} />
      </form>

      {/* Password */}
      <form className="card" onSubmit={handleChangePassword}>
        <h3 className="section-title" style={{ marginBottom: 12 }}>
          <KeyRound size={16} style={{ verticalAlign: 'middle', marginRight: 6 }} />Change Password
        </h3>
        <div className="form-group">
          <label className="form-label">Current Password</label>
          <input className="input" type="password" autoComplete="current-password" value={currentPw} onChange={e => setCurrentPw(e.target.value)} required />
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <div className="form-group">
            <label className="form-label">New Password</label>
            <input className="input" type="password" autoComplete="new-password" value={newPw} onChange={e => setNewPw(e.target.value)} required />
          </div>
          <div className="form-group">
            <label className="form-label">Confirm New Password</label>
            <input className="input" type="password" autoComplete="new-password" value={confirmPw} onChange={e => setConfirmPw(e.target.value)} required />
          </div>
        </div>
        <button type="submit" className="btn btn-primary" disabled={savingPw}>
          {savingPw ? 'Updating...' : 'Update Password'}
        </button>
        <NoticeBox notice={pwNotice} />
      </form>
    </div>
  );
};

export default Profile;
