import React, { useEffect, useState } from 'react';
import { Users as UsersIcon, UserPlus, Shield, CheckCircle, XCircle, Search, X, AlertTriangle, Pencil, Trash2 } from 'lucide-react';
import PageHeader from '../components/ui/PageHeader';
import api from '../api/client';
import { useAuth } from '../contexts/AuthContext';
import type { User } from '../types';

const Users: React.FC = () => {
  const { user: me } = useAuth();
  const [editUser, setEditUser] = useState<User | null>(null);
  const [editForm, setEditForm] = useState({ full_name: '', email: '', password: '', apartment_number: '', phone: '' });
  const [deleteTarget, setDeleteTarget] = useState<User | null>(null);
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [roleFilter, setRoleFilter] = useState('');
  const [showAddModal, setShowAddModal] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [toast, setToast] = useState<{ msg: string; type: 'success' | 'error' } | null>(null);

  const [formData, setFormData] = useState({
    email: '',
    password: '',
    full_name: '',
    role: 'guard',
    apartment_number: '',
    phone: '',
  });

  const showToast = (msg: string, type: 'success' | 'error' = 'success') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3500);
  };

  const fetchUsers = async () => {
    setLoading(true);
    try {
      const data = await api.getUsers();
      setUsers(data);
    } catch (err: any) {
      console.error(err);
      showToast('Failed to load users', 'error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  const handleCreateUser = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setErrorMsg(null);
    try {
      const newUser = await api.createUser(formData);
      setUsers(prev => [newUser, ...prev]);
      setShowAddModal(false);
      setFormData({
        email: '', password: '', full_name: '', role: 'guard', apartment_number: '', phone: '',
      });
      showToast(`User ${newUser.full_name} created successfully!`);
    } catch (err: any) {
      setErrorMsg(err.message || 'Error creating user');
    } finally {
      setSubmitting(false);
    }
  };

  const handleRoleChange = async (userId: number, newRole: string) => {
    try {
      const updated = await api.updateUser(userId, { role: newRole });
      setUsers(prev => prev.map(u => u.id === userId ? { ...u, role: updated.role } : u));
      showToast(`Updated role to ${newRole}`);
    } catch (err: any) {
      showToast(err.message || 'Failed to update role', 'error');
    }
  };

  const handleToggleActive = async (userId: number, currentActive: boolean) => {
    try {
      const updated = await api.updateUser(userId, { is_active: !currentActive });
      setUsers(prev => prev.map(u => u.id === userId ? { ...u, is_active: updated.is_active } : u));
      showToast(`User ${updated.is_active ? 'activated' : 'deactivated'}`);
    } catch (err: any) {
      showToast(err.message || 'Failed to update status', 'error');
    }
  };

  const openEdit = (u: User) => {
    setEditUser(u);
    setErrorMsg(null);
    setEditForm({ full_name: u.full_name, email: u.email, password: '', apartment_number: u.apartment_number || '', phone: u.phone || '' });
  };

  const handleEditUser = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editUser) return;
    setSubmitting(true);
    setErrorMsg(null);
    try {
      const payload: Record<string, string> = {
        full_name: editForm.full_name, email: editForm.email,
        apartment_number: editForm.apartment_number, phone: editForm.phone,
      };
      if (editForm.password) payload.password = editForm.password;
      const updated = await api.updateUser(editUser.id, payload);
      setUsers(prev => prev.map(u => u.id === editUser.id ? { ...u, ...updated } : u));
      setEditUser(null);
      showToast('User updated');
    } catch (err: any) {
      setErrorMsg(err.message || 'Error updating user');
    } finally {
      setSubmitting(false);
    }
  };

  const handleDeleteUser = async () => {
    if (!deleteTarget) return;
    setSubmitting(true);
    try {
      await api.deleteUser(deleteTarget.id);
      setUsers(prev => prev.filter(u => u.id !== deleteTarget.id));
      showToast(`Deleted ${deleteTarget.full_name}`);
      setDeleteTarget(null);
    } catch (err: any) {
      showToast(err.message || 'Failed to delete user', 'error');
    } finally {
      setSubmitting(false);
    }
  };

  const filteredUsers = users.filter(u => {
    const matchesSearch = !search ||
      u.full_name.toLowerCase().includes(search.toLowerCase()) ||
      u.email.toLowerCase().includes(search.toLowerCase()) ||
      (u.apartment_number && u.apartment_number.toLowerCase().includes(search.toLowerCase()));
    const matchesRole = !roleFilter || u.role === roleFilter;
    return matchesSearch && matchesRole;
  });

  return (
    <div>
      {/* Toast Notification */}
      {toast && (
        <div style={{
          position: 'fixed', top: 20, right: 20, zIndex: 2000,
          padding: '12px 18px', borderRadius: 10, fontWeight: 600, fontSize: 13,
          background: toast.type === 'success' ? 'var(--color-success-bg)' : 'var(--color-danger-bg)',
          color: toast.type === 'success' ? 'var(--color-success)' : 'var(--color-danger)',
          border: `1px solid ${toast.type === 'success' ? 'var(--color-success-border)' : 'var(--color-danger-border)'}`,
          display: 'flex', alignItems: 'center', gap: 8, boxShadow: '0 4px 24px rgba(0,0,0,0.3)',
        }}>
          {toast.type === 'success' ? <CheckCircle size={15} /> : <AlertTriangle size={15} />}
          {toast.msg}
        </div>
      )}

      {/* Header */}
      <PageHeader
        title="User Management"
        description="Manage security staff, guards, and resident system access"
        action={{ label: 'Add User', icon: UserPlus, onClick: () => setShowAddModal(true) }}
      />

      {/* Filters */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 20, flexWrap: 'wrap' }}>
        <div style={{ position: 'relative', flex: 1, minWidth: 250 }}>
          <Search size={16} style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-dim)' }} />
          <input
            className="input"
            style={{ paddingLeft: 36 }}
            placeholder="Search users by name, email, or apartment..."
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </div>
        <select className="select" style={{ width: 160 }} value={roleFilter} onChange={e => setRoleFilter(e.target.value)}>
          <option value="">All Roles</option>
          <option value="admin">Admin</option>
          <option value="guard">Security Guard</option>
          <option value="resident">Resident</option>
        </select>
      </div>

      {/* Users Table */}
      {loading ? (
        <div style={{ textAlign: 'center', padding: 60, color: 'var(--text-muted)' }}>Loading users...</div>
      ) : (
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>User</th>
                <th>Role</th>
                <th>Apartment / Phone</th>
                <th>Status</th>
                <th>Last Login</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredUsers.map(user => (
                <tr key={user.id}>
                  <td>
                    <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{user.full_name}</div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{user.email}</div>
                  </td>
                  <td>
                    <select
                      className="select"
                      style={{ fontSize: 12, padding: '3px 8px', height: 'auto', width: 'auto' }}
                      value={user.role}
                      onChange={e => handleRoleChange(user.id, e.target.value)}
                    >
                      <option value="admin">System Admin</option>
                      <option value="guard">Security Guard</option>
                      <option value="resident">Resident</option>
                    </select>
                  </td>
                  <td>
                    <div style={{ fontSize: 13, color: 'var(--text-primary)' }}>
                      {user.apartment_number ? `Apt ${user.apartment_number}` : 'N/A'}
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      {user.phone || 'No phone'}
                    </div>
                  </td>
                  <td>
                    <span style={{
                      display: 'inline-flex', alignItems: 'center', gap: 4,
                      padding: '2px 8px', borderRadius: 12, fontSize: 11, fontWeight: 700,
                      background: user.is_active ? 'var(--color-success-bg)' : 'var(--color-danger-bg)',
                      color: user.is_active ? 'var(--color-success)' : 'var(--color-danger)',
                      border: `1px solid ${user.is_active ? 'var(--color-success-border)' : 'var(--color-danger-border)'}`,
                    }}>
                      {user.is_active ? <CheckCircle size={11} /> : <XCircle size={11} />}
                      {user.is_active ? 'Active' : 'Inactive'}
                    </span>
                  </td>
                  <td className="mono" style={{ fontSize: 12 }}>
                    {user.last_login ? new Date(user.last_login).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }) : 'Never'}
                  </td>
                  <td style={{ whiteSpace: 'nowrap' }}>
                    <button className="btn btn-sm btn-ghost" style={{ fontSize: 11, padding: '4px 10px', marginRight: 4 }} onClick={() => openEdit(user)}>
                      <Pencil size={12} /> Edit
                    </button>
                    <button
                      className={`btn btn-sm ${user.is_active ? 'btn-ghost' : 'btn-outline'}`}
                      style={{ fontSize: 11, padding: '4px 10px' }}
                      onClick={() => handleToggleActive(user.id, user.is_active)}
                    >
                      {user.is_active ? 'Deactivate' : 'Activate'}
                    </button>
                    {user.id !== me?.id && (
                      <button className="btn btn-sm btn-ghost" style={{ fontSize: 11, padding: '4px 10px', marginLeft: 4, color: 'var(--color-danger)' }} onClick={() => setDeleteTarget(user)}>
                        <Trash2 size={12} /> Delete
                      </button>
                    )}
                  </td>
                </tr>
              ))}
              {filteredUsers.length === 0 && (
                <tr>
                  <td colSpan={6} style={{ textAlign: 'center', padding: 40, color: 'var(--text-muted)' }}>
                    No users found matching query
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Edit User Modal */}
      {editUser && (
        <div className="modal-overlay" onClick={() => setEditUser(null)}>
          <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 500 }}>
            <div className="modal-header">
              <h3 className="modal-title">Edit User</h3>
              <button className="btn btn-ghost btn-sm" onClick={() => setEditUser(null)}><X size={16} /></button>
            </div>
            <form onSubmit={handleEditUser}>
              <div className="modal-body">
                {errorMsg && (
                  <div style={{ marginBottom: 16, padding: '10px 14px', borderRadius: 8, background: 'var(--color-critical-bg)', border: '1px solid var(--color-critical-border)', color: 'var(--color-critical)', fontSize: 13 }}>
                    {errorMsg}
                  </div>
                )}
                <div className="form-group">
                  <label className="form-label">Full Name</label>
                  <input className="input" value={editForm.full_name} onChange={e => setEditForm({ ...editForm, full_name: e.target.value })} required />
                </div>
                <div className="form-group">
                  <label className="form-label">Email Address</label>
                  <input className="input" type="email" value={editForm.email} onChange={e => setEditForm({ ...editForm, email: e.target.value })} required />
                </div>
                <div className="form-group">
                  <label className="form-label">New Password (leave blank to keep)</label>
                  <input className="input" type="password" value={editForm.password} onChange={e => setEditForm({ ...editForm, password: e.target.value })} />
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                  <div className="form-group">
                    <label className="form-label">Apartment #</label>
                    <input className="input" value={editForm.apartment_number} onChange={e => setEditForm({ ...editForm, apartment_number: e.target.value })} />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Phone Number</label>
                    <input className="input" value={editForm.phone} onChange={e => setEditForm({ ...editForm, phone: e.target.value })} />
                  </div>
                </div>
              </div>
              <div className="modal-footer">
                <button type="button" className="btn btn-ghost" onClick={() => setEditUser(null)}>Cancel</button>
                <button type="submit" className="btn btn-primary" disabled={submitting}>{submitting ? 'Saving...' : 'Save Changes'}</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Delete Confirmation */}
      {deleteTarget && (
        <div className="modal-overlay" onClick={() => setDeleteTarget(null)}>
          <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 420 }}>
            <div className="modal-header">
              <h3 className="modal-title">Delete User</h3>
              <button className="btn btn-ghost btn-sm" onClick={() => setDeleteTarget(null)}><X size={16} /></button>
            </div>
            <div className="modal-body" style={{ fontSize: 14 }}>
              Permanently delete <strong>{deleteTarget.full_name}</strong> ({deleteTarget.email})? Their registered people, vehicles and requests are removed too. This cannot be undone.
            </div>
            <div className="modal-footer">
              <button className="btn btn-ghost" onClick={() => setDeleteTarget(null)}>Cancel</button>
              <button className="btn btn-primary" style={{ background: 'var(--color-danger)' }} disabled={submitting} onClick={handleDeleteUser}>
                {submitting ? 'Deleting...' : 'Delete'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Add User Modal */}
      {showAddModal && (
        <div className="modal-overlay" onClick={() => setShowAddModal(false)}>
          <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 500 }}>
            <div className="modal-header">
              <h3 className="modal-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <UsersIcon size={18} style={{ color: 'var(--color-info)' }} /> Add System User
              </h3>
              <button className="btn btn-ghost btn-sm" onClick={() => setShowAddModal(false)}>
                <X size={16} />
              </button>
            </div>
            <form onSubmit={handleCreateUser}>
              <div className="modal-body">
                {errorMsg && (
                  <div style={{
                    marginBottom: 16, padding: '10px 14px', borderRadius: 8,
                    background: 'var(--color-critical-bg)', border: '1px solid var(--color-critical-border)',
                    color: 'var(--color-critical)', fontSize: 13, display: 'flex', alignItems: 'center', gap: 8,
                  }}>
                    <AlertTriangle size={16} /> {errorMsg}
                  </div>
                )}
                <div className="form-group">
                  <label className="form-label">Full Name</label>
                  <input
                    className="input"
                    value={formData.full_name}
                    onChange={e => setFormData({ ...formData, full_name: e.target.value })}
                    placeholder="e.g., Alex Mercer"
                    required
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Email Address</label>
                  <input
                    className="input"
                    type="email"
                    value={formData.email}
                    onChange={e => setFormData({ ...formData, email: e.target.value })}
                    placeholder="e.g., alex@example.com"
                    required
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Password</label>
                  <input
                    className="input"
                    type="password"
                    value={formData.password}
                    onChange={e => setFormData({ ...formData, password: e.target.value })}
                    placeholder="••••••••"
                    required
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">System Role</label>
                  <select
                    className="select"
                    value={formData.role}
                    onChange={e => setFormData({ ...formData, role: e.target.value })}
                  >
                    <option value="admin">System Administrator</option>
                    <option value="guard">Security Guard</option>
                    <option value="resident">Resident</option>
                  </select>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                  <div className="form-group">
                    <label className="form-label">Apartment #</label>
                    <input
                      className="input"
                      value={formData.apartment_number}
                      onChange={e => setFormData({ ...formData, apartment_number: e.target.value })}
                      placeholder="e.g., B-402"
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Phone Number</label>
                    <input
                      className="input"
                      value={formData.phone}
                      onChange={e => setFormData({ ...formData, phone: e.target.value })}
                      placeholder="e.g., +1 555-0192"
                    />
                  </div>
                </div>
              </div>
              <div className="modal-footer">
                <button type="button" className="btn btn-ghost" onClick={() => setShowAddModal(false)}>Cancel</button>
                <button type="submit" className="btn btn-primary" disabled={submitting}>
                  {submitting ? 'Creating...' : 'Create User'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default Users;
