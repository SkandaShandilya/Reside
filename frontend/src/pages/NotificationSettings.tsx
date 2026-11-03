import React, { useEffect, useState } from 'react';
import { Save } from 'lucide-react';
import PageHeader from '../components/ui/PageHeader';
import api from '../api/client';
import type { NotificationSetting } from '../types';

const NotificationSettings: React.FC = () => {
  const [settings, setSettings] = useState<NotificationSetting[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getNotificationSettings().then(res => {
      setSettings(res.settings);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  const toggleSetting = async (id: number, field: string, value: boolean) => {
    try {
      await api.updateNotificationSetting(id, { [field]: value });
      setSettings(prev => prev.map(s => s.id === id ? { ...s, [field]: value } : s));
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div>
      <PageHeader title="Notification Settings" description="Configure which alerts are sent to each role." />

      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Activity</th>
              <th style={{ textAlign: 'center' }}>Resident</th>
              <th style={{ textAlign: 'center' }}>Guard</th>
              <th style={{ textAlign: 'center' }}>Admin</th>
              <th style={{ textAlign: 'center' }}>Channel</th>
              <th style={{ textAlign: 'center' }}>Active</th>
            </tr>
          </thead>
          <tbody>
            {settings.map(s => (
              <tr key={s.id}>
                <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{s.activity_type}</td>
                <td style={{ textAlign: 'center' }}>
                  <div
                    className={`toggle ${s.notify_resident ? 'active' : ''}`}
                    style={{ margin: '0 auto' }}
                    onClick={() => toggleSetting(s.id, 'notify_resident', !s.notify_resident)}
                  />
                </td>
                <td style={{ textAlign: 'center' }}>
                  <div
                    className={`toggle ${s.notify_guard ? 'active' : ''}`}
                    style={{ margin: '0 auto' }}
                    onClick={() => toggleSetting(s.id, 'notify_guard', !s.notify_guard)}
                  />
                </td>
                <td style={{ textAlign: 'center' }}>
                  <div
                    className={`toggle ${s.notify_admin ? 'active' : ''}`}
                    style={{ margin: '0 auto' }}
                    onClick={() => toggleSetting(s.id, 'notify_admin', !s.notify_admin)}
                  />
                </td>
                <td style={{ textAlign: 'center' }}>
                  <span className="mono" style={{ fontSize: 11, textTransform: 'uppercase' }}>{s.channel.replace('_', ' ')}</span>
                </td>
                <td style={{ textAlign: 'center' }}>
                  <div
                    className={`toggle ${s.is_active ? 'active' : ''}`}
                    style={{ margin: '0 auto' }}
                    onClick={() => toggleSetting(s.id, 'is_active', !s.is_active)}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {settings.length === 0 && !loading && (
        <div className="empty-state">
          <h3>No notification rules configured</h3>
          <p>Add notification rules to control alert routing.</p>
        </div>
      )}
    </div>
  );
};

export default NotificationSettings;
