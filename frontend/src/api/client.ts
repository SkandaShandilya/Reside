// 127.0.0.1, not localhost: on this dev machine "localhost" resolves to the
// IPv6 loopback (::1) first, and since the backend only binds IPv4, every
// request pays a ~2s connect-timeout before falling back to IPv4 — with
// several calls on page load that reads as "the app won't load" even though
// it's actually loading, just very slowly. 127.0.0.1 skips DNS entirely.
const API_BASE = '' + '/api';

class ApiClient {
  private token: string | null = null;

  constructor() {
    this.token = localStorage.getItem('sentinel_token');
  }

  setToken(token: string | null) {
    this.token = token;
    if (token) {
      localStorage.setItem('sentinel_token', token);
    } else {
      localStorage.removeItem('sentinel_token');
    }
  }

  getToken(): string | null {
    return this.token;
  }

  private async request<T>(path: string, options: RequestInit = {}): Promise<T> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(options.headers as Record<string, string>),
    };

    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    const response = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Network error' }));
      if (response.status === 401) {
        // The JWT expired or is invalid. Every page's API calls just throw
        // on this and get silently swallowed by their own .catch(), which
        // leaves the page rendering an empty shell (e.g. "Camera Management
        // shows nothing") with zero indication the session died. Clear the
        // stale token and tell AuthContext so it can force a redirect to
        // /login instead of leaving every page to fail open into a blank
        // screen.
        this.setToken(null);
        window.dispatchEvent(new CustomEvent('sentinel:unauthorized'));
      }
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    if (response.status === 204) {
      return undefined as T;
    }

    return response.json();
  }

  // Auth
  async login(email: string, password: string) {
    return this.request<{ access_token: string; user: any }>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
  }

  async getMe() {
    return this.request<any>('/auth/me');
  }

  async register(data: any) {
    return this.request<any>('/auth/register', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // Cameras
  async getCameras() {
    return this.request<any[]>('/cameras');
  }

  async getCamera(id: number) {
    return this.request<any>(`/cameras/${id}`);
  }

  async getCameraTelemetry(id: number) {
    return this.request<any>(`/stream/${id}/telemetry`);
  }

  async createCamera(data: any) {
    return this.request<any>('/cameras', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async uploadCameraVideo(formData: FormData) {
    const token = this.getToken() || localStorage.getItem('sentinel_token');
    const response = await fetch(`${API_BASE}/cameras/upload-video`, {
      method: 'POST',
      headers: {
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: formData,
    });
    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `Upload failed (HTTP ${response.status})`);
    }
    return response.json();
  }



  async updateCamera(id: number, data: any) {
    return this.request<any>(`/cameras/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteCamera(id: number) {
    return this.request<void>(`/cameras/${id}`, { method: 'DELETE' });
  }

  async testCamera(id: number) {
    return this.request<any>(`/cameras/${id}/test`, { method: 'POST' });
  }

  // Camera Zones
  async getCameraZones(cameraId: number) {
    return this.request<any[]>(`/cameras/${cameraId}/zones`);
  }

  async createCameraZone(cameraId: number, data: {
    name: string;
    zone_type: string;
    polygon_points: { x: number; y: number }[];
    alert_severity?: string;
    description?: string;
  }) {
    return this.request<any>(`/cameras/${cameraId}/zones`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async deleteCameraZone(cameraId: number, zoneId: number) {
    return this.request<void>(`/cameras/${cameraId}/zones/${zoneId}`, { method: 'DELETE' });
  }

  // System Settings
  async getSystemSettings() {
    return this.request<any>('/settings');
  }

  async updateSystemSettings(data: { ai_analysis_fps: number; confidence_threshold: number; alert_cooldown_seconds: number }) {
    return this.request<any>('/settings', {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  // Alerts
  async getAlerts(params: Record<string, string> = {}) {
    const query = new URLSearchParams(params).toString();
    return this.request<any>(`/alerts${query ? `?${query}` : ''}`);
  }

  async getAlert(id: number) {
    return this.request<any>(`/alerts/${id}`);
  }

  async reportSuspiciousActivity(location: string, description: string) {
    return this.request<any>(`/alerts/report`, {
      method: 'POST',
      body: JSON.stringify({ location, description }),
    });
  }

  async acknowledgeAlert(id: number, notes?: string) {
    return this.request<any>(`/alerts/${id}/acknowledge`, {
      method: 'POST',
      body: JSON.stringify({ notes }),
    });
  }

  async resolveAlert(id: number, notes?: string) {
    return this.request<any>(`/alerts/${id}/resolve`, {
      method: 'POST',
      body: JSON.stringify({ notes }),
    });
  }

  async escalateAlert(id: number, data?: any) {
    return this.request<any>(`/alerts/${id}/escalate`, {
      method: 'POST',
      body: JSON.stringify(data || {}),
    });
  }

  async markFalsePositive(id: number, notes?: string) {
    return this.request<any>(`/alerts/${id}/false-positive`, {
      method: 'POST',
      body: JSON.stringify({ notes }),
    });
  }

  // Incidents
  async getIncidents(params: Record<string, string> = {}) {
    const query = new URLSearchParams(params).toString();
    return this.request<any>(`/incidents${query ? `?${query}` : ''}`);
  }

  // Review
  async getReviewItems(params: Record<string, string> = {}) {
    const query = new URLSearchParams(params).toString();
    return this.request<any>(`/review${query ? `?${query}` : ''}`);
  }

  async verifyReview(id: number, data: any) {
    return this.request<any>(`/review/${id}/verify`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // Models
  async getTrainingData() {
    return this.request<{ total: number; original: number; verified_cctv: number; class_count: number; classes: Array<{ name: string; original: number; verified_cctv: number; total: number }> }>('/models/training-data');
  }

  async getModels() {
    return this.request<any>('/models');
  }

  async getTrainingStatus() {
    return this.request<any>('/models/training-status');
  }

  async startTraining(data: any) {
    return this.request<any>('/models/train', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async trainModel(data: any) {
    return this.startTraining(data);
  }

  async deployModel(id: number) {
    return this.request<any>(`/models/${id}/deploy`, { method: 'POST' });
  }

  async rollbackModel(id: number) {
    return this.request<any>(`/models/${id}/rollback`, { method: 'POST' });
  }

  async deleteModel(id: number) {
    return this.request<void>(`/models/${id}`, { method: 'DELETE' });
  }

  // Analytics
  async getAnalytics(params: Record<string, string> = {}) {
    const query = new URLSearchParams(params).toString();
    return this.request<any>(`/analytics${query ? `?${query}` : ''}`);
  }

  // Export helpers
  async downloadIncidentsCSV(params: Record<string, string> = {}) {
    const query = new URLSearchParams(params).toString();
    const token = this.getToken();
    const res = await fetch(`${API_BASE}/incidents/export${query ? `?${query}` : ''}`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    if (!res.ok) throw new Error('Export failed');
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'security_incidents_report.csv';
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  }

  async downloadAlertsCSV(params: Record<string, string> = {}) {
    const query = new URLSearchParams(params).toString();
    const token = this.getToken();
    const res = await fetch(`${API_BASE}/alerts/export${query ? `?${query}` : ''}`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    if (!res.ok) throw new Error('Export failed');
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'security_alerts_report.csv';
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  }


  // Notifications
  async getNotificationSettings() {
    return this.request<any>('/notifications/settings');
  }

  async updateNotificationSetting(id: number, data: any) {
    return this.request<any>(`/notifications/settings/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  // Users
  async getUsers() {
    return this.request<any[]>('/users');
  }

  async createUser(data: any) {
    return this.request<any>('/users', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateUser(id: number, data: any) {
    return this.request<any>(`/users/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteUser(id: number) {
    return this.request<void>(`/users/${id}`, { method: 'DELETE' });
  }

  // Own profile
  async updateProfile(data: { full_name?: string; phone?: string }) {
    return this.request<any>('/profile', { method: 'PUT', body: JSON.stringify(data) });
  }

  async changePassword(currentPassword: string, newPassword: string) {
    return this.request<{ success: boolean }>('/profile/password', {
      method: 'POST',
      body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
    });
  }

  async uploadAvatar(file: File | Blob) {
    return this.uploadImage<any>('/profile/avatar', file, 'avatar.jpg');
  }

  async removeAvatar() {
    return this.request<any>('/profile/avatar', { method: 'DELETE' });
  }

  // Health
  async health() {
    return this.request<any>('/health');
  }

  // Faces
  private async uploadImage<T>(path: string, file: File | Blob, filename = 'capture.jpg', extraFields?: Record<string, string>) {
    const formData = new FormData();
    formData.append('file', file, filename);
    if (extraFields) {
      Object.entries(extraFields).forEach(([k, v]) => formData.append(k, v));
    }
    const token = this.getToken();
    const response = await fetch(`${API_BASE}${path}`, {
      method: 'POST',
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      body: formData,
    });
    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      // FastAPI's own 422 validation errors carry `detail` as an array
      // of {msg, loc, ...}, not a plain string — surface the first
      // message instead of letting it stringify to "[object Object]".
      const detail = Array.isArray(errData.detail)
        ? (errData.detail[0]?.msg || 'Invalid request.')
        : errData.detail;
      throw new Error(detail || `HTTP ${response.status}`);
    }
    return response.json() as Promise<T>;
  }

  async registerPerson(file: File | Blob, fullName: string, phone: string, relationship: string) {
    return this.uploadImage<import('../types').PersonRegisterResult>(
      '/faces/register', file, 'face.jpg', { full_name: fullName, phone, relationship },
    );
  }

  async getHousehold() {
    return this.request<import('../types').HouseholdListResult>('/faces/household');
  }

  async deleteHouseholdMember(personId: number) {
    return this.request<{ success: boolean; message: string }>(`/faces/household/${personId}`, { method: 'DELETE' });
  }

  async verifyFace(file: File | Blob) {
    return this.uploadImage<import('../types').FaceVerifyResult>('/faces/verify', file);
  }

  // Unregistered-visitor face requests
  async validateVisitorFace(file: Blob) {
    return this.uploadImage<{ success: boolean; face_detected: boolean }>('/visitor-requests/validate-face', file);
  }

  async createVisitorRequest(file: Blob, residentId: number, location: string, visitorName = '') {
    return this.uploadImage<import('../types').VisitorFaceRequest>(
      '/visitor-requests', file, 'visitor.jpg', { resident_id: String(residentId), location, visitor_name: visitorName },
    );
  }

  async getVisitorRequest(id: number) {
    return this.request<import('../types').VisitorFaceRequest>(`/visitor-requests/${id}`);
  }

  async cancelVisitorRequest(id: number) {
    return this.request<{ success: boolean }>(`/visitor-requests/${id}/cancel`, { method: 'POST' });
  }

  async getResidentVisitorRequests() {
    return this.request<{ success: boolean; requests: import('../types').VisitorFaceRequest[] }>('/visitor-requests/resident');
  }

  async respondVisitorRequest(id: number, action: 'approve' | 'reject') {
    return this.request<{ success: boolean }>(`/visitor-requests/${id}/respond`, {
      method: 'PATCH',
      body: JSON.stringify({ action }),
    });
  }

  // Vehicles
  async verifyPlate(file: File | Blob) {
    return this.uploadImage<import('../types').PlateVerifyResult>('/vehicles/verify-plate', file);
  }

  async registerVehicle(data: { plate_number: string; owner_name?: string; apartment_number?: string; vehicle_type?: string }) {
    const formData = new FormData();
    Object.entries(data).forEach(([k, v]) => { if (v !== undefined) formData.append(k, v); });
    const token = this.getToken();
    const response = await fetch(`${API_BASE}/vehicles/register`, {
      method: 'POST',
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      body: formData,
    });
    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP ${response.status}`);
    }
    return response.json();
  }

  async getMyVehicles() {
    return this.request<{ success: boolean; count: number; vehicles: import('../types').Vehicle[] }>('/vehicles/my');
  }

  async deleteVehicle(id: number) {
    return this.request<{ success: boolean; message: string }>(`/vehicles/${id}`, { method: 'DELETE' });
  }

  // Admin — Resident & Visitor Directory
  async getResidentDirectory() {
    return this.request<{ success: boolean; count: number; residents: import('../types').AdminResidentSummary[] }>('/admin/residents');
  }

  // Vehicle requests
  async getActiveResidents() {
    return this.request<{ success: boolean; residents: import('../types').ResidentSummary[] }>('/vehicle-requests/residents-list');
  }

  async createVehicleRequest(data: { plate_number: string; resident_id: number; visitor_name?: string; purpose?: string; guard_notes?: string }) {
    return this.request<import('../types').VehicleRequest>('/vehicle-requests', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getGuardVehicleRequests() {
    return this.request<{ success: boolean; count: number; requests: import('../types').VehicleRequest[] }>('/vehicle-requests/guard');
  }

  async getResidentVehicleRequests() {
    return this.request<{ success: boolean; count: number; requests: import('../types').VehicleRequest[] }>('/vehicle-requests/residents');
  }

  async respondVehicleRequest(id: number, action: 'approve' | 'reject', resident_notes?: string) {
    return this.request<{ success: boolean; message: string; entry_status: string; request: import('../types').VehicleRequest }>(`/vehicle-requests/${id}/respond`, {
      method: 'PATCH',
      body: JSON.stringify({ action, resident_notes }),
    });
  }
}

export const api = new ApiClient();
export default api;
