export interface User {
  id: number;
  email: string;
  full_name: string;
  role: 'resident' | 'guard' | 'admin' | 'super_admin';
  is_active: boolean;
  last_login: string | null;
  created_at: string;
  apartment_number?: string;
  phone?: string;
  face_registered?: boolean;
  avatar_url?: string | null;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface Camera {
  id: number;
  name: string;
  location: string;
  description?: string;
  source_type: 'rtsp' | 'webcam' | 'upload' | 'network' | 'nvr';
  resolution?: string;
  fps: number;
  ai_processing_fps: number;
  is_online: boolean;
  is_active: boolean;
  ai_enabled: boolean;
  recording_enabled: boolean;
  motion_detection: boolean;
  retention_days: number;
  detection_config?: Record<string, any>;
  alert_confidence_threshold: number;
  cooldown_seconds: number;
  has_credentials: boolean;
  created_at: string;
  updated_at: string;
}

export interface CameraZone {
  id: number;
  camera_id: number;
  name: string;
  zone_type: 'restricted' | 'monitoring' | 'normal' | 'asset';
  polygon_points: Array<{ x: number; y: number }>;
  detection_rules?: Record<string, any>;
  time_restrictions?: Record<string, any>;
  alert_severity: string;
  description?: string;
  is_active: boolean;
  created_at: string;
}

export interface Alert {
  id: number;
  event_id?: number;
  camera_id: number;
  camera_name: string;
  camera_location: string;
  activity: string;
  confidence: number;
  severity: 'critical' | 'warning' | 'info';
  status: 'unresolved' | 'acknowledged' | 'resolved' | 'escalated' | 'false_positive';
  persons_detected: number;
  duration_seconds?: number;
  video_clip_path?: string;
  thumbnail_path?: string;
  assigned_to?: number;
  acknowledged_by?: number;
  acknowledged_at?: string;
  resolved_by?: number;
  resolved_at?: string;
  notes?: string;
  source?: 'ai' | 'resident_report';
  reported_by?: number;
  created_at: string;
  updated_at: string;
}

export interface SecurityEvent {
  id: number;
  camera_id: number;
  activity_class: string;
  confidence: number;
  severity: string;
  persons_detected: number;
  duration_seconds?: number;
  video_clip_path?: string;
  thumbnail_path?: string;
  model_version?: string;
  timestamp: string;
  created_at: string;
}

export interface ReviewItem {
  id: number;
  event_id?: number;
  alert_id?: number;
  video_path?: string;
  thumbnail_path?: string;
  camera_id: number;
  camera_name: string;
  original_prediction: string;
  original_confidence: number;
  all_predictions?: string;
  verified_label?: string;
  reviewer_id?: number;
  model_version?: string;
  reviewed_at?: string;
  added_to_training: boolean;
  timestamp: string;
  created_at: string;
}

export interface ModelVersion {
  id: number;
  version: string;
  name: string;
  architecture: string;
  dataset_name?: string;
  weights_path?: string;

  accuracy?: number;
  val_accuracy?: number;
  f1_score?: number;
  precision_score?: number;
  recall_score?: number;
  confusion_matrix?: Record<string, any>;
  per_class_metrics?: Record<string, { precision: number; recall: number; f1: number }>;
  activity_classes?: string[];
  training_samples_count?: number;
  validation_samples_count?: number;
  epochs?: number;
  training_duration_seconds?: number;
  is_deployed: boolean;
  is_baseline: boolean;
  trained_at?: string;
  deployed_at?: string;
  notes?: string;
  created_at: string;
}

export interface NotificationSetting {
  id: number;
  activity_type: string;
  notify_resident: boolean;
  notify_guard: boolean;
  notify_admin: boolean;
  channel: string;
  is_active: boolean;
  severity_filter?: string;
  message_template?: string;
  created_at: string;
}

export interface AnalyticsOverview {
  active_cameras: number;
  cameras_online: number;
  cameras_offline: number;
  alerts_today: number;
  critical_alerts_today: number;
  events_under_review: number;
  total_events: number;
  false_positive_rate?: number;
}

export interface AlertsPerDay {
  date: string;
  count: number;
  critical: number;
  warning: number;
  info: number;
}

export interface ActivityDistribution {
  activity: string;
  count: number;
  percentage: number;
}

export interface AnalyticsData {
  overview: AnalyticsOverview;
  alerts_per_day: AlertsPerDay[];
  activity_distribution: ActivityDistribution[];
  camera_activity: Array<{ camera_id: number; camera_name: string; event_count: number }>;
  detection_performance: Array<{ model_version: string; accuracy: number; f1_score: number }>;
}

export type PersonRelationship = 'self' | 'spouse' | 'child' | 'parent' | 'other';

export interface RegisteredPerson {
  id: number;
  full_name: string;
  phone?: string;
  relationship: PersonRelationship;
  face_registered: boolean;
  face_registered_at: string | null;
  is_active: boolean;
  created_at: string;
}

export interface PersonRegisterResult {
  success: boolean;
  message: string;
  person: RegisteredPerson;
}

export interface HouseholdListResult {
  success: boolean;
  count: number;
  people: RegisteredPerson[];
}

export interface FaceVerifyResult {
  success: boolean;
  matched: boolean;
  similarity: number;
  person?: RegisteredPerson;
  resident?: { id: number; full_name: string; apartment_number?: string };
  message?: string;
}

export interface Vehicle {
  id: number;
  plate_number: string;
  owner_name: string;
  apartment_number: string;
  vehicle_type: 'car' | 'bike' | 'other';
  is_active: boolean;
  source: 'resident' | 'visitor';
  created_at: string;
}

export interface PlateVerifyResult {
  success: boolean;
  plate_detected: boolean;
  detected_number: string | null;
  confidence: number;
  raw_texts: string[];
  is_valid_format?: boolean;
  registered: boolean;
  vehicle?: Vehicle;
  entry_status: 'ALLOWED' | 'REQUIRES_APPROVAL' | 'OCR_FAILED';
  message: string;
}

export interface ResidentSummary {
  id: number;
  full_name: string;
  apartment_number?: string;
}

export interface VehicleRequest {
  id: number;
  plate_number: string;
  guard: ResidentSummary;
  resident: ResidentSummary;
  apartment_number: string;
  visitor_name: string;
  purpose: string;
  // "used" = the resident's approval already granted its one entry and a
  // guard scan consumed it; a fresh VehicleRequest is required for the
  // same plate's next visit.
  status: 'pending' | 'approved' | 'rejected' | 'used';
  responded_at: string | null;
  guard_notes: string;
  resident_notes: string;
  created_at: string;
}

export interface AdminHouseholdMemberSummary {
  id: number;
  full_name: string;
  phone?: string;
  relationship: PersonRelationship | string;
  face_registered: boolean;
  created_at: string;
}

export interface AdminVehicleSummary {
  id: number;
  plate_number: string;
  vehicle_type: 'car' | 'bike' | 'other' | string;
  is_active: boolean;
  created_at: string;
}

export interface AdminVisitorSummary {
  id: number;
  visitor_name: string;
  plate_number: string;
  purpose: string;
  status: 'pending' | 'approved' | 'rejected' | string;
  guard_name: string;
  created_at: string;
  responded_at: string | null;
}

export interface AdminResidentSummary {
  id: number;
  full_name: string;
  email: string;
  apartment_number: string | null;
  phone: string | null;
  is_active: boolean;
  face_registered: boolean;
  face_registered_at: string | null;
  created_at: string;
  household_members: AdminHouseholdMemberSummary[];
  vehicles: AdminVehicleSummary[];
  visitors: AdminVisitorSummary[];
}

export interface WSSecurityAlert {
  type: 'security_alert' | 'connection_established' | 'pong';
  camera_id?: number;
  camera_name?: string;
  camera_location?: string;
  activity?: string;
  confidence?: number;
  severity?: string;
  message?: string;
  timestamp?: string;
  alert_id?: number;
}

// "used" = the resident's approval already granted its one entry and a
// guard scan consumed it; a fresh VisitorFaceRequest is required for the
// same visitor's next visit.
export type VisitorFaceRequestStatus = 'pending' | 'approved' | 'rejected' | 'expired' | 'used';

export interface VisitorFaceRequest {
  id: number;
  status: VisitorFaceRequestStatus;
  image_url: string;
  visitor_name?: string | null;
  location: string;
  guard: { id: number; full_name: string };
  resident: { id: number; full_name: string; apartment_number?: string | null };
  resident_notes: string;
  responded_at: string | null;
  created_at: string;
}
