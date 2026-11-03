"""
Admin-only aggregated views. Reuses the existing User/Vehicle/
VehicleRequest/RegisteredPerson models exactly as they already are —
VehicleRequest already IS the system's "visitor" record (visitor_name +
plate_number + status + which resident it's for), so no new visitor
model was created. Never includes hashed_password, face_embedding, or
any token/secret.
"""
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class AdminHouseholdMemberSummary(BaseModel):
    """A registered household member (self, spouse, child, parent,
    other) — sourced from RegisteredPerson. Status only, never the
    embedding itself."""
    id: int
    full_name: str
    phone: Optional[str] = None
    relationship: str
    face_registered: bool
    created_at: datetime


class AdminVehicleSummary(BaseModel):
    id: int
    plate_number: str
    vehicle_type: str
    is_active: bool
    created_at: datetime


class AdminVisitorSummary(BaseModel):
    """A saved visitor record — sourced from VehicleRequest, the
    existing guard-initiated visitor-approval model. Not a new table."""
    id: int
    visitor_name: str
    plate_number: str
    purpose: str
    status: str
    guard_name: str
    created_at: datetime
    responded_at: Optional[datetime] = None


class AdminResidentSummary(BaseModel):
    id: int
    full_name: str
    email: str
    apartment_number: Optional[str] = None
    phone: Optional[str] = None
    is_active: bool
    face_registered: bool
    face_registered_at: Optional[datetime] = None
    created_at: datetime
    household_members: List[AdminHouseholdMemberSummary] = []
    vehicles: List[AdminVehicleSummary] = []
    visitors: List[AdminVisitorSummary] = []


class AdminResidentDirectoryResponse(BaseModel):
    success: bool = True
    count: int
    residents: List[AdminResidentSummary]
