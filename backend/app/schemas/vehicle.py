from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class VehicleCreate(BaseModel):
    plate_number: str
    owner_name: Optional[str] = None
    apartment_number: Optional[str] = None
    vehicle_type: str = "car"


class VehicleResponse(BaseModel):
    id: int
    plate_number: str
    owner_name: str
    apartment_number: str
    vehicle_type: str
    is_active: bool
    source: str = "resident"  # "resident" | "visitor"
    created_at: datetime


class VehicleListResponse(BaseModel):
    success: bool = True
    count: int
    vehicles: List[VehicleResponse]


class PlateVerifyResponse(BaseModel):
    success: bool = True
    plate_detected: bool
    detected_number: Optional[str] = None
    confidence: float = 0.0
    raw_texts: List[str] = []
    is_valid_format: Optional[bool] = None
    registered: bool = False
    vehicle: Optional[VehicleResponse] = None
    entry_status: str  # ALLOWED | REQUIRES_APPROVAL | OCR_FAILED
    message: str


class VehicleRequestCreate(BaseModel):
    plate_number: str
    resident_id: int
    apartment_number: Optional[str] = None
    visitor_name: Optional[str] = ""
    purpose: Optional[str] = ""
    guard_notes: Optional[str] = ""


class VehicleRequestRespond(BaseModel):
    action: str  # approve | reject
    resident_notes: Optional[str] = ""


class VehicleRequestUserSummary(BaseModel):
    id: int
    full_name: str
    apartment_number: Optional[str] = None


class VehicleRequestResponse(BaseModel):
    id: int
    plate_number: str
    guard: VehicleRequestUserSummary
    resident: VehicleRequestUserSummary
    apartment_number: str
    visitor_name: str
    purpose: str
    status: str
    responded_at: Optional[datetime] = None
    guard_notes: str
    resident_notes: str
    created_at: datetime


class VehicleRequestListResponse(BaseModel):
    success: bool = True
    count: int
    requests: List[VehicleRequestResponse]
