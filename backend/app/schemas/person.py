from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class PersonSummary(BaseModel):
    id: int
    full_name: str
    phone: Optional[str] = None
    relationship: str
    face_registered: bool
    face_registered_at: Optional[datetime] = None
    is_active: bool
    created_at: datetime


class PersonRegisterResponse(BaseModel):
    success: bool = True
    message: str
    person: PersonSummary


class PersonListResponse(BaseModel):
    success: bool = True
    count: int
    people: List[PersonSummary]


class HouseholdMatchResident(BaseModel):
    id: int
    full_name: str
    apartment_number: Optional[str] = None


class HouseholdFaceVerifyResponse(BaseModel):
    success: bool = True
    matched: bool
    similarity: float
    person: Optional[PersonSummary] = None
    resident: Optional[HouseholdMatchResident] = None
    message: Optional[str] = None
