import enum
from sqlalchemy import String, Boolean, DateTime, JSON, Enum, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from app.database import Base, TimestampMixin


class PersonRelationship(str, enum.Enum):
    SELF = "self"
    SPOUSE = "spouse"
    CHILD = "child"
    PARENT = "parent"
    OTHER = "other"


class RegisteredPerson(Base, TimestampMixin):
    """
    A single face-registered individual belonging to a Resident's
    household (self, spouse, child, parent, or other authorized family
    member sharing the apartment) — additive, brand-new table. One
    Resident (`resident_id`, the household-owning User) can have many
    RegisteredPerson rows, each with its OWN name/phone/embedding.

    This replaces writing a resident's face data onto
    User.face_embedding/face_registered/face_registered_at (which only
    ever supported one face per account) — those User columns are kept
    for backward compatibility with data registered before this model
    existed and are migrated into a `relationship=self` row here (see
    app/database.py's init_db()), but are no longer written to by new
    registrations.
    """
    __tablename__ = "registered_people"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    resident_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    phone: Mapped[str | None] = mapped_column(String(20), nullable=True)
    relationship: Mapped[PersonRelationship] = mapped_column(
        Enum(PersonRelationship, name="person_relationship"),
        nullable=False,
        default=PersonRelationship.OTHER,
    )
    face_embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
    face_registered: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    face_registered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Soft-delete flag (mirrors Vehicle.is_active) — removing a household
    # member hides them from the household list/count and from guard
    # verification without destroying the record.
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    # "household" — the resident explicitly registered this person via
    # POST /api/faces/register: permanent, matches on every future scan.
    # "visitor" — created automatically when a resident approves a
    # guard-raised VisitorFaceRequest: grants exactly ONE future match
    # (POST /api/faces/verify deactivates it the moment it's matched),
    # mirroring Vehicle.source's resident/visitor split.
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="household")
    # Traceability back to the VisitorFaceRequest this one-time pass was
    # created from (NULL for source="household" rows, which were never
    # derived from a request). Lets POST /api/faces/verify flip the
    # originating request's status to USED the moment the pass is
    # consumed, instead of leaving it stuck on APPROVED forever.
    source_request_id: Mapped[int | None] = mapped_column(
        ForeignKey("visitor_face_requests.id", ondelete="SET NULL"), nullable=True
    )

    def __repr__(self) -> str:
        return f"<RegisteredPerson(id={self.id}, full_name='{self.full_name}', resident_id={self.resident_id})>"
