from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import DateTime, event, func
from datetime import datetime
from typing import AsyncGenerator

from app.config import settings


is_sqlite = "sqlite" in settings.DATABASE_URL
engine_kwargs = {"echo": settings.DEBUG}
if not is_sqlite:
    engine_kwargs.update({"pool_size": 20, "max_overflow": 10})
else:
    engine_kwargs.update({"connect_args": {"check_same_thread": False}})

engine = create_async_engine(
    settings.DATABASE_URL,
    **engine_kwargs
)

if is_sqlite:
    # SQLite has FK enforcement OFF by default per-connection, so the
    # `ondelete="CASCADE"` declared on Alert.camera_id, CameraZone.camera_id
    # and SecurityEvent.camera_id are silently no-ops without this — a
    # deleted camera would leave orphaned rows instead of cascading.
    @event.listens_for(engine.sync_engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class TimestampMixin:
    """Mixin that adds created_at and updated_at columns."""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def _add_column_if_missing(conn, table: str, column: str, ddl_type: str) -> None:
    """
    Add a column to an existing table if it isn't there yet.
    `Base.metadata.create_all` only creates missing TABLES, not missing
    COLUMNS on tables that already exist — so evolving a model (e.g. adding
    a field to `Alert`) needs this instead of a full migration tool, since
    this project doesn't run Alembic migrations in dev.
    """
    from sqlalchemy import text, inspect as sa_inspect

    def _has_column(sync_conn):
        return column in [c["name"] for c in sa_inspect(sync_conn).get_columns(table)]

    exists = await conn.run_sync(_has_column)
    if not exists:
        await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}"))


async def _make_alert_camera_nullable() -> None:
    """
    Resident reports carry a typed location and no camera, so
    `alerts.camera_id` must be nullable. SQLite can't drop NOT NULL with
    ALTER, so rebuild the table once (idempotent: no-op when already nullable).
    """
    from sqlalchemy import text
    if not is_sqlite:
        try:
            async with engine.begin() as conn:
                await conn.execute(text("ALTER TABLE alerts ALTER COLUMN camera_id DROP NOT NULL"))
        except Exception:
            pass  # table not created yet; create_all builds it nullable
        return

    import sqlite3
    path = settings.DATABASE_URL.split("///", 1)[1]
    con = sqlite3.connect(path, isolation_level=None)
    try:
        cols = {r[1]: r[3] for r in con.execute("PRAGMA table_info(alerts)")}
        if not cols or cols.get("camera_id") == 0:
            return
        ddl = con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='alerts'").fetchone()[0]
        extras = [r[0] for r in con.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='alerts' AND sql IS NOT NULL")]
        new_ddl = ddl.replace("camera_id INTEGER NOT NULL", "camera_id INTEGER", 1).replace(
            "CREATE TABLE alerts", "CREATE TABLE alerts_new", 1)
        con.execute("PRAGMA foreign_keys=OFF")
        con.execute("BEGIN")
        con.execute(new_ddl)
        con.execute("INSERT INTO alerts_new SELECT * FROM alerts")
        con.execute("DROP TABLE alerts")
        con.execute("ALTER TABLE alerts_new RENAME TO alerts")
        for stmt in extras:
            con.execute(stmt)
        con.execute("COMMIT")
    finally:
        con.close()


async def _rebase_stale_storage_paths(conn) -> None:
    """
    Absolute file paths stored in the DB (camera uploads, model weights) break
    when the project is moved to another folder/PC. For any such path whose
    file is missing, re-root the part after ".../storage/" onto the current
    STORAGE_PATH - only when the re-rooted file actually exists.
    """
    import re
    from pathlib import Path
    from sqlalchemy import text

    base = Path(settings.STORAGE_PATH)
    for table, column in (("cameras", "upload_path"), ("model_versions", "weights_path")):
        rows = (await conn.execute(text(f"SELECT id, {column} FROM {table} WHERE {column} IS NOT NULL"))).all()
        for row_id, value in rows:
            if not value or Path(value).exists():
                continue
            m = re.split(r"[\\/]storage[\\/]", value, maxsplit=1, flags=re.IGNORECASE)
            if len(m) != 2:
                continue
            fixed = base / Path(m[1].replace("\\", "/"))
            if fixed.exists():
                await conn.execute(
                    text(f"UPDATE {table} SET {column} = :v WHERE id = :i"),
                    {"v": str(fixed), "i": row_id},
                )


async def init_db():
    """Create all tables. Used for development only."""
    await _make_alert_camera_nullable()
    async with engine.begin() as conn:
        from app.models import (  # noqa: F401
            user, camera, camera_zone, security_event,
            alert, incident_review, training_sample,
            model_version, notification_setting, audit_log,
            system_setting, vehicle, vehicle_request, registered_person,
            visitor_face_request,
        )
        await conn.run_sync(Base.metadata.create_all)

        # Backfill columns added to existing models after their tables were
        # first created (SQLite and Postgres both support plain ADD COLUMN).
        await _add_column_if_missing(conn, "alerts", "source", "VARCHAR(20) NOT NULL DEFAULT 'ai'")
        await _add_column_if_missing(conn, "alerts", "reported_by", "INTEGER")
        await _add_column_if_missing(conn, "users", "avatar_path", "VARCHAR(500)")
        await _rebase_stale_storage_paths(conn)
        await _add_column_if_missing(conn, "visitor_face_requests", "visitor_name", "VARCHAR(120)")

        # Face Verification — additive columns on the existing `users`
        # table (see app/models/user.py). All nullable/defaulted, so
        # every pre-existing user row is valid with no backfill value
        # needed; `Vehicle`/`VehicleRequest` above are brand-new tables,
        # already handled by create_all.
        await _add_column_if_missing(conn, "users", "face_embedding", "JSON")
        await _add_column_if_missing(conn, "users", "face_registered", "BOOLEAN NOT NULL DEFAULT 0")
        await _add_column_if_missing(conn, "users", "face_registered_at", "TIMESTAMP WITH TIME ZONE")

        # Phase 2 — timeline-aware event fields. All nullable, so existing
        # rows (and non-lifecycle-tracked sources like rule-based/fire-smoke
        # alerts) are valid with no backfill value needed.
        for table in ("alerts", "security_events"):
            await _add_column_if_missing(conn, table, "start_time", "TIMESTAMP WITH TIME ZONE")
            await _add_column_if_missing(conn, table, "end_time", "TIMESTAMP WITH TIME ZONE")
            await _add_column_if_missing(conn, table, "peak_confidence", "FLOAT")
            await _add_column_if_missing(conn, table, "average_confidence", "FLOAT")

        # Visitor/resident parking distinction — existing rows default to
        # "resident" (self-registered), matching their prior behavior.
        await _add_column_if_missing(conn, "vehicles", "source", "VARCHAR(20) NOT NULL DEFAULT 'resident'")

        # Household/visitor face distinction, mirroring vehicles.source —
        # existing rows (all registered via POST /api/faces/register before
        # this column existed) default to "household" (permanent), matching
        # their prior behavior.
        await _add_column_if_missing(conn, "registered_people", "source", "VARCHAR(20) NOT NULL DEFAULT 'household'")

        # One-time-pass traceability + terminal "used" tracking (see
        # RegisteredPerson.source_request_id / Vehicle.source_request_id /
        # VisitorFaceRequestStatus.USED / VehicleRequestStatus.USED
        # docstrings). All nullable, so existing rows are valid untouched —
        # `status` itself needs no migration since SQLite's Enum column has
        # no CHECK constraint here (plain VARCHAR), so the new "used" value
        # is already writable.
        await _add_column_if_missing(conn, "registered_people", "source_request_id", "INTEGER")
        await _add_column_if_missing(conn, "vehicles", "source_request_id", "INTEGER")
        await _add_column_if_missing(conn, "visitor_face_requests", "consumed_at", "TIMESTAMP WITH TIME ZONE")
        await _add_column_if_missing(conn, "vehicle_requests", "consumed_at", "TIMESTAMP WITH TIME ZONE")


async def migrate_legacy_face_registrations() -> int:
    """
    One-time, idempotent data migration: a Resident who registered a
    face before the household-members model existed has their embedding
    sitting on User.face_embedding (single-face-per-account design).
    Copy — never move — that into a `relationship=self`
    RegisteredPerson row so they show up in the new multi-person
    household list/Guard verification/Admin directory, without ever
    touching or dropping the original User columns (kept intact per
    the "do not lose existing registered faces" requirement).

    Idempotent via a simple "does this resident already have ANY
    RegisteredPerson row" check — safe to call on every startup.
    Returns the number of residents migrated (0 on a normal restart
    after the first run).
    """
    from sqlalchemy import select
    from app.models.user import User, UserRole
    from app.models.registered_person import RegisteredPerson, PersonRelationship

    migrated = 0
    async with async_session_factory() as db:
        residents_result = await db.execute(
            select(User).where(User.role == UserRole.RESIDENT, User.face_registered == True)  # noqa: E712
        )
        residents = residents_result.scalars().all()

        for resident in residents:
            existing = await db.execute(
                select(RegisteredPerson.id).where(RegisteredPerson.resident_id == resident.id).limit(1)
            )
            if existing.scalar_one_or_none() is not None:
                continue  # already migrated (or already has household members registered normally)

            db.add(RegisteredPerson(
                resident_id=resident.id,
                full_name=resident.full_name,
                phone=resident.phone,
                relationship=PersonRelationship.SELF,
                face_embedding=resident.face_embedding,
                face_registered=resident.face_registered,
                face_registered_at=resident.face_registered_at,
            ))
            migrated += 1

        if migrated:
            await db.commit()
    return migrated


async def close_db():
    """Dispose the engine connection pool."""
    await engine.dispose()
