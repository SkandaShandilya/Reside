from pydantic_settings import BaseSettings
from typing import List
from pathlib import Path
import os

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Database (canonical absolute path)
    DATABASE_URL: str = f"sqlite+aiosqlite:///{BACKEND_DIR.as_posix()}/sentinel.db"

    # JWT
    JWT_SECRET_KEY: str = "change-this-to-a-random-secret-key-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Storage (canonical absolute path)
    STORAGE_TYPE: str = "local"
    STORAGE_PATH: str = str((BACKEND_DIR / "storage").resolve())

    # AI Service
    AI_SERVICE_URL: str = "http://localhost:8001"
    AI_PROCESSING_FPS: int = 10
    CONFIDENCE_THRESHOLD: float = 0.80
    ALERT_COOLDOWN_SECONDS: int = 30

    # Face Verification / Vehicle OCR — a SEPARATE Python process
    # (face-service/, its own venv) from ai_service, deliberately: its
    # pinned deps (numpy==1.26.4, opencv-python==4.10.0.84) directly
    # conflict with the numpy 2.x / opencv 5.x this backend's ai_service
    # (YOLO, Fighting CNN, tracker.py) already depends on — installing
    # both into one environment would force a downgrade and risk
    # breaking Behaviour/Fire-Smoke. Communication is HTTP-only (httpx),
    # matching the original migration's Node→Python pattern. Port 8010,
    # not 8001 (AI_SERVICE_URL above), to avoid any confusion with that
    # already-declared-but-unused setting.
    FACE_SERVICE_URL: str = "http://localhost:8010"
    # Cosine similarity threshold for a face match — same default (0.50)
    # and same env-var name (FACE_MATCH_THRESHOLD) as the original
    # migration, so an operator's existing expectations carry over.
    FACE_MATCH_THRESHOLD: float = 0.50

    # CORS
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    class Config:
        env_file = [".env", "../.env"]
        env_file_encoding = "utf-8"


settings = Settings()
