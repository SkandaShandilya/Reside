import os
import shutil
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from app.config import settings


class StorageBackend(ABC):
    """Abstract storage interface for S3-compatible replacement later."""

    @abstractmethod
    async def save_file(self, file_data: bytes, path: str, filename: str) -> str:
        ...

    @abstractmethod
    async def get_file(self, file_path: str) -> bytes:
        ...

    @abstractmethod
    async def delete_file(self, file_path: str) -> bool:
        ...

    @abstractmethod
    async def file_exists(self, file_path: str) -> bool:
        ...


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage for development."""

    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path or settings.STORAGE_PATH)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def save_file(self, file_data: bytes, path: str, filename: str) -> str:
        dir_path = self.base_path / path
        dir_path.mkdir(parents=True, exist_ok=True)
        # Generate unique filename
        ext = Path(filename).suffix
        unique_name = f"{uuid.uuid4().hex}{ext}"
        file_path = dir_path / unique_name
        file_path.write_bytes(file_data)
        return str(file_path.relative_to(self.base_path))

    async def get_file(self, file_path: str) -> bytes:
        full_path = self.base_path / file_path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return full_path.read_bytes()

    async def delete_file(self, file_path: str) -> bool:
        full_path = self.base_path / file_path
        if full_path.exists():
            full_path.unlink()
            return True
        return False

    async def file_exists(self, file_path: str) -> bool:
        return (self.base_path / file_path).exists()

    def get_absolute_path(self, file_path: str) -> str:
        return str(self.base_path / file_path)


def get_storage() -> StorageBackend:
    """Factory for storage backend — swap to S3 implementation later."""
    return LocalStorageBackend()
