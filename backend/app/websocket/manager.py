from typing import Dict, Set, Optional
from fastapi import WebSocket
import json
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    """WebSocket connection manager with role-based message filtering."""

    def __init__(self):
        # Map of user_id -> set of WebSocket connections
        self.active_connections: Dict[int, Set[WebSocket]] = {}
        # Map of user_id -> user role
        self.user_roles: Dict[int, str] = {}
        # All connections for broadcast
        self.all_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket, user_id: int, role: str):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(websocket)
        self.user_roles[user_id] = role
        self.all_connections.add(websocket)
        logger.info(f"WebSocket connected: user={user_id}, role={role}")

    async def disconnect(self, websocket: WebSocket, user_id: int):
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
                self.user_roles.pop(user_id, None)
        self.all_connections.discard(websocket)
        logger.info(f"WebSocket disconnected: user={user_id}")

    async def send_personal(self, message: dict, user_id: int):
        if user_id in self.active_connections:
            for ws in self.active_connections[user_id]:
                try:
                    await ws.send_json(message)
                except Exception:
                    pass

    async def broadcast(self, message: dict):
        """Send to all connected users."""
        disconnected = set()
        for ws in self.all_connections:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.add(ws)
        self.all_connections -= disconnected

    async def broadcast_to_roles(self, message: dict, roles: list[str]):
        """Send to users with specific roles only."""
        for user_id, role in self.user_roles.items():
            if role in roles:
                await self.send_personal(message, user_id)

    async def broadcast_alert(self, alert_data: dict):
        """Behaviour/Fire-Smoke alerts are a security-operator concern at
        any severity — Residents are not security operators and must not
        receive them (not even a sanitized version), per the platform's
        strict role separation. Guards/admins only, always. In practice a
        Resident can never be a key in user_roles anyway, since
        security_alerts_websocket now refuses their connection outright —
        this filter is the belt to that suspenders."""
        await self.broadcast_to_roles(alert_data, ["guard", "admin", "super_admin"])

    @property
    def connection_count(self) -> int:
        return len(self.all_connections)


# Global connection manager instance
ws_manager = ConnectionManager()
