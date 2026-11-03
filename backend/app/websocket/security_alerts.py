from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from app.websocket.manager import ws_manager
from app.utils.security import decode_access_token
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/security-alerts")
async def security_alerts_websocket(
    websocket: WebSocket,
    token: str = Query(None),
):
    """WebSocket endpoint for real-time security alerts."""
    # Authenticate via query parameter token
    if not token:
        await websocket.close(code=4001, reason="Missing authentication token")
        return

    payload = decode_access_token(token)
    if payload is None:
        await websocket.close(code=4001, reason="Invalid authentication token")
        return

    user_id = int(payload.get("sub", 0))
    role = payload.get("role", "resident")

    # Behaviour and Fire/Smoke alerts are a security-operator concern —
    # a Resident must never receive them, over this channel or any other.
    # Rejecting the connection outright (rather than connecting and
    # relying on broadcast_alert to filter) means a Resident's valid JWT
    # can't be used to open this socket at all, matching the same
    # backend-enforced restriction as the live stream endpoints.
    if role == "resident":
        await websocket.close(code=4403, reason="Residents do not receive security alerts")
        return

    await ws_manager.connect(websocket, user_id, role)

    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection_established",
            "user_id": user_id,
            "role": role,
            "message": "Connected to security alert feed",
        })

        # Keep connection alive and listen for client messages
        while True:
            data = await websocket.receive_text()
            # Handle client messages (e.g., acknowledge receipt)
            if data == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket, user_id)
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        await ws_manager.disconnect(websocket, user_id)
