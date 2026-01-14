"""
WebSocket Router for Real-Time Pipeline Updates
Handles WebSocket connections for live pipeline status broadcasting
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
import logging
from datetime import datetime

router = APIRouter()
logger = logging.getLogger(__name__)

# Store active WebSocket connections per file_id
active_connections: Dict[str, Set[WebSocket]] = {}


class ConnectionManager:
    """Manages WebSocket connections for pipeline updates"""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, file_id: str):
        """Accept new WebSocket connection"""
        await websocket.accept()
        
        if file_id not in self.active_connections:
            self.active_connections[file_id] = set()
        
        self.active_connections[file_id].add(websocket)
        logger.info(f"WebSocket connected for file_id: {file_id}")
        
        # Send initial connection success message
        await self.send_personal_message(
            {
                "event": "connected",
                "file_id": file_id,
                "message": "WebSocket connection established",
                "timestamp": datetime.utcnow().isoformat()
            },
            websocket
        )

    def disconnect(self, websocket: WebSocket, file_id: str):
        """Remove WebSocket connection"""
        if file_id in self.active_connections:
            self.active_connections[file_id].discard(websocket)
            
            # Clean up empty sets
            if not self.active_connections[file_id]:
                del self.active_connections[file_id]
        
        logger.info(f"WebSocket disconnected for file_id: {file_id}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific WebSocket"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")

    async def broadcast(self, message: dict, file_id: str):
        """Broadcast message to all connections for a file_id"""
        if file_id not in self.active_connections:
            return

        disconnected = set()
        
        for connection in self.active_connections[file_id]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.add(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection, file_id)

    def get_connection_count(self, file_id: str) -> int:
        """Get number of active connections for file_id"""
        return len(self.active_connections.get(file_id, set()))


# Global connection manager instance
manager = ConnectionManager()


@router.websocket("/ws/pipeline/{file_id}")
async def pipeline_websocket(websocket: WebSocket, file_id: str):
    """
    WebSocket endpoint for real-time pipeline updates
    
    Args:
        websocket: WebSocket connection
        file_id: File identifier for pipeline tracking
    """
    await manager.connect(websocket, file_id)
    
    try:
        while True:
            # Wait for messages from client (keepalive, etc.)
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                
                # Handle ping/pong for keepalive
                if message.get("type") == "ping":
                    await manager.send_personal_message(
                        {
                            "type": "pong",
                            "timestamp": datetime.utcnow().isoformat()
                        },
                        websocket
                    )
                
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received: {data}")
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, file_id)
        logger.info(f"Client disconnected: {file_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {file_id}: {e}")
        manager.disconnect(websocket, file_id)


async def broadcast_pipeline_event(
    file_id: str,
    event: str,
    message: str = "",
    step: str = None,
    status: str = None,
    progress: int = None,
    data: dict = None
):
    """
    Broadcast pipeline event to all connected clients
    
    Args:
        file_id: File identifier
        event: Event type (status, progress, log, step, error, complete)
        message: Event message
        step: Current pipeline step
        status: Step status
        progress: Progress percentage (0-100)
        data: Additional event data
    """
    payload = {
        "event": event,
        "file_id": file_id,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if step:
        payload["step"] = step
    if status:
        payload["status"] = status
    if progress is not None:
        payload["progress"] = progress
    if data:
        payload["data"] = data
    
    await manager.broadcast(payload, file_id)
    logger.debug(f"Broadcast event '{event}' for file_id: {file_id}")


# Convenience functions for common events

async def broadcast_step_start(file_id: str, step: str, message: str = ""):
    """Broadcast step start event"""
    await broadcast_pipeline_event(
        file_id=file_id,
        event="step",
        message=message or f"Starting {step}...",
        step=step,
        status="running"
    )


async def broadcast_step_complete(file_id: str, step: str, message: str = ""):
    """Broadcast step completion event"""
    await broadcast_pipeline_event(
        file_id=file_id,
        event="step",
        message=message or f"Completed {step}",
        step=step,
        status="success"
    )


async def broadcast_step_error(file_id: str, step: str, error: str):
    """Broadcast step error event"""
    await broadcast_pipeline_event(
        file_id=file_id,
        event="error",
        message=error,
        step=step,
        status="error"
    )


async def broadcast_progress(file_id: str, progress: int, message: str = ""):
    """Broadcast progress update"""
    await broadcast_pipeline_event(
        file_id=file_id,
        event="progress",
        message=message,
        progress=progress
    )


async def broadcast_log(file_id: str, message: str, level: str = "info"):
    """Broadcast log message"""
    await broadcast_pipeline_event(
        file_id=file_id,
        event="log",
        message=message,
        data={"level": level}
    )


async def broadcast_complete(file_id: str, message: str = "", data: dict = None):
    """Broadcast pipeline completion"""
    await broadcast_pipeline_event(
        file_id=file_id,
        event="complete",
        message=message or "Pipeline completed successfully",
        status="success",
        progress=100,
        data=data
    )
