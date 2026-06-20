"""WebSocket connection manager for the Anomaly API.

Manages a set of active WebSocket clients and broadcasts scored-event
messages to all of them. Dead connections are pruned automatically.
"""
from __future__ import annotations

import asyncio
from typing import Set

from fastapi import WebSocket


class ConnectionManager:
    """Thread-safe (asyncio-level) manager for active WebSocket connections."""

    def __init__(self) -> None:
        self.active: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        """Accept the WebSocket handshake and register the connection."""
        await ws.accept()
        async with self._lock:
            self.active.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        """Remove a connection (sync — safe to call from exception handlers)."""
        self.active.discard(ws)

    async def broadcast(self, message: dict) -> None:
        """Send a JSON message to all connected clients.

        Dead sockets (clients that closed without a clean handshake) raise an
        exception on send; they are collected and pruned from the active set.
        """
        async with self._lock:
            snapshot = set(self.active)

        dead: list[WebSocket] = []
        for ws in snapshot:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    self.active.discard(ws)


# Module-level singleton used by main.py and the background Kafka task
manager = ConnectionManager()

__all__ = ["ConnectionManager", "manager"]
