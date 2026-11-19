from __future__ import annotations

import asyncio
import threading
from collections import deque
from typing import Any

from fastapi import WebSocket


class EventHub:
    def __init__(self, max_replay_events: int = 10_000) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[WebSocket] = set()
        self._recent: deque[dict[str, Any]] = deque(maxlen=max_replay_events)
        self._recent_lock = threading.Lock()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(
        self,
        websocket: WebSocket,
        initial_status: dict[str, Any],
        replay: bool = False,
    ) -> None:
        await websocket.accept()
        self._clients.add(websocket)
        await websocket.send_json({"event": "snapshot", "status": initial_status})
        if replay:
            with self._recent_lock:
                recent = list(self._recent)
            for event in recent:
                await websocket.send_json(event)

    def disconnect(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)

    def publish_from_thread(self, event: dict[str, Any]) -> None:
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(lambda: asyncio.create_task(self._publish(event)))

    async def _publish(self, event: dict[str, Any]) -> None:
        with self._recent_lock:
            self._recent.append(event)
        stale: list[WebSocket] = []
        for client in self._clients:
            try:
                await client.send_json(event)
            except RuntimeError:
                stale.append(client)
        for client in stale:
            self.disconnect(client)

    def clear_replay(self) -> None:
        with self._recent_lock:
            self._recent.clear()
