from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from mcscram_server.config import CalculationConfig
from mcscram_server.event_hub import EventHub
from mcscram_server.fault_tree_payload import build_fault_tree_payload, resolve_model_xml_path
from mcscram_server.supervisor import CalculationSupervisor


def create_router(supervisor: CalculationSupervisor, event_hub: EventHub) -> APIRouter:
    router = APIRouter()

    @router.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/status")
    def status() -> dict[str, Any]:
        return supervisor.state.snapshot(alive=supervisor.alive())

    @router.get("/model/fault-tree")
    def fault_tree_model(model_id: str | None = Query(default=None)) -> dict[str, Any]:
        if model_id:
            try:
                path = resolve_model_xml_path(model_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return build_fault_tree_payload([str(path)])
        return build_fault_tree_payload(CalculationConfig.from_env().input_files)

    @router.post("/reset")
    def reset_calculation() -> dict[str, Any]:
        return supervisor.restart(CalculationConfig.from_env())

    @router.post("/start")
    def start_sampling() -> dict[str, Any]:
        return supervisor.start_sampling(CalculationConfig.from_env())

    @router.post("/stop")
    def stop_sampling() -> dict[str, Any]:
        return supervisor.stop_sampling()

    @router.websocket("/ws")
    async def websocket_stream(websocket: WebSocket) -> None:
        replay = websocket.query_params.get("replay", "").lower() in {"1", "true", "yes", "on"}
        await event_hub.connect(
            websocket,
            supervisor.state.snapshot(alive=supervisor.alive()),
            replay=replay,
        )
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            event_hub.disconnect(websocket)

    return router
