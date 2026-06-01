from __future__ import annotations

import asyncio
import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mcscram_server.config import CalculationConfig, env_bool
from mcscram_server.event_hub import EventHub
from mcscram_server.routes import create_router
from mcscram_server.supervisor import CalculationSupervisor


def create_app(
    *,
    event_hub: EventHub | None = None,
    supervisor: CalculationSupervisor | None = None,
    autostart: bool | None = None,
) -> FastAPI:
    hub = event_hub or EventHub()
    calc_supervisor = supervisor or CalculationSupervisor(hub)

    app = FastAPI(title="mcSCRAM Server")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.state.event_hub = hub
    app.state.supervisor = calc_supervisor
    app.include_router(create_router(calc_supervisor, hub))

    should_autostart = autostart if autostart is not None else env_bool("MCSCRAM_AUTOSTART", False)

    @app.on_event("startup")
    async def on_startup() -> None:
        hub.bind_loop(asyncio.get_running_loop())
        if should_autostart:
            calc_supervisor.start(CalculationConfig.from_env())

    @app.on_event("shutdown")
    def on_shutdown() -> None:
        calc_supervisor.stop()

    return app


app = create_app()


def main() -> None:
    uvicorn.run(
        "mcscram_server.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
