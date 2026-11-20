from __future__ import annotations

import asyncio
import os
import threading
import time
import webbrowser

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fuzzybits_server.bench import create_bench_router
from fuzzybits_server.config import CalculationConfig, apply_server_cli, env_bool
from fuzzybits_server.event_hub import EventHub
from fuzzybits_server.native import shutdown_native_workers
from fuzzybits_server.routes import create_router
from fuzzybits_server.supervisor import CalculationSupervisor
from fuzzybits_server.ui import mount_ui


def create_app(
    *,
    event_hub: EventHub | None = None,
    supervisor: CalculationSupervisor | None = None,
    autostart: bool | None = None,
) -> FastAPI:
    hub = event_hub or EventHub()
    calc_supervisor = supervisor or CalculationSupervisor(hub)

    app = FastAPI(
        title="fuzzybits-server",
        description=(
            "Compile OpenPSA/MEF fault trees to a PDAG (.pcfb) and sample the top event. "
            "Monte Carlo estimates with confidence intervals. "
            "Event-tree sequences are not compiled or sampled."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.state.event_hub = hub
    app.state.supervisor = calc_supervisor
    app.include_router(create_router(calc_supervisor, hub))
    app.include_router(create_bench_router())
    mount_ui(app)

    should_autostart = autostart if autostart is not None else env_bool("MCSCRAM_AUTOSTART", False)

    @app.on_event("startup")
    async def on_startup() -> None:
        hub.bind_loop(asyncio.get_running_loop())
        if should_autostart:
            calc_supervisor.start(CalculationConfig.from_env())

    @app.on_event("shutdown")
    def on_shutdown() -> None:
        calc_supervisor.stop()
        shutdown_native_workers()

    return app


app = create_app()


def _browser_url(host: str, port: int) -> str:
    browse_host = "127.0.0.1" if host in {"0.0.0.0", "::", "[::]"} else host
    return f"http://{browse_host}:{port}/"


def main() -> None:
    args = apply_server_cli()
    if args.open:
        url = _browser_url(args.host, args.port)

        def _open() -> None:
            time.sleep(0.6)
            webbrowser.open(url)

        threading.Thread(target=_open, daemon=True).start()
    uvicorn.run(
        "fuzzybits_server.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        log_level=os.getenv("LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
