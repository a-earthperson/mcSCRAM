from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from mcscram_server.config import CalculationConfig, parse_watch_value
from mcscram_server.event_hub import EventHub
from mcscram_server.fault_tree_payload import (
    build_dag_bundle_payload,
    build_dag_payload,
    build_execution_plan_payload,
    build_fault_tree_payload,
    build_kernel_dag_payload,
    build_sample_shape_payload,
    build_source_dag_payload,
    list_aralia_xml_inputs,
    resolve_model_xml_path,
)
from mcscram_server.native import InvalidPcfbError, resolve_uploaded_pcfb, store_uploaded_pcfb
from mcscram_server.supervisor import CalculationSupervisor


class SamplingControlBody(BaseModel):
    model_id: str | None = None
    compilation_level: int | None = Field(default=None, ge=0, le=8)
    expand_atleast_gates: bool | None = None
    expand_xor_gates: bool | None = None
    recognize_atleast_groups: bool | None = None
    recognize_xor_groups: bool | None = None
    ccf: bool | None = None
    keep_null_gates: bool | None = None
    # Total trial budget (0 → auto: unbounded / backend default).
    num_trials: float | None = Field(default=None, ge=0)
    # Optimistic per-node trials-per-pass target (0 → auto: fill device memory).
    requested_trials_per_pass: float | None = Field(default=None, ge=0)
    confidence: float | None = Field(default=None, gt=0, lt=1)
    delta: float | None = Field(default=None, gt=0)
    oracle: float | None = Field(default=None, ge=-1, le=1)
    early_stop: bool | None = None
    # Host-side convergence policy that gates early-stop ("wald" | "bayes" | "gbas").
    policy: str | None = Field(default=None, pattern="^(wald|bayes|gbas)$")
    burn_in: float | None = Field(default=None, ge=0)
    pcfb_id: str | None = None
    watch: list[str] | str | None = None


def _pcfb_path_from_id(pcfb_id: str | None) -> str | None:
    if not pcfb_id:
        return None
    try:
        return str(resolve_uploaded_pcfb(pcfb_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _calculation_config(body: SamplingControlBody | None = None) -> CalculationConfig:
    config = CalculationConfig.from_env()
    if body is None:
        return config
    input_files = config.input_files
    pcfb_path = config.pcfb_path
    if body.pcfb_id:
        pcfb_path = _pcfb_path_from_id(body.pcfb_id)
    if body.model_id:
        try:
            input_files = [str(resolve_model_xml_path(body.model_id))]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    options = _compile_options(
        config.options,
        compilation_level=body.compilation_level,
        expand_atleast_gates=body.expand_atleast_gates,
        expand_xor_gates=body.expand_xor_gates,
        recognize_atleast_groups=body.recognize_atleast_groups,
        recognize_xor_groups=body.recognize_xor_groups,
        ccf=body.ccf,
        keep_null_gates=body.keep_null_gates,
    )
    overrides: dict[str, Any] = {}
    if body.num_trials is not None:
        overrides["num_trials"] = body.num_trials
    if body.requested_trials_per_pass is not None:
        overrides["requested_trials_per_pass"] = body.requested_trials_per_pass
    if body.confidence is not None:
        overrides["confidence"] = body.confidence
    if body.delta is not None:
        overrides["delta"] = body.delta
    if body.oracle is not None:
        overrides["oracle"] = body.oracle
    if body.early_stop is not None:
        overrides["early_stop"] = body.early_stop
    if body.policy is not None:
        overrides["policy"] = body.policy
    if body.burn_in is not None:
        overrides["burn_in"] = body.burn_in
    if body.watch is not None:
        _, names = parse_watch_value(body.watch)
        if names:
            overrides["watch"] = names
    if overrides:
        options.update(overrides)
    if input_files == config.input_files and options == config.options and pcfb_path == config.pcfb_path:
        return config
    return CalculationConfig(input_files=input_files, options=options, pcfb_path=pcfb_path)


def create_router(supervisor: CalculationSupervisor, event_hub: EventHub) -> APIRouter:
    router = APIRouter(prefix="/api/v1")

    @router.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/status")
    def status() -> dict[str, Any]:
        return supervisor.state.snapshot(alive=supervisor.alive())

    @router.get("/model/xml-inputs")
    def xml_inputs() -> dict[str, Any]:
        return list_aralia_xml_inputs()

    @router.post("/model/pcfb")
    async def upload_pcfb(request: Request) -> dict[str, Any]:
        """Accept a `.pcfb` circuit, validate with ``fuzzybits.inspect``, and store it.

        Body: raw `.pcfb` bytes (``Content-Type: application/octet-stream``).
        On success: ``{"id": "<sha256>", "pcfb_id": "<sha256>", "inspect": {...}}``.
        Start a run with ``POST /api/v1/start`` and ``{"pcfb_id": "<sha256>"}``.
        Invalid circuits return HTTP 400 with ``diagnostics`` from ``pcfb::validate``.
        """
        data = await request.body()
        try:
            digest, path, summary = store_uploaded_pcfb(data)
        except InvalidPcfbError as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": str(exc),
                    "diagnostics": exc.diagnostics,
                    "inspect": exc.inspect,
                },
            ) from exc
        return {"id": digest, "pcfb_id": digest, "path": str(path), "inspect": summary}

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

    @router.get("/model/dag")
    def dag_model(
        model_id: str | None = Query(default=None),
        pcfb_id: str | None = Query(default=None),
        compilation_level: int | None = Query(default=None, ge=0, le=8),
        expand_atleast_gates: bool | None = Query(default=None),
        expand_xor_gates: bool | None = Query(default=None),
        recognize_atleast_groups: bool | None = Query(default=None),
        recognize_xor_groups: bool | None = Query(default=None),
        ccf: bool | None = Query(default=None),
        keep_null_gates: bool | None = Query(default=None),
    ) -> dict[str, Any]:
        config = CalculationConfig.from_env()
        options = _compile_options(
            config.options,
            compilation_level=compilation_level,
            expand_atleast_gates=expand_atleast_gates,
            expand_xor_gates=expand_xor_gates,
            recognize_atleast_groups=recognize_atleast_groups,
            recognize_xor_groups=recognize_xor_groups,
            ccf=ccf,
            keep_null_gates=keep_null_gates,
        )
        pcfb_path = _pcfb_path_from_id(pcfb_id)
        if model_id:
            try:
                path = resolve_model_xml_path(model_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return build_dag_payload([str(path)], options, pcfb_path=pcfb_path)
        return build_dag_payload(config.input_files, options, pcfb_path=pcfb_path)

    @router.get("/model/source-dag")
    def source_dag_model(model_id: str | None = Query(default=None)) -> dict[str, Any]:
        if model_id:
            try:
                path = resolve_model_xml_path(model_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return build_source_dag_payload([str(path)])
        return build_source_dag_payload(CalculationConfig.from_env().input_files)

    @router.get("/model/kernel-dag")
    def kernel_dag_model(
        model_id: str | None = Query(default=None),
        pcfb_id: str | None = Query(default=None),
        compilation_level: int | None = Query(default=None, ge=0, le=8),
        expand_atleast_gates: bool | None = Query(default=None),
        expand_xor_gates: bool | None = Query(default=None),
        recognize_atleast_groups: bool | None = Query(default=None),
        recognize_xor_groups: bool | None = Query(default=None),
        ccf: bool | None = Query(default=None),
        keep_null_gates: bool | None = Query(default=None),
    ) -> dict[str, Any]:
        config = CalculationConfig.from_env()
        options = _compile_options(
            config.options,
            compilation_level=compilation_level,
            expand_atleast_gates=expand_atleast_gates,
            expand_xor_gates=expand_xor_gates,
            recognize_atleast_groups=recognize_atleast_groups,
            recognize_xor_groups=recognize_xor_groups,
            ccf=ccf,
            keep_null_gates=keep_null_gates,
        )
        if model_id:
            try:
                path = resolve_model_xml_path(model_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return build_kernel_dag_payload([str(path)], options, pcfb_path=_pcfb_path_from_id(pcfb_id))
        return build_kernel_dag_payload(config.input_files, options, pcfb_path=_pcfb_path_from_id(pcfb_id))

    @router.get("/model/sample-shape")
    def sample_shape_model(
        model_id: str | None = Query(default=None),
        pcfb_id: str | None = Query(default=None),
        compilation_level: int | None = Query(default=None, ge=0, le=8),
        expand_atleast_gates: bool | None = Query(default=None),
        expand_xor_gates: bool | None = Query(default=None),
        recognize_atleast_groups: bool | None = Query(default=None),
        recognize_xor_groups: bool | None = Query(default=None),
        ccf: bool | None = Query(default=None),
        keep_null_gates: bool | None = Query(default=None),
        num_trials: float | None = Query(default=None, ge=0),
        requested_trials_per_pass: float | None = Query(default=None, ge=0),
    ) -> dict[str, Any]:
        config = CalculationConfig.from_env()
        options = _compile_options(
            config.options,
            compilation_level=compilation_level,
            expand_atleast_gates=expand_atleast_gates,
            expand_xor_gates=expand_xor_gates,
            recognize_atleast_groups=recognize_atleast_groups,
            recognize_xor_groups=recognize_xor_groups,
            ccf=ccf,
            keep_null_gates=keep_null_gates,
        )
        if num_trials is not None:
            options["num_trials"] = num_trials
        if requested_trials_per_pass is not None:
            options["requested_trials_per_pass"] = requested_trials_per_pass
        if model_id:
            try:
                path = resolve_model_xml_path(model_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return build_sample_shape_payload([str(path)], options, pcfb_path=_pcfb_path_from_id(pcfb_id))
        return build_sample_shape_payload(config.input_files, options, pcfb_path=_pcfb_path_from_id(pcfb_id))

    @router.get("/model/execution-plan")
    def execution_plan_model(
        model_id: str | None = Query(default=None),
        pcfb_id: str | None = Query(default=None),
        compilation_level: int | None = Query(default=None, ge=0, le=8),
        expand_atleast_gates: bool | None = Query(default=None),
        expand_xor_gates: bool | None = Query(default=None),
        recognize_atleast_groups: bool | None = Query(default=None),
        recognize_xor_groups: bool | None = Query(default=None),
        ccf: bool | None = Query(default=None),
        keep_null_gates: bool | None = Query(default=None),
        num_trials: float | None = Query(default=None, ge=0),
        requested_trials_per_pass: float | None = Query(default=None, ge=0),
    ) -> dict[str, Any]:
        config = CalculationConfig.from_env()
        options = _compile_options(
            config.options,
            compilation_level=compilation_level,
            expand_atleast_gates=expand_atleast_gates,
            expand_xor_gates=expand_xor_gates,
            recognize_atleast_groups=recognize_atleast_groups,
            recognize_xor_groups=recognize_xor_groups,
            ccf=ccf,
            keep_null_gates=keep_null_gates,
        )
        if num_trials is not None:
            options["num_trials"] = num_trials
        if requested_trials_per_pass is not None:
            options["requested_trials_per_pass"] = requested_trials_per_pass
        if model_id:
            try:
                path = resolve_model_xml_path(model_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return build_execution_plan_payload([str(path)], options, pcfb_path=_pcfb_path_from_id(pcfb_id))
        return build_execution_plan_payload(config.input_files, options, pcfb_path=_pcfb_path_from_id(pcfb_id))

    @router.get("/model/dag-bundle")
    def dag_bundle_model(
        model_id: str | None = Query(default=None),
        pcfb_id: str | None = Query(default=None),
        compilation_level: int | None = Query(default=None, ge=0, le=8),
        expand_atleast_gates: bool | None = Query(default=None),
        expand_xor_gates: bool | None = Query(default=None),
        recognize_atleast_groups: bool | None = Query(default=None),
        recognize_xor_groups: bool | None = Query(default=None),
        ccf: bool | None = Query(default=None),
        keep_null_gates: bool | None = Query(default=None),
    ) -> dict[str, Any]:
        config = CalculationConfig.from_env()
        options = _compile_options(
            config.options,
            compilation_level=compilation_level,
            expand_atleast_gates=expand_atleast_gates,
            expand_xor_gates=expand_xor_gates,
            recognize_atleast_groups=recognize_atleast_groups,
            recognize_xor_groups=recognize_xor_groups,
            ccf=ccf,
            keep_null_gates=keep_null_gates,
        )
        if model_id:
            try:
                path = resolve_model_xml_path(model_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return build_dag_bundle_payload([str(path)], options, pcfb_path=_pcfb_path_from_id(pcfb_id))
        return build_dag_bundle_payload(config.input_files, options, pcfb_path=_pcfb_path_from_id(pcfb_id))

    @router.post("/reset")
    def reset_calculation(body: SamplingControlBody | None = Body(default=None)) -> dict[str, Any]:
        return supervisor.restart(_calculation_config(body))

    @router.post("/start")
    def start_sampling(body: SamplingControlBody | None = Body(default=None)) -> dict[str, Any]:
        return supervisor.start_sampling(_calculation_config(body))

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


def _compile_options(
    base: dict[str, Any],
    *,
    compilation_level: int | None,
    expand_atleast_gates: bool | None,
    expand_xor_gates: bool | None,
    recognize_atleast_groups: bool | None,
    recognize_xor_groups: bool | None,
    ccf: bool | None,
    keep_null_gates: bool | None,
) -> dict[str, Any]:
    options = dict(base)
    if compilation_level is not None:
        options["compilation_level"] = compilation_level
    if expand_atleast_gates is not None:
        options["expand_atleast_gates"] = expand_atleast_gates
    if expand_xor_gates is not None:
        options["expand_xor_gates"] = expand_xor_gates
    if recognize_atleast_groups is not None:
        options["recognize_atleast_groups"] = recognize_atleast_groups
    if recognize_xor_groups is not None:
        options["recognize_xor_groups"] = recognize_xor_groups
    if ccf is not None:
        options["ccf"] = ccf
    if keep_null_gates is not None:
        options["keep_null_gates"] = keep_null_gates
    return options
