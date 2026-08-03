from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, cast

from orchestrator.brain_runtime import (
    AsyncJsonCompletion,
    build_async_agent_gate,
    build_async_context_compactor,
    build_async_memory_candidate_extractor,
    build_async_response_coordinator,
)
from orchestrator.config import OrchestratorConfig, load_config_from_env
from orchestrator.ids import SessionId
from orchestrator.observability import OnsiteObservability
from orchestrator.onsite_bridge import build_onsite_bridge
from orchestrator.scheduler_runtime import SessionRuntime
from orchestrator.task_registry import SchedulerTaskConfig, TaskKind
from orchestrator.transport_config import load_transport_config_from_env
from orchestrator.transport_runtime import TransportRuntime

if TYPE_CHECKING:
    from collections.abc import Mapping


async def run_transport() -> None:
    config = load_config_from_env(os.environ)

    bridge = None

    observability = None

    if _onsite_bridge_enabled(config):
        observability = OnsiteObservability(config)

        bridge = build_onsite_bridge(
            config,
            voice=os.environ.get("ORCHESTRATOR_TTS_VOICE", ""),
            ref_audio=os.environ.get("ORCHESTRATOR_TTS_REF_AUDIO", ""),
            ref_text=os.environ.get("ORCHESTRATOR_TTS_REF_TEXT", ""),
        )

    transport_config = load_transport_config_from_env(os.environ)

    brain_completion = getattr(bridge, "llm", None)

    def create_session_runtime(session_id: SessionId) -> SessionRuntime:
        session_runtime = SessionRuntime.create(
            session_id=session_id,
            turn_id_prefix="turn",
            task_config=SchedulerTaskConfig(frozenset(TaskKind), 1),
        )
        if brain_completion is not None:
            session_runtime.async_agent_gate = build_async_agent_gate(
                cast("AsyncJsonCompletion", brain_completion)
            )
            session_runtime.async_response_coordinator = (
                build_async_response_coordinator(
                    cast("AsyncJsonCompletion", brain_completion),
                    session_runtime.interaction_ingress.data.retrieval,
                )
            )
            session_runtime.memory_candidate_extractor = (
                build_async_memory_candidate_extractor(
                    cast("AsyncJsonCompletion", brain_completion)
                )
            )
            session_runtime.context_compactor = build_async_context_compactor(
                cast("AsyncJsonCompletion", brain_completion)
            )
        return session_runtime

    session_runtime = create_session_runtime(
        SessionId(f"{config.session_id_prefix}-control")
    )

    runtime = TransportRuntime(transport_config, onsite_bridge=bridge)

    try:
        runtime.set_session_runtime(session_runtime)
        runtime.set_session_runtime_factory(create_session_runtime)

        if observability is not None:
            runtime.set_observability(observability)

        await runtime.start()

        await asyncio.Future[None]()

    finally:
        await runtime.close()


def _onsite_bridge_enabled(config: OrchestratorConfig) -> bool:
    return (
        config.llm_provider == "openai_compatible"
        and config.tts_provider == "vllm_omni"
    )


def main() -> None:
    logging.basicConfig(
        level=_log_level(os.environ),
        format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    configure_dependency_loggers()
    try:
        asyncio.run(run_transport())

    except KeyboardInterrupt:
        return


def _log_level(env: Mapping[str, str]) -> int:
    """Return the development-selectable level; production defaults to INFO."""
    return getattr(logging, env.get("BITNP_LOG_LEVEL", "INFO").upper(), logging.INFO)


def configure_dependency_loggers() -> None:
    """Keep provider diagnostics readable without exposing request payloads."""
    for logger_name in ("openai", "httpcore"):
        logging.getLogger(logger_name).setLevel(logging.INFO)
