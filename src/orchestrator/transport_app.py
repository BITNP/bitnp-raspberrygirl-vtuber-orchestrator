
from __future__ import annotations

import asyncio
import logging
import os

from orchestrator.config import OrchestratorConfig, load_config_from_env
from orchestrator.ids import SessionId
from orchestrator.observability import OnsiteObservability
from orchestrator.onsite_bridge import build_onsite_bridge
from orchestrator.scheduler_runtime import SessionRuntime
from orchestrator.task_registry import SchedulerTaskConfig, TaskKind
from orchestrator.transport_config import load_transport_config_from_env
from orchestrator.transport_runtime import TransportRuntime


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

    session_runtime = SessionRuntime.create(
        session_id=SessionId(f"{config.session_id_prefix}-control"),
        turn_id_prefix="turn-control",
        task_config=SchedulerTaskConfig(frozenset(TaskKind), 1),
    )

    runtime = TransportRuntime(transport_config, onsite_bridge=bridge)

    try:
        runtime.set_session_runtime(session_runtime)

        if observability is not None:
            runtime.set_observability(observability)

        await runtime.start()

        await asyncio.Future[None]()

    finally:
        await runtime.close()


def _onsite_bridge_enabled(config: OrchestratorConfig) -> bool:
    return (
        config.asr_provider in {"openai_compatible", "funasr"}
        and config.llm_provider == "openai_compatible"
        and config.tts_provider == "vllm_omni"
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    try:
        asyncio.run(run_transport())

    except KeyboardInterrupt:
        return
