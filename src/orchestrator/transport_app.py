"""Executable composition root for the Orchestrator control and RTP transport."""

from __future__ import annotations

import asyncio
import os

from orchestrator.config import load_config_from_env
from orchestrator.ids import SessionId
from orchestrator.modes import OrchestratorMode, parse_orchestrator_mode
from orchestrator.observability import OnsiteObservability
from orchestrator.onsite_bridge import build_onsite_bridge
from orchestrator.scheduler_runtime import SessionRuntime
from orchestrator.task_registry import SchedulerTaskConfig, TaskKind
from orchestrator.transport_config import load_transport_config_from_env
from orchestrator.transport_runtime import TransportRuntime


async def run_transport() -> None:
    """Start the configured listener pair and close both on cancellation."""
    mode = parse_orchestrator_mode(
        os.environ.get("ORCHESTRATOR_MODE", OrchestratorMode.VIRTUAL_STREAMER.value)
    )
    config = load_config_from_env(os.environ)
    bridge = None
    observability = None
    if mode is OrchestratorMode.ONSITE_EXPLAINER:
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
        mode=mode,
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


def main() -> None:
    """Run the production WSS/UDP transport process."""
    try:
        asyncio.run(run_transport())
    except KeyboardInterrupt:
        return
