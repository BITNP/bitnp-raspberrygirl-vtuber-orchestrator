"""Executable composition root for the Orchestrator control and RTP transport."""

from __future__ import annotations

import asyncio
import os

from orchestrator.config import load_config_from_env
from orchestrator.modes import OrchestratorMode, parse_orchestrator_mode
from orchestrator.onsite_bridge import build_onsite_bridge
from orchestrator.transport_config import load_transport_config_from_env
from orchestrator.transport_runtime import TransportRuntime


async def run_transport() -> None:
    """Start the configured listener pair and close both on cancellation."""
    mode = parse_orchestrator_mode(
        os.environ.get("ORCHESTRATOR_MODE", OrchestratorMode.VIRTUAL_STREAMER.value)
    )
    bridge = None
    if mode is OrchestratorMode.ONSITE_EXPLAINER:
        bridge = build_onsite_bridge(
            load_config_from_env(os.environ),
            voice=os.environ.get("ORCHESTRATOR_TTS_VOICE", ""),
            ref_audio=os.environ.get("ORCHESTRATOR_TTS_REF_AUDIO", ""),
            ref_text=os.environ.get("ORCHESTRATOR_TTS_REF_TEXT", ""),
        )
    runtime = TransportRuntime(
        load_transport_config_from_env(os.environ), onsite_bridge=bridge
    )
    await runtime.start()
    try:
        await asyncio.Future[None]()
    finally:
        await runtime.close()


def main() -> None:
    """Run the production WSS/UDP transport process."""
    try:
        asyncio.run(run_transport())
    except KeyboardInterrupt:
        return
