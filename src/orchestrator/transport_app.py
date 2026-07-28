"""Executable composition root for the Orchestrator control and RTP transport."""

from __future__ import annotations

import asyncio
import os

from orchestrator.transport_config import load_transport_config_from_env
from orchestrator.transport_runtime import TransportRuntime


async def run_transport() -> None:
    """Start the configured listener pair and close both on cancellation."""
    runtime = TransportRuntime(load_transport_config_from_env(os.environ))
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
