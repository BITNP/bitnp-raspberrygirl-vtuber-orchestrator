import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial

import pytest

from orchestrator import transport_app
from orchestrator.interaction_ingress import SessionInteractionIngress
from orchestrator.scheduler_runtime import SessionRuntime
from orchestrator.sessions import SessionScheduler


@dataclass(frozen=True, slots=True)
class _Config:
    session_id_prefix: str


@dataclass
class _Runtime:
    ingress: SessionInteractionIngress | None = None
    closed: bool = False

    async def start(self) -> None:
        raise asyncio.CancelledError

    def set_session_runtime(self, session_runtime: SessionRuntime) -> None:
        self.ingress = session_runtime.interaction_ingress

    async def close(self) -> None:
        self.closed = True


def _test_config(_env: Mapping[str, str]) -> _Config:
    return _Config("test")


def _test_transport_config(_env: Mapping[str, str]) -> None:
    return None


def _test_runtime(runtime: _Runtime, *_args: str, **_kwargs: str) -> _Runtime:
    return runtime


def test_transport_composes_one_scheduler_control_ingress_before_listening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: deterministic configuration and listener stop at the transport entrypoint.
    schedulers: list[SessionScheduler] = []
    original_create = SessionInteractionIngress.create
    runtime = _Runtime()

    def capture_scheduler(scheduler: SessionScheduler) -> SessionInteractionIngress:
        schedulers.append(scheduler)
        return original_create(scheduler)

    monkeypatch.setattr(
        transport_app,
        "load_config_from_env",
        _test_config,
    )
    monkeypatch.setattr(
        transport_app,
        "load_transport_config_from_env",
        _test_transport_config,
    )
    monkeypatch.setattr(
        SessionInteractionIngress,
        "create",
        staticmethod(capture_scheduler),
    )
    monkeypatch.setattr(
        transport_app,
        "TransportRuntime",
        partial(_test_runtime, runtime),
    )

    # When: production transport composition reaches its listener lifecycle.
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(transport_app.run_transport())

    # Then: exactly one scheduler-owned ingress is installed before listener startup.
    assert schedulers[0].snapshot.session_id == "test-control"
    assert schedulers[0].snapshot.active_turn_id is None
    assert runtime.ingress is not None
    assert runtime.closed is True
