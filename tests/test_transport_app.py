
import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial

import pytest

from orchestrator import transport_app
from orchestrator.interaction_ingress import SessionInteractionIngress
from orchestrator.observability import OnsiteObservability
from orchestrator.scheduler_runtime import SessionRuntime
from orchestrator.sessions import SessionScheduler


@dataclass(frozen=True, slots=True)
class _Config:

    session_id_prefix: str

    asr_provider: str = "mock"

    llm_provider: str = "mock"

    tts_provider: str = "mock"


@dataclass
class _Bridge:
    ...


@dataclass
class _Runtime:

    ingress: SessionInteractionIngress | None = None

    closed: bool = False

    onsite_bridge: _Bridge | None = None

    observability_set: bool = False

    async def start(self) -> None:

        raise asyncio.CancelledError

    def set_session_runtime(self, session_runtime: SessionRuntime) -> None:

        self.ingress = session_runtime.interaction_ingress

    def set_observability(self, _observability: OnsiteObservability) -> None:

        self.observability_set = True

    async def close(self) -> None:

        self.closed = True


def _test_config(_env: Mapping[str, str]) -> _Config:

    return _Config("test")


def _onsite_config(_env: Mapping[str, str]) -> _Config:

    return _Config(
        session_id_prefix="test",
        asr_provider="openai_compatible",
        llm_provider="openai_compatible",
        tts_provider="vllm_omni",
    )


def _test_transport_config(_env: Mapping[str, str]) -> None:

    return


def _test_runtime(
    runtime: _Runtime,
    _transport_config: None,
    *,
    onsite_bridge: _Bridge | None = None,
) -> _Runtime:

    runtime.onsite_bridge = onsite_bridge

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


def test_transport_skips_onsite_bridge_for_mock_provider_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: credential-free mock provider configuration.

    runtime = _Runtime()

    monkeypatch.setattr(transport_app, "load_config_from_env", _test_config)

    monkeypatch.setattr(
        transport_app,
        "load_transport_config_from_env",
        _test_transport_config,
    )

    monkeypatch.setattr(
        transport_app,
        "TransportRuntime",
        partial(_test_runtime, runtime),
    )

    # When: transport startup reaches its listener lifecycle.

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(transport_app.run_transport())

    # Then: no onsite bridge is requested for normal development startup.

    assert runtime.onsite_bridge is None

    assert runtime.observability_set is False


def test_transport_enables_onsite_bridge_for_real_provider_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the real ASR/LLM/TTS provider combination for onsite spoken dialogue.

    runtime = _Runtime()

    bridge = _Bridge()

    def build_bridge(
        _config: _Config,
        *,
        voice: str,
        ref_audio: str,
        ref_text: str,
    ) -> _Bridge:

        assert voice == ""

        assert ref_audio == ""

        assert ref_text == ""

        return bridge

    monkeypatch.setattr(transport_app, "load_config_from_env", _onsite_config)

    monkeypatch.setattr(
        transport_app,
        "load_transport_config_from_env",
        _test_transport_config,
    )

    monkeypatch.setattr(transport_app, "build_onsite_bridge", build_bridge)

    monkeypatch.setattr(
        transport_app,
        "TransportRuntime",
        partial(_test_runtime, runtime),
    )

    # When: transport startup reaches its listener lifecycle.

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(transport_app.run_transport())

    # Then: the bridge is enabled without any product-mode selector.

    assert runtime.onsite_bridge is bridge

    assert runtime.observability_set is True
