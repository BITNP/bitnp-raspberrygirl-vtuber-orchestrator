import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import pytest

from orchestrator import transport_app
from orchestrator.ids import SessionId
from orchestrator.interaction_ingress import SessionInteractionIngress
from orchestrator.llm import LLMRequest
from orchestrator.modes import AnswerCandidate, AudienceInput, AudienceSource
from orchestrator.observability import OnsiteObservability
from orchestrator.retrieval import VersionedRetrievalProvider
from orchestrator.scheduler_runtime import SessionRuntime
from orchestrator.sessions import SessionScheduler


@dataclass(frozen=True, slots=True)
class _Config:
    session_id_prefix: str

    asr_provider: str = "mock"

    llm_provider: str = "mock"

    tts_provider: str = "mock"

    ppt_deck_catalog: frozenset[str] = frozenset()


@dataclass
class _Bridge: ...


@dataclass
class _Runtime:
    ingress: SessionInteractionIngress | None = None
    session_runtimes: list[SessionRuntime] = field(default_factory=list)
    factory: Callable[[SessionId], SessionRuntime] | None = None

    closed: bool = False

    onsite_bridge: _Bridge | None = None

    observability_set: bool = False

    async def start(self) -> None:

        raise asyncio.CancelledError

    def set_session_runtime(self, session_runtime: SessionRuntime) -> None:

        self.ingress = session_runtime.interaction_ingress
        self.session_runtimes.append(session_runtime)

    def set_session_runtime_factory(
        self, factory: Callable[[SessionId], SessionRuntime]
    ) -> None:
        self.factory = factory

    def set_observability(self, _observability: OnsiteObservability) -> None:

        self.observability_set = True

    async def close(self) -> None:

        self.closed = True


def _test_config(_env: Mapping[str, str]) -> _Config:

    return _Config("test")


def _onsite_config(_env: Mapping[str, str]) -> _Config:

    return _Config(
        session_id_prefix="test",
        asr_provider="funasr",
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

    def capture_scheduler(
        scheduler: SessionScheduler,
        *,
        retrieval: VersionedRetrievalProvider | None = None,
    ) -> SessionInteractionIngress:

        schedulers.append(scheduler)

        return original_create(scheduler, retrieval=retrieval)

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


def test_transport_enables_onsite_bridge_for_llm_tts_provider_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the Mic-ASR, Orchestrator-LLM/TTS provider combination.

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


class _KnowledgeCompletion:
    async def complete_json(
        self, request: LLMRequest, *, schema_name: str, schema: dict[str, object]
    ) -> str:
        _ = request, schema_name, schema
        return '{"decision":"accept","speech":"产品支持演示","operation":null}'


@dataclass
class _KnowledgeBridge:
    llm: _KnowledgeCompletion = field(default_factory=_KnowledgeCompletion)


def test_production_shares_startup_corpus_and_enables_static_mcp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "knowledge"
    corpus.mkdir()
    source = corpus / "product.md"
    _ = source.write_text("展会周六举行", encoding="utf-8")
    monkeypatch.setenv("ORCHESTRATOR_KNOWLEDGE_DIR", str(corpus))
    monkeypatch.setenv(
        "ORCHESTRATOR_MCP_CONFIG",
        str(Path(__file__).parents[1] / "samples/mcp.example.json"),
    )
    monkeypatch.setenv("CATALOG_MCP_TOKEN", "test-token")
    monkeypatch.setenv("ORCHESTRATOR_STATE_DIR", str(tmp_path / "state"))
    runtime = _Runtime()

    def build_bridge(
        _config: object, *, voice: str, ref_audio: str, ref_text: str
    ) -> _KnowledgeBridge:
        _ = voice, ref_audio, ref_text
        return _KnowledgeBridge()

    monkeypatch.setattr(transport_app, "load_config_from_env", _onsite_config)
    monkeypatch.setattr(
        transport_app, "load_transport_config_from_env", _test_transport_config
    )
    monkeypatch.setattr(transport_app, "build_onsite_bridge", build_bridge)
    monkeypatch.setattr(
        transport_app, "TransportRuntime", partial(_test_runtime, runtime)
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(transport_app.run_transport())
    first = runtime.session_runtimes[0]
    _ = source.write_text("展会周日举行", encoding="utf-8")
    assert runtime.factory is not None
    second = runtime.factory(SessionId("second-session"))
    assert (
        second.interaction_ingress.data.retrieval
        is first.interaction_ingress.data.retrieval
    )
    query = AnswerCandidate(AudienceInput(AudienceSource.COMMENT, "展会", 1))
    assert (
        "周六" in second.interaction_ingress.data.retrieval.retrieve(query).refs[0].text
    )
    assert {"mcp:catalog/lookup", "catalog.read"}.issubset(second.agent_capabilities)
    coordinator = second.async_response_coordinator
    assert coordinator is not None
    assert coordinator.retrieval is first.interaction_ingress.data.retrieval
    assert coordinator.router.specs[0].intent_id == "mcp.catalog_lookup"
    assert second.agent_mcp_allowlist == frozenset({"catalog/lookup"})
