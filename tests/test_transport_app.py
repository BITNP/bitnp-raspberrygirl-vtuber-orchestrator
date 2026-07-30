"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

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
    """类契约说明.

    职责: 保存 _Config 不可变数据结构,用类型标注表达字段契约。
    契约: 字段: session_id_prefix。
    """

    session_id_prefix: str

    asr_provider: str = "mock"

    llm_provider: str = "mock"

    tts_provider: str = "mock"


@dataclass
class _Bridge:
    """类契约说明.

    职责: 保存 _Bridge 不可变数据结构,用类型标注表达字段契约。
    契约: 无字段。
    """


@dataclass
class _Runtime:
    """类契约说明.

    职责: 保存 _Runtime 不可变数据结构,用类型标注表达字段契约。
    契约: 字段: ingress、closed。 方法:
    start、set_session_runtime、close。
    """

    ingress: SessionInteractionIngress | None = None

    closed: bool = False

    onsite_bridge: _Bridge | None = None

    observability_set: bool = False

    async def start(self) -> None:
        """函数契约说明.

        功能: 执行 start 的异步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 异步调用。 返回 `None`。
        """

        raise asyncio.CancelledError

    def set_session_runtime(self, session_runtime: SessionRuntime) -> None:
        """函数契约说明.

        功能: 执行 set_session_runtime
        的同步逻辑,并产出 ingress。
        参数: self 表示当前实例。
        session_runtime: SessionRuntime。
        必填。
        契约: 同步调用。 返回 `None`。
        """

        self.ingress = session_runtime.interaction_ingress

    def set_observability(self, _observability: OnsiteObservability) -> None:
        """函数契约说明.

        功能: 记录测试运行时是否收到 observability。
        参数: _observability: OnsiteObservability。 必填。
        契约: 同步调用。 返回 `None`。
        """

        self.observability_set = True

    async def close(self) -> None:
        """函数契约说明.

        功能: 执行 close 的异步逻辑,并产出 closed。
        参数: self 表示当前实例。
        契约: 异步调用。 返回 `None`。
        """

        self.closed = True


def _test_config(_env: Mapping[str, str]) -> _Config:
    """函数契约说明.

    功能: 执行 _test_config 的同步逻辑,并协调
    _Config。
    参数: _env: Mapping[str, str]。 必填。
    契约: 同步调用。 返回 `_Config`。
    """

    return _Config("test")


def _onsite_config(_env: Mapping[str, str]) -> _Config:
    """函数契约说明.

    功能: 构造启用现场语音桥接的测试配置。
    参数: _env: Mapping[str, str]。 必填。
    契约: 同步调用。 返回 `_Config`。
    """

    return _Config(
        session_id_prefix="test",
        asr_provider="openai_compatible",
        llm_provider="openai_compatible",
        tts_provider="vllm_omni",
    )


def _test_transport_config(_env: Mapping[str, str]) -> None:
    """函数契约说明.

    功能: 执行 _test_transport_config
    的同步逻辑,并维持签名契约。
    参数: _env: Mapping[str, str]。 必填。
    契约: 同步调用。 返回 `None`。
    """

    return


def _test_runtime(
    runtime: _Runtime,
    _transport_config: None,
    *,
    onsite_bridge: _Bridge | None = None,
) -> _Runtime:
    """函数契约说明.

    功能: 执行 _test_runtime 的同步逻辑,并维持签名契约。
    参数: runtime: _Runtime。 必填。
    _transport_config: None。 必填。
    onsite_bridge: _Bridge | None。 可省略。
    契约: 同步调用。 返回 `_Runtime`。
    """

    runtime.onsite_bridge = onsite_bridge

    return runtime


def test_transport_composes_one_scheduler_control_ingress_before_listening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: deterministic configuration and listener stop at the transport entrypoint.

    """函数契约说明.

    功能: 验证 transport composes one
    scheduler control ingress before
    listening 的回归场景和可观察结果。
    参数: monkeypatch: pytest.MonkeyPatch。
    必填。
    契约: 同步调用。 返回 `None`。
    """

    schedulers: list[SessionScheduler] = []

    original_create = SessionInteractionIngress.create

    runtime = _Runtime()

    def capture_scheduler(scheduler: SessionScheduler) -> SessionInteractionIngress:
        """函数契约说明.

        功能: 执行 capture_scheduler
        的同步逻辑,并协调 append,
        original_create。
        参数: scheduler: SessionScheduler。
        必填。
        契约: 同步调用。 返回
        `SessionInteractionIngress`。
        """

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
        """函数契约说明.

        功能: 返回测试现场语音桥接对象。
        参数: _config: _Config。 必填。 voice: str。
        必填。 ref_audio: str。 必填。 ref_text: str。 必填。
        契约: 同步调用。 返回 `_Bridge`。
        """

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
