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
from orchestrator.scheduler_runtime import SessionRuntime
from orchestrator.sessions import SessionScheduler


@dataclass(frozen=True, slots=True)
class _Config:
    """类契约说明.

    职责: 保存 _Config 不可变数据结构,用类型标注表达字段契约。
    契约: 字段: session_id_prefix。
    """

    session_id_prefix: str


@dataclass
class _Runtime:
    """类契约说明.

    职责: 保存 _Runtime 不可变数据结构,用类型标注表达字段契约。
    契约: 字段: ingress、closed。 方法:
    start、set_session_runtime、close。
    """

    ingress: SessionInteractionIngress | None = None

    closed: bool = False

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


def _test_transport_config(_env: Mapping[str, str]) -> None:
    """函数契约说明.

    功能: 执行 _test_transport_config
    的同步逻辑,并维持签名契约。
    参数: _env: Mapping[str, str]。 必填。
    契约: 同步调用。 返回 `None`。
    """

    return


def _test_runtime(runtime: _Runtime, *_args: str, **_kwargs: str) -> _Runtime:
    """函数契约说明.

    功能: 执行 _test_runtime 的同步逻辑,并维持签名契约。
    参数: runtime: _Runtime。 必填。 *_args:
    str。 必填。 **_kwargs: str。 必填。
    契约: 同步调用。 返回 `_Runtime`。
    """

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
