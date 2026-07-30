"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from orchestrator.funasr_adapter import FunASRWebSocketAdapter
from orchestrator.llm import CancellationToken
from orchestrator.media_adapters import ASRPartialEvent, ASRStreamRequest
from orchestrator.pipeline_contracts import ASRAudienceEvent
from orchestrator.provider_streaming import ProviderResponseError

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(slots=True)
class _FakeFunASRConnection:
    """类契约说明.

    职责: 保存 _FakeFunASRConnection
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: responses、sent、closed、entere
    d_receive、release_receive。 方法:
    send、recv、close。
    """

    responses: list[str]

    sent: list[str | bytes] = field(default_factory=list)

    closed: bool = False

    entered_receive: threading.Event = field(default_factory=threading.Event)

    release_receive: threading.Event = field(default_factory=threading.Event)

    def send(self, message: str | bytes) -> None:
        """函数契约说明.

        功能: 发送协议消息或媒体数据。
        参数: self 表示当前实例。 message: str |
        bytes。 必填。
        契约: 同步调用。 返回 `None`。
        """

        self.sent.append(message)

    def recv(self, timeout: float | None = None) -> str:
        """函数契约说明.

        功能: 执行 recv 的同步逻辑,并协调 set, pop,
        wait, OSError。
        参数: self 表示当前实例。 timeout: float
        | None。 可省略。
        契约: 同步调用。 返回 `str`。 可能抛出
        OSError。
        """

        _ = timeout

        self.entered_receive.set()

        if not self.responses:
            _ = self.release_receive.wait(timeout=1.0)

            message = "closed"

            raise OSError(message)

        return self.responses.pop(0)

    def close(self) -> None:
        """函数契约说明.

        功能: 执行 close 的同步逻辑,并协调 set。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """

        self.closed = True

        self.release_receive.set()


def _request() -> ASRStreamRequest:
    """函数契约说明.

    功能: 执行 _request 的同步逻辑,并协调
    ASRStreamRequest。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `ASRStreamRequest`。
    """

    return ASRStreamRequest(
        audio=b"canonical-l16-audio",
        filename="utterance.pcm",
        received_at_ms=40,
        segment_id="segment-1",
        seq=2,
    )


def _connect(
    connection: _FakeFunASRConnection,
) -> object:
    """函数契约说明.

    功能: 执行 _connect 的同步逻辑,并产出 _。
    参数: connection:
    _FakeFunASRConnection。 必填。
    契约: 同步调用。 返回 `object`。
    """

    def factory(*args: object, **kwargs: object) -> _FakeFunASRConnection:
        """函数契约说明.

        功能: 执行 factory 的同步逻辑,并产出 _。
        参数: *args: object。 必填。 **kwargs:
        object。 必填。
        契约: 同步调用。 返回
        `_FakeFunASRConnection`。
        """

        _ = (args, kwargs)

        return connection

    return factory


def test_funasr_normalizes_vad_partial_then_one_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a local FunASR FSMN/Paraformer session with VAD and two-pass results.

    """函数契约说明.

    功能: 验证 funasr normalizes vad partial
    then one final 的回归场景和可观察结果。
    参数: monkeypatch: pytest.MonkeyPatch。
    必填。
    契约: 同步调用。 返回 `None`。
    """

    connection = _FakeFunASRConnection(
        [
            '{"mode":"2pass-online","text":"你好"}',
            '{"mode":"2pass-offline","text":"你好世界","is_final":true}',
        ]
    )

    monkeypatch.setattr("orchestrator.funasr_adapter.connect", _connect(connection))

    # When: the native adapter sends an endpointed canonical utterance.

    adapter = FunASRWebSocketAdapter("ws://127.0.0.1:10095", "paraformer")

    events = tuple(adapter.stream(_request()))

    # Then: VAD progress is internal event data and exactly one final is normalized.

    assert events == (
        ASRPartialEvent("你好", 40, "segment-1", 2),
        ASRAudienceEvent("你好世界", 40, "segment-1", 2),
    )

    assert connection.sent == [
        json.dumps(
            {
                "mode": "2pass",
                "chunk_size": [5, 10, 5],
                "wav_name": "utterance.pcm",
                "wav_format": "pcm",
                "audio_fs": 16_000,
                "is_speaking": True,
            },
            separators=(",", ":"),
        ),
        b"canonical-l16-audio",
        json.dumps({"is_speaking": False}, separators=(",", ":")),
    ]

    assert connection.closed is True


def test_funasr_rejects_duplicate_final_before_another_turn_can_be_admitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a broken native stream that emits two completed results.

    """函数契约说明.

    功能: 验证 funasr rejects duplicate
    final before another turn can be
    admitted 的回归场景和可观察结果。
    参数: monkeypatch: pytest.MonkeyPatch。
    必填。
    契约: 同步调用。 返回 `None`。
    """

    connection = _FakeFunASRConnection(
        [
            '{"text":"first","is_final":true}',
            '{"text":"second","is_final":true}',
        ]
    )

    monkeypatch.setattr("orchestrator.funasr_adapter.connect", _connect(connection))

    # When / Then: the second final is rejected at the provider boundary.

    adapter = FunASRWebSocketAdapter("ws://127.0.0.1:10095", "paraformer")

    with pytest.raises(ProviderResponseError, match="duplicate_final"):
        _ = tuple(adapter.stream(_request()))


def test_funasr_cancellation_closes_native_session_without_stale_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a native FunASR read blocked before a provider result arrives.

    """函数契约说明.

    功能: 验证 funasr cancellation closes
    native session without stale final
    的回归场景和可观察结果。
    参数: monkeypatch: pytest.MonkeyPatch。
    必填。
    契约: 同步调用。 返回 `None`。
    """

    connection = _FakeFunASRConnection([])

    cancellation = CancellationToken()

    result: list[ASRPartialEvent | ASRAudienceEvent] = []

    failure: list[BaseException] = []

    monkeypatch.setattr("orchestrator.funasr_adapter.connect", _connect(connection))

    # When: a newer turn cancels the per-stream provider session.

    worker = threading.Thread(
        target=_consume,
        args=(
            FunASRWebSocketAdapter("ws://127.0.0.1:10095", "paraformer").stream(
                _request(), cancellation=cancellation
            ),
            result,
            failure,
        ),
    )

    worker.start()

    assert connection.entered_receive.wait(timeout=1.0)

    assert cancellation.cancel(reason="newer_turn") is True

    worker.join(timeout=1.0)

    # Then: cancellation closes the native state and emits no stale input event.

    assert worker.is_alive() is False

    assert connection.closed is True

    assert result == []

    assert failure == []


def _consume(
    stream: Iterator[ASRPartialEvent | ASRAudienceEvent],
    result: list[ASRPartialEvent | ASRAudienceEvent],
    failure: list[BaseException],
) -> None:
    """函数契约说明.

    功能: 执行 _consume 的同步逻辑,并协调 extend,
    append。
    参数: stream: Iterator[ASRPartialEvent
    | ASRAudienceEvent]。 必填。 result:
    list[ASRPartialEvent |
    ASRAudienceEvent]。 必填。 failure:
    list[BaseException]。 必填。
    契约: 同步调用。 返回 `None`。
    """

    try:
        result.extend(stream)

    except BaseException as error:  # noqa: BLE001 - captures worker failure for test assertion.
        failure.append(error)
