"""模块契约说明.

职责: 提供 orchestrator.operational_journal
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Final

_REDACTED_ID_LENGTH: Final = 16


@dataclass(frozen=True, slots=True)
class OperationalRecord:
    """类契约说明.

    职责: 保存 OperationalRecord
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: stage、trace_id、session_id、tu
    rn_id、segment_id、task_id。
    """

    stage: str

    trace_id: str

    session_id: str

    turn_id: str | None

    segment_id: str | None

    task_id: str | None

    outcome: str


@dataclass(frozen=True, slots=True)
class RedactedOperationalRecord:
    """类契约说明.

    职责: 保存 RedactedOperationalRecord
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: stage、trace_id、session_id、tu
    rn_id、segment_id、task_id。
    """

    stage: str

    trace_id: str

    session_id: str

    turn_id: str | None

    segment_id: str | None

    task_id: str | None

    outcome: str


class OperationalJournal:
    """类契约说明.

    职责: 定义 OperationalJournal
    的状态、行为和对外协作边界。
    契约: 方法: __init__、records、append。
    """

    def __init__(self) -> None:
        """函数契约说明.

        功能: 初始化 OperationalJournal
        的字段并建立实例不变式。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """
        self._records: list[RedactedOperationalRecord] = []

    @property
    def records(self) -> tuple[RedactedOperationalRecord, ...]:
        """函数契约说明.

        功能: 执行 records 的同步逻辑,并协调 tuple。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `tuple[RedactedOper
        ationalRecord, ...]`。
        """
        return tuple(self._records)

    def append(self, record: OperationalRecord) -> None:
        """函数契约说明.

        功能: 执行 append 的同步逻辑,并协调 append,
        RedactedOperationalRecord,
        _redact, _redact_optional。
        参数: self 表示当前实例。 record:
        OperationalRecord。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._records.append(
            RedactedOperationalRecord(
                stage=record.stage,
                trace_id=_redact(record.trace_id),
                session_id=_redact(record.session_id),
                turn_id=_redact_optional(record.turn_id),
                segment_id=_redact_optional(record.segment_id),
                task_id=_redact_optional(record.task_id),
                outcome=record.outcome,
            )
        )


def _redact(value: str) -> str:
    """函数契约说明.

    功能: 执行 _redact 的同步逻辑,并协调 hexdigest,
    sha256, encode。
    参数: value: str。 必填。
    契约: 同步调用。 返回 `str`。
    """
    return sha256(value.encode()).hexdigest()[:_REDACTED_ID_LENGTH]


def _redact_optional(value: str | None) -> str | None:
    """函数契约说明.

    功能: 执行 _redact_optional 的同步逻辑,并协调
    _redact。
    参数: value: str | None。 必填。
    契约: 同步调用。 返回 `str | None`。
    """
    if value is None:
        return None

    return _redact(value)
