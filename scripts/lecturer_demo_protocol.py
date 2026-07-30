"""模块契约说明.

职责: 提供命令行脚本的参数处理、验证或运维流程。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

from typing import Literal, TypeAlias

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)

JsonObject: TypeAlias = dict[str, JsonValue]

ModuleName: TypeAlias = Literal["orchestrator", "sound", "frontend"]


def event(
    *,
    event_type: str,
    source: ModuleName,
    seq: int,
    data: JsonObject,
    turn_id: str | None = None,
    segment_id: str | None = None,
) -> JsonObject:
    """函数契约说明.

    功能: 执行 event 的同步逻辑,并产出 payload。
    参数: event_type: str。 必填。 source:
    ModuleName。 必填。 seq: int。 必填。 data:
    JsonObject。 必填。 turn_id: str | None。
    可省略。 segment_id: str | None。 可省略。
    契约: 同步调用。 返回 `JsonObject`。
    """
    payload: JsonObject = {
        "schema_version": "1.0.0",
        "event_type": event_type,
        "event_id": f"evt-demo-{seq:04d}",
        "source": source,
        "time": "2026-07-14T00:00:00Z",
        "trace_id": "trace-lecturer-demo",
        "session_id": "session-lecturer-demo",
        "seq": seq,
        "data": data,
    }

    if turn_id is not None:
        payload["turn_id"] = turn_id

    if segment_id is not None:
        payload["segment_id"] = segment_id

    return payload


def is_peer_edge(edge: JsonObject) -> bool:
    """函数契约说明.

    功能: 执行 is_peer_edge 的同步逻辑,并维持签名契约。
    参数: edge: JsonObject。 必填。
    契约: 同步调用。 返回 `bool`。
    """
    return edge["source"] != "orchestrator" and edge["target"] != "orchestrator"
