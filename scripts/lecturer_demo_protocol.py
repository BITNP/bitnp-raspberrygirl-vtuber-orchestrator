from __future__ import annotations

from typing import Literal, TypeAlias

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
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
    return edge["source"] != "orchestrator" and edge["target"] != "orchestrator"
