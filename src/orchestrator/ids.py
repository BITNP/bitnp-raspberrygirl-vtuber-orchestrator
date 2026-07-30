"""模块契约说明.

职责: 提供 orchestrator.ids
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from typing import NewType

ConnectionId = NewType("ConnectionId", str)

SessionId = NewType("SessionId", str)

TraceId = NewType("TraceId", str)

TurnId = NewType("TurnId", str)

SegmentId = NewType("SegmentId", str)
