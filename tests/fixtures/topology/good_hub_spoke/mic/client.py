"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations


class FakeOrchestrator:
    """类契约说明.

    职责: 定义 FakeOrchestrator
    的状态、行为和对外协作边界。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """



ORCHESTRATOR_URL = "ws://orchestrator:8000/events"

SCHEMA_SUBJECT = "media.stream.command"
