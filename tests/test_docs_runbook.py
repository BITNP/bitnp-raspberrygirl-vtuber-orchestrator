"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]


def test_orchestrator_docs_name_local_canonical_contract_and_explicit_paths() -> None:
    # Given: Orchestrator-owned architecture and runbook documentation.

    """函数契约说明.

    功能: 验证 orchestrator docs name local
    canonical contract and explicit
    paths 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    architecture = (ROOT / "docs/architecture.md").read_text(encoding="utf-8")

    runbook = (ROOT / "docs/runbook.md").read_text(encoding="utf-8")

    docs = architecture + runbook

    # When: canonical protocol and independent command surfaces are inspected.

    required = (
        "schemas/protocol/envelope.schema.json",
        "schemas/protocol/event-data.schema.json",
        "media.stream.command",
        "RTP",
        "page",
        "--frontend-path",
        "--sibling-root",
    )

    # Then: all root-owned contract material points to this checkout.

    assert all(term in docs for term in required)
