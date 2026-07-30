"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

from orchestrator.operational_journal import OperationalJournal, OperationalRecord


def test_journal_redacts_correlation_and_sensitive_sentinels() -> None:
    # Given: a complete lifecycle record with identifiers adjacent to sensitive values.

    """函数契约说明.

    功能: 验证 journal redacts correlation
    and sensitive sentinels 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    journal = OperationalJournal()

    record = OperationalRecord(
        stage="provider_timeout",
        trace_id="raw-audio-sentinel",
        session_id="voice-template-sentinel",
        turn_id="prompt-sentinel",
        segment_id="tool-output-sentinel",
        task_id="credential-sentinel",
        outcome="timeout",
    )

    # When: the operational artifact records the transition.

    journal.append(record)

    # Then: every joinable identifier is redacted and no sensitive sentinel survives.

    artifact = journal.records[0]

    assert artifact.stage == "provider_timeout"

    assert artifact.outcome == "timeout"

    assert artifact.trace_id != record.trace_id

    assert artifact.session_id != record.session_id

    assert artifact.turn_id != record.turn_id

    assert artifact.segment_id != record.segment_id

    assert artifact.task_id != record.task_id

    assert all(
        sentinel not in repr(artifact)
        for sentinel in (
            "raw-audio-sentinel",
            "voice-template-sentinel",
            "prompt-sentinel",
            "tool-output-sentinel",
            "credential-sentinel",
        )
    )


def test_journal_preserves_redacted_task_correlation_across_lifecycle() -> None:
    # Given: accepted and rejected lifecycle transitions for one task.

    """函数契约说明.

    功能: 验证 journal preserves redacted
    task correlation across lifecycle
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    journal = OperationalJournal()

    scheduled = OperationalRecord(
        stage="task_scheduled",
        trace_id="trace-001",
        session_id="session-001",
        turn_id="turn-001",
        segment_id="segment-001",
        task_id="task-001",
        outcome="accepted",
    )

    rejected = OperationalRecord(
        stage="task_result",
        trace_id="trace-001",
        session_id="session-001",
        turn_id="turn-001",
        segment_id="segment-001",
        task_id="task-001",
        outcome="rejected",
    )

    # When: both bounded lifecycle facts are recorded.

    journal.append(scheduled)

    journal.append(rejected)

    # Then: operators can join the safe artifacts without the original identifiers.

    first, second = journal.records

    assert first.trace_id == second.trace_id

    assert first.session_id == second.session_id

    assert first.turn_id == second.turn_id

    assert first.segment_id == second.segment_id

    assert first.task_id == second.task_id
