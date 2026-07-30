"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

import pytest

from .e2e_replay_harness import (
    ReplayError,
    event_types,
    is_peer_edge,
)
from .e2e_replay_scenarios import (
    all_adaptive_scenarios,
    negative_peer_harness,
    negative_stale_harness,
)


def test_replay_harness_runs_adaptive_surfaces_without_peer_communication() -> None:
    # Given: deterministic in-process scenarios for every Task 17 input surface.

    """函数契约说明.

    功能: 验证 replay harness runs adaptive surfaces
    without peer communication
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    scenarios = all_adaptive_scenarios()

    # When / Then: every replay emits media, sound, and frontend controls with IDs.

    for summary in scenarios:
        assert summary.turn_ids != ()

        assert summary.segment_ids != ()

        assert event_types(summary, "media.stream.command") == (
            "media.stream.command",
        ) * len(
            summary.turn_ids,
        )

        assert "media.stream.state" in event_types(summary, "media.stream.state")

        assert "vtuber.caption.command" in event_types(
            summary,
            "vtuber.caption.command",
        )

        assert "vtuber.action.command" in event_types(summary, "vtuber.action.command")

        assert "vtuber.expression.command" in event_types(
            summary,
            "vtuber.expression.command",
        )

        assert all(event.latency_ms == 0 for event in summary.timeline)

        assert all(not is_peer_edge(edge) for edge in summary.edges)


def test_replay_harness_fails_on_injected_peer_edge() -> None:
    # Given: a valid replay with a synthetic ASR-to-Sound peer edge injected.

    """函数契约说明.

    功能: 验证 replay harness fails on
    injected peer edge 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    harness = negative_peer_harness()

    # When / Then: the harness fails the topology ledger check.

    with pytest.raises(ReplayError, match="peer communication edge"):
        harness.assert_no_peer_edges()


def test_replay_harness_fails_on_stale_segment_acceptance_requirement() -> None:
    # Given: a first turn cancelled by a second ASR input.

    """函数契约说明.

    功能: 验证 replay harness fails on stale
    segment acceptance requirement
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    harness, first = negative_stale_harness()

    # When / Then: trying to finish the stale segment fails the replay.

    with pytest.raises(ReplayError, match="fresh synthesis result"):
        _ = harness.require_synthesis_cues(first)
