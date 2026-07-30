"""模块契约说明.

职责: 提供 orchestrator.transport_app
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

import asyncio
import os

from orchestrator.config import load_config_from_env
from orchestrator.ids import SessionId
from orchestrator.modes import OrchestratorMode, parse_orchestrator_mode
from orchestrator.observability import OnsiteObservability
from orchestrator.onsite_bridge import build_onsite_bridge
from orchestrator.scheduler_runtime import SessionRuntime
from orchestrator.task_registry import SchedulerTaskConfig, TaskKind
from orchestrator.transport_config import load_transport_config_from_env
from orchestrator.transport_runtime import TransportRuntime


async def run_transport() -> None:
    """函数契约说明.

    功能: 运行流程并协调其依赖步骤。
    参数: 无显式业务参数。
    契约: 异步调用。 可能等待 I/O 或协程结果。 返回 `None`。
    """
    mode = parse_orchestrator_mode(
        os.environ.get("ORCHESTRATOR_MODE", OrchestratorMode.VIRTUAL_STREAMER.value)
    )

    config = load_config_from_env(os.environ)

    bridge = None

    observability = None

    if mode is OrchestratorMode.ONSITE_EXPLAINER:
        observability = OnsiteObservability(config)

        bridge = build_onsite_bridge(
            config,
            voice=os.environ.get("ORCHESTRATOR_TTS_VOICE", ""),
            ref_audio=os.environ.get("ORCHESTRATOR_TTS_REF_AUDIO", ""),
            ref_text=os.environ.get("ORCHESTRATOR_TTS_REF_TEXT", ""),
        )

    transport_config = load_transport_config_from_env(os.environ)

    session_runtime = SessionRuntime.create(
        session_id=SessionId(f"{config.session_id_prefix}-control"),
        turn_id_prefix="turn-control",
        task_config=SchedulerTaskConfig(frozenset(TaskKind), 1),
        mode=mode,
    )

    runtime = TransportRuntime(transport_config, onsite_bridge=bridge)

    try:
        runtime.set_session_runtime(session_runtime)

        if observability is not None:
            runtime.set_observability(observability)

        await runtime.start()

        await asyncio.Future[None]()

    finally:
        await runtime.close()


def main() -> None:
    """函数契约说明.

    功能: 执行命令行或服务入口流程并返回进程级结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """
    try:
        asyncio.run(run_transport())

    except KeyboardInterrupt:
        return
