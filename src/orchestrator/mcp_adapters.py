"""模块契约说明.

职责: 提供 orchestrator.mcp_adapters
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum, unique
from typing import Protocol

from orchestrator.interactions import (
    CommandId,
    McpCapability,
    McpDispatchAccepted,
    McpDispatchProposal,
    PresentationCommand,
    PresentationResult,
    SessionInteractionReducer,
)
from orchestrator.provider_streaming import ProviderCancellationHandle


@unique
class McpResultKind(StrEnum):
    """类契约说明.

    职责: 定义 McpResultKind 的状态、行为和对外协作边界。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """

    SUCCEEDED = "succeeded"

    AMBIGUOUS = "ambiguous"

    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class McpAdapterResult:
    """类契约说明.

    职责: 保存 McpAdapterResult
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: kind。 方法:
    succeeded、ambiguous。
    """

    kind: McpResultKind

    @classmethod
    def succeeded(cls) -> "McpAdapterResult":
        """函数契约说明.

        功能: 执行 succeeded 的同步逻辑,并协调 cls。
        参数: cls 表示当前类。
        契约: 同步调用。 返回
        `'McpAdapterResult'`。
        """
        return cls(McpResultKind.SUCCEEDED)

    @classmethod
    def ambiguous(cls) -> "McpAdapterResult":
        """函数契约说明.

        功能: 执行 ambiguous 的同步逻辑,并协调 cls。
        参数: cls 表示当前类。
        契约: 同步调用。 返回
        `'McpAdapterResult'`。
        """
        return cls(McpResultKind.AMBIGUOUS)


@dataclass(frozen=True, slots=True)
class McpIntent:
    """类契约说明.

    职责: 保存 McpIntent
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段:
    proposal、command、deadline_ms。
    """

    proposal: McpDispatchProposal

    command: PresentationCommand

    deadline_ms: int


class McpAdapter(Protocol):
    """类契约说明.

    职责: 声明 McpAdapter 协议接口,约束实现方必须提供的行为。
    契约: 方法: execute、reconcile。
    """

    def execute(self, intent: McpIntent) -> McpAdapterResult:
        """函数契约说明.

        功能: 执行 execute 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 intent:
        McpIntent。 必填。
        契约: 同步调用。 返回 `McpAdapterResult`。
        """
        ...

    def reconcile(self, intent: McpIntent) -> McpAdapterResult:
        """函数契约说明.

        功能: 执行 reconcile 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 intent:
        McpIntent。 必填。
        契约: 同步调用。 返回 `McpAdapterResult`。
        """
        ...


@dataclass(frozen=True, slots=True)
class LocalDeckAdapter:
    """类契约说明.

    职责: 保存 LocalDeckAdapter
    不可变数据结构,用类型标注表达字段契约。
    契约: 方法:
    execute、reconcile、execute_async。
    """

    def execute(self, intent: McpIntent) -> McpAdapterResult:
        """函数契约说明.

        功能: 执行 execute 的同步逻辑,并协调
        succeeded。
        参数: self 表示当前实例。 intent:
        McpIntent。 必填。
        契约: 同步调用。 返回 `McpAdapterResult`。
        """
        _ = intent

        return McpAdapterResult.succeeded()

    def reconcile(self, intent: McpIntent) -> McpAdapterResult:
        """函数契约说明.

        功能: 执行 reconcile 的同步逻辑,并协调
        execute。
        参数: self 表示当前实例。 intent:
        McpIntent。 必填。
        契约: 同步调用。 返回 `McpAdapterResult`。
        """
        return self.execute(intent)

    async def execute_async(
        self, intent: McpIntent, cancellation: ProviderCancellationHandle
    ) -> McpAdapterResult:
        """函数契约说明.

        功能: 执行 execute_async 的异步逻辑,并协调
        succeeded, sleep,
        McpAdapterResult。
        参数: self 表示当前实例。 intent:
        McpIntent。 必填。 cancellation:
        ProviderCancellationHandle。 必填。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `McpAdapterResult`。
        """
        _ = intent

        await asyncio.sleep(0)

        if cancellation.cancelled:
            return McpAdapterResult(McpResultKind.FAILED)

        return McpAdapterResult.succeeded()


@unique
class McpJournalKind(StrEnum):
    """类契约说明.

    职责: 定义 McpJournalKind 的状态、行为和对外协作边界。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """

    INTENT_DISPATCHED = "intent_dispatched"

    RESULT_SUCCEEDED = "result_succeeded"

    RESULT_AMBIGUOUS = "result_ambiguous"

    RESULT_FAILED = "result_failed"

    RECONCILED = "reconciled"

    TIMED_OUT = "timed_out"

    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class McpJournalEntry:
    """类契约说明.

    职责: 保存 McpJournalEntry
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: kind、command_id、capability。
    """

    kind: McpJournalKind

    command_id: CommandId

    capability: McpCapability


@dataclass(frozen=True, slots=True)
class McpDispatchOutcome:
    """类契约说明.

    职责: 保存 McpDispatchOutcome
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: accepted、completion。
    """

    accepted: bool

    completion: PresentationResult | None = None


@dataclass(slots=True)
class ScopedMcpAdapterDispatcher:
    """类契约说明.

    职责: 保存 ScopedMcpAdapterDispatcher
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: reducer、adapters、_journal、_p
    ending、_issued、_active。 方法: journal、
    dispatch、_admit、dispatch_async、cance
    l、reconcile。
    """

    reducer: SessionInteractionReducer

    adapters: Mapping[McpCapability, McpAdapter]

    _journal: list[McpJournalEntry] = field(default_factory=list)

    _pending: dict[CommandId, McpIntent] = field(default_factory=dict)

    _issued: set[CommandId] = field(default_factory=set)

    _active: dict[CommandId, ProviderCancellationHandle] = field(default_factory=dict)

    @property
    def journal(self) -> tuple[McpJournalEntry, ...]:
        """函数契约说明.

        功能: 执行 journal 的同步逻辑,并协调 tuple。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `tuple[McpJournalEntry, ...]`。
        """
        return tuple(self._journal)

    def dispatch(self, intent: McpIntent, *, now_ms: int) -> McpDispatchOutcome:
        """函数契约说明.

        功能: 执行 dispatch 的同步逻辑,并协调
        _admit, _consume,
        McpDispatchOutcome, execute。
        参数: self 表示当前实例。 intent:
        McpIntent。 必填。 now_ms: int。 必填。
        契约: 同步调用。 返回
        `McpDispatchOutcome`。
        """
        adapter = self._admit(intent, now_ms=now_ms)

        if adapter is None:
            return McpDispatchOutcome(accepted=False)

        return self._consume(adapter.execute(intent), intent, reconciled=False)

    def _admit(self, intent: McpIntent, *, now_ms: int) -> McpAdapter | None:
        """函数契约说明.

        功能: 执行 _admit 的同步逻辑,并协调
        reduce_mcp, get, add, _record。
        参数: self 表示当前实例。 intent:
        McpIntent。 必填。 now_ms: int。 必填。
        契约: 同步调用。 返回 `McpAdapter |
        None`。
        """
        if not self.reducer.presentation_intent_is_pending(intent.command):
            self._record(McpJournalKind.REJECTED, intent)

            return None

        if intent.command.command_id in self._issued:
            self._record(McpJournalKind.REJECTED, intent)

            return None

        admission = self.reducer.reduce_mcp(intent.proposal)

        if (
            not isinstance(admission, McpDispatchAccepted)
            or intent.proposal.command_id != intent.command.command_id
        ):
            self._record(McpJournalKind.REJECTED, intent)

            return None

        adapter = self.adapters.get(intent.proposal.capability)

        if adapter is None:
            self._record(McpJournalKind.REJECTED, intent)

            return None

        if now_ms > intent.deadline_ms:
            self._record(McpJournalKind.TIMED_OUT, intent)

            return None

        self._issued.add(intent.command.command_id)

        self._record(McpJournalKind.INTENT_DISPATCHED, intent)

        return adapter

    async def dispatch_async(
        self, intent: McpIntent, *, now_ms: int
    ) -> McpDispatchOutcome:
        """函数契约说明.

        功能: 执行 dispatch_async 的异步逻辑,并协调
        _admit,
        ProviderCancellationHandle,
        isinstance, McpDispatchOutcome。
        参数: self 表示当前实例。 intent:
        McpIntent。 必填。 now_ms: int。 必填。
        契约: 异步调用。 可能等待 I/O 或协程结果。 返回
        `McpDispatchOutcome`。
        """
        adapter = self._admit(intent, now_ms=now_ms)

        if not isinstance(adapter, LocalDeckAdapter):
            return McpDispatchOutcome(accepted=False)

        cancellation = ProviderCancellationHandle()

        self._active[intent.command.command_id] = cancellation

        try:
            result = await adapter.execute_async(intent, cancellation)

            if cancellation.cancelled:
                self._record(McpJournalKind.REJECTED, intent)

                return McpDispatchOutcome(accepted=False)

            return self._consume(result, intent, reconciled=False)

        finally:
            _ = self._active.pop(intent.command.command_id, None)

    def cancel(self, command_id: CommandId) -> bool:
        """函数契约说明.

        功能: 执行 cancel 的同步逻辑,并协调 get,
        cancel。
        参数: self 表示当前实例。 command_id:
        CommandId。 必填。
        契约: 同步调用。 返回 `bool`。
        """
        cancellation = self._active.get(command_id)

        if cancellation is None:
            return False

        return cancellation.cancel(reason="task_cancelled")

    def reconcile(self, command_id: CommandId, *, now_ms: int) -> McpDispatchOutcome:
        """函数契约说明.

        功能: 执行 reconcile 的同步逻辑,并协调 get,
        _consume, McpDispatchOutcome,
        _record。
        参数: self 表示当前实例。 command_id:
        CommandId。 必填。 now_ms: int。 必填。
        契约: 同步调用。 返回
        `McpDispatchOutcome`。
        """
        intent = self._pending.get(command_id)

        if intent is None:
            return McpDispatchOutcome(accepted=False)

        if now_ms > intent.deadline_ms:
            del self._pending[command_id]

            self._record(McpJournalKind.TIMED_OUT, intent)

            return McpDispatchOutcome(accepted=False)

        adapter = self.adapters[intent.proposal.capability]

        return self._consume(adapter.reconcile(intent), intent, reconciled=True)

    def _consume(
        self, result: McpAdapterResult, intent: McpIntent, *, reconciled: bool
    ) -> McpDispatchOutcome:
        """函数契约说明.

        功能: 执行 _consume 的同步逻辑,并协调 pop,
        _record, McpDispatchOutcome,
        PresentationResult。
        参数: self 表示当前实例。 result:
        McpAdapterResult。 必填。 intent:
        McpIntent。 必填。 reconciled: bool。
        必填。
        契约: 同步调用。 返回
        `McpDispatchOutcome`。
        """
        match result.kind:
            case McpResultKind.SUCCEEDED:
                _ = self._pending.pop(intent.command.command_id, None)

                self._record(
                    (
                        McpJournalKind.RECONCILED
                        if reconciled
                        else McpJournalKind.RESULT_SUCCEEDED
                    ),
                    intent,
                )

                return McpDispatchOutcome(
                    accepted=True,
                    completion=PresentationResult(
                        intent.command.command_id, succeeded=True
                    ),
                )

            case McpResultKind.AMBIGUOUS:
                self._pending[intent.command.command_id] = intent

                self._record(McpJournalKind.RESULT_AMBIGUOUS, intent)

                return McpDispatchOutcome(accepted=False)

            case McpResultKind.FAILED:
                _ = self._pending.pop(intent.command.command_id, None)

                self._record(McpJournalKind.RESULT_FAILED, intent)

                return McpDispatchOutcome(accepted=False)

    def _record(self, kind: McpJournalKind, intent: McpIntent) -> None:
        """函数契约说明.

        功能: 执行 _record 的同步逻辑,并协调 append,
        McpJournalEntry。
        参数: self 表示当前实例。 kind:
        McpJournalKind。 必填。 intent:
        McpIntent。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._journal.append(
            McpJournalEntry(kind, intent.command.command_id, intent.proposal.capability)
        )
