"""模块契约说明.

职责: 提供 orchestrator.interactions
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import NewType, final

from orchestrator.sessions import (
    EventCorrelation,
    SchedulerEvent,
    SessionScheduler,
    StartTurn,
    TransitionAccepted,
)
from orchestrator.state_snapshots import TaskStateSnapshot
from orchestrator.task_reducer import TaskReductionResult, TaskResult, TaskResultReducer

CommandId = NewType("CommandId", str)


@unique
class InteractionRejectionReason(StrEnum):
    """类契约说明.

    职责: 定义 InteractionRejectionReason
    的状态、行为和对外协作边界。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """

    UNSUPPORTED_ACTION = "unsupported_action"

    DUPLICATE = "duplicate"

    INVALID_PRESENTATION_STATE = "invalid_presentation_state"

    FRONTEND_REJECTED = "frontend_rejected"


@dataclass(frozen=True, slots=True)
class InteractionAccepted:
    """类契约说明.

    职责: 保存 InteractionAccepted
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: command_id、turn_id。
    """

    command_id: CommandId | None = None

    turn_id: str | None = None


@dataclass(frozen=True, slots=True)
class InteractionRejection:
    """类契约说明.

    职责: 保存 InteractionRejection
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: reason。
    """

    reason: InteractionRejectionReason


@dataclass(frozen=True, slots=True)
class CommentProposal:
    """类契约说明.

    职责: 保存 CommentProposal
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: text、correlation。
    """

    text: str

    correlation: EventCorrelation


@dataclass(frozen=True, slots=True)
class ActionProposal:
    """类契约说明.

    职责: 保存 ActionProposal
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: action、command_id。
    """

    action: str

    command_id: CommandId


@final
class ActionCapabilityRegistry:
    """类契约说明.

    职责: 定义 ActionCapabilityRegistry
    的状态、行为和对外协作边界。
    契约: 方法: __init__、permits。
    """

    def __init__(self, actions: frozenset[str]) -> None:
        """函数契约说明.

        功能: 初始化 ActionCapabilityRegistry
        的字段并建立实例不变式。
        参数: self 表示当前实例。 actions:
        frozenset[str]。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._actions = actions

    def permits(self, action: str) -> bool:
        """函数契约说明.

        功能: 执行 permits 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 action: str。
        必填。
        契约: 同步调用。 返回 `bool`。
        """
        return action in self._actions


@unique
class PresentationCommandKind(StrEnum):
    """类契约说明.

    职责: 定义 PresentationCommandKind
    的状态、行为和对外协作边界。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """

    LOAD = "load"

    PLAY = "play"

    NAVIGATE = "navigate"


@dataclass(frozen=True, slots=True)
class PresentationCommand:
    """类契约说明.

    职责: 保存 PresentationCommand
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: kind、deck_id、page、command_id
    、deck_version。
    """

    kind: PresentationCommandKind

    deck_id: str

    page: int

    command_id: CommandId

    deck_version: str = "v1"


@dataclass(frozen=True, slots=True)
class PresentationResult:
    """类契约说明.

    职责: 保存 PresentationResult
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: command_id、succeeded。
    """

    command_id: CommandId

    succeeded: bool


@unique
class McpCapability(StrEnum):
    """类契约说明.

    职责: 定义 McpCapability 的状态、行为和对外协作边界。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """

    KNOWLEDGE_LOOKUP = "knowledge_lookup"

    PRESENTATION_DECK = "presentation_deck"


@dataclass(frozen=True, slots=True)
class McpDispatchProposal:
    """类契约说明.

    职责: 保存 McpDispatchProposal
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段:
    capability、command_id、cancelled。
    """

    capability: McpCapability

    command_id: CommandId

    cancelled: bool


@unique
class McpDispatchRejection(StrEnum):
    """类契约说明.

    职责: 定义 McpDispatchRejection
    的状态、行为和对外协作边界。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """

    UNSUPPORTED_CAPABILITY = "unsupported_capability"

    CANCELLED = "cancelled"

    DUPLICATE = "duplicate"


@dataclass(frozen=True, slots=True)
class McpDispatchAccepted:
    """类契约说明.

    职责: 保存 McpDispatchAccepted
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: command_id、capability。
    """

    command_id: CommandId

    capability: McpCapability


@dataclass(frozen=True, slots=True)
class McpDispatchRejected:
    """类契约说明.

    职责: 保存 McpDispatchRejected
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: reason。
    """

    reason: McpDispatchRejection


@final
class SessionInteractionReducer:
    """类契约说明.

    职责: 定义 SessionInteractionReducer
    的状态、行为和对外协作边界。
    契约: 方法: __init__、presentation_state、
    reduce_comment、reduce_action、reduce_
    presentation、reduce_presentation_res
    ult。
    """

    def __init__(
        self,
        *,
        scheduler: SessionScheduler,
        actions: ActionCapabilityRegistry,
        mcp_capabilities: frozenset[McpCapability],
    ) -> None:
        """函数契约说明.

        功能: 初始化
        SessionInteractionReducer
        的字段并建立实例不变式。
        参数: self 表示当前实例。 scheduler:
        SessionScheduler。 必填。 actions:
        ActionCapabilityRegistry。 必填。
        mcp_capabilities:
        frozenset[McpCapability]。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._scheduler = scheduler

        self._actions = actions

        self._mcp_capabilities = mcp_capabilities

        self._command_ids: set[CommandId] = set()

        self._mcp_command_ids: set[CommandId] = set()

        self._pending_presentations: dict[CommandId, PresentationCommand] = {}

        self._presentation_state: tuple[str, str, int] | None = None

    @property
    def presentation_state(self) -> tuple[str, str, int] | None:
        """函数契约说明.

        功能: 执行 presentation_state
        的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `tuple[str, str,
        int] | None`。
        """
        return self._presentation_state

    def reduce_comment(
        self,
        proposal: CommentProposal,
    ) -> InteractionAccepted | InteractionRejection:
        """函数契约说明.

        功能: 执行 reduce_comment 的同步逻辑,并协调
        apply, StartTurn,
        InteractionAccepted,
        InteractionRejection。
        参数: self 表示当前实例。 proposal:
        CommentProposal。 必填。
        契约: 同步调用。 返回
        `InteractionAccepted |
        InteractionRejection`。
        """
        transition = self._scheduler.apply(
            StartTurn(
                expected_revision=self._scheduler.snapshot.revision,
                event=SchedulerEvent("audience.input", proposal.correlation),
            )
        )

        match transition:
            case TransitionAccepted(accepted_event=accepted):
                return InteractionAccepted(turn_id=str(accepted.turn_id))

            case _:
                return InteractionRejection(InteractionRejectionReason.DUPLICATE)

    def reduce_action(
        self,
        proposal: ActionProposal,
    ) -> InteractionAccepted | InteractionRejection:
        """函数契约说明.

        功能: 执行 reduce_action 的同步逻辑,并协调
        add, InteractionAccepted,
        InteractionRejection, permits。
        参数: self 表示当前实例。 proposal:
        ActionProposal。 必填。
        契约: 同步调用。 返回
        `InteractionAccepted |
        InteractionRejection`。
        """
        if proposal.command_id in self._command_ids:
            return InteractionRejection(InteractionRejectionReason.DUPLICATE)

        if not self._actions.permits(proposal.action):
            return InteractionRejection(InteractionRejectionReason.UNSUPPORTED_ACTION)

        self._command_ids.add(proposal.command_id)

        return InteractionAccepted(command_id=proposal.command_id)

    def reduce_presentation(
        self,
        proposal: PresentationCommand,
    ) -> InteractionAccepted | InteractionRejection:
        """函数契约说明.

        功能: 执行 reduce_presentation
        的同步逻辑,并协调 add,
        InteractionAccepted,
        InteractionRejection, strip。
        参数: self 表示当前实例。 proposal:
        PresentationCommand。 必填。
        契约: 同步调用。 返回
        `InteractionAccepted |
        InteractionRejection`。
        """
        if proposal.command_id in self._command_ids:
            return InteractionRejection(InteractionRejectionReason.DUPLICATE)

        if (
            proposal.page < 1
            or proposal.deck_id.strip() == ""
            or proposal.deck_version.strip() == ""
        ):
            return InteractionRejection(
                InteractionRejectionReason.INVALID_PRESENTATION_STATE
            )

        state = self._presentation_state

        if proposal.kind is not PresentationCommandKind.LOAD and (
            state is None or (proposal.deck_id, proposal.deck_version) != state[:2]
        ):
            return InteractionRejection(
                InteractionRejectionReason.INVALID_PRESENTATION_STATE
            )

        self._command_ids.add(proposal.command_id)

        self._pending_presentations[proposal.command_id] = proposal

        return InteractionAccepted(command_id=proposal.command_id)

    def reduce_presentation_result(
        self,
        result: PresentationResult,
    ) -> InteractionAccepted | InteractionRejection:
        """函数契约说明.

        功能: 执行
        reduce_presentation_result
        的同步逻辑,并协调 pop,
        InteractionAccepted,
        InteractionRejection。
        参数: self 表示当前实例。 result:
        PresentationResult。 必填。
        契约: 同步调用。 返回
        `InteractionAccepted |
        InteractionRejection`。
        """
        proposal = self._pending_presentations.pop(result.command_id, None)

        if proposal is None:
            return InteractionRejection(InteractionRejectionReason.DUPLICATE)

        if not result.succeeded:
            return InteractionRejection(InteractionRejectionReason.FRONTEND_REJECTED)

        self._presentation_state = (
            proposal.deck_id,
            proposal.deck_version,
            proposal.page,
        )

        return InteractionAccepted(command_id=result.command_id)

    def reduce_mcp(
        self,
        proposal: McpDispatchProposal,
    ) -> McpDispatchAccepted | McpDispatchRejected:
        """函数契约说明.

        功能: 执行 reduce_mcp 的同步逻辑,并协调 add,
        McpDispatchAccepted,
        McpDispatchRejected。
        参数: self 表示当前实例。 proposal:
        McpDispatchProposal。 必填。
        契约: 同步调用。 返回
        `McpDispatchAccepted |
        McpDispatchRejected`。
        """
        if proposal.command_id in self._mcp_command_ids:
            return McpDispatchRejected(McpDispatchRejection.DUPLICATE)

        if proposal.cancelled:
            return McpDispatchRejected(McpDispatchRejection.CANCELLED)

        if (
            proposal.capability is not McpCapability.PRESENTATION_DECK
            or proposal.capability not in self._mcp_capabilities
        ):
            return McpDispatchRejected(McpDispatchRejection.UNSUPPORTED_CAPABILITY)

        self._mcp_command_ids.add(proposal.command_id)

        return McpDispatchAccepted(proposal.command_id, proposal.capability)

    def presentation_intent_is_pending(self, proposal: PresentationCommand) -> bool:
        """函数契约说明.

        功能: 执行
        presentation_intent_is_pending
        的同步逻辑,并协调 get。
        参数: self 表示当前实例。 proposal:
        PresentationCommand。 必填。
        契约: 同步调用。 返回 `bool`。
        """
        return self._pending_presentations.get(proposal.command_id) == proposal

    def cancel_presentation(self, command_id: CommandId) -> None:
        """函数契约说明.

        功能: 执行 cancel_presentation
        的同步逻辑,并协调 pop。
        参数: self 表示当前实例。 command_id:
        CommandId。 必填。
        契约: 同步调用。 返回 `None`。
        """
        _ = self._pending_presentations.pop(command_id, None)

    def reduce_mcp_result(
        self,
        result: TaskResult,
        *,
        task_reducer: TaskResultReducer,
        now_ms: int,
        data_snapshot: TaskStateSnapshot | None = None,
    ) -> TaskReductionResult:
        """函数契约说明.

        功能: 执行 reduce_mcp_result
        的同步逻辑,并协调 reduce。
        参数: self 表示当前实例。 result:
        TaskResult。 必填。 task_reducer:
        TaskResultReducer。 必填。 now_ms:
        int。 必填。 data_snapshot:
        TaskStateSnapshot | None。 可省略。
        契约: 同步调用。 返回
        `TaskReductionResult`。
        """
        return task_reducer.reduce(
            result,
            snapshot=self._scheduler.snapshot,
            now_ms=now_ms,
            data_snapshot=data_snapshot,
        )
