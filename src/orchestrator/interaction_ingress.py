"""模块契约说明.

职责: 提供 orchestrator.interaction_ingress
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from dataclasses import dataclass, field
from hashlib import sha256
from os import environ
from pathlib import Path

from orchestrator.ids import SessionId, TraceId
from orchestrator.interactions import (
    ActionCapabilityRegistry,
    ActionProposal,
    CommentProposal,
    InteractionAccepted,
    InteractionRejection,
    McpCapability,
    McpDispatchAccepted,
    McpDispatchProposal,
    McpDispatchRejected,
    PresentationCommand,
    PresentationResult,
    SessionInteractionReducer,
)
from orchestrator.json_boundary import JsonBoundaryError, parse_json_value
from orchestrator.memory_store import JsonMemoryStore
from orchestrator.profile_store import JsonVoiceProfileStore
from orchestrator.retrieval import RetrievalFixtureProvider
from orchestrator.session_data import ProfilePersistence, SessionDataState
from orchestrator.sessions import EventCorrelation, EventSequence, SessionScheduler
from orchestrator.voice_profile_service import VoiceProfileService


@dataclass(frozen=True, slots=True)
class SessionInteractionIngress:
    """类契约说明.

    职责: 保存 SessionInteractionIngress
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: data、profiles、reducer、_consu
    med_correlations。 方法: create、receive
    _comment、receive_action、receive_pres
    entation、receive_presentation_result
    、receive_mcp。
    """

    data: SessionDataState

    profiles: VoiceProfileService

    reducer: SessionInteractionReducer

    _consumed_correlations: set[EventCorrelation] = field(default_factory=set)

    @classmethod
    def create(cls, scheduler: SessionScheduler) -> "SessionInteractionIngress":
        """函数契约说明.

        功能: 执行 create 的同步逻辑,并协调
        session_storage_root, create,
        cls, RetrievalFixtureProvider。
        参数: cls 表示当前类。 scheduler:
        SessionScheduler。 必填。
        契约: 同步调用。 返回
        `'SessionInteractionIngress'`。
        """
        session_root = session_storage_root(scheduler.snapshot.session_id)

        data = SessionDataState.create(
            session_id=scheduler.snapshot.session_id,
            retrieval=RetrievalFixtureProvider(refs=()),
            memory_store=JsonMemoryStore(session_root / "memory.json"),
            profile_persistence=ProfilePersistence(
                store=JsonVoiceProfileStore(session_root / "voice-profiles.json"),
                vault_directory=session_root / "voice-templates",
            ),
        )

        return cls(
            data=data,
            profiles=data.profiles,
            reducer=SessionInteractionReducer(
                scheduler=scheduler,
                actions=ActionCapabilityRegistry(
                    frozenset({"breathe", "dance", "explain_point", "speak"})
                ),
                mcp_capabilities=frozenset({McpCapability.PRESENTATION_DECK}),
            ),
        )

    def receive_comment(
        self,
        *,
        text: str,
        correlation: EventCorrelation,
    ) -> InteractionAccepted | InteractionRejection:
        """函数契约说明.

        功能: 执行 receive_comment 的同步逻辑,并协调
        reduce_comment, CommentProposal。
        参数: self 表示当前实例。 text: str。 必填。
        correlation: EventCorrelation。
        必填。
        契约: 同步调用。 返回
        `InteractionAccepted |
        InteractionRejection`。
        """
        return self.reducer.reduce_comment(CommentProposal(text, correlation))

    def receive_action(
        self,
        proposal: ActionProposal,
    ) -> InteractionAccepted | InteractionRejection:
        """函数契约说明.

        功能: 执行 receive_action 的同步逻辑,并协调
        reduce_action。
        参数: self 表示当前实例。 proposal:
        ActionProposal。 必填。
        契约: 同步调用。 返回
        `InteractionAccepted |
        InteractionRejection`。
        """
        return self.reducer.reduce_action(proposal)

    def receive_presentation(
        self,
        proposal: PresentationCommand,
    ) -> InteractionAccepted | InteractionRejection:
        """函数契约说明.

        功能: 执行 receive_presentation
        的同步逻辑,并协调 reduce_presentation。
        参数: self 表示当前实例。 proposal:
        PresentationCommand。 必填。
        契约: 同步调用。 返回
        `InteractionAccepted |
        InteractionRejection`。
        """
        return self.reducer.reduce_presentation(proposal)

    def receive_presentation_result(
        self,
        result: PresentationResult,
    ) -> InteractionAccepted | InteractionRejection:
        """函数契约说明.

        功能: 执行
        receive_presentation_result
        的同步逻辑,并协调
        reduce_presentation_result。
        参数: self 表示当前实例。 result:
        PresentationResult。 必填。
        契约: 同步调用。 返回
        `InteractionAccepted |
        InteractionRejection`。
        """
        return self.reducer.reduce_presentation_result(result)

    def receive_mcp(
        self,
        proposal: McpDispatchProposal,
    ) -> McpDispatchAccepted | McpDispatchRejected:
        """函数契约说明.

        功能: 执行 receive_mcp 的同步逻辑,并协调
        reduce_mcp。
        参数: self 表示当前实例。 proposal:
        McpDispatchProposal。 必填。
        契约: 同步调用。 返回
        `McpDispatchAccepted |
        McpDispatchRejected`。
        """
        return self.reducer.reduce_mcp(proposal)

    def receive_control(self, raw_message: str) -> bool:
        """函数契约说明.

        功能: 执行 receive_control 的同步逻辑,并协调
        parse_comment_proposal,
        receive_comment, isinstance,
        add。
        参数: self 表示当前实例。 raw_message:
        str。 必填。
        契约: 同步调用。 返回 `bool`。
        """
        proposal = parse_comment_proposal(raw_message)

        if proposal is None:
            return False

        correlation = proposal.correlation

        if correlation in self._consumed_correlations:
            return True

        outcome = self.receive_comment(
            text=proposal.text,
            correlation=correlation,
        )

        if isinstance(outcome, InteractionAccepted):
            self._consumed_correlations.add(correlation)

        return True


def _state_root() -> Path:
    """函数契约说明.

    功能: 执行 _state_root 的同步逻辑,并协调 Path,
    get。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `Path`。
    """
    return Path(environ.get("ORCHESTRATOR_STATE_DIR", ".orchestrator-state"))


def session_storage_root(session_id: SessionId) -> Path:
    """函数契约说明.

    功能: 执行 session_storage_root
    的同步逻辑,并协调 hexdigest, _state_root,
    sha256, encode。
    参数: session_id: SessionId。 必填。
    契约: 同步调用。 返回 `Path`。
    """
    storage_key = sha256(str(session_id).encode()).hexdigest()

    return _state_root() / storage_key


def parse_comment_proposal(raw_message: str) -> CommentProposal | None:
    """函数契约说明.

    功能: 从边界输入解析类型化值。
    参数: raw_message: str。 必填。
    契约: 同步调用。 返回 `CommentProposal |
    None`。
    """
    try:
        value = parse_json_value(raw_message)

    except JsonBoundaryError:
        return None

    if not isinstance(value, dict):
        return None

    if value.get("event_type") != "audience.input" or value.get("source") != "comments":
        return None

    data = value.get("data")

    if not isinstance(data, dict):
        return None

    text = data.get("text")

    trace_id = value.get("trace_id")

    session_id = value.get("session_id")

    sequence = value.get("seq")

    if (
        not isinstance(text, str)
        or text.strip() == ""
        or not isinstance(trace_id, str)
        or not isinstance(session_id, str)
        or type(sequence) is not int
        or sequence < 0
    ):
        return None

    return CommentProposal(
        text=text,
        correlation=EventCorrelation(
            trace_id=TraceId(trace_id),
            session_id=SessionId(session_id),
            sequence=EventSequence(sequence),
        ),
    )
