"""模块契约说明.

职责: 提供 orchestrator.sessions
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import NewType

from orchestrator.ids import SessionId, TraceId, TurnId

StateRevision = NewType("StateRevision", int)

EventSequence = NewType("EventSequence", int)


@dataclass(frozen=True, slots=True)
class EventCorrelation:
    """类契约说明.

    职责: 保存 EventCorrelation
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段:
    trace_id、session_id、sequence。
    """

    trace_id: TraceId

    session_id: SessionId

    sequence: EventSequence


@dataclass(frozen=True, slots=True)
class SchedulerEvent:
    """类契约说明.

    职责: 保存 SchedulerEvent
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: event_type、correlation。
    """

    event_type: str

    correlation: EventCorrelation


@dataclass(frozen=True, slots=True)
class StartTurn:
    """类契约说明.

    职责: 保存 StartTurn
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: expected_revision、event。
    """

    expected_revision: StateRevision

    event: SchedulerEvent


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    """类契约说明.

    职责: 保存 SessionSnapshot
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段:
    session_id、revision、active_turn_id。
    """

    session_id: SessionId

    revision: StateRevision

    active_turn_id: TurnId | None


@dataclass(frozen=True, slots=True)
class AcceptedEvent:
    """类契约说明.

    职责: 保存 AcceptedEvent
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: event、turn_id。
    """

    event: SchedulerEvent

    turn_id: TurnId


@unique
class TransitionRejection(StrEnum):
    """类契约说明.

    职责: 定义 TransitionRejection
    的状态、行为和对外协作边界。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """

    STALE_REVISION = "stale_revision"

    SESSION_MISMATCH = "session_mismatch"


@dataclass(frozen=True, slots=True)
class TransitionAccepted:
    """类契约说明.

    职责: 保存 TransitionAccepted
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: snapshot、accepted_event。
    """

    snapshot: SessionSnapshot

    accepted_event: AcceptedEvent


@dataclass(frozen=True, slots=True)
class TransitionRejected:
    """类契约说明.

    职责: 保存 TransitionRejected
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: snapshot、reason。
    """

    snapshot: SessionSnapshot

    reason: TransitionRejection


type TransitionResult = TransitionAccepted | TransitionRejected


@dataclass(frozen=True, slots=True)
class Session:
    """类契约说明.

    职责: 保存 Session 不可变数据结构,用类型标注表达字段契约。
    契约: 字段: session_id。
    """

    session_id: SessionId


class SessionManager:
    """类契约说明.

    职责: 定义 SessionManager 的状态、行为和对外协作边界。
    契约: 方法: __init__、create_session。
    """

    def __init__(self, *, session_id_prefix: str) -> None:
        """函数契约说明.

        功能: 初始化 SessionManager
        的字段并建立实例不变式。
        参数: self 表示当前实例。
        session_id_prefix: str。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._session_id_prefix: str = session_id_prefix

        self._next_seq: int = 1

    def create_session(self) -> Session:
        """函数契约说明.

        功能: 执行 create_session 的同步逻辑,并协调
        Session, SessionId。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `Session`。
        """
        session = Session(
            session_id=SessionId(f"{self._session_id_prefix}-{self._next_seq:04d}"),
        )

        self._next_seq += 1

        return session


class SessionScheduler:
    """类契约说明.

    职责: 定义 SessionScheduler
    的状态、行为和对外协作边界。
    契约: 方法: __init__、snapshot、event_hist
    ory、apply。
    """

    def __init__(self, *, session_id: SessionId, turn_id_prefix: str) -> None:
        """函数契约说明.

        功能: 初始化 SessionScheduler
        的字段并建立实例不变式。
        参数: self 表示当前实例。 session_id:
        SessionId。 必填。 turn_id_prefix:
        str。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._turn_id_prefix: str = turn_id_prefix

        self._turn_sequence: int = 0

        self._snapshot: SessionSnapshot = SessionSnapshot(
            session_id=session_id,
            revision=StateRevision(0),
            active_turn_id=None,
        )

        self._event_history: list[AcceptedEvent] = []

    @property
    def snapshot(self) -> SessionSnapshot:
        """函数契约说明.

        功能: 执行 snapshot 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `SessionSnapshot`。
        """
        return self._snapshot

    @property
    def event_history(self) -> tuple[AcceptedEvent, ...]:
        """函数契约说明.

        功能: 执行 event_history 的同步逻辑,并协调
        tuple。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `tuple[AcceptedEvent, ...]`。
        """
        return tuple(self._event_history)

    def apply(self, transition: StartTurn) -> TransitionResult:
        """函数契约说明.

        功能: 执行 apply 的同步逻辑,并协调 TurnId,
        AcceptedEvent, SessionSnapshot,
        append。
        参数: self 表示当前实例。 transition:
        StartTurn。 必填。
        契约: 同步调用。 返回 `TransitionResult`。
        """
        if transition.expected_revision != self._snapshot.revision:
            return TransitionRejected(
                snapshot=self._snapshot,
                reason=TransitionRejection.STALE_REVISION,
            )

        if transition.event.correlation.session_id != self._snapshot.session_id:
            return TransitionRejected(
                snapshot=self._snapshot,
                reason=TransitionRejection.SESSION_MISMATCH,
            )

        self._turn_sequence += 1

        turn_id = TurnId(f"{self._turn_id_prefix}-{self._turn_sequence:04d}")

        accepted_event = AcceptedEvent(event=transition.event, turn_id=turn_id)

        snapshot = SessionSnapshot(
            session_id=self._snapshot.session_id,
            revision=StateRevision(self._snapshot.revision + 1),
            active_turn_id=turn_id,
        )

        self._event_history.append(accepted_event)

        self._snapshot = snapshot

        return TransitionAccepted(snapshot=snapshot, accepted_event=accepted_event)
