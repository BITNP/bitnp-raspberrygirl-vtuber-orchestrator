"""模块契约说明.

职责: 提供 orchestrator.pipeline_contracts
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from dataclasses import dataclass
from typing import Literal

from orchestrator.ids import SegmentId, TurnId


@dataclass(frozen=True, slots=True)
class CommentAudienceEvent:
    """类契约说明.

    职责: 保存 CommentAudienceEvent
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段:
    platform、source、user、text、timestamp。
    """

    platform: str

    source: str

    user: str

    text: str

    timestamp: str


@dataclass(frozen=True, slots=True)
class ASRAudienceEvent:
    """类契约说明.

    职责: 保存 ASRAudienceEvent
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段:
    text、received_at_ms、segment_id、seq。
    """

    text: str

    received_at_ms: int

    segment_id: str

    seq: int


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """类契约说明.

    职责: 保存 PipelineConfig
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: queue_capacity、turn_id_prefi
    x、segment_id_prefix。
    """

    queue_capacity: int

    turn_id_prefix: str

    segment_id_prefix: str


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    """类契约说明.

    职责: 保存 AudioMetadata
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: sample_rate、channels、codec、d
    uration_ms、byte_length。
    """

    sample_rate: int

    channels: int

    codec: str

    duration_ms: int

    byte_length: int


@dataclass(frozen=True, slots=True)
class MediaStreamCommand:
    """类契约说明.

    职责: 保存 MediaStreamCommand
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: turn_id、segment_id、stream_id
    、audio、start_at_ms、event_type。
    """

    turn_id: TurnId

    segment_id: SegmentId

    stream_id: str

    audio: AudioMetadata

    start_at_ms: int

    event_type: Literal["media.stream.command"] = "media.stream.command"


@dataclass(frozen=True, slots=True)
class MediaStreamState:
    """类契约说明.

    职责: 保存 MediaStreamState
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: turn_id、segment_id、stream_id
    、state、playback_position_ms、event_ty
    pe。
    """

    turn_id: TurnId

    segment_id: SegmentId

    stream_id: str

    state: Literal["queued", "playing", "finished", "cancelled"]

    playback_position_ms: int

    event_type: Literal["media.stream.state"] = "media.stream.state"


@dataclass(frozen=True, slots=True)
class VtuberCaptionCommand:
    """类契约说明.

    职责: 保存 VtuberCaptionCommand
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: turn_id、segment_id、text、star
    t_at_ms、event_type。
    """

    turn_id: TurnId

    segment_id: SegmentId

    text: str

    start_at_ms: int = 0

    event_type: Literal["vtuber.caption.command"] = "vtuber.caption.command"


@dataclass(frozen=True, slots=True)
class VtuberActionCommand:
    """类契约说明.

    职责: 保存 VtuberActionCommand
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: turn_id、segment_id、action、st
    art_at_ms、event_type。
    """

    turn_id: TurnId

    segment_id: SegmentId

    action: str

    start_at_ms: int = 0

    event_type: Literal["vtuber.action.command"] = "vtuber.action.command"


@dataclass(frozen=True, slots=True)
class VtuberExpressionCommand:
    """类契约说明.

    职责: 保存 VtuberExpressionCommand
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: turn_id、segment_id、expressio
    n、start_at_ms、event_type。
    """

    turn_id: TurnId

    segment_id: SegmentId

    expression: str

    start_at_ms: int = 0

    event_type: Literal["vtuber.expression.command"] = "vtuber.expression.command"


@dataclass(frozen=True, slots=True)
class VtuberSceneCommand:
    """类契约说明.

    职责: 保存 VtuberSceneCommand
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: turn_id、segment_id、scene、sli
    de_id、slide_title、slide_page。
    """

    turn_id: TurnId

    segment_id: SegmentId

    scene: str

    slide_id: str

    slide_title: str

    slide_page: int = 1

    start_at_ms: int = 0

    event_type: Literal["vtuber.scene.command"] = "vtuber.scene.command"


@dataclass(frozen=True, slots=True)
class MockSynthesisResult:
    """类契约说明.

    职责: 保存 MockSynthesisResult
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: turn_id、segment_id、audio、exp
    ression、action、scene。
    """

    turn_id: TurnId

    segment_id: SegmentId

    audio: AudioMetadata | None

    expression: str

    action: str

    scene: str

    slide_page: int

    offset_samples: int = 0


@dataclass(frozen=True, slots=True)
class SynthesisCueResult:
    """类契约说明.

    职责: 保存 SynthesisCueResult
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: media、caption、expression、act
    ion、scene。
    """

    media: MediaStreamCommand | None

    caption: VtuberCaptionCommand

    expression: VtuberExpressionCommand

    action: VtuberActionCommand

    scene: VtuberSceneCommand


@dataclass(frozen=True, slots=True)
class CancelCommand:
    """类契约说明.

    职责: 保存 CancelCommand
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: turn_id、segment_id、target、re
    ason、event_type。
    """

    turn_id: TurnId

    segment_id: SegmentId

    target: Literal["llm", "media_stream", "frontend"]

    reason: str

    event_type: Literal["cancel"] = "cancel"


@dataclass(frozen=True, slots=True)
class TurnResult:
    """类契约说明.

    职责: 保存 TurnResult
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: turn_id、segment_id、answer_te
    xt、used_fallback。
    """

    turn_id: TurnId

    segment_id: SegmentId

    answer_text: str

    used_fallback: bool


type AudienceEvent = CommentAudienceEvent | ASRAudienceEvent
