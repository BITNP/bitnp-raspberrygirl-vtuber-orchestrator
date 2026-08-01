from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Protocol, cast

from orchestrator.ids import SegmentId, TurnId
from orchestrator.llm import (
    CancellationToken,
    LLMAdapter,
    LLMChunk,
    LLMError,
    LLMFinal,
    LLMStreamEvent,
    build_llm_request,
)
from orchestrator.modes import AnswerCandidate, AudienceInput, AudienceSource
from orchestrator.pipeline_contracts import (
    ASRAudienceEvent,
    AudienceEvent,
    CancelCommand,
    CommentAudienceEvent,
    MediaStreamCommand,
    MockSynthesisResult,
    PipelineConfig,
    SynthesisCueResult,
    TurnResult,
    VtuberActionCommand,
    VtuberCaptionCommand,
    VtuberExpressionCommand,
    VtuberSceneCommand,
)
from orchestrator.retrieval import RetrievalProvider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class AnswerPolicy(Protocol):
    def select_answer_candidate(
        self,
        audience_inputs: tuple[AudienceInput, ...],
    ) -> AnswerCandidate | None: ...


@dataclass(frozen=True, slots=True)
class PipelineAdapters:
    mode_policy: AnswerPolicy

    llm: LLMAdapter

    retrieval: RetrievalProvider


class OrchestratorTurnPipeline:
    def __init__(
        self,
        *,
        adapters: PipelineAdapters,
        config: PipelineConfig,
    ) -> None:
        self._mode_policy: AnswerPolicy = adapters.mode_policy

        self._llm: LLMAdapter = adapters.llm

        self._retrieval: RetrievalProvider = adapters.retrieval

        self._queue_capacity: int = config.queue_capacity

        self._turn_id_prefix: str = config.turn_id_prefix

        self._segment_id_prefix: str = config.segment_id_prefix

        self._queue: deque[AudienceEvent] = deque()

        self._turn_seq: int = 0

        self._active: _ActiveTurn | None = None

        self._stale_segments: set[SegmentId] = set()

        self._rejections: list[str] = []

        self._cancel_commands: list[CancelCommand] = []

    @property
    def rejections(self) -> tuple[str, ...]:
        return tuple(self._rejections)

    @property
    def cancel_commands(self) -> tuple[CancelCommand, ...]:
        return tuple(self._cancel_commands)

    def accept_audience_input(self, event: AudienceEvent) -> bool:
        if self._active is not None:
            self._cancel_active(reason="user_interrupt")

        if len(self._queue) >= self._queue_capacity:
            self._rejections.append("queue_full")

            return False

        self._queue.append(event)

        return True

    def process_next_turn(
        self, cancellation: CancellationToken | None = None
    ) -> TurnResult | None:
        if len(self._queue) == 0:
            return None

        event = self._queue.popleft()

        audience_input = _to_audience_input(event)

        candidate = self._mode_policy.select_answer_candidate((audience_input,))

        if candidate is None:
            return None

        self._turn_seq += 1

        turn_id = TurnId(f"{self._turn_id_prefix}-{self._turn_seq:04d}")

        segment_id = SegmentId(f"{self._segment_id_prefix}-{self._turn_seq:04d}")

        token = CancellationToken() if cancellation is None else cancellation

        text_parts: list[str] = []

        final: LLMFinal | None = None

        try:
            request = build_llm_request(
                candidate,
                retrieval=self._retrieval.retrieve(candidate),
            )

            for llm_event in self._llm.stream(
                request,
                cancellation=token,
            ):
                match llm_event:
                    case LLMChunk(text=text):
                        text_parts.append(text)

                    case LLMError() as error:
                        if error.cancel_pending_media:
                            self._cancel_commands.append(
                                _cancel(
                                    _CancelIntent(
                                        turn_id,
                                        segment_id,
                                        "media_stream",
                                        "llm_timeout",
                                    ),
                                ),
                            )

                    case LLMFinal() as llm_final:
                        final = llm_final
        except OSError:
            # The bridge retries transient LLM failures once.  Returning the
            # exact event to the queue makes that retry semantically identical
            # to the original turn rather than silently turning into no work.
            self._queue.appendleft(event)
            self._turn_seq -= 1
            raise

        answer_text = final.text if final is not None else "".join(text_parts)

        self._active = _ActiveTurn(
            turn_id=turn_id,
            segment_id=segment_id,
            text=answer_text,
            cancellation=token,
        )

        return TurnResult(
            turn_id=turn_id,
            segment_id=segment_id,
            answer_text=answer_text,
            used_fallback=final.used_fallback if final is not None else False,
        )

    async def process_next_turn_async(  # noqa: C901
        self, cancellation: CancellationToken | None = None
    ) -> TurnResult | None:
        """Process a turn with an async live LLM, retaining mock compatibility."""
        if len(self._queue) == 0:
            return None
        event = self._queue.popleft()
        candidate = self._mode_policy.select_answer_candidate(
            (_to_audience_input(event),)
        )
        if candidate is None:
            return None
        self._turn_seq += 1
        turn_id = TurnId(f"{self._turn_id_prefix}-{self._turn_seq:04d}")
        segment_id = SegmentId(f"{self._segment_id_prefix}-{self._turn_seq:04d}")
        token = CancellationToken() if cancellation is None else cancellation
        text_parts: list[str] = []
        final: LLMFinal | None = None
        try:
            request = build_llm_request(
                candidate, retrieval=self._retrieval.retrieve(candidate)
            )
            stream = self._llm.stream(request, cancellation=token)
            if hasattr(stream, "__aiter__"):
                async_stream = cast(
                    "AsyncIterator[LLMStreamEvent]", cast("object", stream)
                )
                async for llm_event in async_stream:
                    if isinstance(llm_event, LLMChunk):
                        text_parts.append(llm_event.text)
                    elif isinstance(llm_event, LLMFinal):
                        final = llm_event
                    elif (
                        isinstance(llm_event, LLMError)
                        and llm_event.cancel_pending_media
                    ):
                        self._cancel_commands.append(
                            _cancel(
                                _CancelIntent(
                                    turn_id, segment_id, "media_stream", "llm_timeout"
                                )
                            )
                        )
            else:
                for llm_event in stream:
                    if isinstance(llm_event, LLMChunk):
                        text_parts.append(llm_event.text)
                    elif isinstance(llm_event, LLMFinal):
                        final = llm_event
        except OSError:
            self._queue.appendleft(event)
            self._turn_seq -= 1
            raise
        answer_text = final.text if final is not None else "".join(text_parts)
        self._active = _ActiveTurn(turn_id, segment_id, answer_text, token)
        return TurnResult(
            turn_id=turn_id,
            segment_id=segment_id,
            answer_text=answer_text,
            used_fallback=final.used_fallback if final is not None else False,
        )

    def complete_synthesis(
        self,
        synthesis: MockSynthesisResult,
        *,
        rtp_stream_start_ms: int,
        stream_id: str = "rtp-local",
    ) -> SynthesisCueResult | None:
        active = self._active

        if active is None or synthesis.segment_id in self._stale_segments:
            return None

        if (
            synthesis.turn_id != active.turn_id
            or synthesis.segment_id != active.segment_id
        ):
            return None

        if synthesis.audio is None:
            return None

        offset_ms = synthesis.offset_samples * 1_000 // synthesis.audio.sample_rate

        start_at_ms = rtp_stream_start_ms + offset_ms

        return SynthesisCueResult(
            media=MediaStreamCommand(
                turn_id=synthesis.turn_id,
                segment_id=synthesis.segment_id,
                stream_id=stream_id,
                audio=synthesis.audio,
                start_at_ms=start_at_ms,
            ),
            caption=VtuberCaptionCommand(
                turn_id=synthesis.turn_id,
                segment_id=synthesis.segment_id,
                text=active.text,
                start_at_ms=start_at_ms,
            ),
            expression=VtuberExpressionCommand(
                turn_id=synthesis.turn_id,
                segment_id=synthesis.segment_id,
                expression=synthesis.expression,
                start_at_ms=start_at_ms,
            ),
            action=VtuberActionCommand(
                turn_id=synthesis.turn_id,
                segment_id=synthesis.segment_id,
                action=synthesis.action,
                start_at_ms=start_at_ms,
            ),
            scene=VtuberSceneCommand(
                turn_id=synthesis.turn_id,
                segment_id=synthesis.segment_id,
                scene=synthesis.scene,
                slide_id="",
                slide_title="",
                slide_page=synthesis.slide_page,
                start_at_ms=start_at_ms,
            ),
        )

    def _cancel_active(self, *, reason: str) -> None:
        active = self._active

        if active is None:
            return

        _ = active.cancellation.cancel(reason=reason)

        self._stale_segments.add(active.segment_id)

        self._cancel_commands.extend(
            (
                _cancel(
                    _CancelIntent(active.turn_id, active.segment_id, target, reason),
                )
                for target in ("media_stream", "frontend")
            ),
        )

        self._active = None


@dataclass(frozen=True, slots=True)
class _ActiveTurn:
    turn_id: TurnId

    segment_id: SegmentId

    text: str

    cancellation: CancellationToken


@dataclass(frozen=True, slots=True)
class _CancelIntent:
    turn_id: TurnId

    segment_id: SegmentId

    target: Literal["media_stream", "frontend"]

    reason: str


def _to_audience_input(event: AudienceEvent) -> AudienceInput:
    match event:
        case CommentAudienceEvent(text=text, timestamp=timestamp):
            return AudienceInput(
                source=AudienceSource.COMMENT,
                text=text,
                received_at_ms=_timestamp_ms(timestamp),
            )

        case ASRAudienceEvent(text=text, received_at_ms=received_at_ms):
            return AudienceInput(
                source=AudienceSource.ASR,
                text=text,
                received_at_ms=received_at_ms,
            )


def _timestamp_ms(raw_timestamp: str) -> int:
    parsed = datetime.fromisoformat(raw_timestamp)

    return int(parsed.timestamp() * 1000)


def _cancel(intent: _CancelIntent) -> CancelCommand:
    match intent.target:
        case "media_stream" | "frontend":
            return CancelCommand(
                turn_id=intent.turn_id,
                segment_id=intent.segment_id,
                target=intent.target,
                reason=intent.reason,
            )
